"""v3 因子腿。四条全新来源,每条先有定价逻辑再有代码。

v1/v2 的教训:前两轮的信号本质上都是「价格水平 + 流动性」的变形,所以一做
中性化就归零,一放大规模就付不起成本。v3 换方向——找与价格水平正交的定价偏离:

  iv_rv         期权隐含波动率 vs 正股实际波动率(转债套利的核心定价关系)
  ts_val        个券估值相对自身历史(时序均值回归,不是截面比价)
  call_state    强赎条款状态(发行人期权的行权风险,事件驱动)
  issuer_q      发行人财务质量(债底可靠性)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cb_miner_v2 import cs_rank, cs_z, roll_by  # noqa: E402
from tushare_data import DATA_DIR  # noqa: E402

CB_DIR = DATA_DIR / "cb"
RISK_FREE = 0.02


# ---------------------------------------------------------------------------
# Black-Scholes 与隐含波动率反解
# ---------------------------------------------------------------------------
def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26,精度 1.5e-7,足够反解 IV。"""
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def bs_call(S, K, T, sigma, r=RISK_FREE):
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
        d2 = d1 - sq
        return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)


def implied_vol(price, S, K, T, lo=0.02, hi=2.0, iters=40):
    """向量化二分反解。转债 moneyness 跨度极大(转股价值 26~331),
    用 ATM 近似 sigma≈C/(0.4·S·√T) 会在深度实值/虚值处失真,必须真反解。"""
    lo = np.full_like(price, lo, dtype=float)
    hi = np.full_like(price, hi, dtype=float)
    ok = np.isfinite(price) & (price > 0) & np.isfinite(S) & (S > 0) & \
        np.isfinite(K) & (K > 0) & np.isfinite(T) & (T > 0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = bs_call(S, K, T, mid)
        up = c < price
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    out = 0.5 * (lo + hi)
    # 贴边即无解(期权价低于内在价值、或高到 200% 波动率仍不够),置 NaN 而非留假值
    edge = (out < 0.025) | (out > 1.95)
    return np.where(ok & ~edge, out, np.nan)


# ---------------------------------------------------------------------------
# 因子腿
# ---------------------------------------------------------------------------
def leg_iv_rv(p: pd.DataFrame, rv_window: int = 60) -> pd.Series:
    """期权贵贱:隐含波动率减正股实际波动率,越低越便宜。

    转债 = 纯债 + 看涨期权。把纯债价值剥掉得到期权价格,按转股比例折成每股,
    再用 BS 反解 IV,与正股已实现波动率比较。这是可转债套利最核心的定价关系,
    也是与「价格水平」最正交的一条——便宜的期权可以出现在任何价位的券上。

    补偿的风险:承担 gamma 的对手方要为条款不确定性(下修/强赎)与转债流动性定价。
    """
    opt_per_bond = p["b_close"] - p["debt_puredebt_ratio"]
    conv_ratio = p["conv_ratio"].replace(0, np.nan)
    opt_per_share = opt_per_bond / conv_ratio
    S = p["s_close"].to_numpy()
    K = p["conv_price"].to_numpy()
    T = p["year_to_mat"].clip(lower=0.1).to_numpy()
    iv = implied_vol(opt_per_share.to_numpy(), S, K, T)
    rv = roll_by(p["s_log_ret"], p["b_sym"], rv_window, "std") * np.sqrt(252)
    spread = pd.Series(iv, index=p.index) - rv
    return -cs_rank(spread, p["date"])


def leg_ts_val(p: pd.DataFrame, window: int = 250) -> pd.Series:
    """时序估值偏离:转股溢价率相对该券自身 250 日历史的位置,越低越优。

    与截面比价的关键差别:不同券的合理溢价率本就不同(取决于正股波动率、
    剩余期限、条款)。跟自己比,把这些个券固定效应差掉,因而不会像截面双低
    那样退化成「买低价券」——v2 的中性化诊断显示后者一剥就归零。

    补偿的风险:估值均值回归需要时间,期间承担条款与信用风险。
    """
    prem = p["conv_prem"]
    mu = roll_by(prem, p["b_sym"], window, "mean")
    sd = roll_by(prem, p["b_sym"], window, "std")
    z = (prem - mu) / sd.replace(0, np.nan)
    return -cs_rank(z, p["date"])


def leg_call_state(p: pd.DataFrame) -> pd.Series:
    """强赎条款状态。发行人手里的赎回权是投资者的空头期权。

      已满足强赎条件但未表态  -> 悬顶未决,期权价值被压制,负向
      公告不强赎              -> 悬顶解除,期权价值恢复,正向(通常给 3~6 个月豁免期)

    事件驱动、状态持续,天然低换手。用公告日单向掩码,不含未来信息。
    """
    call = pd.read_parquet(CB_DIR / "cb_call.parquet")
    call = call.rename(columns={"ts_code": "b_sym"})
    call["ann_date"] = pd.to_datetime(call["ann_date"], errors="coerce")
    call = call.dropna(subset=["ann_date"])

    score = pd.Series(0.0, index=p.index)
    key = pd.MultiIndex.from_arrays([p["b_sym"], p["date"]])

    for label, val, decay in (("公告不强赎", 1.0, 120), ("已满足强赎条件", -1.0, 60)):
        ev = call[call["is_call"].astype(str).str.contains(label, na=False)]
        ev = ev[["b_sym", "ann_date"]].dropna().drop_duplicates()
        if not len(ev):
            continue
        ev = ev.sort_values("ann_date")
        # 每只券可能多次公告,取每个日期之前最近一次
        m = pd.Series(np.nan, index=key)
        tmp = pd.DataFrame({"b_sym": p["b_sym"].values, "date": p["date"].values})
        tmp = tmp.merge(ev.assign(_e=ev["ann_date"]), on="b_sym", how="left")
        tmp = tmp[tmp["_e"] <= tmp["date"]]
        if not len(tmp):
            continue
        last = tmp.groupby(["b_sym", "date"])["_e"].max()
        age = (pd.Series(p["date"].values, index=key)
               - last.reindex(key)).dt.days
        w = np.exp(-age.to_numpy() / decay)
        score = score + np.nan_to_num(val * w)

    return cs_rank(pd.Series(score.values, index=p.index), p["date"])


def leg_issuer_q(p: pd.DataFrame) -> pd.Series:
    """发行人财务质量。转债的债底只有在发行人不违约时才成立。

    2024 年信用事件后,低价转债的定价里信用风险占比显著上升,债底不再是无风险的。
    用 PIT 财务(ann_date 对齐)构造 ROE + 低杠杆 + 盈利稳定的综合分。

    补偿的风险:违约与退市风险。
    """
    fin_dir = DATA_DIR / "fina_indicator_vip"
    files = sorted(fin_dir.glob("*.parquet"))
    if not files:
        return pd.Series(np.nan, index=p.index)
    cols = ["ts_code", "ann_date", "end_date", "roe", "debt_to_assets", "netprofit_margin"]
    frames = []
    for f in files:
        try:
            d = pd.read_parquet(f, columns=cols)
        except Exception:  # noqa: BLE001
            continue
        frames.append(d)
    fin = pd.concat(frames, ignore_index=True)
    fin["ann_date"] = pd.to_datetime(fin["ann_date"], errors="coerce")
    fin = fin.dropna(subset=["ann_date"]).sort_values(["ts_code", "ann_date"])
    fin = fin.drop_duplicates(subset=["ts_code", "ann_date"], keep="last")
    fin = fin.rename(columns={"ts_code": "stk_code"})

    # 按公告日 merge_asof,严格取「公告日 <= 当日」的最近一期,无未来信息
    left = p[["date", "stk_code"]].copy().reset_index().sort_values("date")
    right = fin[["ann_date", "stk_code", "roe", "debt_to_assets", "netprofit_margin"]].sort_values("ann_date")
    merged = pd.merge_asof(left, right, left_on="date", right_on="ann_date",
                           by="stk_code", direction="backward")
    merged = merged.set_index("index").reindex(p.index)

    q = (cs_rank(merged["roe"], p["date"])
         - cs_rank(merged["debt_to_assets"], p["date"])
         + cs_rank(merged["netprofit_margin"], p["date"])) / 3.0
    return q


LEGS_V3 = {
    "iv_rv": leg_iv_rv,
    "ts_val": leg_ts_val,
    "call_state": leg_call_state,
    "issuer_q": leg_issuer_q,
}


def build_v3(p: pd.DataFrame, clip: float = 2.5) -> pd.DataFrame:
    return pd.DataFrame({k: cs_z(fn(p), p["date"], clip=clip) for k, fn in LEGS_V3.items()},
                        index=p.index)

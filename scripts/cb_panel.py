"""转债因子面板装配 + 评价引擎。

面板口径对齐内部 alphafactory 的 cvbd_derivative,列名沿用内部命名,
使挖出的因子表达式可以直接搬到 CVBondAlphaBase 上跑。

产出 data_tushare/cb/cb_panel.parquet,长表 (date, b_sym) 索引。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tushare_data import DATA_DIR  # noqa: E402

CB_DIR = DATA_DIR / "cb"
CB_DAILY_DIR = CB_DIR / "cb_daily"
PANEL_PATH = CB_DIR / "cb_panel.parquet"

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 面板装配
# ---------------------------------------------------------------------------
def _load_cb_daily(start: str, end: str) -> pd.DataFrame:
    files = sorted(CB_DAILY_DIR.glob("*.parquet"))
    files = [f for f in files if start <= f.stem <= end]
    if not files:
        raise FileNotFoundError(f"no cb_daily in {start}~{end}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def _load_stock_daily(dates: list[str]) -> pd.DataFrame:
    """正股日线 + 后复权价。只读面板需要的日期,控制内存。"""
    rows = []
    for d in dates:
        p = DATA_DIR / "daily" / f"{d}.parquet"
        if not p.exists():
            continue
        px = pd.read_parquet(p, columns=["ts_code", "trade_date", "close", "vol", "amount", "pct_chg"])
        a = DATA_DIR / "adj_factor" / f"{d}.parquet"
        if a.exists():
            adj = pd.read_parquet(a, columns=["ts_code", "adj_factor"])
            px = px.merge(adj, on="ts_code", how="left")
        else:
            px["adj_factor"] = np.nan
        b = DATA_DIR / "daily_basic" / f"{d}.parquet"
        if b.exists():
            db = pd.read_parquet(b, columns=["ts_code", "turnover_rate", "total_mv", "circ_mv", "pe_ttm", "pb"])
            px = px.merge(db, on="ts_code", how="left")
        rows.append(px)
    st = pd.concat(rows, ignore_index=True)
    st = st.rename(columns={
        "ts_code": "stk_code", "close": "s_close", "vol": "s_vol",
        "amount": "s_amount", "pct_chg": "s_pct_chg",
        "turnover_rate": "s_turnover", "total_mv": "s_total_mv",
        "circ_mv": "s_circ_mv", "pe_ttm": "s_pe_ttm", "pb": "s_pb",
    })
    st["s_close_adj"] = st["s_close"] * st["adj_factor"]
    return st.drop(columns=["adj_factor"])


def build_panel(start: str = "20190101", end: str = "20260301") -> pd.DataFrame:
    cb = _load_cb_daily(start, end)
    cb = cb.rename(columns={
        "ts_code": "b_sym",
        "close": "b_close", "open": "b_open", "high": "b_high", "low": "b_low",
        "pre_close": "b_pre_close", "vol": "b_vol", "amount": "b_amount",
        "pct_chg": "b_pct_chg", "change": "b_change",
        # 对齐 alphafactory 命名
        "bond_value": "debt_puredebt_ratio",   # 纯债价值
        "cb_value": "conv_value",              # 转股价值
        "cb_over_rate": "bond_prem_ratio",     # 转股溢价率(%)
        "bond_over_rate": "puredebt_prem_ratio",  # 纯债溢价率(%)
    })
    cb["date"] = pd.to_datetime(cb["trade_date"])
    cb = cb.drop(columns=["trade_date"])

    basic = pd.read_parquet(CB_DIR / "cb_basic.parquet")
    keep = ["ts_code", "stk_code", "bond_short_name", "list_date", "delist_date",
            "maturity_date", "issue_size", "remain_size", "coupon_rate",
            "conv_start_date", "conv_end_date", "maturity_put_price"]
    basic = basic[[c for c in keep if c in basic.columns]].rename(columns={"ts_code": "b_sym"})
    cb = cb.merge(basic, on="b_sym", how="left")

    dates = sorted(cb["date"].dt.strftime("%Y%m%d").unique().tolist())
    st = _load_stock_daily(dates)
    st["date"] = pd.to_datetime(st["trade_date"])
    st = st.drop(columns=["trade_date"])
    cb = cb.merge(st, on=["date", "stk_code"], how="left")

    # 强赎:公告日起该券进入执行窗口,价格被锁定,必须能在 universe 里剔除
    call = pd.read_parquet(CB_DIR / "cb_call.parquet")
    call = call[call["is_call"].astype(str).str.contains("实施|已满足|行使", na=False)]
    if len(call):
        c = call.rename(columns={"ts_code": "b_sym"})[["b_sym", "ann_date", "call_date"]]
        c = c.dropna(subset=["ann_date"]).sort_values("ann_date").drop_duplicates("b_sym", keep="first")
        c["call_ann_date"] = pd.to_datetime(c["ann_date"], errors="coerce")
        cb = cb.merge(c[["b_sym", "call_ann_date"]], on="b_sym", how="left")
    else:
        cb["call_ann_date"] = pd.NaT

    for col in ("list_date", "delist_date", "maturity_date", "conv_start_date", "conv_end_date"):
        if col in cb.columns:
            cb[col] = pd.to_datetime(cb[col], errors="coerce")

    cb = cb.sort_values(["b_sym", "date"]).reset_index(drop=True)

    # 未上市/停牌占位行 close=0(vol 也为 0)。这些行本就在 universe 之外,
    # 但留着会让 fwd_ret 出现 inf,任何漏掉二次掩码的下游脚本都会被污染。
    for col in ("b_close", "b_pre_close", "s_close", "s_close_adj"):
        cb.loc[cb[col] == 0, col] = np.nan

    # ---- 衍生字段 ----
    # 转股比例 = 100 / 转股价;由转股价值反推,天然含下修与派息调整,无需 cb_price_chg
    cb["conv_ratio"] = cb["conv_value"] / cb["s_close"].replace(0, np.nan)
    cb["conv_price"] = 100.0 / cb["conv_ratio"].replace(0, np.nan)
    cb["floor_prem"] = cb["puredebt_prem_ratio"] / 100.0     # 纯债溢价率(小数)
    cb["conv_prem"] = cb["bond_prem_ratio"] / 100.0          # 转股溢价率(小数)
    cb["dbl_low_factor"] = cb["b_close"] + cb["bond_prem_ratio"]
    cb["year_to_mat"] = (cb["maturity_date"] - cb["date"]).dt.days / 365.25
    # 近似 YTM:面值 100 + 剩余期限内票息,单利化
    cb["ytm_approx"] = ((100.0 + cb["coupon_rate"].fillna(0) * cb["year_to_mat"]) / cb["b_close"] - 1.0) \
        / cb["year_to_mat"].replace(0, np.nan)
    cb["embedded_option_price"] = (cb["b_close"] - cb["debt_puredebt_ratio"]) / cb["conv_ratio"].replace(0, np.nan)
    # 距强赎触发距离:正股价 / (1.3 * 转股价)
    cb["call_trigger_dist"] = cb["s_close"] / (1.3 * cb["conv_price"].replace(0, np.nan))
    # 距下修/回售触发距离:正股价 / (0.7 * 转股价)
    cb["put_trigger_dist"] = cb["s_close"] / (0.7 * cb["conv_price"].replace(0, np.nan))

    g = cb.groupby("b_sym", sort=False)
    cb["b_log_ret"] = np.log(cb["b_close"] / g["b_close"].shift(1))
    cb["s_log_ret"] = np.log(cb["s_close_adj"] / g["s_close_adj"].shift(1))
    cb["stock_volatility"] = g["s_log_ret"].transform(lambda x: x.rolling(30, min_periods=15).std())
    cb["bond_volatility"] = g["b_log_ret"].transform(lambda x: x.rolling(30, min_periods=15).std())
    cb["b_turnover"] = cb["b_amount"] / (cb["remain_size"].replace(0, np.nan) * 1000.0)
    cb["days_listed"] = (cb["date"] - cb["list_date"]).dt.days

    # 前视收益:T 日信号吃 T->T+1 的转债收益
    cb["fwd_ret"] = g["b_close"].shift(-1) / cb["b_close"] - 1.0

    return cb


# ---------------------------------------------------------------------------
# Universe —— 内部 univ_AA 的本地代理
# ---------------------------------------------------------------------------
def add_universe(df: pd.DataFrame, min_days: int = 20, min_amount: float = 500.0,
                 exclude_call: bool = True) -> pd.DataFrame:
    """min_amount 单位千元(tushare amount 口径),500 千元 = 50 万元。"""
    ok = (
        (df["days_listed"] >= min_days)
        & (df["b_vol"].fillna(0) > 0)
        & (df["b_amount"].fillna(0) >= min_amount)
        & df["b_close"].notna()
        & df["s_close"].notna()
        & df["conv_value"].notna()
        & df["debt_puredebt_ratio"].notna()
        & (df["year_to_mat"] > 0.25)          # 临到期券价格行为退化为纯债
    )
    if exclude_call:
        # 已公告强赎:进入执行窗口后价格被锁死,留在池子里会污染任何估值类因子
        ok &= ~(df["call_ann_date"].notna() & (df["date"] >= df["call_ann_date"]))
    df = df.copy()
    df["in_univ"] = ok
    return df


# ---------------------------------------------------------------------------
# 评价引擎
# ---------------------------------------------------------------------------
def cs_rank(s: pd.Series, min_count: int = 10) -> pd.Series:
    """截面 rank 标准化,与示例代码 _cs_rank 一致。"""
    if s.notna().sum() < min_count:
        return pd.Series(np.nan, index=s.index)
    return (s.rank(pct=True) - 0.5) * np.sqrt(12.0)


def cs_zscore(s: pd.Series, eps: float = 0.0004) -> pd.Series:
    """截面 z-score,与示例代码 _cal_sd 一致(分母加 eps 防爆)。"""
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / (sd + eps)


def evaluate(panel: pd.DataFrame, alpha: pd.Series, roll_days: int = 1,
             start: str | None = None, end: str | None = None) -> dict:
    """alpha: 与 panel 同索引的原始因子值(越大越看多)。

    流程复刻示例:截面 rank/原值 -> 按券 roll_mean -> 截面 z-score -> 打分。
    """
    df = panel[["date", "b_sym", "fwd_ret", "in_univ"]].copy()
    df["raw"] = alpha.values if isinstance(alpha, pd.Series) else alpha
    df.loc[~df["in_univ"], "raw"] = np.nan

    df = df.sort_values(["b_sym", "date"])
    if roll_days > 1:
        df["raw"] = df.groupby("b_sym", sort=False)["raw"].transform(
            lambda x: x.rolling(roll_days, min_periods=1).mean())

    df["sig"] = df.groupby("date", sort=False)["raw"].transform(cs_zscore)
    df.loc[~df["in_univ"], "sig"] = np.nan

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    df = df.dropna(subset=["sig"])
    if df.empty:
        return {"n_days": 0}

    # --- IC ---
    def _ic(gp):
        v = gp.dropna(subset=["fwd_ret"])
        if len(v) < 10:
            return np.nan
        return v["sig"].corr(v["fwd_ret"])

    def _ric(gp):
        v = gp.dropna(subset=["fwd_ret"])
        if len(v) < 10:
            return np.nan
        return v["sig"].corr(v["fwd_ret"], method="spearman")

    ic = df.groupby("date").apply(_ic, include_groups=False).dropna()
    ric = df.groupby("date").apply(_ric, include_groups=False).dropna()

    # --- 权重与 PnL ---
    df["w"] = df["sig"] / df.groupby("date")["sig"].transform(lambda x: x.abs().sum())
    df["pnl"] = df["w"] * df["fwd_ret"]
    pnl = df.groupby("date")["pnl"].sum()

    # --- turnover ---
    # w 已按 Σ|w|=1 归一,故 Σ|Δw|/2 即「单边换手占账面比例」,这是闸门 <0.2 的口径。
    # 同时报 gross(双边)供换算,避免两边对不上。
    wide = df.pivot_table(index="date", columns="b_sym", values="w").fillna(0.0)
    dw = wide.diff().abs().sum(axis=1)
    gross = wide.abs().sum(axis=1)
    turnover_gross = (dw / gross.replace(0, np.nan)).iloc[1:]

    breadth = df.groupby("date").size()

    return {
        "n_days": int(len(pnl)),
        "breadth": float(breadth.mean()),
        "IC": float(ic.mean()),
        "RankIC": float(ric.mean()),
        "IC_std": float(ic.std()),
        "ICIR": float(ic.mean() / ic.std() * np.sqrt(TRADING_DAYS)) if ic.std() > 0 else np.nan,
        "RankICIR": float(ric.mean() / ric.std() * np.sqrt(TRADING_DAYS)) if ric.std() > 0 else np.nan,
        "IC_pos_rate": float((ic > 0).mean()),
        "Sharpe": float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS)) if pnl.std() > 0 else np.nan,
        "AnnRet": float(pnl.mean() * TRADING_DAYS),
        "MaxDD": float((pnl.cumsum().cummax() - pnl.cumsum()).max()),
        "turnover": float(turnover_gross.mean() / 2.0),   # 单边,闸门口径
        "turnover_gross": float(turnover_gross.mean()),   # 双边 Σ|Δw|
    }


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--start", default="20190101")
    ap.add_argument("--end", default="20260301")
    args = ap.parse_args()

    if args.build:
        p = build_panel(args.start, args.end)
        p = add_universe(p)
        p.to_parquet(PANEL_PATH, index=False)
        print(f"panel: {p.shape} -> {PANEL_PATH} ({PANEL_PATH.stat().st_size / 1e6:.1f} MB)")
        u = p[p["in_univ"]]
        print(f"universe 日均 {u.groupby('date').size().mean():.0f} 只, "
              f"{u['date'].min().date()} ~ {u['date'].max().date()}")

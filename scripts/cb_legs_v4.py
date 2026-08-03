"""v4 因子腿:本身就慢的信号。

v3 的教训是结构性的——把 3 日平滑拉到 120 日,ICIR 从 2.55 掉到 1.62。
平滑一个快信号只会损失信息,不会创造持续性。要在日频下同时满足闸门与容量,
必须让腿的构造本身就是长周期的:信号自相关高 → 目标权重天然稳定 → 换手低,
而不是靠事后平滑压换手。

每条腿附带 half_life 诊断(信号自相关衰减到 0.5 所需天数)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cb_legs_v3 import implied_vol  # noqa: E402
from cb_miner_v2 import cs_rank, roll_by  # noqa: E402


def _bond_iv(p: pd.DataFrame) -> pd.Series:
    """从转债价格反解每股期权的隐含波动率。多条腿共用,只算一次。"""
    conv_ratio = (p["conv_value"] / p["s_close"].replace(0, np.nan)).replace(0, np.nan)
    opt_per_share = (p["b_close"] - p["debt_puredebt_ratio"]) / conv_ratio
    iv = implied_vol(opt_per_share.to_numpy(),
                     p["s_close"].to_numpy(),
                     p["conv_price"].to_numpy(),
                     p["year_to_mat"].clip(lower=0.1).to_numpy())
    return pd.Series(iv, index=p.index)


# ---------------------------------------------------------------------------
def leg_iv_ts(p: pd.DataFrame, window: int = 250) -> pd.Series:
    """期权贵贱的时序版:隐含波动率相对该券自身历史的位置。

    截面版(iv_rv)要跨券比 IV 减 RV,受个券固定效应干扰且变化快。跟自己比,
    问的是「这只券的期权现在比它自己平时便宜多少」,是一个天然慢变量。

    补偿的风险:同 iv_rv —— 承接 gamma 的一方为条款不确定性与转债流动性定价。
    """
    iv = _bond_iv(p)
    mu = roll_by(iv, p["b_sym"], window, "mean")
    sd = roll_by(iv, p["b_sym"], window, "std")
    return -cs_rank((iv - mu) / sd.replace(0, np.nan), p["date"])


def leg_iv_rv_long(p: pd.DataFrame, rv_window: int = 250) -> pd.Series:
    """IV 减长周期 RV。用 250 日实际波动率而非 60 日,分母本身更稳,
    信号随之更慢。经济含义与 iv_rv 相同,只是把「正股波动率」定义在更长的窗口上。
    """
    iv = _bond_iv(p)
    rv = roll_by(p["s_log_ret"], p["b_sym"], rv_window, "std") * np.sqrt(252)
    return -cs_rank(iv - rv, p["date"])


def leg_ts_val_long(p: pd.DataFrame, window: int = 500) -> pd.Series:
    """时序估值偏离,窗口拉到 500 日(约两年)。覆盖一个完整的转债估值周期。"""
    prem = p["conv_prem"]
    mu = roll_by(prem, p["b_sym"], window, "mean")
    sd = roll_by(prem, p["b_sym"], window, "std")
    return -cs_rank((prem - mu) / sd.replace(0, np.nan), p["date"])


def leg_peer_prem(p: pd.DataFrame, n_vol: int = 3, n_mat: int = 3) -> pd.Series:
    """同类比价:溢价率相对「同波动率档 × 同剩余期限档」同侪的位置。

    纯截面比价会把高波动正股、长期限的券系统性判为贵——它们本就该贵。
    先分档再组内比,把这部分结构性差异差掉,剩下的才是真偏离。
    分档用当期截面分位,不引入未来信息。

    补偿的风险:同侪定价错误的收敛需要时间。
    """
    vol = roll_by(p["s_log_ret"], p["b_sym"], 120, "std")
    df = pd.DataFrame({"date": p["date"], "prem": p["conv_prem"],
                       "vol": vol, "mat": p["year_to_mat"]}, index=p.index)

    def _bucket(x, n):
        try:
            return pd.qcut(x.rank(method="first"), n, labels=False)
        except ValueError:
            return pd.Series(0, index=x.index)

    df["vb"] = df.groupby("date")["vol"].transform(lambda x: _bucket(x, n_vol))
    df["mb"] = df.groupby("date")["mat"].transform(lambda x: _bucket(x, n_mat))
    key = df["date"].astype(str) + "_" + df["vb"].astype(str) + "_" + df["mb"].astype(str)
    grp_med = df.groupby(key)["prem"].transform("median")
    return -cs_rank(df["prem"] - grp_med, p["date"])


def leg_lt_rev(p: pd.DataFrame, window: int = 250, skip: int = 20) -> pd.Series:
    """转债长期反转,跳过最近 20 日以避开短期动量。

    补偿的风险:长周期估值修复,承担期间的条款与信用风险。构造上是慢变量。
    """
    lr = roll_by(p["b_log_ret"], p["b_sym"], window, "sum")
    lr_recent = roll_by(p["b_log_ret"], p["b_sym"], skip, "sum")
    return -cs_rank(lr - lr_recent, p["date"])


def leg_moneyness_ts(p: pd.DataFrame, window: int = 250) -> pd.Series:
    """转股价值相对自身历史的位置。度量该券在其自身周期中所处的位置,
    与价格水平的截面排序不同——低位不等于低价。
    """
    m = p["conv_value"] / 100.0
    mu = roll_by(m, p["b_sym"], window, "mean")
    sd = roll_by(m, p["b_sym"], window, "std")
    return -cs_rank((m - mu) / sd.replace(0, np.nan), p["date"])


LEGS_V4 = {
    "iv_ts": leg_iv_ts,
    "iv_rv_long": leg_iv_rv_long,
    "ts_val_long": leg_ts_val_long,
    "peer_prem": leg_peer_prem,
    "lt_rev": leg_lt_rev,
    "moneyness_ts": leg_moneyness_ts,
}


def half_life(sig: pd.Series, p: pd.DataFrame, max_lag: int = 60) -> float:
    """信号自相关衰减到 0.5 所需交易日。持续性越高,目标权重越稳,换手越低。"""
    tmp = pd.DataFrame({"b": p["b_sym"].values, "v": sig.values}, index=p.index)
    for lag in range(1, max_lag + 1):
        lagged = tmp.groupby("b", sort=False)["v"].shift(lag)
        m = tmp["v"].notna() & lagged.notna()
        if m.sum() < 1000:
            continue
        if np.corrcoef(tmp["v"][m], lagged[m])[0, 1] < 0.5:
            return float(lag)
    return float(max_lag)

"""容量与滑点。

把「Σ|w|=1 的无摩擦组合」翻译成「AUM 元的真实账户」,再问两件事:
  1. 目标仓位放得进去吗(仓位 / ADV、仓位 / 剩余余额)
  2. 每天的调仓吃掉多少(价差 + 平方根冲击)

参数不做精确假装:全部外露,并按区间给敏感性。转债无买卖盘口数据,
价差只能按流动性分位近似,这一点在报告里必须写明。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# --- 成本模型参数 ---
COMMISSION_BP = 1.0        # 佣金+经手费,单边 bp
SPREAD_BP_REF = 8.0        # 参考流动性(ADV 中位数)处的单边半价差 bp
SPREAD_ADV_REF = 4.0e6     # 参考 ADV(元)
SPREAD_EXP = 0.35          # 半价差随 ADV 衰减的幂次
SPREAD_BP_CAP = (2.0, 60.0)
IMPACT_Y = 0.8             # 平方根冲击系数,行业常用 0.5~1.0


def half_spread_bp(adv: np.ndarray) -> np.ndarray:
    """半价差随流动性递减。转债无盘口数据,只能用 ADV 近似,是本模块最弱的一环。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = SPREAD_BP_REF * (SPREAD_ADV_REF / np.where(adv > 0, adv, np.nan)) ** SPREAD_EXP
    return np.clip(s, *SPREAD_BP_CAP)


def apply_caps(w: np.ndarray, adv: np.ndarray, outstanding: np.ndarray, aum: float,
               adv_days_cap: float = 5.0, own_cap: float = 0.10) -> np.ndarray:
    """按「仓位不超过 N 日 ADV」与「不超过剩余余额的 x%」截断权重后重新归一。

    截断后归一会把被截掉的资金推给还有余量的券,因此这是一个保守下界:
    真实组合还要受制于被推向的那些券本身的容量。
    """
    if aum <= 0:
        return w
    cap_val = np.minimum(adv_days_cap * adv, own_cap * outstanding)
    cap_w = np.where(np.isfinite(cap_val), cap_val / aum, np.inf)
    out = np.sign(w) * np.minimum(np.abs(w), cap_w)
    gross = np.nansum(np.abs(out))
    return out / gross if gross > 0 else out


def cost_bp(trade_notional: np.ndarray, adv: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """单笔交易的总成本(bp),= 佣金 + 半价差 + 平方根冲击。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        part = np.where(adv > 0, trade_notional / adv, np.nan)
    impact = IMPACT_Y * sigma * 1e4 * np.sqrt(np.clip(part, 0, None))
    return COMMISSION_BP + half_spread_bp(adv) + np.nan_to_num(impact)


def simulate(dates, w_wide: np.ndarray, ret_wide: np.ndarray, adv_wide: np.ndarray,
             out_wide: np.ndarray, sig_wide: np.ndarray, aum: float,
             adv_days_cap: float = 5.0, own_cap: float = 0.10,
             use_caps: bool = True) -> dict:
    """逐日跑一遍受容量约束的组合,返回毛/净口径指标。

    w_wide     T×N 目标权重(Σ|w|=1)
    ret_wide   T×N 次日收益
    adv_wide   T×N 日成交额(元)
    out_wide   T×N 剩余余额(元,PIT)
    sig_wide   T×N 个券波动率(日频,小数)
    """
    T, N = w_wide.shape
    w_prev = np.zeros(N)
    gross_pnl, net_pnl, cost_ser, part_p95, cap_hit = [], [], [], [], []

    for t in range(T):
        w = np.nan_to_num(w_wide[t])
        adv = np.nan_to_num(adv_wide[t])
        outq = np.nan_to_num(out_wide[t])
        sg = np.nan_to_num(sig_wide[t], nan=0.02)

        w_t = apply_caps(w, adv, outq, aum, adv_days_cap, own_cap) if use_caps else w
        cap_hit.append(float(np.nansum(np.abs(w_t - w)) / 2.0))

        dv = np.abs(w_t - w_prev) * aum
        c_bp = cost_bp(dv, adv, sg)
        cost = float(np.nansum(dv * c_bp / 1e4) / aum)

        r = np.nan_to_num(ret_wide[t])
        g = float(np.nansum(w_t * r))
        gross_pnl.append(g)
        net_pnl.append(g - cost)
        cost_ser.append(cost)

        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(adv > 0, dv / adv, np.nan)
        part_p95.append(float(np.nanpercentile(pr[dv > 0], 95)) if (dv > 0).any() else np.nan)
        w_prev = w_t

    g = np.array(gross_pnl)
    n = np.array(net_pnl)
    c = np.array(cost_ser)
    return {
        "aum_yi": aum / 1e8,
        "gross_SH": float(g.mean() / g.std(ddof=1) * np.sqrt(TRADING_DAYS)) if g.std() > 0 else np.nan,
        "net_SH": float(n.mean() / n.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n.std() > 0 else np.nan,
        "gross_ret": float(g.mean() * TRADING_DAYS),
        "net_ret": float(n.mean() * TRADING_DAYS),
        "cost_ret": float(c.mean() * TRADING_DAYS),
        "part_p95": float(np.nanmean(part_p95)),
        "cap_hit": float(np.mean(cap_hit)),
    }

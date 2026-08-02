"""转债因子挖掘 v2 —— 容量优先、带正则、每条腿有经济学解释。

与 v1 的三处根本差别:

1. 数据。剩余余额改用 cb_share 的 PIT 口径。v1 用的 cb_basic.remain_size 是
   当前时点快照,零值 99.9% 对应「该券最终退市」,是未来函数。

2. 目标。v1 优化 IS ICIR,turnover 只当作 <0.2 的闸门。但 0.063 的单边日换手
   等于 31.8 倍年化双边换手,在毛年化 8% 的策略上必然被成本吃光。v2 直接优化
   目标 AUM 下的净 Sharpe。

3. 选腿。剔除 illiq。它的经济含义就是流动性溢价——百亿规模下这笔钱不是你赚的,
   是你付的。保留它会让回测好看而实盘不可执行。

正则化分四层:信号截尾、暴露中性化、权重向等权收缩、时间平滑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IS_START, IS_END = "20210601", "20240630"
OOS_START, OOS_END = "20240701", "20260301"


# ---------------------------------------------------------------------------
# 正则化算子
# ---------------------------------------------------------------------------
def cs_rank(s: pd.Series, by: pd.Series, min_count: int = 10) -> pd.Series:
    def _f(x):
        if x.notna().sum() < min_count:
            return pd.Series(np.nan, index=x.index)
        return (x.rank(pct=True) - 0.5) * np.sqrt(12.0)
    return s.groupby(by).transform(_f)


def cs_z(s: pd.Series, by: pd.Series, clip: float | None = None) -> pd.Series:
    """截面 z-score。clip 是第一层正则:截尾直接压住单券权重上限,
    既降低对尾部个券的依赖,也让容量约束更容易满足。"""
    out = s.groupby(by).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8) if np.isfinite(x.std()) and x.std() > 0 else x * 0)
    return out.clip(-clip, clip) if clip else out


def cs_neutralize(s: pd.Series, by: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    """第二层正则:对给定暴露做逐日截面回归取残差。

    用途是把「流动性押注」从 alpha 里剥掉——否则组合会在不知情的情况下
    整体做多小盘低流动性券,这正是回测好看、实盘做不了的典型来源。
    """
    # 解释变量先做截面 rank 标准化。直接用原始值(转债价格可达 400+、ADV 跨三个
    # 数量级)会让逐日 OLS 的 beta 被离群点带着抖,残差换手比原信号高一个量级,
    # 那是实现噪声而非真实的暴露剥离。
    df = pd.DataFrame({"y": s.values}, index=s.index)
    cols = list(exposures.columns)
    for c in cols:
        df[c] = cs_rank(exposures[c], by).values
    df["_by"] = by.values

    def _resid(g):
        m = g["y"].notna() & g[cols].notna().all(axis=1)
        if m.sum() < len(cols) + 10:
            return pd.Series(np.nan, index=g.index)
        X = np.column_stack([np.ones(m.sum())] + [g.loc[m, c].to_numpy() for c in cols])
        y = g.loc[m, "y"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out = pd.Series(np.nan, index=g.index)
        out.loc[m] = y - X @ beta
        return out

    return df.groupby("_by", group_keys=False).apply(_resid, include_groups=False)


def roll_by(s: pd.Series, key: pd.Series, window: int, how: str = "mean") -> pd.Series:
    tmp = pd.DataFrame({"k": key.values, "v": s.values}, index=s.index)
    return tmp.groupby("k", sort=False)["v"].transform(
        lambda x: getattr(x.rolling(window, min_periods=max(2, window // 2)), how)())


def shrink(w: dict, lam: float) -> dict:
    """第三层正则:组合权重向等权收缩。lam=1 即完全等权。

    IS 网格最优与等权在 v1 里表现几乎一致,说明最优点附近很平——
    这种情况下向等权收缩几乎不损失 IS,却明显降低权重被样本噪声挑出来的风险。
    """
    n = len(w)
    return {k: (1 - lam) * v + lam / n for k, v in w.items()}


# ---------------------------------------------------------------------------
# 因子腿:每条都必须先说清补偿的是什么风险
# ---------------------------------------------------------------------------
def leg_valuation(p: pd.DataFrame) -> pd.Series:
    """估值。转债 = 纯债 + 转股期权,两个溢价率衡量市场为这两部分付的价格。

    补偿的风险:发行人信用风险、强赎条款被行使的风险、二级流动性风险。
    容量友好——溢价率高低与个券流动性无系统性关联。
    """
    floor_prem = p["b_close"] / p["debt_puredebt_ratio"].where(p["debt_puredebt_ratio"] > 0) - 1
    conv_prem = p["b_close"] / p["conv_value"].where(p["conv_value"] > 0) - 1
    return -(0.5 * cs_rank(floor_prem, p["date"]) + 0.5 * cs_rank(conv_prem, p["date"]))


def leg_conv_progress(p: pd.DataFrame) -> pd.Series:
    """转股进度(PIT)。累计转股比例高 = 该券长期在转股价值之上,已逼近强赎。

    事前假设方向为负:期权时间价值被压缩,且继续转股对正股构成供给压力。
    容量代价:高转股比例的券剩余存量小,天然容量差,权重需受存量上限约束。
    """
    return -cs_rank(p["acc_conv_ratio_pit"], p["date"])


def leg_delta_gap(p: pd.DataFrame, window: int = 20) -> pd.Series:
    """弹性残差。转债实际收益减去 delta×正股收益的累计偏离,取负做反转。

    补偿的风险:转债二级市场深度远逊于正股,价格对正股信息的吸收滞后,
    承接这段滞后需要承担隔夜与流动性风险。属于做市/统计套利性质。
    """
    delta = (p["conv_value"] / p["b_close"].replace(0, np.nan)).clip(0, 1)
    resid = p["b_log_ret"] - delta * p["s_log_ret"]
    return -cs_rank(roll_by(resid, p["b_sym"], window, "sum"), p["date"])


def leg_credit_floor(p: pd.DataFrame) -> pd.Series:
    """债底保护。到期收益率越高、纯债溢价率越低,下行保护越厚。

    补偿的风险:信用下沉。与估值腿相关但不重合——估值看期权便宜,这条看债性安全。
    """
    return 0.5 * cs_rank(p["ytm_approx"], p["date"]) - 0.5 * cs_rank(p["floor_prem"], p["date"])


LEGS = {
    "val": leg_valuation,
    "conv": leg_conv_progress,
    "dgap": leg_delta_gap,
    "floor": leg_credit_floor,
}


def build_legs(p: pd.DataFrame, clip: float = 2.5, neutralize: bool = True) -> pd.DataFrame:
    out = {}
    if neutralize:
        exp = pd.DataFrame({
            "log_adv": np.log(p["adv_yuan"].clip(lower=1e4)),
            "price": p["b_close"],
        }, index=p.index)
    for k, fn in LEGS.items():
        s = fn(p)
        if neutralize:
            s = cs_neutralize(s, p["date"], exp)
        out[k] = cs_z(s, p["date"], clip=clip)
    return pd.DataFrame(out, index=p.index)

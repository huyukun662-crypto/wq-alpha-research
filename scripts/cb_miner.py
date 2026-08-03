"""转债 D1 因子挖掘器。

目标闸门(IS 20210601-20240630 / OOS 20240701-20260301):
    IC > 0.015, Sharpe > 1.25, turnover < 0.2, ICIR > 2

用法:
    python scripts/cb_miner.py --batch 1
    python scripts/cb_miner.py --factor cb_dual_prem --grid
    python scripts/cb_miner.py --report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cb_panel import CB_DIR, cs_rank, evaluate, load_panel  # noqa: E402

IS_START, IS_END = "20210601", "20240630"
OOS_START, OOS_END = "20240701", "20260301"

GATE = {"IC": 0.015, "Sharpe": 1.25, "turnover": 0.2, "ICIR": 2.0}

RESULTS_PATH = CB_DIR / "cb_mining_results.json"


# ---------------------------------------------------------------------------
# 因子算子
# ---------------------------------------------------------------------------
def _rank(df: pd.DataFrame, col_or_series) -> pd.Series:
    s = df[col_or_series] if isinstance(col_or_series, str) else col_or_series
    return s.groupby(df["date"]).transform(cs_rank)


def _ts(df: pd.DataFrame, col: str, window: int, fn: str = "mean") -> pd.Series:
    g = df.groupby("b_sym", sort=False)[col]
    return getattr(g.transform(lambda x: getattr(x.rolling(window, min_periods=max(2, window // 2)), fn)()), "__call__", None) \
        or g.transform(lambda x: getattr(x.rolling(window, min_periods=max(2, window // 2)), fn)())


def _roll(df: pd.DataFrame, s: pd.Series, window: int, fn: str = "mean") -> pd.Series:
    tmp = pd.DataFrame({"b_sym": df["b_sym"].values, "v": s.values}, index=df.index)
    return tmp.groupby("b_sym", sort=False)["v"].transform(
        lambda x: getattr(x.rolling(window, min_periods=max(2, window // 2)), fn)())


def _delta(df: pd.DataFrame, col: str, lag: int) -> pd.Series:
    return df[col] - df.groupby("b_sym", sort=False)[col].shift(lag)


# ---------------------------------------------------------------------------
# 因子库
# ---------------------------------------------------------------------------
FACTORS: dict[str, dict] = {}


def factor(name: str, desc: str, definition: str, roll: int = 3):
    def deco(fn):
        FACTORS[name] = {"fn": fn, "desc": desc, "definition": definition, "roll": roll}
        return fn
    return deco


# ---- 批次 1:估值基线 ----
@factor("cb_dual_prem",
        "双溢价率截面 rank 混合(复刻示例基线)",
        "-(0.5*rank(floor_prem) + 0.5*rank(conv_prem))", roll=3)
def f_dual_prem(df, w_floor: float = 0.5):
    return -(w_floor * _rank(df, "floor_prem") + (1 - w_floor) * _rank(df, "conv_prem"))


@factor("cb_dbl_low",
        "双低:价格 + 转股溢价率,越低越优",
        "-rank(b_close + bond_prem_ratio)", roll=3)
def f_dbl_low(df, w_prem: float = 1.0):
    return -_rank(df, df["b_close"] + w_prem * df["bond_prem_ratio"])


@factor("cb_floor_yield",
        "债底保护:纯债溢价率越低 + 近似 YTM 越高越优",
        "0.5*rank(ytm_approx) - 0.5*rank(floor_prem)", roll=3)
def f_floor_yield(df, w_ytm: float = 0.5):
    return w_ytm * _rank(df, "ytm_approx") - (1 - w_ytm) * _rank(df, "floor_prem")


# ---- 批次 2:条款与期权结构 ----
@factor("cb_call_dist",
        "距强赎触发距离:正股价/(1.3*转股价),远离触发线的券不被赎回风险压制",
        "-rank(call_trigger_dist)", roll=5)
def f_call_dist(df, sign: float = -1.0):
    return sign * _rank(df, "call_trigger_dist")


@factor("cb_opt_cheap",
        "内含期权便宜度:期权价格/(正股波动率*剩余期限),越便宜越优",
        "-rank(embedded_option_price/(stock_volatility*sqrt(year_to_mat)))", roll=5)
def f_opt_cheap(df):
    denom = (df["stock_volatility"] * np.sqrt(df["year_to_mat"].clip(lower=0.1))).replace(0, np.nan)
    return -_rank(df, df["embedded_option_price"] / denom)


@factor("cb_downward_exp",
        "下修预期:正股长期低于转股价的程度,越深越可能触发下修",
        "-rank(ts_mean(put_trigger_dist, 60))", roll=5)
def f_downward_exp(df, window: int = 60):
    return -_rank(df, _roll(df, df["put_trigger_dist"], window))


# ---- 批次 3:正股传导 ----
@factor("cb_stk_rev",
        "正股中期反转经转债滞后传导",
        "-rank(ts_sum(s_log_ret, 20))", roll=5)
def f_stk_rev(df, window: int = 20):
    return -_rank(df, _roll(df, df["s_log_ret"], window, "sum"))


@factor("cb_stk_vol",
        "正股波动率:期权 vega,高波动正股的转债期权价值常被低估",
        "rank(stock_volatility)", roll=5)
def f_stk_vol(df, sign: float = 1.0):
    return sign * _rank(df, "stock_volatility")


@factor("cb_delta_gap",
        "弹性偏离:转债实际涨跌 vs 理论 delta*正股涨跌 的累计残差",
        "-rank(ts_sum(b_log_ret - delta*s_log_ret, 20))", roll=5)
def f_delta_gap(df, window: int = 20):
    delta = (df["conv_value"] / df["b_close"].replace(0, np.nan)).clip(0, 1)
    resid = df["b_log_ret"] - delta * df["s_log_ret"]
    return -_rank(df, _roll(df, resid, window, "sum"))


# ---- 批次 4:规模与流动性 ----
@factor("cb_size",
        "剩余规模:小盘转债的条款博弈弹性更高",
        "-rank(log(remain_size))", roll=5)
def f_size(df, sign: float = -1.0):
    return sign * _rank(df, np.log(df["remain_size"].clip(lower=0.01)))


@factor("cb_illiq",
        "转债 amihud:非流动性溢价",
        "rank(ts_mean(|b_pct_chg|/b_amount, 20))", roll=5)
def f_illiq(df, window: int = 20):
    il = df["b_log_ret"].abs() / df["b_amount"].replace(0, np.nan)
    return _rank(df, _roll(df, il, window))


@factor("cb_turn_cool",
        "转债换手率:过热规避",
        "-rank(ts_mean(b_turnover, 20))", roll=5)
def f_turn_cool(df, window: int = 20):
    return -_rank(df, _roll(df, df["b_turnover"], window))


# ---- 批次 5:信用质量控制 ----
# 假设:双低/纯债溢价率天然做多低价券,而低价券在 2024H2 信用恐慌中被无差别抛售,
# 这是 OOS 衰减的主因。用正股市值/股价作为信用质量代理去中和这一暴露。
@factor("cb_credit_q",
        "信用质量:正股市值越大、股价离面值退市线越远,违约与退市风险越低",
        "0.5*rank(log(s_total_mv)) + 0.5*rank(s_close)", roll=5)
def f_credit_q(df, w_mv: float = 0.5):
    mv = _rank(df, np.log(df["s_total_mv"].clip(lower=1.0)))
    px = _rank(df, df["s_close"])
    return w_mv * mv + (1 - w_mv) * px


@factor("cb_value_credit",
        "双溢价率估值 + 信用质量控制",
        "(1-w)*cb_dual_prem + w*cb_credit_q", roll=3)
def f_value_credit(df, w_credit: float = 0.3, w_floor: float = 0.5):
    return (1 - w_credit) * f_dual_prem(df, w_floor=w_floor) + w_credit * f_credit_q(df)


@factor("cb_dbl_low_credit",
        "双低 + 信用质量控制",
        "(1-w)*cb_dbl_low + w*cb_credit_q", roll=3)
def f_dbl_low_credit(df, w_credit: float = 0.3):
    return (1 - w_credit) * f_dbl_low(df) + w_credit * f_credit_q(df)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def run_factor(panel: pd.DataFrame, name: str, roll: int | None = None, **params) -> dict:
    spec = FACTORS[name]
    raw = spec["fn"](panel, **params)
    roll = roll if roll is not None else spec["roll"]
    is_ = evaluate(panel, raw, roll_days=roll, start=IS_START, end=IS_END)
    oos = evaluate(panel, raw, roll_days=roll, start=OOS_START, end=OOS_END)
    return {"name": name, "roll": roll, "params": params, "IS": is_, "OOS": oos}


def passes(m: dict) -> bool:
    return (m.get("IC", -9) > GATE["IC"] and m.get("Sharpe", -9) > GATE["Sharpe"]
            and m.get("turnover", 9) < GATE["turnover"] and m.get("ICIR", -9) > GATE["ICIR"])


def fmt(r: dict) -> str:
    def row(tag, m):
        if not m.get("n_days"):
            return f"  {tag}: (empty)"
        flag = "PASS" if passes(m) else "----"
        return (f"  {tag} {flag} IC={m['IC']:+.4f} ICIR={m['ICIR']:+.2f} SH={m['Sharpe']:+.2f} "
                f"TO={m['turnover']:.3f} RankIC={m['RankIC']:+.4f} ret={m['AnnRet']:+.3%} "
                f"n={m['breadth']:.0f}")
    p = f" {r['params']}" if r["params"] else ""
    return f"[{r['name']} roll={r['roll']}{p}]\n{row('IS ', r['IS'])}\n{row('OOS', r['OOS'])}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int)
    ap.add_argument("--factor")
    ap.add_argument("--rolls", default="3")
    args = ap.parse_args()

    panel = load_panel()
    print(f"panel {panel.shape}, universe 日均 "
          f"{panel[panel['in_univ']].groupby('date').size().mean():.0f}\n")

    names = [args.factor] if args.factor else list(FACTORS)
    rolls = [int(x) for x in args.rolls.split(",")]
    out = []
    for n in names:
        for r in rolls:
            res = run_factor(panel, n, roll=r)
            out.append(res)
            print(fmt(res))
            print()
    RESULTS_PATH.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

"""v2 运行器:流动性分层 × 平滑窗口 × 目标 AUM,以净 Sharpe 为准绳。

    python scripts/cb_run_v2.py --scan       # 分层 × 平滑,毛口径与换手
    python scripts/cb_run_v2.py --capacity   # 容量曲线(净 Sharpe vs AUM)
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import cb_capacity as cc  # noqa: E402
import cb_miner_v2 as v2  # noqa: E402
from cb_fast import FastEval  # noqa: E402
from cb_panel import load_panel  # noqa: E402

TIERS = {"all": 0.0, "adv500w": 5e6, "adv1000w": 1e7, "adv2000w": 2e7}


def liquid_mask(p: pd.DataFrame, thresh: float) -> pd.Series:
    """滚动 20 日均 ADV 门槛。用历史均值而非当日值,避免用到当日成交信息。"""
    adv20 = v2.roll_by(p["adv_yuan"], p["b_sym"], 20, "mean")
    adv20 = adv20.groupby(p["b_sym"], sort=False).shift(1)
    return p["in_univ"] & (adv20 >= thresh)


def make_weights(fe: FastEval, raw: np.ndarray, univ: np.ndarray, roll: int) -> np.ndarray:
    s = np.where(univ, raw, np.nan)
    s = pd.DataFrame(s).rolling(roll, min_periods=1).mean().to_numpy()
    s = np.where(univ, s, np.nan)
    s = (s - np.nanmean(s, axis=1, keepdims=True)) / (np.nanstd(s, axis=1, keepdims=True) + 4e-4)
    s = np.where(np.isfinite(fe.ret) & univ, s, np.nan)
    gross = np.nansum(np.abs(s), axis=1, keepdims=True)
    return np.divide(s, gross, out=np.full_like(s, np.nan), where=gross > 0)


def gross_stats(fe: FastEval, w: np.ndarray, sl: slice) -> dict:
    w0 = np.nan_to_num(w[sl])
    r = np.nan_to_num(fe.ret[sl])
    pnl = np.nansum(w0 * r, axis=1)
    dw = np.abs(np.diff(w0, axis=0)).sum(axis=1)
    ic = []
    for t in range(w[sl].shape[0]):
        m = np.isfinite(w[sl][t]) & np.isfinite(fe.ret[sl][t])
        if m.sum() >= 10:
            ic.append(np.corrcoef(w[sl][t][m], fe.ret[sl][t][m])[0, 1])
    ic = np.array(ic)
    return {
        "breadth": float(np.isfinite(w[sl]).sum(axis=1).mean()),
        "gross_SH": float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if pnl.std() > 0 else np.nan,
        "gross_ret": float(pnl.mean() * 252),
        "to_1side": float(dw.mean() / 2),
        "to_annual": float(dw.mean() * 252),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--capacity", action="store_true")
    ap.add_argument("--clip", type=float, default=2.5)
    ap.add_argument("--shrink", type=float, default=0.5)
    args = ap.parse_args()

    p = load_panel().sort_values(["b_sym", "date"]).reset_index(drop=True)
    legs = v2.build_legs(p, clip=args.clip, neutralize=True)
    W = v2.shrink({"val": 0.4, "conv": 0.2, "dgap": 0.2, "floor": 0.2}, args.shrink)
    sig = sum(v * legs[k] for k, v in W.items())

    fe = FastEval(p)
    raw = fe.wide(sig)
    lo_is, hi_is = fe.window(v2.IS_START, v2.IS_END)
    lo_o, hi_o = fe.window(v2.OOS_START, v2.OOS_END)

    if args.scan:
        print(f"权重(向等权收缩 lam={args.shrink}): "
              + " ".join(f"{k}={v:.2f}" for k, v in W.items()))
        print(f"{'tier':>9} {'roll':>5} {'breadth':>8} {'毛SH_IS':>8} {'毛SH_OOS':>9} "
              f"{'毛年化OOS':>10} {'单边换手':>9} {'年化换手':>9}")
        for tname, th in TIERS.items():
            um = liquid_mask(p, th)
            univ = fe.wide(um.astype(float)) > 0.5
            for roll in (5, 20, 60):
                w = make_weights(fe, raw, univ, roll)
                a = gross_stats(fe, w, slice(lo_is, hi_is))
                b = gross_stats(fe, w, slice(lo_o, hi_o))
                print(f"{tname:>9} {roll:5d} {b['breadth']:8.0f} {a['gross_SH']:8.2f} "
                      f"{b['gross_SH']:9.2f} {b['gross_ret']:10.2%} {b['to_1side']:9.4f} "
                      f"{b['to_annual']:9.1f}")
        return

    if args.capacity:
        adv = fe.wide(p["adv_yuan"])
        out = fe.wide(p["remain_size_pit"])
        bvol = v2.roll_by(p["b_log_ret"], p["b_sym"], 60, "std")
        vol = fe.wide(bvol)
        print(f"{'tier':>9} {'roll':>5} {'AUM亿':>7} {'毛SH':>7} {'净SH':>7} {'净年化':>9} "
              f"{'成本':>8} {'p95参与':>8} {'截断':>7}")
        for tname, roll in (("adv1000w", 20), ("adv1000w", 60), ("adv2000w", 60)):
            um = liquid_mask(p, TIERS[tname])
            univ = fe.wide(um.astype(float)) > 0.5
            w = make_weights(fe, raw, univ, roll)
            S = slice(lo_o, hi_o)
            for aum in (5, 10, 20, 30, 50, 100):
                r = cc.simulate(fe.dates[S], w[S], fe.ret[S], adv[S], out[S], vol[S], aum=aum * 1e8)
                print(f"{tname:>9} {roll:5d} {aum:7d} {r['gross_SH']:7.2f} {r['net_SH']:7.2f} "
                      f"{r['net_ret']:9.2%} {r['cost_ret']:8.2%} {r['part_p95']:8.1%} "
                      f"{r['cap_hit']:7.1%}")
        return


if __name__ == "__main__":
    main()

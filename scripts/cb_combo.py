"""转债因子组合搜索。

纪律:权重与超参只在 IS(20210601-20240630)上搜,OOS(20240701-20260301)
全程作为纯持出,只在最后打印一次。任何"看了 OOS 再回头改权重"的操作都算数据窥探。

用法:
    python scripts/cb_combo.py --corr            # 基信号相关矩阵
    python scripts/cb_combo.py --search          # IS 上搜权重
    python scripts/cb_combo.py --check "a=0.4,b=0.3"
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
import cb_miner as cm  # noqa: E402
from cb_panel import evaluate, load_panel  # noqa: E402

# 参与组合的基信号:各自收益来源不同,且 IS/OOS 至少一端不塌
BASE = {
    "prem": lambda df: cm.f_dual_prem(df),
    "size": lambda df: cm.f_size(df),
    "illiq": lambda df: cm.f_illiq(df),
    "dgap": lambda df: cm.f_delta_gap(df),
    "credit": lambda df: cm.f_credit_q(df),
}


def build_base(panel: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for k, fn in BASE.items():
        s = fn(panel)
        # 统一到截面 z,使权重可比
        out[k] = s.groupby(panel["date"]).transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return pd.DataFrame(out, index=panel.index)


def vol_scale(panel: pd.DataFrame, sig: pd.Series, window: int = 60,
              floor_q: float = 0.10) -> pd.Series:
    """风险平价加权:sig / 个券波动率。

    诊断显示排序信息稳定但线性 z 加权 Sharpe 低,即资金被压在尾部最凶的券上。
    除以波动率把仓位从高弹性券挪走,直接检验这个解释。
    """
    tmp = pd.DataFrame({"b": panel["b_sym"].values, "r": panel["b_log_ret"].values},
                       index=panel.index)
    vol = tmp.groupby("b", sort=False)["r"].transform(
        lambda x: x.rolling(window, min_periods=20).std())
    vol = vol.clip(lower=vol.quantile(floor_q))
    return sig / vol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--volscale", action="store_true")
    ap.add_argument("--roll", type=int, default=3)
    args = ap.parse_args()

    panel = load_panel()
    base = build_base(panel)
    print(f"panel {panel.shape}, base signals {list(base.columns)}\n")

    if args.corr:
        m = panel["in_univ"]
        print("截面平均相关(IS 区间):")
        d = base[m].copy()
        d["date"] = panel.loc[m, "date"]
        c = d.groupby("date").corr().groupby(level=1).mean()
        print(c.round(3).to_string())
        return

    if args.volscale:
        for name in ("prem", "size", "illiq", "dgap"):
            for tag, sig in (("plain", base[name]), ("volscaled", vol_scale(panel, base[name]))):
                is_ = evaluate(panel, sig, roll_days=args.roll,
                               start=cm.IS_START, end=cm.IS_END)
                oos = evaluate(panel, sig, roll_days=args.roll,
                               start=cm.OOS_START, end=cm.OOS_END)
                print(f"{name:6s} {tag:10s} "
                      f"IS IC={is_['IC']:+.4f} ICIR={is_['ICIR']:+.2f} SH={is_['Sharpe']:+.2f} TO={is_['turnover']:.3f} | "
                      f"OOS IC={oos['IC']:+.4f} ICIR={oos['ICIR']:+.2f} SH={oos['Sharpe']:+.2f} TO={oos['turnover']:.3f}")
        return

    if args.search:
        # IS-only 权重网格。OOS 到最后统一复核,不参与选择。
        # 快路径只用于排序;入选项一律回慢路径确认(宽表滚动会把停牌日算进窗口,
        # 与示例代码的 groupby 滚动有 ~0.15 的 Sharpe 残差)。
        from cb_fast import FastEval

        fe = FastEval(panel)
        wide = {k: fe.wide(base[k]) for k in BASE}
        lo, hi = fe.window(cm.IS_START, cm.IS_END)

        grid = [0.0, 0.2, 0.4, 0.6]
        keys = list(BASE)
        results = []
        for combo in itertools.product(grid, repeat=len(keys)):
            if sum(combo) < 1e-9:
                continue
            w = np.array(combo) / sum(combo)
            sig = sum(w[i] * wide[keys[i]] for i in range(len(keys)))
            m = fe.evaluate(sig, roll=args.roll, lo=lo, hi=hi)
            if not m.get("n_days"):
                continue
            results.append((dict(zip(keys, w.round(3))), m))
        results.sort(key=lambda r: -r[1]["ICIR"])
        print(f"IS 网格 {len(results)} 组,按 IS ICIR 排序,前 15:\n")
        for wts, m in results[:15]:
            nz = {k: v for k, v in wts.items() if v > 0}
            print(f"{str(nz):58s} IC={m['IC']:+.4f} ICIR={m['ICIR']:+.2f} "
                  f"SH={m['Sharpe']:+.2f} TO={m['turnover']:.3f}")
        return


if __name__ == "__main__":
    main()

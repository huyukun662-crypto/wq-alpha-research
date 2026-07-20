"""用 tushare 本地数据挖掘 A股因子 —— SKILL.md 流程的本地回测版。

与 mine_chn_alphas.py 的关系：同一套「表达式 → 回测 → 检查 → 相关性 → 报告」
方法论，只是把 WorldQuant BRAIN 模拟器换成本地数据 + 本地算子。

回测口径（WQ 风格，delay=1）:
    - 信号用 t 日收盘数据计算，t+1 日持仓（w 取 shift(1)），收益按 close-close。
    - Universe：每日流通市值前 2000、上市满 120 个交易日、非 ST、当日有成交。
    - 行业中性：申万风格行业（stock_basic.industry）内 rank / 去均值。
    - 权重归一：sum|w| = 2（多空各 1）。
    - Turnover = sum|Δw| / 4（完全换仓 = 100%）。
    - Fitness = Sharpe × √(|年化收益| / max(TO, 0.125))（SKILL 5.1）。
    - 成本敏感性：净值口径按单边 13bp（佣金+印花+冲击）计提。

用法:
    python scripts/tushare_data.py --download          # 先下载数据
    python scripts/mine_tushare_alphas.py --mine       # 全量因子回测 + 报告
    python scripts/mine_tushare_alphas.py --mine --only reversal
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from tushare_data import DATA_DIR, load_panel  # noqa: E402

REPORT_PATH = SKILL_DIR / "tushare_mining_report.md"
RESULTS_PATH = SKILL_DIR / "tushare_mining_results.json"
FACTOR_RET_PATH = DATA_DIR / "factor_returns.parquet"

COST_RATE = 0.0013  # 单边成本：佣金 ~3bp + 卖出印花 5bp + 冲击 ~5bp

EPS = 1e-9


# ---------------------------------------------------------------------------
# FASTEXPR 风格算子（宽表：index=date, columns=ts_code）
# ---------------------------------------------------------------------------
def rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def ts_rank(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return df.rolling(w, min_periods=max(2, w // 2)).rank(pct=True)


def ts_delta(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return df.diff(w)


def ts_mean(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return df.rolling(w, min_periods=max(2, w // 2)).mean()


def ts_std_dev(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return df.rolling(w, min_periods=max(2, w // 2)).std()


def ts_min(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return df.rolling(w, min_periods=max(2, w // 2)).min()


def ts_max(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return df.rolling(w, min_periods=max(2, w // 2)).max()


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(3, w // 2)).corr(y)


def ts_decay_linear(df: pd.DataFrame, d: int) -> pd.DataFrame:
    if d <= 1:
        return df
    num = None
    den = 0.0
    for i in range(d):
        weight = d - i
        term = df.shift(i) * weight
        num = term if num is None else num.add(term, fill_value=0.0)
        den += weight
    return num / den


def group_apply(df: pd.DataFrame, groups: dict[str, list[str]], fn) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype="float32")
    for _, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 3:
            continue
        out[cols] = fn(df[cols])
    return out


def group_rank(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    return group_apply(df, groups, lambda sub: sub.rank(axis=1, pct=True))


def group_demean(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    return group_apply(df, groups, lambda sub: sub.sub(sub.mean(axis=1), axis=0))


# ---------------------------------------------------------------------------
# Universe / 行业分组
# ---------------------------------------------------------------------------
def build_universe(data: dict) -> pd.DataFrame:
    close = data["close"]
    sb = data["_stock_basic"].set_index("ts_code")
    st_mask = sb["name"].str.contains("ST", na=False)
    st_cols = [c for c in close.columns if c in st_mask.index and st_mask.loc[c]]

    traded = (data["vol"] > 0) & close.notna()
    listed_days = close.notna().cumsum()  # 有行情的累计天数近似上市天数
    seasoned = listed_days > 120

    mv = data["circ_mv"].where(traded & seasoned)
    top2000 = mv.rank(axis=1, ascending=False) <= 2000

    uni = traded & seasoned & top2000
    uni[st_cols] = False
    return uni


def build_groups(data: dict) -> dict[str, list[str]]:
    sb = data["_stock_basic"].set_index("ts_code")
    cols = data["close"].columns
    ind = sb.reindex(cols)["industry"].fillna("其他")
    groups: dict[str, list[str]] = {}
    for c, g in ind.items():
        groups.setdefault(g, []).append(c)
    return groups


# ---------------------------------------------------------------------------
# 因子库（references/chn_candidate_alphas.json 的 tushare 字段适配版）
# ---------------------------------------------------------------------------
def factor_library(data: dict, groups: dict[str, list[str]]) -> dict[str, dict]:
    close, vol = data["close_adj"], data["vol"]
    returns = data["returns"]
    ep = 1.0 / data["pe_ttm"]          # 盈利收益率（负 PE -> 负 EP，保留其排序含义）
    sp = 1.0 / data["ps_ttm"]          # 营收收益率
    ret5 = close / close.shift(5) - 1.0

    lib: dict[str, dict] = {
        "reversal_5d": {
            "family": "technical/reversal",
            "decay": 15,
            "signal": lambda: group_rank(-ret5, groups),
        },
        "reversal_volatility_scaled": {
            "family": "technical/reversal",
            "decay": 15,
            "signal": lambda: group_rank(-ret5 / (ts_std_dev(returns, 20) + 1e-4), groups),
        },
        "range_position": {
            "family": "technical/position",
            "decay": 15,
            "signal": lambda: rank(
                -(close - ts_min(data["low_adj"], 20))
                / (ts_max(data["high_adj"], 20) - ts_min(data["low_adj"], 20) + EPS)
            ),
        },
        "abnormal_turnover": {
            "family": "liquidity/turnover",
            "decay": 10,
            "signal": lambda: group_rank(
                -ts_mean(data["turnover_rate_f"], 20) / (ts_mean(data["turnover_rate_f"], 120) + EPS),
                groups,
            ),
        },
        "amihud_illiquidity": {
            "family": "liquidity",
            "decay": 15,
            "signal": lambda: group_rank(ts_mean(returns.abs() / (data["amount"] + 1.0), 20), groups),
        },
        "low_volatility": {
            "family": "risk/low-vol",
            "decay": 5,
            "signal": lambda: group_rank(-ts_std_dev(returns, 60), groups),
        },
        "price_volume_divergence": {
            "family": "technical/price-volume",
            "decay": 20,
            "signal": lambda: rank(-ts_corr(rank(close), rank(vol), 10)),
        },
        "vwap_deviation": {
            "family": "technical/mean-reversion",
            "decay": 12,
            "signal": lambda: rank(-(data["close"] / data["vwap"] - 1.0)),
        },
        "ep_trend": {
            "family": "fundamental/profitability-trend",
            "decay": 4,
            "signal": lambda: group_rank(ts_rank(ep, 126), groups),
        },
        "earnings_yield": {
            "family": "fundamental/value",
            "decay": 4,
            "signal": lambda: group_rank(ep, groups),
        },
        "sales_yield": {
            "family": "fundamental/value",
            "decay": 4,
            "signal": lambda: group_rank(sp, groups),
        },
        "dividend_yield": {
            "family": "fundamental/income",
            "decay": 4,
            "signal": lambda: group_rank(data["dv_ttm"].fillna(0.0), groups),
        },
        "quality_value_mix": {
            "family": "mix/value+income",
            "decay": 4,
            "signal": lambda: 0.5 * group_rank(ep, groups) + 0.5 * group_rank(data["dv_ttm"].fillna(0.0), groups),
        },
        "reversal_open_close_mix": {
            "family": "mix/technical+fundamental",
            "decay": 10,
            "signal": lambda: 0.5 * rank(-(data["close"] / data["open"] - 1.0)) + 0.5 * rank(ts_rank(ep, 126)),
        },
        "volume_ratio_reversal": {
            "family": "liquidity/volume-ratio",
            "decay": 10,
            "signal": lambda: group_rank(-data["volume_ratio"], groups),
        },
    }

    # ---- 财务指标簇（point-in-time，按公告日生效；SKILL 模板 A/C/D 的真字段版）----
    if "roe" in data:
        roe = data["roe"]
        lib["roe_trend"] = {
            "family": "fundamental/profitability",
            "decay": 4,
            "signal": lambda: group_rank(ts_rank(roe, 252), groups),
        }
        lib["roe_level"] = {
            "family": "fundamental/quality",
            "decay": 4,
            "signal": lambda: group_rank(roe, groups),
        }
        lib["quality_value_roe_mix"] = {
            "family": "mix/quality+value",
            "decay": 4,
            "signal": lambda: 0.5 * group_rank(ts_rank(roe, 252), groups) + 0.5 * group_rank(ep, groups),
        }
    if "ocfps" in data:
        lib["ocf_yield"] = {
            "family": "fundamental/cashflow",
            "decay": 4,
            "signal": lambda: group_rank(data["ocfps"] / data["close"], groups),
        }
    if "netprofit_yoy" in data:
        lib["netprofit_growth"] = {
            "family": "fundamental/growth",
            "decay": 4,
            "signal": lambda: group_rank(data["netprofit_yoy"], groups),
        }
    if "grossprofit_margin" in data:
        lib["gross_margin_trend"] = {
            "family": "fundamental/quality",
            "decay": 4,
            "signal": lambda: group_rank(ts_rank(data["grossprofit_margin"], 252), groups),
        }

    # ---- 高频日频化簇（5 分钟线聚合，仅覆盖分钟数据区间）----
    if "late_vol_share" in data:
        lib["late_volume_share"] = {
            "family": "intraday/volume-structure",
            "decay": 10,
            "signal": lambda: group_rank(ts_mean(data["late_vol_share"], 20), groups),
        }
        lib["intraday_skew"] = {
            "family": "intraday/skewness",
            "decay": 10,
            "signal": lambda: group_rank(-ts_mean(data["intraday_skew"], 20), groups),
        }
        lib["open30_reversal"] = {
            "family": "intraday/open-momentum",
            "decay": 10,
            "signal": lambda: rank(-ts_mean(data["open30_ret"], 5)),
        }

    # ---- 资金流簇（大单+特大单净流入占成交额比例）----
    if "net_lg_amount" in data:
        # net_lg_amount 单位万元, amount 单位千元 -> ×10 统一
        lg_ratio = data["net_lg_amount"] * 10.0 / (data["amount"] + 1.0)
        lib["smart_money_flow"] = {
            "family": "moneyflow",
            "decay": 10,
            "signal": lambda: group_rank(ts_mean(lg_ratio, 20), groups),
        }
    return lib


# ---------------------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------------------
def backtest(signal: pd.DataFrame, data: dict, groups: dict, universe: pd.DataFrame, decay: int) -> dict:
    sig = signal.where(universe)
    sig = ts_decay_linear(sig, decay)
    sig = sig.where(universe)

    w = group_demean(sig, groups)                       # 行业中性
    gross = w.abs().sum(axis=1)
    w = w.div(gross.replace(0, np.nan), axis=0) * 2.0   # sum|w| = 2

    w_lag = w.shift(1)                                  # delay-1
    ret_matrix = w_lag * data["returns"]
    pnl = ret_matrix.sum(axis=1, min_count=10)

    dw = (w - w.shift(1)).abs().sum(axis=1)
    to = (dw / 4.0).reindex(pnl.index)

    valid = pnl.dropna()
    if len(valid) < 120:
        return {"error": "insufficient_data", "days": len(valid)}
    to_valid = to.reindex(valid.index).fillna(0.0)

    cost = to_valid * 4.0 / 2.0 * COST_RATE * 2.0       # 交易额×成本（双边近似）
    net = valid - cost

    def metrics(series: pd.Series) -> dict:
        ann = float(series.mean() * 252)
        sharpe = float(series.mean() / (series.std() + EPS) * np.sqrt(252))
        eq = series.cumsum()
        dd = float((eq.cummax() - eq).max())
        avg_to = float(to_valid.mean())
        fitness = sharpe * float(np.sqrt(abs(ann) / max(avg_to, 0.125)))
        return {"ann_return": ann, "sharpe": sharpe, "max_drawdown": dd, "turnover": avg_to, "fitness": fitness}

    gross_m = metrics(valid)
    net_m = metrics(net)

    # IC：信号 rank 与次日收益的秩相关（逐日）
    sig_ranked = sig.rank(axis=1, pct=True)
    fwd = data["returns"].shift(-1).rank(axis=1, pct=True)
    ic_series = sig_ranked.corrwith(fwd, axis=1)
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / (ic_series.std() + EPS) * np.sqrt(252))

    checks = []
    if gross_m["sharpe"] < 1.25:
        checks.append("LOW_SHARPE")
    if gross_m["fitness"] < 1.0:
        checks.append("LOW_FITNESS")
    if gross_m["turnover"] > 0.70:
        checks.append("HIGH_TURNOVER")
    if gross_m["turnover"] < 0.01:
        checks.append("LOW_TURNOVER")
    if gross_m["max_drawdown"] > 0.15:
        checks.append("HIGH_DRAWDOWN")

    return {
        "days": len(valid),
        "gross": gross_m,
        "net": net_m,
        "ic": ic,
        "icir": icir,
        "failed_checks": checks,
        "pass_all": not checks,
        "pnl": valid,
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def low_corr_portfolio(results: dict[str, dict], corr: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """贪心：按 Fitness 降序，只纳入与已选因子日收益 |corr|<threshold 的（SKILL 8.2/8.3）。"""
    ranked = sorted(
        (n for n, r in results.items() if "gross" in r),
        key=lambda n: results[n]["gross"]["fitness"],
        reverse=True,
    )
    chosen: list[str] = []
    for n in ranked:
        if results[n]["gross"]["sharpe"] < 0.5:
            continue
        if all(abs(corr.loc[n, c]) < threshold for c in chosen):
            chosen.append(n)
    return chosen


def write_report(results: dict[str, dict], corr: pd.DataFrame, portfolio: list[str], period: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "—"
    pct = lambda v: f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"

    lines = [
        f"# A股因子挖掘报告（tushare 本地数据） — {now}",
        "",
        f"- 数据：tushare 全市场日频，回测区间 {period}",
        "- Universe：每日流通市值前 2000（TOP2000U 口径）、上市>120 交易日、非 ST",
        "- 口径：delay-1、行业中性、多空各 1 元（sum|w|=2）、完全换仓=100% TO",
        f"- 净值含单边 {COST_RATE*10000:.0f}bp 成本；检查阈值沿用 SKILL.md §5",
        "",
        "## 结果总览（按 Fitness 排序）",
        "",
        "| 因子 | 簇 | Sharpe | Fitness | 年化 | TO(日) | 最大回撤 | IC | ICIR | 净Sharpe | 未过检查 |",
        "|------|----|--------|---------|------|--------|----------|----|------|----------|----------|",
    ]
    ordered = sorted(
        (n for n, r in results.items() if "gross" in r),
        key=lambda n: results[n]["gross"]["fitness"],
        reverse=True,
    )
    for n in ordered:
        r = results[n]
        g, net = r["gross"], r["net"]
        lines.append(
            f"| {n} | {r['family']} | {fmt(g['sharpe'])} | {fmt(g['fitness'])} | {pct(g['ann_return'])} "
            f"| {pct(g['turnover'])} | {pct(g['max_drawdown'])} | {r['ic']:.3f} | {fmt(r['icir'])} "
            f"| {fmt(net['sharpe'])} | {', '.join(r['failed_checks']) or '—'} |"
        )
    for n, r in results.items():
        if "error" in r:
            lines.append(f"| {n} | — | — | — | — | — | — | — | — | — | {r['error']} |")

    lines += [
        "",
        "## 因子日收益相关性（SKILL §7.2：日收益而非累计 PnL）",
        "",
        "| |" + "|".join(corr.columns) + "|",
        "|--|" + "--|" * len(corr.columns),
    ]
    for idx, row in corr.iterrows():
        lines.append(f"| **{idx}** |" + "|".join(f"{v:+.2f}" for v in row.values) + "|")

    lines += [
        "",
        f"## 低相关组合建议（贪心，|corr|<0.5，SKILL §8）",
        "",
        "入选：" + (", ".join(f"`{n}`" for n in portfolio) if portfolio else "无"),
        "",
    ]
    if len(portfolio) >= 2:
        combo = sum(results[n]["pnl"] for n in portfolio) / len(portfolio)
        ann = combo.mean() * 252
        sharpe = combo.mean() / (combo.std() + EPS) * np.sqrt(252)
        eq = combo.cumsum()
        dd = (eq.cummax() - eq).max()
        lines += [f"等权组合：Sharpe **{sharpe:.2f}**，年化 **{ann*100:.1f}%**，最大回撤 **{dd*100:.1f}%**", ""]

    lines += [
        "## 结论要点",
        "",
        "- 阈值沿用 SKILL.md §5（Sharpe≥1.25 / Fitness≥1.0 / TO 1%–70% / DD<15%)；"
        "本地回测无 BRAIN 的 sub-universe 与 self-correlation 官方检查，提交 BRAIN 前仍需走 mine_chn_alphas.py 流程。",
        "- 涨跌停/停牌导致的不可成交未建模，反转与流动性簇的真实容量会低于回测值。",
        "- 基本面簇用 daily_basic 估值字段（EP/SP/股息率）代理 SKILL 的报表字段（ROE/现金流），逻辑同簇但字段不同。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_mine(only: str | None, start: str | None, end: str | None) -> None:
    print("loading panel...", flush=True)
    data = load_panel(start, end)
    try:
        from tushare_minute import load_minute_features

        data.update(load_minute_features(data["close"].index, data["close"].columns))
    except Exception as e:
        print(f"minute features unavailable: {e}", flush=True)
    print(f"panel: {data['close'].shape[0]} days x {data['close'].shape[1]} stocks", flush=True)
    universe = build_universe(data)
    groups = build_groups(data)
    lib = factor_library(data, groups)
    if only:
        lib = {k: v for k, v in lib.items() if only.lower() in k.lower()}

    results: dict[str, dict] = {}
    for i, (name, spec) in enumerate(lib.items()):
        print(f"[{i+1}/{len(lib)}] {name} ...", flush=True)
        try:
            sig = spec["signal"]()
            r = backtest(sig, data, groups, universe, spec["decay"])
            r["family"] = spec["family"]
            results[name] = r
            if "gross" in r:
                g = r["gross"]
                print(
                    f"    sharpe={g['sharpe']:.2f} fitness={g['fitness']:.2f} ann={g['ann_return']*100:.1f}% "
                    f"to={g['turnover']*100:.1f}% dd={g['max_drawdown']*100:.1f}% ic={r['ic']:.3f} "
                    f"fail={r['failed_checks']}",
                    flush=True,
                )
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}", "family": spec["family"]}
            print(f"    ERROR: {e}", flush=True)

    pnls = pd.DataFrame({n: r["pnl"] for n, r in results.items() if "pnl" in r})
    corr = pnls.corr()
    pnls.to_parquet(FACTOR_RET_PATH)
    portfolio = low_corr_portfolio(results, corr)

    period = f"{pnls.index.min()}..{pnls.index.max()}"
    write_report(results, corr, portfolio, period)

    dump = {
        n: {k: v for k, v in r.items() if k != "pnl"}
        for n, r in results.items()
    }
    RESULTS_PATH.write_text(json.dumps(dump, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH.name}; factor returns -> {FACTOR_RET_PATH}", flush=True)
    print(f"low-corr portfolio: {portfolio}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine A股 alphas on local tushare data")
    parser.add_argument("--mine", action="store_true")
    parser.add_argument("--only", metavar="NAME")
    parser.add_argument("--start", help="YYYYMMDD")
    parser.add_argument("--end", help="YYYYMMDD")
    args = parser.parse_args()
    if args.mine:
        run_mine(args.only, args.start, args.end)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

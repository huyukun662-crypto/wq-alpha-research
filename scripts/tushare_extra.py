"""tushare 扩展数据层 —— 对齐 WorldQuant BRAIN 的数据类别。

BRAIN 类别 -> tushare 接口映射(A股)：

| BRAIN 类别    | tushare 接口                              | 频率   | 说明 |
|---------------|-------------------------------------------|--------|------|
| Fundamental   | income/balancesheet/cashflow_vip          | 季频   | 三大报表全科目(84/152/97 列) |
| Earnings      | forecast/express_vip, disclosure_date, dividend | 季/日 | 预告快报、披露日历、分红 |
| Analyst       | report_rc                                 | 日频   | 卖方盈利预测(目标价/评级/预测EPS) |
| Model         | stk_factor_pro(裁剪), cyq_perf            | 日频   | 技术指标库、筹码分布 |
| Sentiment     | margin_detail, hk_hold, limit_list_d, top_list/inst, stk_holdernumber | 日/季 | 两融、北向、涨跌停、龙虎榜、股东户数 |
| News          | anns_d                                    | 日频   | 个股公告(标题) |
| Option        | opt_daily                                 | 日频   | ETF/指数期权(市场级,A股无个股期权) |
| Social Media  | stk_surv                                  | 日频   | 机构调研(关注度代理;tushare 无真社媒) |
| Price Volume  | (已在 tushare_data.py 完成)               | —      | 日线/指标/复权/资金流/5分钟 |

用法:
    python scripts/tushare_extra.py --list                  # 查看各接口状态与体积
    python scripts/tushare_extra.py --quarterly             # 季频(便宜,先跑)
    python scripts/tushare_extra.py --daily cyq_perf,margin_detail
    python scripts/tushare_extra.py --daily all --start 20150101 --end 20260717

所有下载按「接口 × 日期」粒度断点续传,重跑自动跳过已有文件。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tushare_data import DATA_DIR, call_with_retry, get_pro, quarter_periods, trade_dates  # noqa: E402

# ---------------------------------------------------------------------------
# stk_factor_pro 列裁剪:261 列 -> 后复权(hfq)口径的核心指标,控制磁盘
# ---------------------------------------------------------------------------
FACTOR_PRO_KEEP = [
    "ts_code", "trade_date",
    # 趋势/均线类
    "ma_hfq_5", "ma_hfq_10", "ma_hfq_20", "ma_hfq_60", "ma_hfq_250",
    "ema_hfq_5", "ema_hfq_20", "ema_hfq_60", "bbi_hfq",
    # 动量/振荡
    "macd_hfq", "macd_dif_hfq", "macd_dea_hfq",
    "kdj_k_hfq", "kdj_d_hfq", "kdj_hfq",
    "rsi_hfq_6", "rsi_hfq_12", "rsi_hfq_24",
    "cci_hfq", "wr_hfq", "roc_hfq", "mtm_hfq", "trix_hfq", "dpo_hfq",
    "bias1_hfq", "bias2_hfq", "bias3_hfq", "psy_hfq", "psyma_hfq",
    # 波动/通道
    "atr_hfq", "boll_upper_hfq", "boll_mid_hfq", "boll_lower_hfq",
    "ktn_upper_hfq", "ktn_mid_hfq", "ktn_down_hfq",
    # 量能/资金
    "obv_hfq", "mfi_hfq", "vr_hfq", "emv_hfq", "asi_hfq", "cr_hfq", "brar_ar_hfq", "brar_br_hfq",
    # 趋向/强弱
    "dmi_adx_hfq", "dmi_pdi_hfq", "dmi_mdi_hfq", "mass_hfq",
    # 计数型
    "updays", "downdays", "topdays", "lowdays",
]

# 接口配置: name -> (参数名, 保留列 or None=全部, 说明)
PER_DATE = {
    # Model
    "cyq_perf": ("trade_date", None, "筹码分布/获利盘"),
    "stk_factor_pro": ("trade_date", FACTOR_PRO_KEEP, "技术指标库(裁剪至hfq核心)"),
    # Sentiment
    "margin_detail": ("trade_date", None, "融资融券明细"),
    "hk_hold": ("trade_date", ["ts_code", "trade_date", "vol", "ratio"], "沪深港通持股"),
    "limit_list_d": ("trade_date", None, "涨跌停/炸板"),
    "top_list": ("trade_date", None, "龙虎榜个股"),
    "top_inst": ("trade_date", None, "龙虎榜机构席位"),
    # Analyst
    "report_rc": ("report_date", None, "卖方盈利预测"),
    # News
    "anns_d": ("trade_date", ["ann_date", "ts_code", "title"], "个股公告标题"),
    # Social(代理)
    "stk_surv": ("trade_date", ["ts_code", "surv_date", "fund_visitors", "rece_org", "org_type"], "机构调研"),
    # Earnings
    "dividend": ("ex_date", None, "分红送股"),
    # Option(市场级)
    "opt_daily": ("trade_date", None, "期权日线(ETF/指数)"),
}

PER_PERIOD = {
    "income_vip": (None, "利润表(84列)"),
    "balancesheet_vip": (None, "资产负债表(152列)"),
    "cashflow_vip": (None, "现金流量表(97列)"),
    "stk_holdernumber": (["ts_code", "ann_date", "end_date", "holder_num"], "股东户数"),
    "disclosure_date": (None, "财报披露日历"),
}


def _prune(df: pd.DataFrame, keep) -> pd.DataFrame:
    if keep is None or df is None or df.empty:
        return df
    cols = [c for c in keep if c in df.columns]
    return df[cols] if cols else df


def download_per_date(names: list[str], start: str, end: str) -> None:
    pro = get_pro()
    dates = trade_dates(pro, start, end)
    print(f"per-date endpoints: {names} over {len(dates)} dates", flush=True)
    for name in names:
        param, keep, desc = PER_DATE[name]
        out = DATA_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        fn = getattr(pro, name)
        done = t0 = 0
        t0 = time.time()
        for i, d in enumerate(dates):
            p = out / f"{d}.parquet"
            if p.exists():
                continue
            df = call_with_retry(fn, **{param: d})
            _prune(df, keep).to_parquet(p)
            done += 1
            time.sleep(0.12)
            if done and done % 100 == 0:
                mb = sum(f.stat().st_size for f in out.glob("*.parquet")) / 1e6
                print(f"  {name}: {i+1}/{len(dates)} dates, {done} new, {mb:.0f}MB, {time.time()-t0:.0f}s", flush=True)
        mb = sum(f.stat().st_size for f in out.glob("*.parquet")) / 1e6
        print(f"{name} ({desc}): {len(list(out.glob('*.parquet')))}/{len(dates)} files, {mb:.0f}MB", flush=True)


def download_per_period(names: list[str], start: str, end: str) -> None:
    pro = get_pro()
    periods = quarter_periods(start, end)
    for name in names:
        keep, desc = PER_PERIOD[name]
        out = DATA_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        fn = getattr(pro, name)
        for p_ in periods:
            path = out / f"{p_}.parquet"
            if path.exists() and p_ < periods[-2]:
                continue
            kw = {"end_date": p_} if name == "disclosure_date" else {"period": p_}
            df = call_with_retry(fn, **kw)
            _prune(df, keep).to_parquet(path)
            time.sleep(0.4)
        mb = sum(f.stat().st_size for f in out.glob("*.parquet")) / 1e6
        print(f"{name} ({desc}): {len(list(out.glob('*.parquet')))}/{len(periods)} periods, {mb:.0f}MB", flush=True)


def show_status() -> None:
    print(f"{'endpoint':22s} {'files':>7s} {'MB':>8s}  说明")
    for group, cfg in (("per-date", PER_DATE), ("per-period", PER_PERIOD)):
        print(f"-- {group} --")
        for name, spec in cfg.items():
            desc = spec[-1]
            d = DATA_DIR / name
            n = len(list(d.glob("*.parquet"))) if d.exists() else 0
            mb = sum(f.stat().st_size for f in d.glob("*.parquet")) / 1e6 if d.exists() else 0
            print(f"{name:22s} {n:7d} {mb:8.0f}  {desc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download extra tushare categories (BRAIN-aligned)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quarterly", action="store_true", help="下载全部季频接口")
    ap.add_argument("--daily", metavar="NAMES", help="逗号分隔的日频接口名,或 all")
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--end", default="20260717")
    a = ap.parse_args()

    if a.list:
        show_status()
        return 0
    if a.quarterly:
        download_per_period(list(PER_PERIOD), a.start, a.end)
    if a.daily:
        names = list(PER_DATE) if a.daily == "all" else [n.strip() for n in a.daily.split(",")]
        bad = [n for n in names if n not in PER_DATE]
        if bad:
            print(f"unknown endpoints: {bad}\navailable: {list(PER_DATE)}")
            return 1
        download_per_date(names, a.start, a.end)
    if not (a.quarterly or a.daily):
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""可转债数据层 —— tushare 落地,对齐内部 alphafactory 的 cvbd_derivative 字段口径。

字段映射(已实测核对数值一致):

| alphafactory 字段        | tushare cb_daily | 含义       |
|--------------------------|------------------|------------|
| debt_puredebt_ratio      | bond_value       | 纯债价值   |
| conv_value               | cb_value         | 转股价值   |
| bond_prem_ratio          | cb_over_rate     | 转股溢价率 |
| puredebt_prem_ratio      | bond_over_rate   | 纯债溢价率 |

用法:
    python scripts/cb_data.py --basic                       # 静态表(cb_basic/issue/call/price_chg/share)
    python scripts/cb_data.py --daily --start 20190101 --end 20260301
    python scripts/cb_data.py --status

按「日期」粒度断点续传,重跑跳过已有文件。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tushare_data import DATA_DIR, call_with_retry, get_pro, trade_dates  # noqa: E402

CB_DIR = DATA_DIR / "cb"
CB_DAILY_DIR = CB_DIR / "cb_daily"

# tushare 的坑:cb_* 接口不显式传 fields 会返回空表,必须逐个写全
CB_DAILY_FIELDS = (
    "ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount,"
    "bond_value,bond_over_rate,cb_value,cb_over_rate"
)
CB_BASIC_FIELDS = (
    "ts_code,bond_short_name,stk_code,stk_short_name,list_date,delist_date,par,"
    "issue_size,remain_size,value_date,maturity_date,rate_type,coupon_rate,"
    "conv_start_date,conv_end_date,conv_price,maturity_put_price"
)
CB_ISSUE_FIELDS = (
    "ts_code,ann_date,res_ann_date,plan_issue_size,issue_size,issue_price,issue_type,"
    "issue_cost,onl_code,onl_name,onl_date,onl_size"
)
CB_CALL_FIELDS = (
    "ts_code,call_type,is_call,ann_date,call_date,call_price,call_price_tax,"
    "call_vol,call_amount,payment_date,call_reg_date"
)
CB_PRICE_CHG_FIELDS = (
    "ts_code,bond_short_name,publish_date,change_date,convert_price_initial,"
    "convertprice_bef,convertprice_aft"
)


def download_basic(pro) -> None:
    """静态/事件表。cb_basic 一次拉全,cb_share/cb_price_chg 需逐券。"""
    CB_DIR.mkdir(parents=True, exist_ok=True)

    basic = call_with_retry(pro.cb_basic, fields=CB_BASIC_FIELDS)
    basic.to_parquet(CB_DIR / "cb_basic.parquet", index=False)
    print(f"cb_basic: {len(basic)} 只(在市 {basic['delist_date'].isna().sum()})")

    for api, fields in (("cb_issue", CB_ISSUE_FIELDS), ("cb_call", CB_CALL_FIELDS)):
        out = CB_DIR / f"{api}.parquet"
        if out.exists():
            print(f"{api}: 已存在,跳过")
            continue
        df = call_with_retry(getattr(pro, api), fields=fields)
        df.to_parquet(out, index=False)
        print(f"{api}: {len(df)} 行")

    # 转股价变动(下修/派息调整)——逐券拉,单次返回上限低
    out = CB_DIR / "cb_price_chg.parquet"
    if out.exists():
        print("cb_price_chg: 已存在,跳过")
    else:
        frames = []
        codes = basic["ts_code"].tolist()
        for i, code in enumerate(codes):
            try:
                df = call_with_retry(pro.cb_price_chg, ts_code=code, fields=CB_PRICE_CHG_FIELDS)
                if len(df):
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                print(f"  cb_price_chg {code} 失败: {str(exc)[:60]}")
            if (i + 1) % 200 == 0:
                print(f"  cb_price_chg {i + 1}/{len(codes)}")
            time.sleep(0.12)
        chg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        chg.to_parquet(out, index=False)
        print(f"cb_price_chg: {len(chg)} 行")


def download_daily(pro, start: str, end: str) -> None:
    CB_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    dates = trade_dates(pro, start, end)
    todo = [d for d in dates if not (CB_DAILY_DIR / f"{d}.parquet").exists()]
    print(f"cb_daily: {len(dates)} 个交易日,待下载 {len(todo)}")

    t0 = time.time()
    for i, d in enumerate(todo):
        df = call_with_retry(pro.cb_daily, trade_date=d, fields=CB_DAILY_FIELDS)
        df.to_parquet(CB_DAILY_DIR / f"{d}.parquet", index=False)
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(todo)} {d} rows={len(df)} {rate:.1f}/s "
                  f"eta={(len(todo) - i - 1) / max(rate, 1e-6) / 60:.1f}min")
    print(f"cb_daily 完成,已有 {len(list(CB_DAILY_DIR.glob('*.parquet')))} 天")


def status() -> None:
    for p in sorted(CB_DIR.glob("*.parquet")):
        print(f"{p.name:24s} {p.stat().st_size / 1e6:8.2f} MB")
    if CB_DAILY_DIR.exists():
        files = sorted(CB_DAILY_DIR.glob("*.parquet"))
        size = sum(f.stat().st_size for f in files) / 1e6
        span = f"{files[0].stem}~{files[-1].stem}" if files else "-"
        print(f"{'cb_daily/':24s} {size:8.2f} MB  {len(files)} 天  {span}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic", action="store_true")
    ap.add_argument("--daily", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--start", default="20190101")
    ap.add_argument("--end", default="20260301")
    args = ap.parse_args()

    if args.status:
        status()
        return

    pro = get_pro()
    if args.basic:
        download_basic(pro)
    if args.daily:
        download_daily(pro, args.start, args.end)


if __name__ == "__main__":
    main()

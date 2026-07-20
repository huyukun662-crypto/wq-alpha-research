"""Tushare A股数据下载与本地缓存（wq-alpha-research 数据层）。

用法:
    export TUSHARE_TOKEN=...   # 或在 skill 根目录放 tushare_token.txt（已 gitignore）
    python scripts/tushare_data.py --download --start 20230101 --end 20260717

缓存结构（data_tushare/ 已 gitignore）:
    data_tushare/daily/YYYYMMDD.parquet        # 全市场日线
    data_tushare/daily_basic/YYYYMMDD.parquet  # 换手率/PE/PB/市值等
    data_tushare/adj_factor/YYYYMMDD.parquet   # 复权因子
    data_tushare/stock_basic.parquet           # 股票列表 + 行业

断点续传：已存在的日期文件自动跳过。限流：捕获异常退避重试。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DATA_DIR = SKILL_DIR / "data_tushare"
TOKEN_PATH = SKILL_DIR / "tushare_token.txt"

ENDPOINTS = ("daily", "daily_basic", "adj_factor")


def load_token() -> str:
    token = os.getenv("TUSHARE_TOKEN")
    if token:
        return token.strip()
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Set TUSHARE_TOKEN env var or create untracked tushare_token.txt")


def get_pro():
    import tushare as ts

    return ts.pro_api(load_token())


def call_with_retry(fn, retries: int = 6, **kwargs) -> pd.DataFrame:
    delay = 5
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as e:  # tushare 限流以普通 Exception 抛出
            if attempt == retries - 1:
                raise
            msg = str(e)
            wait = 65 if ("每分钟" in msg or "访问频率" in msg or "权限" not in msg) else delay
            print(f"  retry after {wait}s: {msg[:120]}", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def trade_dates(pro, start: str, end: str) -> list[str]:
    cal = call_with_retry(pro.trade_cal, exchange="SSE", start_date=start, end_date=end, is_open="1")
    return sorted(cal["cal_date"].tolist())


def download(start: str, end: str) -> None:
    pro = get_pro()
    for ep in ENDPOINTS:
        (DATA_DIR / ep).mkdir(parents=True, exist_ok=True)

    sb_path = DATA_DIR / "stock_basic.parquet"
    if not sb_path.exists():
        # 含已退市/暂停上市，避免行业映射的幸存者偏差
        parts = []
        for status in ("L", "D", "P"):
            parts.append(
                call_with_retry(
                    pro.stock_basic,
                    list_status=status,
                    fields="ts_code,name,industry,market,list_date,exchange",
                )
            )
            time.sleep(0.3)
        sb = pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")
        sb.to_parquet(sb_path)
        print(f"stock_basic: {len(sb)} rows (incl. delisted)", flush=True)

    dates = trade_dates(pro, start, end)
    print(f"trade dates: {len(dates)} ({dates[0]}..{dates[-1]})", flush=True)

    n_calls = 0
    t0 = time.time()
    for i, d in enumerate(dates):
        for ep in ENDPOINTS:
            path = DATA_DIR / ep / f"{d}.parquet"
            if path.exists():
                continue
            df = call_with_retry(getattr(pro, ep), trade_date=d)
            df.to_parquet(path)
            n_calls += 1
            time.sleep(0.15)  # ~400 次/分钟以内
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dates)} dates done, {n_calls} calls, {time.time()-t0:.0f}s", flush=True)
    print(f"download complete: {len(dates)} dates, {n_calls} new calls", flush=True)


def load_panel(start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    """把缓存拼成宽表字典：{字段: DataFrame(index=date, columns=ts_code)}。"""
    frames: dict[str, list[pd.DataFrame]] = {ep: [] for ep in ENDPOINTS}
    for ep in ENDPOINTS:
        for p in sorted((DATA_DIR / ep).glob("*.parquet")):
            d = p.stem
            if (start and d < start) or (end and d > end):
                continue
            frames[ep].append(pd.read_parquet(p))
    daily = pd.concat(frames["daily"], ignore_index=True)
    basic = pd.concat(frames["daily_basic"], ignore_index=True)
    adj = pd.concat(frames["adj_factor"], ignore_index=True)

    def pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
        out = df.pivot_table(index="trade_date", columns="ts_code", values=col, aggfunc="last")
        return out.sort_index().astype("float32")

    data: dict[str, pd.DataFrame] = {}
    for col in ("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"):
        data[col] = pivot(daily, col)
    for col in (
        "turnover_rate", "turnover_rate_f", "volume_ratio", "pe_ttm", "pb",
        "ps_ttm", "dv_ttm", "total_mv", "circ_mv",
    ):
        data[col] = pivot(basic, col)
    data["adj_factor"] = pivot(adj, "adj_factor")

    # 对齐所有矩阵到同一 index/columns
    idx = data["close"].index
    cols = data["close"].columns
    for k in list(data):
        data[k] = data[k].reindex(index=idx, columns=cols)

    # 派生字段
    data["returns"] = data["pct_chg"] / 100.0
    data["close_adj"] = data["close"] * data["adj_factor"]
    data["open_adj"] = data["open"] * data["adj_factor"]
    data["high_adj"] = data["high"] * data["adj_factor"]
    data["low_adj"] = data["low"] * data["adj_factor"]
    # vol 单位=手, amount 单位=千元 -> vwap(元) = amount*1000 / (vol*100)
    data["vwap"] = (data["amount"] * 1000.0) / (data["vol"] * 100.0)

    sb = pd.read_parquet(DATA_DIR / "stock_basic.parquet")
    data["_stock_basic"] = sb
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/cache A股 data from tushare")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--start", default="20230101")
    parser.add_argument("--end", default="20260717")
    args = parser.parse_args()
    if args.download:
        download(args.start, args.end)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

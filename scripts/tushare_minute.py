"""Tushare A股分钟线下载 + 日内特征日频化。

分钟数据按单股拉取（stk_mins,单次约 8000 行上限），全市场全历史不可行；
默认方案：5 分钟线 × 全市场 × 近 2 年，用于构建「高频因子日频化」特征。

用法:
    python scripts/tushare_minute.py --download --freq 5min --start 20240701 --end 20260717
    python scripts/tushare_minute.py --aggregate          # 5min -> 日频日内特征

缓存:
    data_tushare/min5/{ts_code}.parquet     # 每股一个文件，断点续传按股票粒度
    data_tushare/min5_features.parquet      # 聚合后的日频特征（长表）

日内特征（经典高频日频化因子）:
    late_vol_share : 尾盘(14:35 后含收盘竞价)成交额占全天比例
    intraday_skew  : 日内 5 分钟收益率偏度
    open30_ret     : 开盘 30 分钟收益
    intraday_vol   : 日内 5 分钟收益率标准差
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tushare_data import DATA_DIR, call_with_retry, get_pro  # noqa: E402

FREQ_DIR = {"1min": "min1", "5min": "min5", "15min": "min15", "60min": "min60"}
FEATURES_PATH = DATA_DIR / "min5_features.parquet"  # 首窗(20240701-20260717)遗留命名

# 历史回溯按窗口分目录，断点续传粒度 = 股票 × 窗口
LEGACY_WINDOW = ("5min", "20240701", "20260717")


def window_dir(freq: str, start: str, end: str) -> Path:
    if (freq, start, end) == LEGACY_WINDOW:
        return DATA_DIR / "min5"
    return DATA_DIR / f"{FREQ_DIR[freq]}_{start}_{end}"


def features_path_for(dir_path: Path) -> Path:
    if dir_path.name == "min5":
        return FEATURES_PATH
    return DATA_DIR / f"min5_features_{dir_path.name}.parquet"


def chunk_ranges(start: str, end: str, days: int = 140) -> list[tuple[str, str]]:
    """把 [start, end] 切成 <=days 天的段，避开单次 8000 行上限。"""
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    out = []
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=days - 1), e)
        out.append((cur.strftime("%Y-%m-%d 09:00:00"), nxt.strftime("%Y-%m-%d 15:30:00")))
        cur = nxt + timedelta(days=1)
    return out


def download(freq: str, start: str, end: str) -> None:
    pro = get_pro()
    out_dir = window_dir(freq, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)

    sb = pd.read_parquet(DATA_DIR / "stock_basic.parquet")
    codes = sorted(sb["ts_code"].unique())
    ranges = chunk_ranges(start, end)
    print(f"{freq}: {len(codes)} stocks x {len(ranges)} chunks", flush=True)

    done = 0
    t0 = time.time()
    for code in codes:
        path = out_dir / f"{code}.parquet"
        if path.exists():
            done += 1
            continue
        parts = []
        for s, e in ranges:
            df = call_with_retry(pro.stk_mins, ts_code=code, freq=freq, start_date=s, end_date=e)
            if df is not None and len(df):
                parts.append(df)
            time.sleep(0.12)
        full = (
            pd.concat(parts, ignore_index=True).drop_duplicates("trade_time").sort_values("trade_time")
            if parts
            else pd.DataFrame(columns=["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"])
        )
        full.to_parquet(path)  # 空文件也落盘=该股完成标记
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(codes)} stocks, {time.time()-t0:.0f}s", flush=True)
    print(f"minute download complete: {done}/{len(codes)}", flush=True)


def _aggregate_one(df: pd.DataFrame) -> pd.DataFrame:
    """单股 5min -> 日频特征（全向量化）。"""
    t = pd.to_datetime(df["trade_time"])
    df = df.assign(day=t.dt.strftime("%Y%m%d"), hm=t.dt.strftime("%H:%M")).sort_values("trade_time")

    g = df.groupby("day")
    # 日内 bar 收益（跨日断开）
    ret = df["close"].pct_change()
    ret[g.cumcount() == 0] = np.nan
    df = df.assign(ret=ret)
    gg = df.groupby("day")

    n_bars = gg["close"].count()
    total_amt = gg["amount"].sum()
    late_amt = df["amount"].where(df["hm"] >= "14:35", 0.0).groupby(df["day"]).sum()
    skew = gg["ret"].skew()
    vol = gg["ret"].std()

    m30 = df[df["hm"] <= "10:00"]
    o30 = m30.groupby("day").agg(first_open=("open", "first"), last_close=("close", "last"))
    open30 = o30["last_close"] / (o30["first_open"] + 1e-12) - 1.0

    out = pd.DataFrame(
        {
            "late_vol_share": late_amt / total_amt.replace(0, np.nan),
            "intraday_skew": skew,
            "open30_ret": open30,
            "intraday_vol": vol,
        }
    )
    out = out[(n_bars >= 20) & total_amt.gt(0)]
    out.index.name = "trade_date"
    return out.reset_index()


def aggregate_dir(src: Path) -> None:
    """单个窗口目录：5min -> 每股每日日内特征（长表落盘，按股票断点续传）。"""
    feat_path = features_path_for(src)
    files = sorted(src.glob("*.parquet"))
    done_codes: set[str] = set()
    parts: list[pd.DataFrame] = []
    if feat_path.exists():
        prev = pd.read_parquet(feat_path)
        done_codes = set(prev["ts_code"].unique())
        parts.append(prev)
    todo = [p for p in files if p.stem not in done_codes]
    print(f"[{src.name}] aggregating {len(todo)}/{len(files)} stocks (resume: {len(done_codes)} done)", flush=True)

    buf: list[pd.DataFrame] = []
    for i, p in enumerate(todo):
        df = pd.read_parquet(p)
        if not df.empty:
            feat = _aggregate_one(df)
            feat.insert(1, "ts_code", p.stem)
            buf.append(feat)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
        if (i + 1) % 1000 == 0:  # 周期性落盘，进程被杀不丢进度
            parts.append(pd.concat(buf, ignore_index=True))
            buf = []
            pd.concat(parts, ignore_index=True).to_parquet(feat_path)
    if buf:
        parts.append(pd.concat(buf, ignore_index=True))
    if not parts:
        return
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(feat_path)
    print(f"saved {len(out)} stock-days -> {feat_path.name}", flush=True)


def aggregate() -> None:
    """聚合所有 5min 窗口目录。"""
    dirs = [DATA_DIR / "min5"] + sorted(DATA_DIR.glob("min5_2*"))
    for src in dirs:
        if src.is_dir():
            aggregate_dir(src)


def load_minute_features(trade_index, columns) -> dict[str, pd.DataFrame]:
    """给 mine_tushare_alphas.load 用：日内特征宽表（合并所有窗口；无数据则空 dict）。"""
    paths = [p for p in [FEATURES_PATH, *sorted(DATA_DIR.glob("min5_features_*.parquet"))] if p.exists()]
    if not paths:
        return {}
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out = {}
    for f in ("late_vol_share", "intraday_skew", "open30_ret", "intraday_vol"):
        wide = df.pivot_table(index="trade_date", columns="ts_code", values=f, aggfunc="last")
        out[f] = wide.reindex(index=trade_index, columns=columns).astype("float32")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/aggregate A股 minute bars from tushare")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--freq", default="5min", choices=list(FREQ_DIR))
    parser.add_argument("--start", default="20240701")
    parser.add_argument("--end", default="20260717")
    args = parser.parse_args()
    if args.download:
        download(args.freq, args.start, args.end)
    if args.aggregate:
        aggregate()
    if not (args.download or args.aggregate):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""BRAIN 九大类别扩展数据 -> 日频宽表面板。

把 tushare_extra.py 下载的各类别长表装配成 index=trade_date, columns=ts_code 的宽表，
缓存到 data_tushare/panel_extra/，供 mine_tushare_alphas.py 的新因子簇使用。

对齐口径（杜绝未来函数）:
    - trade_date 口径的接口（cyq_perf / margin_detail / hk_hold / limit_list_d /
      top_list / top_inst / stk_factor_pro）直接按交易日对齐；信号在 t 日算，
      回测里 w.shift(1) 保证 t+1 才建仓。
    - 非交易日口径的接口（report_rc 的 report_date、dividend 的 ann_date、
      stk_surv 的 surv_date）一律对齐到 **严格晚于该日期的第一个交易日**，
      再向前 ffill —— 研报/公告当日盘中发布，同日入场属于未来函数。

内存策略: 逐文件填 numpy 数组，不做全量 concat（stk_factor_pro 有 4.6GB）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from tushare_data import DATA_DIR  # noqa: E402

PANEL_EXTRA_DIR = DATA_DIR / "panel_extra"

# 计数型字段：接口覆盖了全部交易日，缺失即"当日无此事件"，应填 0 而非 NaN
COUNT_LIKE = {
    "limit_up", "limit_down", "limit_open_times", "limit_fd_amount",
    "lhb_net_amount", "lhb_buy", "lhb_sell", "lhb_net_rate",
    "inst_net_buy", "inst_seats",
    "surv_count", "surv_visitors",
}

# stk_factor_pro 只取最有区分度的技术指标（全 hfq 口径，避免复权断层）
FACTOR_PRO_FIELDS = (
    "ma_hfq_20", "ma_hfq_60",
    "macd_hfq", "macd_dif_hfq",
    "kdj_k_hfq", "kdj_d_hfq",
    "rsi_hfq_6", "rsi_hfq_24",
    "cci_hfq", "wr_hfq", "bias2_hfq", "psy_hfq",
    "atr_hfq", "mfi_hfq", "vr_hfq",
    "dmi_adx_hfq", "brar_br_hfq",
    "boll_upper_hfq", "boll_lower_hfq",
)


def _new_arrs(fields, n_dates, n_codes) -> dict[str, np.ndarray]:
    return {f: np.full((n_dates, n_codes), np.nan, dtype="float32") for f in fields}


def _fill(arrs, fields, df, dates, codes, date_col, date_mode="trade") -> None:
    """长表片段 -> 填入宽表数组。重复 (date, code) 取后值。

    date_mode:
        "trade" —— 接口本身就是交易日口径，直接精确对齐。
        "event" —— 事件日口径（研报/公告/调研）。落到 **严格晚于事件日的第一个
                   交易日**；这样既避免了盘中发布当日生效的未来函数，也不会像
                   精确匹配那样把周末/节假日发生的事件整条丢掉。
    """
    if df.empty or "ts_code" not in df.columns:
        return
    d = df[date_col].astype(str).to_numpy()
    if date_mode == "event":
        r = dates.searchsorted(d, side="right").astype("int64")
        r[r >= len(dates)] = -1
    else:
        r = dates.get_indexer(d)
    c = codes.get_indexer(df["ts_code"].astype(str).to_numpy())
    ok = (r >= 0) & (c >= 0)
    if not ok.any():
        return
    r, c = r[ok], c[ok]
    for f in fields:
        if f not in df.columns:
            continue
        v = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype="float32")[ok]
        arrs[f][r, c] = v


def _iter_files(ep: str):
    d = DATA_DIR / ep
    if not d.exists():
        return
    for p in sorted(d.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if len(df):
            yield df


def _assemble_ep(ep, fields, dates, codes, date_col="trade_date", prep=None, date_mode="trade"):
    """逐文件装配一个接口。prep(df) 可做派生列/聚合。"""
    arrs = _new_arrs(fields, len(dates), len(codes))
    n = 0
    for df in _iter_files(ep):
        if prep is not None:
            df = prep(df)
        if df is None or df.empty:
            continue
        _fill(arrs, fields, df, dates, codes, date_col, date_mode)
        n += 1
    if n == 0:
        return {}
    print(f"  {ep}: {n} files -> {len(fields)} fields", flush=True)
    return {f: pd.DataFrame(arrs[f], index=dates, columns=codes) for f in fields}


# ---------------------------------------------------------------------------
# 各接口的派生/聚合逻辑
# ---------------------------------------------------------------------------
def _prep_hk(df: pd.DataFrame) -> pd.DataFrame:
    # vol 与 daily.vol 重名，改名避免覆盖行情字段
    return df.rename(columns={"ratio": "hk_ratio", "vol": "hk_vol"})


def _prep_limit(df: pd.DataFrame) -> pd.DataFrame:
    lim = df["limit"].astype(str)
    return df.assign(
        limit_up=(lim == "U").astype("float32"),
        limit_down=(lim == "D").astype("float32"),
        limit_open_times=pd.to_numeric(df.get("open_times"), errors="coerce"),
        limit_fd_amount=pd.to_numeric(df.get("fd_amount"), errors="coerce"),
    )


def _prep_top_list(df: pd.DataFrame) -> pd.DataFrame:
    # 同一股票同日可能因多个上榜原因出现多行 -> 汇总
    g = df.groupby(["trade_date", "ts_code"], as_index=False).agg(
        lhb_net_amount=("net_amount", "sum"),
        lhb_buy=("l_buy", "sum"),
        lhb_sell=("l_sell", "sum"),
        lhb_net_rate=("net_rate", "mean"),
    )
    return g


def _prep_top_inst(df: pd.DataFrame) -> pd.DataFrame:
    # 每个席位一行 -> 汇总为机构净买入与席位数
    g = df.groupby(["trade_date", "ts_code"], as_index=False).agg(
        inst_net_buy=("net_buy", "sum"),
        inst_seats=("exalter", "count"),
    )
    return g


def _prep_report_rc(df: pd.DataFrame) -> pd.DataFrame:
    """卖方预测: 每份研报给出多个预测年度，取最近一个年度作为当期一致预期。"""
    d = df.copy()
    d["quarter"] = d["quarter"].astype(str)
    d = d[d["quarter"].str.match(r"^\d{4}Q\d$", na=False)]
    if d.empty:
        return d
    # 每个 (report_date, ts_code) 取最小预测年度 = 当年预测
    d = d.sort_values("quarter")
    near = d.groupby(["report_date", "ts_code"], as_index=False).first()
    cnt = d.groupby(["report_date", "ts_code"], as_index=False).size()
    tp_mid = d.groupby(["report_date", "ts_code"], as_index=False).agg(
        rc_tp_max=("max_price", "mean"), rc_tp_min=("min_price", "mean")
    )
    out = near[["report_date", "ts_code", "eps", "np", "roe"]].rename(
        columns={"eps": "rc_eps", "np": "rc_np", "roe": "rc_roe"}
    )
    out = out.merge(cnt.rename(columns={"size": "rc_count"}), on=["report_date", "ts_code"])
    out = out.merge(tp_mid, on=["report_date", "ts_code"])
    return out


def _prep_surv(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["surv_date", "ts_code"], as_index=False).agg(
        surv_count=("rece_org", "count"),
        surv_visitors=("fund_visitors", "sum"),
    )
    return g


def _prep_dividend(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["div_proc"].astype(str).str.contains("实施", na=False)].copy()
    if d.empty:
        return d
    d = d[d["ann_date"].notna()]
    g = d.groupby(["ann_date", "ts_code"], as_index=False).agg(
        div_cash=("cash_div_tax", "sum"), div_stk=("stk_div", "sum")
    )
    return g


# 事件型状态量的有效期（交易日）：超过就算过期，不再 ffill
STATE_TTL = 180


def _ffill_state(wide: pd.DataFrame, ttl: int = STATE_TTL) -> pd.DataFrame:
    """事件型"状态量"（如一致预期 EPS）在两次事件之间保持不变。

    _fill(date_mode="event") 已把值放到严格晚于事件日的第一个交易日，
    这里只负责向前填充；超过 ttl 个交易日没有新研报就置为 NaN（预期过期）。
    """
    return wide.ffill(limit=ttl)


def build_extra_panel(dates: pd.Index, codes: pd.Index, events_only: bool = False) -> None:
    """装配全部扩展面板并写入 panel_extra/ 缓存。

    events_only=True 时只重建事件日口径的接口（report_rc/stk_surv/dividend），
    避免为了修一个对齐口径而重读 4.6GB 的 stk_factor_pro。
    """
    PANEL_EXTRA_DIR.mkdir(exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    if events_only:
        rc = _assemble_ep(
            "report_rc",
            ("rc_eps", "rc_np", "rc_roe", "rc_count", "rc_tp_max", "rc_tp_min"),
            dates, codes, date_col="report_date", prep=_prep_report_rc, date_mode="event",
        )
        for k, v in rc.items():
            out[k] = _ffill_state(v)
        out.update(_assemble_ep(
            "stk_surv", ("surv_count", "surv_visitors"),
            dates, codes, date_col="surv_date", prep=_prep_surv, date_mode="event",
        ))
        out.update(_assemble_ep(
            "dividend", ("div_cash", "div_stk"),
            dates, codes, date_col="ann_date", prep=_prep_dividend, date_mode="event",
        ))
        for name in list(out):
            if name in COUNT_LIKE:
                out[name] = out[name].fillna(0.0)
        for name, wide in out.items():
            wide.astype("float32").to_parquet(PANEL_EXTRA_DIR / f"{name}.parquet")
        print(f"panel_extra (events only) rebuilt: {len(out)} fields", flush=True)
        return

    # ---- 交易日口径：直接对齐 ----
    out.update(_assemble_ep(
        "cyq_perf",
        ("winner_rate", "cost_15pct", "cost_50pct", "cost_85pct", "weight_avg"),
        dates, codes,
    ))
    out.update(_assemble_ep(
        "margin_detail", ("rzye", "rqye", "rzmre", "rzche", "rzrqye"), dates, codes,
    ))
    out.update(_assemble_ep("hk_hold", ("hk_ratio", "hk_vol"), dates, codes, prep=_prep_hk))
    out.update(_assemble_ep(
        "limit_list_d",
        ("limit_up", "limit_down", "limit_open_times", "limit_fd_amount"),
        dates, codes, prep=_prep_limit,
    ))
    out.update(_assemble_ep(
        "top_list", ("lhb_net_amount", "lhb_buy", "lhb_sell", "lhb_net_rate"),
        dates, codes, prep=_prep_top_list,
    ))
    out.update(_assemble_ep(
        "top_inst", ("inst_net_buy", "inst_seats"), dates, codes, prep=_prep_top_inst,
    ))
    out.update(_assemble_ep("stk_factor_pro", FACTOR_PRO_FIELDS, dates, codes))

    # ---- 事件日口径：落到下一交易日；状态量再 ffill，流量型保持稀疏 ----
    rc = _assemble_ep(
        "report_rc",
        ("rc_eps", "rc_np", "rc_roe", "rc_count", "rc_tp_max", "rc_tp_min"),
        dates, codes, date_col="report_date", prep=_prep_report_rc, date_mode="event",
    )
    # 一致预期是"状态"：两份研报之间应保持上一份的值，否则 ts_delta 全是 NaN
    for k, v in rc.items():
        out[k] = _ffill_state(v)

    # 调研/分红是"流量"（当期发生量），稀疏即真实，不 ffill；
    # 因子侧用 ts_mean 做滚动窗口聚合。
    out.update(_assemble_ep(
        "stk_surv", ("surv_count", "surv_visitors"),
        dates, codes, date_col="surv_date", prep=_prep_surv, date_mode="event",
    ))
    out.update(_assemble_ep(
        "dividend", ("div_cash", "div_stk"),
        dates, codes, date_col="ann_date", prep=_prep_dividend, date_mode="event",
    ))

    # ---- 计数型字段：接口覆盖全交易日，缺失=无事件，填 0 ----
    for name in list(out):
        if name in COUNT_LIKE:
            out[name] = out[name].fillna(0.0)

    for name, wide in out.items():
        wide.astype("float32").to_parquet(PANEL_EXTRA_DIR / f"{name}.parquet")
    print(f"panel_extra cache saved: {len(out)} fields", flush=True)


def load_extra_panel(dates: pd.Index, codes: pd.Index) -> dict[str, pd.DataFrame]:
    """加载扩展面板；无缓存则先构建。返回按 (dates, codes) 对齐的宽表。"""
    if not PANEL_EXTRA_DIR.exists() or not any(PANEL_EXTRA_DIR.glob("*.parquet")):
        return {}
    data: dict[str, pd.DataFrame] = {}
    for p in sorted(PANEL_EXTRA_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        data[p.stem] = df.reindex(index=dates, columns=codes)

    # ---- 派生字段 ----
    if "cost_50pct" in data and "cost_15pct" in data:
        # 筹码集中度: (85分位-15分位)/中位数，越小越集中
        data["chip_concentration"] = (
            (data["cost_85pct"] - data["cost_15pct"]) / (data["cost_50pct"].abs() + 1e-6)
        )
    if "rc_tp_max" in data and "rc_tp_min" in data:
        data["rc_tp_mid"] = (data["rc_tp_max"] + data["rc_tp_min"]) / 2.0
    return data


def main() -> int:
    import argparse

    from tushare_data import load_panel

    ap = argparse.ArgumentParser(description="Build BRAIN extra panel cache")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--events-only", action="store_true", help="只重建事件日口径接口")
    args = ap.parse_args()
    if not (args.build or args.events_only):
        ap.print_help()
        return 1
    base = load_panel(None, None)
    build_extra_panel(base["close"].index, base["close"].columns, events_only=args.events_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())

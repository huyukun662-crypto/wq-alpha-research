"""向量化评估。

把 (date, b_sym) 长表摊成 T×N 宽矩阵后,一次组合评估只剩 numpy 运算,
使权重网格搜索从小时级降到秒级。口径与 cb_panel.evaluate 逐项对齐,
用 --verify 可复核两条路径给出相同数字。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

TRADING_DAYS = 252


class FastEval:
    def __init__(self, panel: pd.DataFrame):
        p = panel.sort_values(["date", "b_sym"])
        self.dates = np.array(sorted(p["date"].unique()))
        self.syms = np.array(sorted(p["b_sym"].unique()))
        self.di = pd.Series(np.arange(len(self.dates)), index=self.dates)
        self.si = pd.Series(np.arange(len(self.syms)), index=self.syms)
        self.T, self.N = len(self.dates), len(self.syms)

        # 排序后行序与传入 panel 的行序不同。外部算出的因子仍带原 panel 索引,
        # 必须按这个索引对齐后再散射,否则信号与收益错位(IC 会塌到 0)。
        self._order = p.index
        self._r = self.di.reindex(p["date"]).to_numpy()
        self._c = self.si.reindex(p["b_sym"]).to_numpy()

        self.ret = self._to_wide(p["fwd_ret"].to_numpy())
        self.univ = self._to_wide(p["in_univ"].astype(float).to_numpy()) > 0.5
        self.valid = self.univ & np.isfinite(self.ret)

    def _to_wide(self, vals: np.ndarray) -> np.ndarray:
        out = np.full((self.T, self.N), np.nan)
        out[self._r, self._c] = vals
        return out

    def wide(self, series) -> np.ndarray:
        if isinstance(series, pd.Series):
            v = series.reindex(self._order).to_numpy()
        else:
            v = np.asarray(series)
        return self._to_wide(v)

    # -- 算子 --
    @staticmethod
    def _row_z(x: np.ndarray, eps: float = 0.0004) -> np.ndarray:
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
        return (x - mu) / (sd + eps)

    @staticmethod
    def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return x
        df = pd.DataFrame(x)
        return df.rolling(w, min_periods=1).mean().to_numpy()

    def evaluate(self, sig_wide: np.ndarray, roll: int = 3,
                 lo: int | None = None, hi: int | None = None) -> dict:
        s = np.where(self.univ, sig_wide, np.nan)
        s = self._roll_mean(s, roll)
        s = np.where(self.univ, s, np.nan)
        s = self._row_z(s)
        s = np.where(self.valid, s, np.nan)

        w = s / np.nansum(np.abs(s), axis=1, keepdims=True)
        w0 = np.nan_to_num(w)

        pnl_all = np.nansum(w0 * np.nan_to_num(self.ret), axis=1)
        dw = np.abs(np.diff(w0, axis=0)).sum(axis=1)
        gross = np.abs(w0).sum(axis=1)

        lo = 0 if lo is None else lo
        hi = self.T if hi is None else hi
        sl = slice(lo, hi)

        ic = np.full(self.T, np.nan)
        ric = np.full(self.T, np.nan)
        for t in range(lo, hi):
            m = np.isfinite(s[t]) & np.isfinite(self.ret[t])
            if m.sum() < 10:
                continue
            a, b = s[t, m], self.ret[t, m]
            ic[t] = np.corrcoef(a, b)[0, 1]
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            ric[t] = np.corrcoef(ra, rb)[0, 1]

        icw = ic[sl][np.isfinite(ic[sl])]
        ricw = ric[sl][np.isfinite(ric[sl])]
        pnl = pnl_all[sl]
        # 与 cb_panel.evaluate 对齐:只取窗口内部的相邻差分,丢掉窗口第一天
        tw = dw[lo:hi - 1] / np.where(gross[lo + 1:hi] == 0, np.nan, gross[lo + 1:hi])
        breadth = np.isfinite(s[sl]).sum(axis=1)

        if len(icw) == 0 or pnl.std() == 0:
            return {"n_days": 0}
        return {
            "n_days": int(len(pnl)),
            "breadth": float(breadth.mean()),
            "IC": float(icw.mean()),
            "RankIC": float(ricw.mean()),
            "ICIR": float(icw.mean() / icw.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "RankICIR": float(ricw.mean() / ricw.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "IC_pos_rate": float((icw > 0).mean()),
            "Sharpe": float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "AnnRet": float(pnl.mean() * TRADING_DAYS),
            "MaxDD": float(np.max(np.maximum.accumulate(pnl.cumsum()) - pnl.cumsum())),
            "turnover": float(np.nanmean(tw) / 2.0),
            "turnover_gross": float(np.nanmean(tw)),
        }

    def window(self, start: str, end: str) -> tuple[int, int]:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        lo = int(np.searchsorted(self.dates, s, side="left"))
        hi = int(np.searchsorted(self.dates, e, side="right"))
        return lo, hi

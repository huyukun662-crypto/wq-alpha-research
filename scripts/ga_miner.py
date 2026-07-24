"""GA 因子挖掘机 —— 遗传算法 × tushare 本地回测引擎。

结合 Worldquant_Mining 仓库的方式(文法约束表达式生成、窗口参数集、
双闸门惩罚、严格 IS/OOS 纪律),升级为完整遗传算法(种群/锦标赛选择/
子树交叉/多种变异/精英保留/名人堂)。

核心纪律:
  - GA 全程只看 IS 段(2015-01 ~ 2022-12)的净指标;
  - OOS 段(2023-01 ~ 2026-07)只在收尾时对名人堂验证一次;
  - fitness = IS 净 Sharpe - 惩罚(高换手 / 无效解 / 复杂度 / 与名人堂高相关)。

用法:
    python scripts/ga_miner.py --smoke            # 冒烟: 种群8 x 2代
    python scripts/ga_miner.py --run              # 标准: 种群60 x 20代(断点续传)
    python scripts/ga_miner.py --run --population 100 --generations 40
    python scripts/ga_miner.py --report           # 只生成报告(含 OOS 验证)

状态文件(均已 gitignore):
    ga_state.json   # 代数/种群/名人堂(含 IS pnl)
    ga_cache.json   # 表达式 md5 -> 指标缓存(跨代、跨进程复用)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal as _signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from tushare_data import load_panel  # noqa: E402
import mine_tushare_alphas as mt  # noqa: E402

STATE_PATH = SKILL_DIR / "ga_state.json"
CACHE_PATH = SKILL_DIR / "ga_cache.json"
REPORT_PATH = SKILL_DIR / "ga_mining_report.md"

IS_END = "20221231"    # GA 只能看到这之前
OOS_START = "20230101"  # 名人堂收尾验证用

COST_RATE = mt.COST_RATE
EPS = 1e-9

WINDOWS = (3, 5, 10, 20, 40, 60, 120, 250)

# ---------------------------------------------------------------------------
# 终端集(叶子字段),按量纲分组;div 优先同组配对
# ---------------------------------------------------------------------------
FIELD_GROUPS = {
    "price": ["close_adj", "open_adj", "high_adj", "low_adj", "vwap"],
    "volume": ["vol", "amount", "turnover_rate_f", "volume_ratio"],
    "ratio": [
        "returns", "ep", "bp", "sp", "dv_ttm", "roe", "netprofit_yoy",
        "grossprofit_margin", "debt_to_assets", "lg_ratio",
        "late_vol_share", "intraday_skew", "open30_ret", "intraday_vol",
    ],
    "size": ["log_circ_mv"],
}
ALL_FIELDS = [f for g in FIELD_GROUPS.values() for f in g]
FIELD_TO_GROUP = {f: g for g, fields in FIELD_GROUPS.items() for f in fields}

TS_OPS = ("ts_rank", "ts_mean", "ts_std_dev", "ts_delta", "ts_min", "ts_max", "ts_decay_linear", "ts_zscore")
BIN_OPS = ("add", "sub", "mul", "div", "ts_corr")
UN_OPS = ("neg", "abs", "sign", "slog1p")
ROOT_OPS = ("rank", "group_rank")

MAX_DEPTH = 4
MAX_NODES = 12


# ---------------------------------------------------------------------------
# 表达式树:JSON 结构 {"op":..,"w":..,"kids":[..]} / {"leaf":..}
# ---------------------------------------------------------------------------
def leaf(field: str) -> dict:
    return {"leaf": field}


def node(op: str, kids: list, w: int | None = None) -> dict:
    d = {"op": op, "kids": kids}
    if w is not None:
        d["w"] = w
    return d


def to_str(t: dict) -> str:
    if "leaf" in t:
        return t["leaf"]
    args = ", ".join(to_str(k) for k in t["kids"])
    if t["op"] == "group_rank":
        return f"group_rank({args}, industry)"
    if "w" in t:
        return f"{t['op']}({args}, {t['w']})"
    return f"{t['op']}({args})"


def md5_of(t: dict) -> str:
    return hashlib.md5(to_str(t).encode()).hexdigest()


def node_count(t: dict) -> int:
    if "leaf" in t:
        return 1
    return 1 + sum(node_count(k) for k in t["kids"])


def depth_of(t: dict) -> int:
    if "leaf" in t:
        return 1
    return 1 + max(depth_of(k) for k in t["kids"])


def all_subtrees(t: dict, _acc=None) -> list[dict]:
    """返回所有节点引用(不含根),用于交叉/变异定位。"""
    if _acc is None:
        _acc = []
    for k in t.get("kids", []):
        _acc.append(k)
        all_subtrees(k, _acc)
    return _acc


# ---------------------------------------------------------------------------
# 文法约束随机生成(参考 repo 权重: 55% ts / 25% 算术 / 12% 一元 / 8% 叶子)
# ---------------------------------------------------------------------------
def rand_leaf(rng: random.Random, group: str | None = None) -> dict:
    fields = FIELD_GROUPS[group] if group else ALL_FIELDS
    return leaf(rng.choice(fields))


def rand_core(rng: random.Random, depth: int) -> dict:
    if depth >= MAX_DEPTH:
        return rand_leaf(rng)
    r = rng.random()
    if r < 0.55:
        op = rng.choice(TS_OPS)
        return node(op, [rand_core(rng, depth + 1)], w=rng.choice(WINDOWS))
    if r < 0.80:
        op = rng.choice(BIN_OPS)
        if op == "ts_corr":
            return node(op, [rand_core(rng, depth + 1), rand_core(rng, depth + 1)], w=rng.choice(WINDOWS[:6]))
        if op == "div":
            g = rng.choice(list(FIELD_GROUPS))  # 同量纲相除
            a = rand_leaf(rng, g) if rng.random() < 0.5 else rand_core(rng, depth + 1)
            b = rand_leaf(rng, g)
            return node(op, [a, b])
        return node(op, [rand_core(rng, depth + 1), rand_core(rng, depth + 1)])
    if r < 0.92:
        return node(rng.choice(UN_OPS), [rand_core(rng, depth + 1)])
    return rand_leaf(rng)


def rand_tree(rng: random.Random) -> dict:
    core = rand_core(rng, 1)
    root = rng.choice(ROOT_OPS)
    t = node(root, [core])
    if node_count(t) > MAX_NODES:
        return rand_tree(rng)
    return t


# ---------------------------------------------------------------------------
# 种子(现有已验证因子的核心,作为进化起点)
# ---------------------------------------------------------------------------
def seed_trees() -> list[dict]:
    S = [
        # intraday_skew(全样本最强): -ts_mean(intraday_skew, 20)
        node("group_rank", [node("neg", [node("ts_mean", [leaf("intraday_skew")], w=20)])]),
        # amihud: ts_mean(|returns|/amount, 20)
        node("group_rank", [node("ts_mean", [node("div", [node("abs", [leaf("returns")]), leaf("amount")])], w=20)]),
        # 盈利收益率
        node("group_rank", [leaf("ep")]),
        # 价量背离: -ts_corr(close, vol, 10)
        node("rank", [node("neg", [node("ts_corr", [leaf("close_adj"), leaf("vol")], w=10)])]),
        # vwap 偏离: -(close/vwap)
        node("rank", [node("neg", [node("div", [leaf("close_adj"), leaf("vwap")])])]),
        # ROE 趋势
        node("group_rank", [node("ts_rank", [leaf("roe")], w=250)]),
        # 异常换手: -ts_mean(to,20)/ts_mean(to,120)
        node("group_rank", [node("neg", [node("div", [node("ts_mean", [leaf("turnover_rate_f")], w=20), node("ts_mean", [leaf("turnover_rate_f")], w=120)])])]),
        # 低波动
        node("group_rank", [node("neg", [node("ts_std_dev", [leaf("returns")], w=60)])]),
        # 5日反转
        node("group_rank", [node("neg", [node("ts_delta", [leaf("close_adj")], w=5)])]),
        # 股息率
        node("group_rank", [leaf("dv_ttm")]),
    ]
    return S


# ---------------------------------------------------------------------------
# 求值器:表达式树 -> 信号宽表(复用 mine_tushare_alphas 的算子)
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self):
        print("loading panel...", flush=True)
        data = load_panel(None, None)
        try:
            from tushare_minute import load_minute_features

            data.update(load_minute_features(data["close"].index, data["close"].columns))
        except Exception as e:
            print(f"minute features unavailable: {e}", flush=True)

        self.universe = mt.build_universe(data)
        self.groups = mt.build_groups(data)

        # 终端字段(派生量预先算好)
        t: dict[str, pd.DataFrame] = {}
        for f in ("close_adj", "open_adj", "high_adj", "low_adj", "vwap", "vol", "amount",
                  "turnover_rate_f", "volume_ratio", "returns", "dv_ttm"):
            t[f] = data[f]
        t["ep"] = 1.0 / data["pe_ttm"]
        t["bp"] = 1.0 / data["pb"]
        t["sp"] = 1.0 / data["ps_ttm"]
        t["log_circ_mv"] = np.log(data["circ_mv"] + 1.0)
        t["lg_ratio"] = data["net_lg_amount"] * 10.0 / (data["amount"] + 1.0)
        for f in ("roe", "netprofit_yoy", "grossprofit_margin", "debt_to_assets"):
            t[f] = data[f] if f in data else None
        for f in ("late_vol_share", "intraday_skew", "open30_ret", "intraday_vol"):
            t[f] = data[f] if f in data else None
        missing = [k for k, v in t.items() if v is None]
        if missing:
            raise RuntimeError(f"missing terminal fields: {missing}")
        self.terms = t

        self.returns = data["returns"]
        self.dates = data["close"].index
        self.is_mask = self.dates <= IS_END
        self.oos_mask = self.dates >= OOS_START
        print(f"panel ready: {len(self.dates)} days, IS={int(self.is_mask.sum())} OOS={int(self.oos_mask.sum())}", flush=True)

    # -- 树求值 ------------------------------------------------------------
    def eval_tree(self, t: dict) -> pd.DataFrame:
        if "leaf" in t:
            return self.terms[t["leaf"]]
        op = t["op"]
        k = [self.eval_tree(c) for c in t["kids"]]
        w = t.get("w")
        if op == "rank":
            return mt.rank(k[0])
        if op == "group_rank":
            return mt.group_rank(k[0], self.groups)
        if op == "ts_rank":
            return mt.ts_rank(k[0], w)
        if op == "ts_mean":
            return mt.ts_mean(k[0], w)
        if op == "ts_std_dev":
            return mt.ts_std_dev(k[0], w)
        if op == "ts_delta":
            return mt.ts_delta(k[0], w)
        if op == "ts_min":
            return mt.ts_min(k[0], w)
        if op == "ts_max":
            return mt.ts_max(k[0], w)
        if op == "ts_decay_linear":
            return mt.ts_decay_linear(k[0], min(w, 30))
        if op == "ts_zscore":
            return (k[0] - mt.ts_mean(k[0], w)) / (mt.ts_std_dev(k[0], w) + EPS)
        if op == "ts_corr":
            return mt.ts_corr(k[0], k[1], w)
        if op == "add":
            return k[0] + k[1]
        if op == "sub":
            return k[0] - k[1]
        if op == "mul":
            return k[0] * k[1]
        if op == "div":
            return k[0] / (k[1].abs() + EPS) * np.sign(k[1])
        if op == "neg":
            return -k[0]
        if op == "abs":
            return k[0].abs()
        if op == "sign":
            return np.sign(k[0])
        if op == "slog1p":
            return np.sign(k[0]) * np.log1p(k[0].abs())
        raise ValueError(f"unknown op {op}")

    # -- 回测(全区间一次,IS 供 fitness,OOS 只在收尾读取) -----------------
    def backtest(self, t: dict, eval_timeout: int = 240) -> dict:
        def _on_alarm(signum, frame):
            raise TimeoutError("eval timeout")

        old = _signal.signal(_signal.SIGALRM, _on_alarm)
        _signal.alarm(eval_timeout)
        try:
            sig = self.eval_tree(t)
        except TimeoutError:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, old)

        sig = sig.where(self.universe)
        w = mt.group_demean(sig, self.groups)
        gross = w.abs().sum(axis=1)
        w = w.div(gross.replace(0, np.nan), axis=0) * 2.0
        w_lag = w.shift(1)
        pnl = (w_lag * self.returns).sum(axis=1, min_count=10)
        dw = (w - w.shift(1)).abs().sum(axis=1)
        to = dw / 4.0
        holdings = (w_lag.abs() > 1e-8).sum(axis=1)

        out = {}
        for name, mask in (("is", self.is_mask), ("oos", self.oos_mask)):
            p = pnl[mask].dropna()
            if len(p) < 60:
                out[name] = None
                continue
            t_ = to.reindex(p.index).fillna(0.0)
            cost = t_ * 4.0 / 2.0 * COST_RATE * 2.0
            net = p - cost
            m = {}
            for tag, series in (("gross", p), ("net", net)):
                ann = float(series.mean() * 252)
                sharpe = float(series.mean() / (series.std() + EPS) * np.sqrt(252))
                m[f"{tag}_sharpe"] = round(sharpe, 3)
                m[f"{tag}_ann"] = round(ann, 4)
            m["turnover"] = round(float(t_.mean()), 4)
            m["days"] = len(p)
            m["avg_holdings"] = round(float(holdings.reindex(p.index).mean()), 1)
            out[name] = m
        out["is_pnl"] = [round(float(x), 6) for x in pnl[self.is_mask].dropna().tolist()]
        return out


# ---------------------------------------------------------------------------
# fitness(IS only;结合参考 repo 双闸门 + SKILL 相关性/简约压力)
# ---------------------------------------------------------------------------
def fitness_of(metrics: dict, tree: dict, hof: list[dict]) -> float:
    if "error" in metrics or not metrics.get("is"):
        return -99.0
    m = metrics["is"]
    f = m["net_sharpe"]
    if m["turnover"] > 0.35:
        f -= 5.0
    if m["days"] < 1000 or m["avg_holdings"] < 300:
        f -= 5.0
    f -= 0.03 * node_count(tree)
    # 与名人堂的 IS 日收益相关闸门(跳过自身)
    pnl = metrics.get("is_pnl") or []
    if pnl and hof:
        self_key = md5_of(tree)
        a = np.array(pnl[-1500:])
        for h in hof:
            if h.get("md5") == self_key:
                continue
            hp = np.array((h.get("is_pnl") or [])[-1500:])
            n = min(len(a), len(hp))
            if n > 200:
                c = abs(float(np.corrcoef(a[-n:], hp[-n:])[0, 1]))
                if c > 0.7:
                    f -= 2.0
                    break
    return round(f, 4)


# ---------------------------------------------------------------------------
# 遗传操作
# ---------------------------------------------------------------------------
def crossover(rng: random.Random, a: dict, b: dict) -> dict:
    child = json.loads(json.dumps(a))
    sa = all_subtrees(child)
    sb = all_subtrees(b)
    if not sa or not sb:
        return child
    target = rng.choice(sa)
    donor = json.loads(json.dumps(rng.choice(sb)))
    target.clear()
    target.update(donor)
    if node_count(child) > MAX_NODES or depth_of(child) > MAX_DEPTH + 2:
        return json.loads(json.dumps(a))
    return child


def mutate(rng: random.Random, t: dict) -> dict:
    t = json.loads(json.dumps(t))
    nodes = all_subtrees(t)
    if not nodes:
        return t
    kind = rng.random()
    target = rng.choice(nodes)
    if kind < 0.30:  # 子树重生成
        new = rand_core(rng, depth=2)
        target.clear()
        target.update(new)
    elif kind < 0.55:  # 算子同类替换
        if "op" in target:
            if target["op"] in TS_OPS:
                target["op"] = rng.choice(TS_OPS)
            elif target["op"] in ("add", "sub", "mul"):
                target["op"] = rng.choice(("add", "sub", "mul"))
            elif target["op"] in UN_OPS:
                target["op"] = rng.choice(UN_OPS)
    elif kind < 0.80:  # 字段同量纲组替换
        leaves = [n for n in nodes if "leaf" in n]
        if leaves:
            lf = rng.choice(leaves)
            g = FIELD_TO_GROUP[lf["leaf"]]
            lf["leaf"] = rng.choice(FIELD_GROUPS[g])
    else:  # 窗口抖动(参考 repo 的窗口调参思想)
        wins = [n for n in nodes if "w" in n]
        if wins:
            wn = rng.choice(wins)
            i = WINDOWS.index(wn["w"]) if wn["w"] in WINDOWS else 3
            wn["w"] = WINDOWS[max(0, min(len(WINDOWS) - 1, i + rng.choice((-1, 1))))]
    if node_count(t) > MAX_NODES:
        return t  # 超限时保持原样(重生成分支可能超)
    return t


def tournament(rng: random.Random, scored: list[tuple[float, dict]], k: int = 3) -> dict:
    picks = rng.sample(scored, min(k, len(scored)))
    return max(picks, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# 状态与缓存
# ---------------------------------------------------------------------------
def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# GA 主循环
# ---------------------------------------------------------------------------
def run_ga(population: int, generations: int, seed: int = 42) -> None:
    ev = Evaluator()
    cache: dict = load_json(CACHE_PATH, {})
    state = load_json(STATE_PATH, None)

    rng = random.Random(seed)
    if state:
        gen0 = state["generation"]
        pop = state["population"]
        hof = state["hof"]
        rng.seed(seed + gen0 * 1000)
        print(f"resume: generation {gen0}, pop={len(pop)}, hof={len(hof)}, cache={len(cache)}", flush=True)
    else:
        gen0 = 0
        pop = seed_trees()
        while len(pop) < population:
            t = rand_tree(rng)
            if md5_of(t) not in {md5_of(p) for p in pop}:
                pop.append(t)
        hof = []

    def evaluate(t: dict) -> dict:
        key = md5_of(t)
        if key in cache:
            return cache[key]
        t0 = time.time()
        m = ev.backtest(t)
        m["_expr"] = to_str(t)
        m["_secs"] = round(time.time() - t0, 1)
        cache[key] = m
        save_json(CACHE_PATH, cache)
        return m

    no_improve = 0
    best_ever = max((h["fitness"] for h in hof), default=-99.0)

    for gen in range(gen0, generations):
        t_gen = time.time()
        scored: list[tuple[float, dict]] = []
        for i, t in enumerate(pop):
            m = evaluate(t)
            f = fitness_of(m, t, hof)
            scored.append((f, t))
            if m.get("is"):
                print(f"  g{gen} [{i+1}/{len(pop)}] f={f:+.2f} netShp={m['is']['net_sharpe']:+.2f} "
                      f"to={m['is']['turnover']:.2f} | {to_str(t)[:90]}", flush=True)
            else:
                print(f"  g{gen} [{i+1}/{len(pop)}] f={f:+.2f} INVALID({m.get('error','no-is')}) | {to_str(t)[:70]}", flush=True)

        scored.sort(key=lambda x: -x[0])

        # 名人堂更新(达标者入堂;按 fitness 截断到 20)
        for f, t in scored[:10]:
            m = cache[md5_of(t)]
            if not m.get("is"):
                continue
            ok = (m["is"]["net_sharpe"] > 0.8 and m["is"]["turnover"] <= 0.35
                  and m["is"]["days"] >= 1000 and m["is"]["avg_holdings"] >= 300)
            if ok and md5_of(t) not in {h["md5"] for h in hof}:
                hof.append({"md5": md5_of(t), "expr": to_str(t), "tree": t, "fitness": f,
                            "is": m["is"], "is_pnl": m["is_pnl"], "gen": gen})
        hof.sort(key=lambda h: -h["fitness"])
        hof = hof[:20]

        gen_best = scored[0][0]
        if gen_best > best_ever + 1e-6:
            best_ever = gen_best
            no_improve = 0
        else:
            no_improve += 1
        print(f"== gen {gen}: best={gen_best:+.3f} mean={np.mean([f for f,_ in scored]):+.3f} "
              f"hof={len(hof)} no_improve={no_improve} ({time.time()-t_gen:.0f}s)", flush=True)

        # 下一代
        nxt = [json.loads(json.dumps(t)) for _, t in scored[:5]]  # 精英
        seen = {md5_of(t) for t in nxt}
        rng.seed(seed + (gen + 1) * 1000)
        attempts = 0
        while len(nxt) < population and attempts < population * 30:
            attempts += 1
            r = rng.random()
            if r < 0.6:
                child = crossover(rng, tournament(rng, scored), tournament(rng, scored))
            elif r < 0.9:
                child = mutate(rng, tournament(rng, scored))
            else:
                child = rand_tree(rng)
            key = md5_of(child)
            if key not in seen:
                seen.add(key)
                nxt.append(child)
        pop = nxt

        save_json(STATE_PATH, {"generation": gen + 1, "population": pop, "hof": hof,
                               "updated": datetime.now(timezone.utc).isoformat()})

        if no_improve >= 5:
            print("early stop: no improvement for 5 generations", flush=True)
            break

    print(f"GA done. HOF={len(hof)}. Run --report for OOS validation.", flush=True)


# ---------------------------------------------------------------------------
# 收尾:名人堂 OOS 验证 + 报告
# ---------------------------------------------------------------------------
def write_report() -> None:
    state = load_json(STATE_PATH, None)
    cache = load_json(CACHE_PATH, {})
    if not state or not state.get("hof"):
        print("no HOF in state; run --run first")
        return
    hof = state["hof"]

    ev = Evaluator()
    rows = []
    for h in hof:
        m = cache.get(h["md5"]) or ev.backtest(h["tree"])
        oos = m.get("oos")
        is_ = m.get("is") or h["is"]
        decay_ok = None
        selected = False
        if oos:
            decay_ok = oos["net_sharpe"] > 0 and (is_["net_sharpe"] <= 0 or oos["net_sharpe"] >= 0.5 * is_["net_sharpe"] - 0.5)
            selected = oos["net_sharpe"] > 0 and oos["net_sharpe"] >= 0.5 * is_["net_sharpe"]
        rows.append({"h": h, "is": is_, "oos": oos, "selected": selected})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fmt = lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) else "—"
    lines = [
        f"# GA 因子挖掘机报告 — {now}",
        "",
        f"- 方式:遗传算法(文法约束表达式树、锦标赛选择、子树交叉、四类变异、精英保留),"
        f"结合 Worldquant_Mining 的窗口集/双闸门/IS-OOS 纪律",
        f"- fitness:IS({'..' + IS_END}) 净 Sharpe - 惩罚(TO>35%、无效解、复杂度、与名人堂相关>0.7)",
        f"- OOS({OOS_START}..)只在此报告中验证一次,GA 过程不可见",
        f"- 进化到第 {state['generation']} 代,评估缓存 {len(cache)} 条表达式",
        "",
        "## 名人堂(按 IS fitness 排序)",
        "",
        "| # | 表达式 | 代 | IS净Shp | IS TO | OOS净Shp | OOS毛Shp | 入选 |",
        "|---|--------|----|---------|-------|----------|----------|------|",
    ]
    for i, r in enumerate(rows):
        h, is_, oos = r["h"], r["is"], r["oos"]
        lines.append(
            f"| {i+1} | `{h['expr'][:80]}` | {h.get('gen','?')} | {fmt(is_['net_sharpe'])} | {is_['turnover']:.2f} "
            f"| {fmt(oos['net_sharpe']) if oos else '—'} | {fmt(oos['gross_sharpe']) if oos else '—'} "
            f"| {'✅' if r['selected'] else '❌'} |"
        )
    sel = [r for r in rows if r["selected"]]
    lines += [
        "",
        f"## 结论:{len(sel)}/{len(rows)} 通过 OOS 闸门(OOS 净 Sharpe>0 且 ≥ IS 的 50%)",
        "",
        "> 入选者可并入 tushare_mining_report 的低相关组合分析;未入选者视为 IS 过拟合。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {REPORT_PATH.name}; selected={len(sel)}/{len(rows)}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="GA alpha miner on local tushare data")
    p.add_argument("--smoke", action="store_true", help="种群8 x 2代冒烟")
    p.add_argument("--run", action="store_true", help="标准运行(断点续传)")
    p.add_argument("--report", action="store_true", help="OOS 验证 + 报告")
    p.add_argument("--population", type=int, default=60)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.smoke:
        run_ga(8, 2, a.seed)
    elif a.run:
        run_ga(a.population, a.generations, a.seed)
    elif a.report:
        write_report()
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

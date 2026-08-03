"""A 股 (CHN region) alpha mining pipeline for the WQ Alpha Research Skill.

Self-contained: only needs `requests`, `numpy`, and credentials supplied via
environment variables or a local untracked `credential.txt` file.

Follows the SKILL.md loop, adapted to CHN:
    fields -> expressions -> simulate -> IS checks -> correlation -> report

Usage (run from the skill root directory):
    # 0. Offline: preview candidate expressions and settings, no API calls
    python scripts/mine_chn_alphas.py --plan

    # 1. Pull the CHN field snapshot (SKILL.md section 2.4: a region change
    #    requires re-pulling fields; the bundled snapshot is USA-only)
    python scripts/mine_chn_alphas.py --fetch-fields

    # 2. Search the local CHN snapshot
    python scripts/mine_chn_alphas.py --search roe

    # 3. Mine: resolve templates against the CHN snapshot, simulate each
    #    candidate, collect metrics + IS checks, write a report
    python scripts/mine_chn_alphas.py --mine
    python scripts/mine_chn_alphas.py --mine --limit 5

    # 4. Daily-return correlation of one alpha vs all ACTIVE alphas
    python scripts/mine_chn_alphas.py --check-corr <alpha_id>

Credentials: WQ_BRAIN_USERNAME / WQ_BRAIN_PASSWORD env vars, or credential.txt
containing ["your_username", "your_password"]. Never commit credentials.

Outputs (all untracked by git):
    references/wq_chn_top2000u_delay1_data_fields.json  (tracked once generated)
    chn_mining_results.json
    chn_mining_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REF_DIR = SKILL_DIR / "references"
CREDENTIAL_PATH = SKILL_DIR / "credential.txt"
CANDIDATES_PATH = REF_DIR / "chn_candidate_alphas.json"
FIELDS_JSON = REF_DIR / "wq_chn_top2000u_delay1_data_fields.json"
FIELDS_CSV = REF_DIR / "wq_chn_top2000u_delay1_data_fields.csv"
FIELDS_SUMMARY = REF_DIR / "wq_chn_top2000u_delay1_data_fields_summary.json"
RESULTS_PATH = SKILL_DIR / "chn_mining_results.json"
REPORT_PATH = SKILL_DIR / "chn_mining_report.md"

API_BASE = "https://api.worldquantbrain.com"

REGION = "CHN"
UNIVERSE = "TOP2000U"
DELAY = 1

HEADERS = {
    "Accept": "application/json;version=2.0",
    "Content-Type": "application/json",
}

# 与 SKILL.md 4.2 一致的默认设置，region/universe 换成 CHN
BASE_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": REGION,
    "universe": UNIVERSE,
    "delay": DELAY,
    "decay": 0,
    "neutralization": "INDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": False,
}

PLACEHOLDER_RE = re.compile(r"\{field:([^}]+)\}")


# ---------------------------------------------------------------------------
# Auth / HTTP helpers (same conventions as evolve_skill.py)
# ---------------------------------------------------------------------------
def load_credentials() -> tuple[str, str]:
    env_user = os.getenv("WQ_BRAIN_USERNAME")
    env_password = os.getenv("WQ_BRAIN_PASSWORD")
    if env_user and env_password:
        return env_user, env_password
    for p in (CREDENTIAL_PATH, Path.cwd() / "credential.txt"):
        if p.exists():
            username, password = json.loads(p.read_text(encoding="utf-8"))
            return str(username), str(password)
    raise FileNotFoundError(
        "BRAIN credentials not found. Set WQ_BRAIN_USERNAME/WQ_BRAIN_PASSWORD "
        'or create an untracked credential.txt with ["your_username", "your_password"].'
    )


def create_session() -> requests.Session:
    username, password = load_credentials()
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    session.headers.update(HEADERS)
    resp = session.post(f"{API_BASE}/authentication")
    if resp.status_code != 201:
        raise RuntimeError(f"BRAIN auth failed: {resp.status_code} {resp.text}")
    return session


def get_with_retry(session: requests.Session, url: str, retries: int = 3, **kwargs) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=(10, 60), **kwargs)
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 5)))
                continue
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {retries} retries")


# ---------------------------------------------------------------------------
# Step 1: CHN field snapshot
# ---------------------------------------------------------------------------
def fetch_chn_fields(session: requests.Session) -> list[dict]:
    """Pull all CHN TOP2000U delay-1 data fields, dataset by dataset."""
    datasets: list[dict] = []
    offset = 0
    while True:
        resp = get_with_retry(
            session,
            f"{API_BASE}/data-sets",
            params={
                "instrumentType": "EQUITY",
                "region": REGION,
                "universe": UNIVERSE,
                "delay": DELAY,
                "limit": 50,
                "offset": offset,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"data-sets failed: {resp.status_code} {resp.text}")
        batch = resp.json().get("results", [])
        if not batch:
            break
        datasets.extend(batch)
        if len(batch) < 50:
            break
        offset += 50
        time.sleep(0.3)
    print(f"datasets: {len(datasets)}", flush=True)

    fields: list[dict] = []
    for ds in datasets:
        ds_id = ds.get("id")
        offset = 0
        while True:
            resp = get_with_retry(
                session,
                f"{API_BASE}/data-fields",
                params={
                    "instrumentType": "EQUITY",
                    "region": REGION,
                    "universe": UNIVERSE,
                    "delay": DELAY,
                    "dataset.id": ds_id,
                    "limit": 50,
                    "offset": offset,
                },
            )
            if resp.status_code != 200:
                print(f"  skip dataset {ds_id}: {resp.status_code}", flush=True)
                break
            batch = resp.json().get("results", [])
            if not batch:
                break
            fields.extend(batch)
            if len(batch) < 50:
                break
            offset += 50
            time.sleep(0.2)
        print(f"  {ds_id}: total fields so far {len(fields)}", flush=True)
        time.sleep(0.3)
    return fields


def save_field_snapshot(fields: list[dict]) -> None:
    REF_DIR.mkdir(exist_ok=True)
    FIELDS_JSON.write_text(json.dumps(fields, ensure_ascii=False, indent=1), encoding="utf-8")

    with FIELDS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "description", "type", "category", "dataset", "coverage", "userCount", "alphaCount"])
        for fd in fields:
            writer.writerow(
                [
                    fd.get("id"),
                    fd.get("description"),
                    fd.get("type"),
                    (fd.get("category") or {}).get("id"),
                    (fd.get("dataset") or {}).get("id"),
                    fd.get("coverage"),
                    fd.get("userCount"),
                    fd.get("alphaCount"),
                ]
            )

    cats = Counter((fd.get("category") or {}).get("id") for fd in fields)
    summary = {
        "region": REGION,
        "universe": UNIVERSE,
        "delay": DELAY,
        "total_fields": len(fields),
        "categories": dict(cats.most_common()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    FIELDS_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(fields)} fields -> {FIELDS_JSON.name}, {FIELDS_CSV.name}, {FIELDS_SUMMARY.name}")


def load_field_snapshot(required: bool = True) -> list[dict]:
    if FIELDS_JSON.exists():
        return json.loads(FIELDS_JSON.read_text(encoding="utf-8"))
    if required:
        raise FileNotFoundError(
            f"{FIELDS_JSON} not found. Run `python scripts/mine_chn_alphas.py --fetch-fields` first "
            "(the bundled snapshot covers USA TOP3000 only, see SKILL.md section 2.4)."
        )
    return []


def search_fields(fields: list[dict], keyword: str, top: int = 15) -> list[dict]:
    kw = keyword.lower()
    matches = [
        f
        for f in fields
        if kw in f.get("id", "").lower() or kw in (f.get("description") or "").lower()
    ]
    matches.sort(key=lambda f: (f.get("alphaCount") or 0, f.get("coverage") or 0), reverse=True)
    return matches[:top]


# ---------------------------------------------------------------------------
# Step 2: candidate templates -> concrete expressions
# ---------------------------------------------------------------------------
def load_candidates() -> list[dict]:
    return json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))["candidates"]


def resolve_expression(expr: str, fields: list[dict]) -> tuple[str | None, dict[str, str]]:
    """Replace {field:kw1|kw2} placeholders with the best-matching CHN field id.

    选择规则：按关键词依次搜索，取 alphaCount 最高（其次 coverage 最高）的
    matrix 类型字段。任何占位符解析失败则返回 None（跳过该候选）。
    """
    resolved: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        keywords = m.group(1).split("|")
        for kw in keywords:
            hits = [f for f in search_fields(fields, kw.strip(), top=50) if f.get("type") in (None, "MATRIX")]
            if hits:
                resolved[m.group(0)] = hits[0]["id"]
                return hits[0]["id"]
        resolved[m.group(0)] = ""
        return ""

    out = PLACEHOLDER_RE.sub(_sub, expr)
    if any(v == "" for v in resolved.values()):
        return None, resolved
    return out, resolved


# ---------------------------------------------------------------------------
# Step 3: simulate + collect IS checks
# ---------------------------------------------------------------------------
def simulate(session: requests.Session, expression: str, settings: dict) -> dict:
    payload = {"type": "REGULAR", "settings": settings, "regular": expression}
    resp = session.post(f"{API_BASE}/simulations", json=payload)
    if resp.status_code == 429:
        time.sleep(int(resp.headers.get("Retry-After", 10)))
        resp = session.post(f"{API_BASE}/simulations", json=payload)
    if resp.status_code != 201:
        return {"error": "simulate_rejected", "status_code": resp.status_code, "detail": resp.text[:500]}
    sim_url = resp.headers["Location"]
    while True:
        data = get_with_retry(session, sim_url).json()
        status = data.get("status")
        if status == "COMPLETE":
            alpha_id = data["alpha"]
            break
        if status in ("ERROR", "FAILED", "FAIL"):
            return {"error": "simulation_error", "detail": json.dumps(data)[:500]}
        time.sleep(8)
    alpha = get_with_retry(session, f"{API_BASE}/alphas/{alpha_id}").json()
    return {"alpha_id": alpha_id, "alpha": alpha}


def summarize_alpha(alpha: dict) -> dict:
    is_ = alpha.get("is", {}) if isinstance(alpha.get("is"), dict) else {}
    checks = is_.get("checks", []) or []
    failed = [c["name"] for c in checks if c.get("result") == "FAIL"]
    return {
        "sharpe": is_.get("sharpe"),
        "fitness": is_.get("fitness"),
        "returns": is_.get("returns"),
        "turnover": is_.get("turnover"),
        "drawdown": is_.get("drawdown"),
        "margin": is_.get("margin"),
        "long_count": is_.get("longCount"),
        "short_count": is_.get("shortCount"),
        # 阈值以 API 返回的 checks 为准（CHN 与 USA 的门槛不同，勿硬编码）
        "checks": [{"name": c.get("name"), "result": c.get("result"), "value": c.get("value"), "limit": c.get("limit")} for c in checks],
        "failed_checks": failed,
        "all_checks_pass": bool(checks) and not failed,
    }


# ---------------------------------------------------------------------------
# Step 4: correlation vs ACTIVE alphas (daily returns, per SKILL.md 7.2)
# ---------------------------------------------------------------------------
def fetch_pnl(session: requests.Session, alpha_id: str) -> list[float]:
    resp = get_with_retry(session, f"{API_BASE}/alphas/{alpha_id}/recordsets/pnl")
    if resp.status_code != 200 or not resp.text.strip():
        return []
    data = resp.json()
    props = data.get("schema", {}).get("properties", [])
    if isinstance(props, list):
        date_idx = next((i for i, p in enumerate(props) if p.get("name", "").lower() == "date"), 0)
        pnl_idx = next((i for i, p in enumerate(props) if p.get("name", "").lower() in ("pnl", "cum_pnl", "returns", "ret")), 1)
    else:
        date_idx = next((v["index"] for k, v in props.items() if k.lower() == "date"), 0)
        pnl_idx = next((v["index"] for k, v in props.items() if k.lower() in ("pnl", "cum_pnl", "returns", "ret")), 1)
    records = sorted(data.get("records", []), key=lambda r: r[date_idx])
    out: list[float] = []
    for row in records:
        rec = row[0] if isinstance(row, list) and len(row) == 1 and isinstance(row[0], list) else row
        try:
            out.append(float(rec[pnl_idx]))
        except Exception:
            continue
    return out


def daily_returns(cum_pnl: list[float]) -> list[float]:
    return [cum_pnl[i + 1] - cum_pnl[i] for i in range(len(cum_pnl) - 1)]


def check_correlation(session: requests.Session, alpha_id: str) -> list[dict]:
    import numpy as np

    new_ret = daily_returns(fetch_pnl(session, alpha_id))
    results: list[dict] = []
    offset = 0
    actives: list[dict] = []
    while True:
        data = get_with_retry(session, f"{API_BASE}/users/self/alphas", params={"limit": 100, "offset": offset}).json()
        batch = data.get("results", data.get("alphas", []))
        if not batch:
            break
        actives.extend(a for a in batch if a.get("status") == "ACTIVE")
        if len(batch) < 100:
            break
        offset += 100
    for old in actives:
        old_id = old.get("id")
        if old_id == alpha_id:
            continue
        old_ret = daily_returns(fetch_pnl(session, old_id))
        if len(new_ret) == len(old_ret) and len(new_ret) > 20:
            corr = float(np.corrcoef(new_ret, old_ret)[0, 1])
            results.append({"alpha_id": old_id, "corr": corr})
        time.sleep(0.3)
    results.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Mining loop
# ---------------------------------------------------------------------------
def build_settings(candidate: dict) -> dict:
    settings = dict(BASE_SETTINGS)
    settings.update(candidate.get("settings_override", {}))
    return settings


def run_mine(session: requests.Session, limit: int | None, only: str | None) -> None:
    fields = load_field_snapshot()
    candidates = load_candidates()
    if only:
        candidates = [c for c in candidates if only.lower() in c["name"].lower()]
    if limit:
        candidates = candidates[:limit]

    results: list[dict] = []
    for i, cand in enumerate(candidates):
        expr, resolved = resolve_expression(cand["expression"], fields)
        entry: dict[str, Any] = {
            "name": cand["name"],
            "family": cand.get("family"),
            "rationale": cand.get("rationale"),
            "template": cand["expression"],
            "resolved_fields": resolved,
        }
        if expr is None:
            entry["decision"] = "unresolved_field"
            results.append(entry)
            print(f"[{i+1}/{len(candidates)}] {cand['name']}: UNRESOLVED {resolved}", flush=True)
            continue

        settings = build_settings(cand)
        entry["expression"] = expr
        entry["settings"] = settings
        print(f"[{i+1}/{len(candidates)}] {cand['name']}: simulating `{expr}`", flush=True)
        sim = simulate(session, expr, settings)
        if "error" in sim:
            entry["decision"] = sim["error"]
            entry["detail"] = sim.get("detail")
            print(f"    -> {sim['error']}: {str(sim.get('detail'))[:160]}", flush=True)
        else:
            entry["alpha_id"] = sim["alpha_id"]
            metrics = summarize_alpha(sim["alpha"])
            entry["metrics"] = metrics
            entry["decision"] = "pass_all_checks" if metrics["all_checks_pass"] else "checks_failed"
            print(
                f"    -> alpha={sim['alpha_id']} sharpe={metrics['sharpe']} fitness={metrics['fitness']} "
                f"to={metrics['turnover']} failed={metrics['failed_checks']}",
                flush=True,
            )
        results.append(entry)
        time.sleep(3)  # SKILL.md 7.6 限流

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results)
    passed = [r for r in results if r.get("decision") == "pass_all_checks"]
    print(f"\ndone: {len(results)} candidates, {len(passed)} pass all IS checks")
    print(f"results -> {RESULTS_PATH.name}, report -> {REPORT_PATH.name}")
    if passed:
        print("next: run --check-corr <alpha_id> before any submission (SKILL.md section 9 checklist)")


def write_report(results: list[dict]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# CHN A股 因子挖掘报告 — {now}",
        "",
        f"Region={REGION} Universe={UNIVERSE} Delay={DELAY}",
        "",
        "| 候选 | 簇 | Sharpe | Fitness | TO | DD | 未通过检查 | 结论 |",
        "|------|----|--------|---------|----|----|------------|------|",
    ]
    for r in results:
        m = r.get("metrics") or {}
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {r['name']} | {r.get('family','')} | {fmt(m.get('sharpe'))} | {fmt(m.get('fitness'))} | "
            f"{fmt(m.get('turnover'))} | {fmt(m.get('drawdown'))} | {', '.join(m.get('failed_checks', [])) or '—'} | {r['decision']} |"
        )
    lines += ["", "## 表达式明细", ""]
    for r in results:
        lines.append(f"### {r['name']}")
        lines.append(f"- 逻辑：{r.get('rationale','')}")
        lines.append(f"- 模板：`{r['template']}`")
        if r.get("expression"):
            lines.append(f"- 解析后：`{r['expression']}`")
        if r.get("alpha_id"):
            lines.append(f"- alpha_id：`{r['alpha_id']}`")
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_plan() -> None:
    """Offline preview: what would be simulated, with which settings."""
    candidates = load_candidates()
    fields = load_field_snapshot(required=False)
    print(f"CHN mining plan: {len(candidates)} candidates | field snapshot: "
          f"{'loaded, ' + str(len(fields)) + ' fields' if fields else 'MISSING (run --fetch-fields first)'}\n")
    for c in candidates:
        settings = build_settings(c)
        print(f"- {c['name']} [{c.get('family')}]")
        print(f"    expr: {c['expression']}")
        print(f"    decay={settings['decay']} neutralization={settings['neutralization']} rationale: {c.get('rationale')}")
        if fields:
            expr, resolved = resolve_expression(c["expression"], fields)
            if resolved:
                print(f"    resolved: {resolved}" + ("" if expr else "  <- UNRESOLVED, would skip"))
    print("\nnext: --fetch-fields (once) then --mine")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine CHN (A股) alphas on WorldQuant BRAIN.")
    parser.add_argument("--plan", action="store_true", help="Offline preview of candidates, no API calls")
    parser.add_argument("--fetch-fields", action="store_true", help="Pull CHN field snapshot into references/")
    parser.add_argument("--search", metavar="KEYWORD", help="Search local CHN field snapshot")
    parser.add_argument("--mine", action="store_true", help="Simulate all candidates and write a report")
    parser.add_argument("--limit", type=int, help="Only mine the first N candidates")
    parser.add_argument("--only", metavar="NAME", help="Only mine candidates whose name contains NAME")
    parser.add_argument("--check-corr", metavar="ALPHA_ID", help="Daily-return correlation vs ACTIVE alphas")
    args = parser.parse_args()

    if args.plan:
        run_plan()
        return 0

    if args.search:
        fields = load_field_snapshot()
        for f in search_fields(fields, args.search):
            print(
                f"{f['id']} | {(f.get('category') or {}).get('id')} | {(f.get('dataset') or {}).get('id')} "
                f"| coverage={f.get('coverage')} | alphaCount={f.get('alphaCount')} | {f.get('description')}"
            )
        return 0

    if not (args.fetch_fields or args.mine or args.check_corr):
        parser.print_help()
        return 1

    session = create_session()
    print("auth ok", flush=True)

    if args.fetch_fields:
        save_field_snapshot(fetch_chn_fields(session))
    if args.mine:
        run_mine(session, args.limit, args.only)
    if args.check_corr:
        corrs = check_correlation(session, args.check_corr)
        for c in corrs[:20]:
            flag = "❌" if abs(c["corr"]) >= 0.7 else ("⚠️" if abs(c["corr"]) >= 0.5 else "✅")
            print(f"{flag} {c['alpha_id']}: {c['corr']:+.3f}")
        if not corrs:
            print("no comparable ACTIVE alphas (or PnL unavailable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

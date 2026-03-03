import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.services.db.database import DatabaseService


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        # SQLite CURRENT_TIMESTAMP: "YYYY-MM-DD HH:MM:SS"
        return datetime.fromisoformat(text.replace("Z", ""))
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
    return None


def topk_recall(candidates: Sequence[str], ground_truth: Set[str], k: int) -> float:
    if not ground_truth or not candidates or k <= 0:
        return 0.0
    topk = set([str(v).strip() for v in candidates[:k] if str(v).strip()])
    return len(topk & ground_truth) / max(1, len(ground_truth))


def load_ranked_ids_from_csv(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return []

        id_col = next((c for c in ("material_id", "id", "mid") if c in fieldnames), None)
        if id_col is None:
            # If it's a single-column CSV, use that column.
            if len(fieldnames) == 1:
                id_col = fieldnames[0]
            else:
                return []

        rows = list(reader)
        if "score" in fieldnames:
            def _score(row: Dict[str, Any]) -> float:
                raw = row.get("score")
                try:
                    return float(raw)
                except Exception:
                    return float("-inf")

            rows.sort(key=_score, reverse=True)

        out: List[str] = []
        for row in rows:
            val = (row.get(id_col) or "").strip()
            if val:
                out.append(val)
        return out


def load_ids_set_from_csv(path: str) -> Set[str]:
    return set(load_ranked_ids_from_csv(path))


def _safe_json_load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_candidate_ids(pack: Dict[str, Any]) -> List[str]:
    ranking = pack.get("ranking_current") or pack.get("ranking_after_gap") or pack.get("ranking_before_gap") or []
    if isinstance(ranking, list) and ranking:
        ids = []
        for item in ranking:
            if isinstance(item, dict) and item.get("material_id"):
                ids.append(str(item.get("material_id")).strip())
            elif isinstance(item, str):
                ids.append(item.strip())
        ids = [v for v in ids if v]
        if ids:
            return ids
    candidates = pack.get("candidate_material_ids") or []
    if isinstance(candidates, list):
        return [str(v).strip() for v in candidates if str(v).strip()]
    return []


def _compute_rag_metrics(pack: Dict[str, Any]) -> Dict[str, Any]:
    rag_items = pack.get("knowledge_rag") or []
    if not isinstance(rag_items, list) or not rag_items:
        return {"rag_results_count": 0, "rag_unique_sources": 0, "rag_avg_score": None, "rag_reference_rate": None}

    total_results = 0
    scored: List[float] = []
    unique_sources: Set[str] = set()
    with_reference = 0

    for item in rag_items:
        if not isinstance(item, dict):
            continue
        results = item.get("results") or []
        if not isinstance(results, list):
            continue
        for res in results:
            if not isinstance(res, dict):
                continue
            total_results += 1
            sid = res.get("id")
            if sid is None:
                sid = res.get("source_id")
            if sid is not None:
                unique_sources.add(str(sid))
            score = res.get("score")
            if isinstance(score, (int, float)):
                scored.append(float(score))
            has_ref = bool(res.get("title") or res.get("url") or res.get("source_id"))
            if has_ref:
                with_reference += 1

    avg_score = round(sum(scored) / len(scored), 4) if scored else None
    ref_rate = round(with_reference / total_results, 4) if total_results else None
    return {
        "rag_results_count": total_results,
        "rag_unique_sources": len(unique_sources),
        "rag_avg_score": avg_score,
        "rag_reference_rate": ref_rate,
    }


def _compute_evidence_delta(pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    before = pack.get("evidence_stats_before_gap")
    after = pack.get("evidence_stats_after_gap")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None

    delta: Dict[str, Any] = {}
    for key in set(before.keys()) | set(after.keys()):
        b = before.get(key)
        a = after.get(key)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            d = a - b
            if d:
                delta[key] = d

    bsrc = before.get("evidence_by_source")
    asrc = after.get("evidence_by_source")
    if isinstance(bsrc, dict) and isinstance(asrc, dict):
        per_src: Dict[str, Any] = {}
        for stype in set(bsrc.keys()) | set(asrc.keys()):
            b = bsrc.get(stype) or 0
            a = asrc.get(stype) or 0
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                d = a - b
                if d:
                    per_src[str(stype)] = d
        if per_src:
            delta["evidence_by_source"] = dict(sorted(per_src.items(), key=lambda x: x[0]))

    return delta or None


@dataclass(frozen=True)
class EvalSuiteConfig:
    db_path: str
    knowledge_dir: str = os.path.join("data", "tasks")
    recent: int = 20
    ks: Tuple[int, ...] = (5, 10, 20)
    ground_truth_ids: Optional[Set[str]] = None
    plan_ids: Optional[Tuple[str, ...]] = None


def build_eval_report(cfg: EvalSuiteConfig) -> Dict[str, Any]:
    db = DatabaseService(db_path=cfg.db_path)

    if cfg.plan_ids:
        plan_rows: List[Dict[str, Any]] = []
        for pid in cfg.plan_ids:
            row = db.get_plan(pid)
            if row:
                plan_rows.append(row)
    else:
        plan_rows = db.list_plans(limit=max(1, int(cfg.recent or 20)))

    try:
        from src.agents.core.theory_agent import TheoryDataConfig
        allowed = TheoryDataConfig().elements
    except Exception:
        allowed = None
    system_evidence_stats = db.get_evidence_stats(allowed_elements=allowed)
    data_integrity_stats = db.get_data_integrity_stats()

    plans_out: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    durations_all: List[float] = []
    durations_completed: List[float] = []
    knowledge_pack_present_count = 0
    recall_by_k: Dict[int, List[float]] = {int(k): [] for k in (cfg.ks or ())}

    for row in plan_rows:
        plan_id = row.get("id")
        if not plan_id:
            continue

        status = row.get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

        last_step_ts = db.get_plan_last_step_created_at(plan_id)
        start_ts = _parse_timestamp(row.get("created_at"))
        end_ts = _parse_timestamp(last_step_ts)
        duration_s = None
        if start_ts and end_ts:
            duration_s = max(0.0, (end_ts - start_ts).total_seconds())
            durations_all.append(duration_s)
            if status == "completed":
                durations_completed.append(duration_s)

        latest_steps = db.list_latest_plan_steps(plan_id)
        step_status_counts: Dict[str, int] = {}
        for s in latest_steps:
            sst = (s.get("status") or "unknown")
            step_status_counts[sst] = step_status_counts.get(sst, 0) + 1

        pack_path = os.path.join(cfg.knowledge_dir, f"knowledge_{plan_id}.json")
        pack_present = os.path.exists(pack_path)
        if pack_present:
            knowledge_pack_present_count += 1
        pack = _safe_json_load(pack_path) if pack_present else None
        pack = pack or {}

        candidate_ids = _extract_candidate_ids(pack)
        rag_metrics = _compute_rag_metrics(pack)
        evidence_delta = _compute_evidence_delta(pack)

        per_plan_recall: Optional[Dict[str, float]] = None
        if cfg.ground_truth_ids:
            per_plan_recall = {}
            for k in cfg.ks:
                val = round(topk_recall(candidate_ids, cfg.ground_truth_ids, int(k)), 4)
                per_plan_recall[str(int(k))] = val
                recall_by_k[int(k)].append(val)

        plans_out.append({
            "plan_id": plan_id,
            "task_type": row.get("task_type"),
            "status": status,
            "created_at": row.get("created_at"),
            "last_step_created_at": last_step_ts,
            "duration_seconds": duration_s,
            "step_count": len(latest_steps),
            "step_status_counts": dict(sorted(step_status_counts.items(), key=lambda x: x[0])),
            "candidate_count": len(candidate_ids),
            "ranking_top_n": len(candidate_ids) if candidate_ids else 0,
            "topk_recall": per_plan_recall,
            "rag_metrics": rag_metrics,
            "evidence_delta": evidence_delta,
            "evaluation_metrics": pack.get("evaluation_metrics"),
            "knowledge_pack_present": pack_present,
            "knowledge_pack_path": pack_path if pack_present else None,
        })

    plan_count = len(plans_out)
    success_rate = round(status_counts.get("completed", 0) / plan_count, 4) if plan_count else 0.0
    avg_duration = round(sum(durations_all) / len(durations_all), 3) if durations_all else None

    def _percentile(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        vals = sorted(float(v) for v in values)
        q = max(0.0, min(1.0, float(q)))
        idx = int(round((len(vals) - 1) * q))
        return vals[idx]

    terminal_statuses = {"completed", "failed", "blocked", "skipped", "cancelled", "canceled"}
    terminal_total = sum(int(status_counts.get(s, 0) or 0) for s in terminal_statuses)
    terminal_success_rate = (
        round(status_counts.get("completed", 0) / terminal_total, 4) if terminal_total else None
    )

    blocked_statuses = ("awaiting_confirmation", "pending", "executing", "running")
    blocked_status_counts = {s: int(status_counts.get(s, 0) or 0) for s in blocked_statuses}
    blocked_total = sum(blocked_status_counts.values())
    blocked_ratio = round(blocked_total / plan_count, 4) if plan_count else 0.0

    duration_p50_all = _percentile(durations_all, 0.5)
    duration_p90_all = _percentile(durations_all, 0.9)
    duration_p50_completed = _percentile(durations_completed, 0.5)
    duration_p90_completed = _percentile(durations_completed, 0.9)

    topk_avg: Optional[Dict[str, float]] = None
    if cfg.ground_truth_ids and recall_by_k:
        topk_avg = {}
        for k, vals in recall_by_k.items():
            if vals:
                topk_avg[str(int(k))] = round(sum(vals) / len(vals), 4)

    return {
        "version": "1.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db_path": cfg.db_path,
        "knowledge_dir": cfg.knowledge_dir,
        "filters": {
            "recent": cfg.recent,
            "plan_ids": list(cfg.plan_ids) if cfg.plan_ids else None,
            "ks": list(cfg.ks),
            "ground_truth_count": (len(cfg.ground_truth_ids) if cfg.ground_truth_ids else 0),
        },
        "summary": {
            "plan_count": plan_count,
            "status_counts": dict(sorted(status_counts.items(), key=lambda x: x[0])),
            "success_rate": success_rate,
            "terminal_success_rate": terminal_success_rate,
            "blocked_ratio": blocked_ratio,
            "blocked_status_counts": {k: v for k, v in blocked_status_counts.items() if v},
            "avg_duration_seconds": avg_duration,
            "duration_p50_seconds": (round(duration_p50_all, 3) if duration_p50_all is not None else None),
            "duration_p90_seconds": (round(duration_p90_all, 3) if duration_p90_all is not None else None),
            "duration_p50_completed_seconds": (round(duration_p50_completed, 3) if duration_p50_completed is not None else None),
            "duration_p90_completed_seconds": (round(duration_p90_completed, 3) if duration_p90_completed is not None else None),
            "knowledge_pack_present_count": int(knowledge_pack_present_count),
            "knowledge_pack_missing_count": int(max(0, plan_count - knowledge_pack_present_count)),
            "topk_avg_recall": topk_avg,
            "system_evidence_stats": system_evidence_stats,
            "data_integrity_stats": data_integrity_stats,
        },
        "plans": plans_out,
    }


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        sec = float(value)
    except Exception:
        return "-"
    if sec < 60:
        return f"{sec:.1f}s"
    return f"{(sec / 60):.1f}m"


def render_markdown_report(report: Dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    status_counts = summary.get("status_counts") or {}
    lines: List[str] = []
    lines.append("# IMCs Evaluation Report")
    lines.append("")
    lines.append(f"- Generated at: {report.get('generated_at')}")
    lines.append(f"- DB: `{report.get('db_path')}`")
    lines.append(f"- Knowledge dir: `{report.get('knowledge_dir')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Plans: {summary.get('plan_count', 0)}")
    lines.append(f"- Success rate: {summary.get('success_rate', 0)}")
    if summary.get("terminal_success_rate") is not None:
        lines.append(f"- Terminal success rate: {summary.get('terminal_success_rate')}")
    if summary.get("blocked_ratio") is not None:
        blocked_counts = summary.get("blocked_status_counts") or {}
        lines.append(f"- Blocked ratio: {summary.get('blocked_ratio')} (by status: `{json.dumps(blocked_counts, ensure_ascii=False)}`)")
    lines.append(f"- Avg duration: {_format_seconds(summary.get('avg_duration_seconds'))}")
    lines.append(
        "- Duration p50/p90 (completed): "
        f"{_format_seconds(summary.get('duration_p50_completed_seconds'))} / "
        f"{_format_seconds(summary.get('duration_p90_completed_seconds'))}"
    )
    pack_present = summary.get("knowledge_pack_present_count")
    pack_missing = summary.get("knowledge_pack_missing_count")
    if pack_present is not None and pack_missing is not None:
        lines.append(f"- Knowledge packs: present={pack_present}, missing={pack_missing}")
    integrity = summary.get("data_integrity_stats") or {}
    if integrity:
        fk = integrity.get("foreign_keys_enabled")
        ev_orphan = integrity.get("evidence_orphan_rows")
        act_orphan = integrity.get("activity_metric_orphan_rows")
        ads_orphan = integrity.get("adsorption_orphan_rows")
        lines.append(f"- Data integrity: fk={fk}, evidence_orphans={ev_orphan}, activity_orphans={act_orphan}, adsorption_orphans={ads_orphan}")
    lines.append(f"- Status counts: `{json.dumps(status_counts, ensure_ascii=False)}`")
    topk_avg = summary.get("topk_avg_recall")
    if topk_avg:
        lines.append(f"- Top‑K avg recall: `{json.dumps(topk_avg, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Plans")
    for p in report.get("plans") or []:
        pid = p.get("plan_id")
        status = p.get("status")
        dur = _format_seconds(p.get("duration_seconds"))
        cand = p.get("candidate_count") or 0
        rag = (p.get("rag_metrics") or {}).get("rag_results_count") or 0
        pack = 1 if p.get("knowledge_pack_present") else 0
        lines.append(f"- `{pid}` | {status} | {dur} | candidates={cand} | rag_results={rag} | pack={pack}")
    lines.append("")
    return "\n".join(lines)


def write_report_files(report: Dict[str, Any], out_dir: str, prefix: str = "eval_report") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"{prefix}_{stamp}.json")
    md_path = os.path.join(out_dir, f"{prefix}_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(report))
    return {"json": json_path, "md": md_path}

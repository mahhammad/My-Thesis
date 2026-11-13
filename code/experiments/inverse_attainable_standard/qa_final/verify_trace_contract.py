#!/usr/bin/env python3
"""Post-handoff independent-structure regression of *archived* §4.10 raw evidence.

Pure Python standard library. No graph, data generation, optimizer, or Test access.
This is AUTHOR_REMEDIATION_WITH_SELF_VALIDATION, not a second independent audit.
Historical raw summary/trace/audit files are never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TOL = 1e-8
MAX_RSS = 1536.0
METHODS = {"Q-only", "Exact-Adam", "Q+Bootstrap-GN"}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, *, tolerance: float = 1e-11) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tolerance


def run() -> dict:
    plan = read(BASE / "standard_plan/plan_108.json")
    coverage = read(BASE / "results/coverage.json")
    rows = {x["run_id"]: x for x in plan["runs"]}
    observed = coverage["records"]
    need(len(rows) == len(observed) == 108, "PLAN_108_MISMATCH")
    need(set(rows) == {r["run_id"] for r in observed}, "RUN_ID_SET_MISMATCH")

    # Regression: reject unescaped underscores in section 4.10
    # text commands; the first two-pass XeLaTeX job failed on this exact case.
    tex = (BASE.parents[2] / "tex/chapter4.tex").read_text(encoding="utf-8")
    start = tex.index(r"\section{بازیابی معکوس سیاست از مراکز دست‌یافتنی}")
    end = tex.index(r"\section{مراکز برچسبی بیزی، فضای قابل‌دستیابی و معیارهای بازنمایی}", start)
    section = tex[start:end]
    escaped_filename = r"\lr{metrics\_full.json}"
    need(escaped_filename in section, "MISSING_ESCAPED_METRICS_FILENAME")
    for match in re.finditer(r"\\(?:lr|texttt)\{[^{}]*\}", section):
        need(re.search(r"(?<!\\)_", match.group()) is None,
             "UNESCAPED_UNDERSCORE_IN_LATEX_TEXT:" + match.group())

    counts: Counter[str] = Counter()
    wall_inversion: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    max_rss = 0.0
    trace_phase_legacy = 0

    for rec in observed:
        run_id = rec["run_id"]
        block = rec["block_id"]
        if "-B01-" in block:
            need(rec["evidence_status"] == "ACCEPT_B01_V2", "B01_HISTORIC_STATUS")
            continue

        counts[block] += 1
        source = BASE / rec["raw_file"]
        need(source.is_file(), f"SUMMARY_MISSING:{run_id}")
        summary = read(source)
        folder = source.parent
        need(summary["run_id"] == run_id and summary["block_id"] == block,
             f"IDENTITY_DRIFT:{run_id}")
        need(summary["attempt_id"] == folder.name, f"ATTEMPT_DRIFT:{run_id}")
        need(summary["status"] == "COMPLETED", f"SUMMARY_NOT_COMPLETED:{run_id}")
        need(summary["solver"] in METHODS, f"METHOD_INVALID:{run_id}")
        need(summary["test_used"] is False and summary["witness_given_to_solver"] is False,
             f"LEAKAGE_FLAG:{run_id}")
        need(summary["initialization"] == rows[run_id]["initialization"] and
             summary["initialization_seed"] == rows[run_id]["initialization_seed"] and
             summary["solver"] == rows[run_id]["solver"] and
             summary["class"] == rows[run_id]["class_index"], f"PLAN_DRIFT:{run_id}")
        need(summary["tol"] == TOL and summary["walk"] ==
             {"H": 12, "beta": 0.4, "lambda": 0.8, "dtype": "float64"},
             f"PROTOCOL_DRIFT:{run_id}")
        best = summary["best_exact_center_JSD"]
        final = summary["final_exact_center_JSD"]
        need(close(summary["selected_exact_center_JSD"], best, tolerance=1e-12),
             f"BEST_SELECTED_DRIFT:{run_id}")
        need(summary["success"] is (best <= TOL) and
             rec["success"] == summary["success"] and
             close(rec["best_exact_center_JSD"], best, tolerance=1e-15) and
             close(rec["final_exact_center_JSD"], final, tolerance=1e-15),
             f"SUCCESS_OR_COVERAGE_DRIFT:{run_id}")
        need(summary["peak_rss_mib"] <= summary["bounded_limits"]["hard_rss_mib"] ==
             MAX_RSS, f"RSS_HARD_LIMIT:{run_id}")
        max_rss = max(max_rss, summary["peak_rss_mib"])
        if summary["success"]:
            successes[summary["solver"]] += 1
        if summary["exclusive_solver_wall_s"] > summary["inclusive_solver_wall_s"]:
            wall_inversion[summary["solver"]] += 1

        trace = read(folder / "trace.json")
        need(trace.get("status") in ("in_progress", "completed", "complete"),
             f"TRACE_PHASE_UNKNOWN:{run_id}")
        traces = trace.get("runs")
        need(isinstance(traces, list) and len(traces) >= 1,
             f"TRACE_RUNS_MISSING:{run_id}")
        steps = []
        for stage in traces:
            need(stage.get("status") in ("complete", "completed") and
                 stage.get("method") == summary["solver"], f"TRACE_STAGE_STATUS:{run_id}")
            trace_steps = stage.get("trace")
            need(isinstance(trace_steps, list) and len(trace_steps) > 0,
                 f"TRACE_STEPS_EMPTY:{run_id}")
            steps.extend(trace_steps)
        direct_min = min(s["exact_center_JSD"] for s in steps)
        need(close(best, direct_min, tolerance=1e-11),
             f"TRACE_BEST_MISMATCH:{run_id}")
        # Only selected/reached checkpoint is authoritative when a restart is used.
        # All frozen new runs have no restart; verify before comparing final trace step.
        if not summary["restart_used"]:
            need(close(final, steps[-1]["exact_center_JSD"], tolerance=1e-11),
                 f"TRACE_FINAL_MISMATCH:{run_id}")
        if trace["status"] == "in_progress":
            trace_phase_legacy += 1

        audit = read(folder / "audit_ref.json")
        pins = read(folder / "input_pins.json")
        need(audit["attempt_id"] == summary["attempt_id"] and
             pins["run_id"] == run_id and
             pins["input_sha256"] == summary["input_sha256"] and
             pins["owner_root_sha256"] == summary["owner_root_sha256"],
             f"AUDIT_PINS_DRIFT:{run_id}")
        start = BASE / "results/audit_attachments" / (summary["attempt_id"] + ".start.json")
        terminal = BASE / "results/audit_attachments" / (summary["attempt_id"] + ".terminal.json")
        need(start.is_file() and terminal.is_file(), f"AUDIT_MISSING:{run_id}")
        need(digest(start) == audit["start_sha256"] and
             digest(terminal) == audit["terminal_sha256"], f"AUDIT_SHA256_DRIFT:{run_id}")
        s, t = read(start), read(terminal)
        need(s["phase"] == "START" and s["run_id"] == run_id and
             s["graph_constructed"] == 0 and s["optimizer_steps"] == 0,
             f"AUDIT_START_INVALID:{run_id}")
        need(t["phase"] == "TERMINAL" and t["run_id"] == run_id and
             t["status"] == "COMPLETED" and t["graph_constructed"] == 1 and
             t["start_digest"] == audit["start_sha256"] and
             close(t["selected_exact_JSD"], best, tolerance=1e-12),
             f"AUDIT_TERMINAL_INVALID:{run_id}")

    need(sum(counts.values()) == 90 and len(counts) == 5 and
         all(v == 18 for v in counts.values()), "RAW_90_BLOCKS")
    need(successes == {"Q-only": 2, "Exact-Adam": 28, "Q+Bootstrap-GN": 30},
         "METHOD_SUCCESS_COUNT_DRIFT")
    return {
        "schema": "SEC410_POSTHANDOFF_TRACE_CONTRACT_V1",
        "status": "SELF_VALIDATED_PURE_ARCHIVE_REGRESSION",
        "new_raw_summaries_checked": sum(counts.values()),
        "section410_escaped_filename_preflight": True,
        "new_raw_trace_minima_checked": sum(counts.values()),
        "new_raw_audit_sha256_pairs_checked": sum(counts.values()),
        "new_successes_by_method": dict(successes),
        "trace_top_level_in_progress_count": trace_phase_legacy,
        "wall_exclusive_gt_inclusive_count": sum(wall_inversion.values()),
        "wall_exclusive_gt_inclusive_by_method": dict(wall_inversion),
        "max_new_rss_mib": max_rss,
        "scope": "immutable archived files, no graph/solver/Test",
        "not_evidence_of_independent_graph_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

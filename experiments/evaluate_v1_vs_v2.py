#!/usr/bin/env python3
"""Synthetic V1-vs-V2 evaluation of QUESTION-SELECTION proxies (not clinical accuracy).

    python experiments/evaluate_v1_vs_v2.py            # full run (30 patients × 6 sessions)
    python experiments/evaluate_v1_vs_v2.py --quick    # smoke run -> outputs/evaluation/quick/

Outputs go to outputs/evaluation/: per_session_metrics.csv, summary.json, comparison.csv,
ablations.csv, run_manifest.json, results.md.

Primary proxies and the decision rule are fixed below, BEFORE results are seen, and are
reported whatever the outcome. Clinical predictive accuracy is NOT evaluated: no outcome
labels exist.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from adaptive_questionnaires.evaluation.harness import compute_metrics, paired_bootstrap, run_v1, run_v2  # noqa: E402
from adaptive_questionnaires.evaluation.synthetic import World, WorldConfig  # noqa: E402
from adaptive_questionnaires.v2.config import V2Config  # noqa: E402
from adaptive_questionnaires.v2.engine import EngineOptions  # noqa: E402
from adaptive_questionnaires.v2.redundancy import LexicalSimilarity  # noqa: E402

OUT = ROOT / "outputs" / "evaluation"

# metric -> (direction, sessions_from, primary?)   direction: +1 higher is better, -1 lower, 0 descriptive
METRICS = {
    "intent_duplicate_rate":        (-1, 0, True),
    "longitudinal_repetition":      (-1, 1, True),
    "change_followup_rate":         (+1, 1, True),
    "min_per_category":             (+1, 0, True),
    "synthetic_rating_quality":     (+1, 0, True),
    "exact_duplicate_rate":         (-1, 0, False),
    "semantic_near_dup_pair_rate":  (-1, 0, False),
    "semantic_diversity":           (+1, 0, False),
    "categories_covered":           (+1, 0, False),
    "category_entropy":             (+1, 0, False),
    "relevance_proxy_new":          (+1, 1, False),
    "replacement_churn":            (0, 1, False),
    "mean_words":                   (0, 0, False),
    "invalid_output_rate":          (-1, 0, False),
    "llm_calls":                    (0, 0, False),
    "retries":                      (-1, 0, False),
    "generated_items":              (0, 0, False),
    "selection_ms":                 (0, 0, False),
}
DECISION_RULE = (
    "Primary proxies: intent_duplicate_rate, longitudinal_repetition, change_followup_rate, "
    "min_per_category, synthetic_rating_quality. 'Better'/'worse' = the 95% paired-bootstrap CI of "
    "(V2 − V1) excludes 0 in the favourable/unfavourable direction. "
    "'V2 improves the measured question-selection proxies' if >=1 primary proxy is better and none is worse; "
    "'V2 produces mixed results' if some are better and some worse; "
    "'V2 does not outperform the baseline on the current evaluation' if none is better."
)


def git_sha():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def semantic_similarity():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from adaptive_questionnaires.v2.redundancy import SentenceTransformerSimilarity
        return SentenceTransformerSimilarity("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print(f"[info] semantic backend unavailable ({type(e).__name__}); semantic metrics and V2-semantic skipped")
        return None


def verdict(rows):
    prim = [r for r in rows if r["primary"]]
    better = [r["metric"] for r in prim if r["assessment"] == "better"]
    worse = [r["metric"] for r in prim if r["assessment"] == "worse"]
    if better and not worse:
        v = "V2 improves the measured question-selection proxies."
    elif better and worse:
        v = "V2 produces mixed results."
    else:
        v = "V2 does not outperform the baseline on the current evaluation."
    return v, better, worse


def compare(df, a, b):
    rows = []
    for m, (direction, s_from, primary) in METRICS.items():
        if m not in df.columns:
            continue
        r = paired_bootstrap(df, m, a, b, sessions_from=s_from)
        if r is None:
            continue
        if direction == 0:
            assess = "descriptive"
        elif r["ci_low"] > 0:
            assess = "better" if direction > 0 else "worse"
        elif r["ci_high"] < 0:
            assess = "better" if direction < 0 else "worse"
        else:
            assess = "no clear difference"
        rows.append({"metric": m, "baseline": a, "method": b, "primary": primary, "sessions_from": s_from,
                     "direction": {1: "higher better", -1: "lower better", 0: "descriptive"}[direction],
                     **r, "assessment": assess})
    return rows


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=30)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--out", default=None, help="output directory (default outputs/evaluation, "
                                                 "or outputs/evaluation/quick with --quick)")
    args = ap.parse_args()
    global OUT
    if args.quick:
        args.patients, args.sessions = 4, 3
        OUT = OUT / "quick"          # never overwrite the recorded full-run results (git-ignored)
    if args.out:
        OUT = pathlib.Path(args.out)

    started = time.time()
    commit = git_sha()   # before any output is written, so the run's own files don't mark it dirty
    wcfg = WorldConfig(seed=args.seed, n_patients=args.patients, n_sessions=args.sessions)
    world = World(wcfg)
    cfg = V2Config(redundancy_backend="lexical")   # V2 default install: no optional extras
    lex = LexicalSimilarity()
    sem = None if args.no_semantic else semantic_similarity()

    runs = {"V1": run_v1(world), "V2": run_v2(world, cfg, lex, "V2")}
    runs["V2-random-control"] = run_v2(world, cfg, lex, "V2-random-control", EngineOptions(random_control=True))
    ablations = {
        "V2 −redundancy": EngineOptions(disabled_components=("redundancy",), use_redundancy_filter=False),
        "V2 −coverage": EngineOptions(disabled_components=("coverage",)),
        "V2 −longitudinal": EngineOptions(disabled_components=("longitudinal",)),
        "V2 −clinician_history": EngineOptions(disabled_components=("clinician_history",)),
    }
    for name, opt in ablations.items():
        runs[name] = run_v2(world, cfg, lex, name, opt)
    # POST-HOC sensitivity (added after seeing the pre-specified results; excluded from the verdict):
    # V1 forces exactly 4 per category; V2's default floor is 2. Does a V1-equivalent floor change things?
    runs["V2 floor=4 (post-hoc)"] = run_v2(world, cfg.with_overrides(min_per_category=4), lex, "V2 floor=4 (post-hoc)")
    if sem is not None:
        runs["V2-semantic"] = run_v2(world, cfg.with_overrides(redundancy_backend="sentence_transformers"),
                                     sem, "V2-semantic")

    records = [r for rs in runs.values() for r in rs]
    df = compute_metrics(world, records, sem)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_session_metrics.csv", index=False)

    main_rows = compare(df, "V1", "V2")
    ctrl_rows = compare(df, "V2-random-control", "V2")
    sem_rows = compare(df, "V1", "V2-semantic") if "V2-semantic" in runs else []
    posthoc_rows = compare(df, "V1", "V2 floor=4 (post-hoc)")
    abl_rows = [row | {"ablation": name} for name in ablations for row in compare(df, "V2", name)]
    pd.DataFrame(main_rows + ctrl_rows + sem_rows + posthoc_rows).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(OUT / "ablations.csv", index=False)
    v, better, worse = verdict(main_rows)

    means = df.groupby("method").mean(numeric_only=True)
    summary = {"verdict": v, "primary_better": better, "primary_worse": worse, "decision_rule": DECISION_RULE,
               "clinical_accuracy": "NOT EVALUATED (no outcome labels; synthetic data only)",
               "method_means": json.loads(means.to_json()), "main": main_rows, "vs_random_control": ctrl_rows,
               "semantic_variant": sem_rows, "posthoc_floor4": posthoc_rows, "ablations": abl_rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "python": platform.python_version(), "platform": platform.platform(),
        "packages": {m: __import__(m).__version__ for m in ("pandas", "numpy")},
        "world_config": wcfg.__dict__, "v2_config": cfg.to_dict(), "prompt_version": cfg.prompt_version,
        "generation_model": "synthetic-fixture (no LLM calls; deterministic replay)",
        "evaluation_similarity": getattr(sem, "model_name", None) or "unavailable",
        "methods": list(runs), "runtime_s": round(time.time() - started, 1),
        "note": "LLM outputs are simulated; real LLM outputs are not deterministic and were not evaluated.",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    # ---- markdown
    def table(rows, a_label, b_label):
        out = [f"| Metric | {a_label} | {b_label} | Difference ({b_label} − {a_label}) [95% CI] | Direction | Assessment |",
               "|---|--:|--:|--:|---|---|"]
        for r in rows:
            star = " **(primary)**" if r["primary"] else ""
            out.append(f"| `{r['metric']}`{star} | {fmt(r['a_mean'])} | {fmt(r['b_mean'])} | "
                       f"{fmt(r['diff'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {r['direction']} | {r['assessment']} |")
        return out

    md = ["# V1 vs V2 — synthetic question-selection evaluation", "",
          f"**Result: {v}**", "", "**Clinical predictive accuracy: NOT EVALUATED.**", "",
          f"Synthetic world: {wcfg.n_patients} patients × {wcfg.n_sessions} sessions, seed {wcfg.seed}. "
          f"Unit of analysis: patient (mean over sessions); paired bootstrap, 2000 resamples. "
          f"Evaluation similarity model: `{manifest['evaluation_similarity']}`. Git: `{manifest['git_commit']}`.", "",
          f"Decision rule (fixed before the run): {DECISION_RULE}", "",
          "## V2 (default install, lexical redundancy) vs V1", "", *table(main_rows, "V1", "V2"), "",
          "## V2 vs random-selection control (same pool, same constraints, no ranking signal)", "",
          *table(ctrl_rows, "control", "V2"), ""]
    if sem_rows:
        md += ["## V2 with semantic redundancy backend vs V1", "",
               "Caution: the semantic evaluation metrics use the same embedding model that this variant "
               "uses for selection, so they are partly circular for this row. Intent-based metrics are not.", "",
               *table(sem_rows, "V1", "V2-semantic"), ""]
    md += ["## Post-hoc sensitivity: V2 with min_per_category = 4 vs V1", "",
           "Added AFTER the pre-specified results were seen, because V2's default floor (2) differs from V1's "
           "fixed grid (4). It is **not** part of the decision rule and must not be read as the primary result.", "",
           *table(posthoc_rows, "V1", "V2 floor=4"), ""]
    md += ["## Ablations (each vs full V2)", "",
           "| Ablation | Metric | V2 | Ablated | Difference [95% CI] | Assessment (of the ablated variant) |",
           "|---|---|--:|--:|--:|---|"]
    for r in abl_rows:
        if r["direction"] == "descriptive" and r["metric"] not in ("replacement_churn",):
            continue
        md.append(f"| {r['ablation']} | `{r['metric']}` | {fmt(r['a_mean'])} | {fmt(r['b_mean'])} | "
                  f"{fmt(r['diff'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {r['assessment']} |")
    md += ["", "Units: `invalid_output_rate` is rejected category calls / calls for V1 and invalid items / "
           "generated items for V2 (different units; not directly comparable). Tokens, cost and API latency "
           "were not measured (no API calls); `selection_ms` is local compute only.", ""]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:8]))
    print(f"wrote {OUT}/ in {manifest['runtime_s']}s")


if __name__ == "__main__":
    main()

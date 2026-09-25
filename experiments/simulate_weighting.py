#!/usr/bin/env python3
"""Compare clinician-score representations before choosing one for V2.

Scenarios (all synthetic): repeated high ratings, repeated low ratings, missing
ratings, a new question, a replaced question, reordered rows, a single outlier,
and a long 50-session sequence. Writes
outputs/evaluation/weighting_simulation.{json,md}.

The representations compared:
  legacy_additive   w += 0.2(f − 4), init 0.5, unscored counted as 0 (upstream)
  legacy_missing_skip  same, but missing ratings skipped (upstream fixed for E)
  clipped_additive  legacy_missing_skip clipped to [0, 1]
  running_mean      mean of (f − 1)/4, init prior 0.5 as one pseudo-rating
  zscore            per-session z-score of f across the questions of that session
  ema               EMA of (f − 1)/4, α = 0.5, init 0.5
  ema_shrunk        EMA shrunk toward 0.5 with k = 2 pseudo-ratings (V2 choice)
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_questionnaires.v2.models import ClinicianFeedback, CLINICIAN_DIMENSIONS  # noqa: E402
from adaptive_questionnaires.v2.scoring import ClinicianHistoryScorer  # noqa: E402

OUT = ROOT / "outputs" / "evaluation"


def fb(f):
    return None if f is None else ClinicianFeedback(scores={d: f for d in CLINICIAN_DIMENSIONS})


# ---- scalar representations: state -> state given composite f (None = missing) ----
def legacy_additive(w, f):
    return w + 0.2 * ((0.0 if f is None else f) - 4.0)


def legacy_missing_skip(w, f):
    return w if f is None else w + 0.2 * (f - 4.0)


def clipped_additive(w, f):
    return min(1.0, max(0.0, legacy_missing_skip(w, f)))


class RunningMean:
    def __init__(self):
        self.s, self.n = 0.5, 1

    def step(self, f):
        if f is not None:
            self.s += (f - 1) / 4
            self.n += 1
        return self.s / self.n


class EMA:
    def __init__(self, a=0.5):
        self.v, self.a = 0.5, a

    def step(self, f):
        if f is not None:
            self.v = (1 - self.a) * self.v + self.a * (f - 1) / 4
        return self.v


class EMAShrunk:
    def __init__(self):
        self.sc = ClinicianHistoryScorer(ema_alpha=0.5, prior_strength=2.0, prior=0.5)
        self.sc.init_question("q")
        self.t = 0

    def step(self, f):
        self.t += 1
        self.sc.update("q", f"s{self.t}", fb(f))
        return self.sc.score("q")


def run_scalar(seq):
    out = {}
    for name, fn in [("legacy_additive", legacy_additive), ("legacy_missing_skip", legacy_missing_skip),
                     ("clipped_additive", clipped_additive)]:
        w, traj = 0.5, []
        for f in seq:
            w = fn(w, f)
            traj.append(w)
        out[name] = traj
    for name, cls in [("running_mean", RunningMean), ("ema", EMA), ("ema_shrunk", EMAShrunk)]:
        obj = cls()
        out[name] = [obj.step(f) for f in seq]
    return out


def zscore_session(fs):
    vals = [f for f in fs if f is not None]
    if len(vals) < 2:
        return [float("nan")] * len(fs)
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
    return [float("nan") if (f is None or sd == 0) else (f - mu) / sd for f in fs]


def finite(xs):
    return all(isinstance(x, float) and math.isfinite(x) for x in xs)


def main():
    rng = random.Random(0)
    scen = {
        "repeated_high_5x10": [5.0] * 10,
        "repeated_low_1x10": [1.0] * 10,
        "average_3x10": [3.0] * 10,
        "missing_after_two": [4.0, 4.0, None, None, None],
        "outlier_1_among_5s": [5.0, 5.0, 5.0, 1.0, 5.0, 5.0],
        "long_50_uniform": [float(rng.choice([1, 2, 3, 4, 5])) for _ in range(50)],
    }
    results = {"scalar": {}, "properties": {}}
    for sname, seq in scen.items():
        results["scalar"][sname] = {k: [round(x, 4) for x in v] for k, v in run_scalar(seq).items()}

    names = list(results["scalar"]["repeated_high_5x10"].keys())
    props = {}
    for n in names:
        hi = results["scalar"]["repeated_high_5x10"][n]
        lo = results["scalar"]["repeated_low_1x10"][n]
        avg = results["scalar"]["average_3x10"][n]
        miss = results["scalar"]["missing_after_two"][n]
        out = results["scalar"]["outlier_1_among_5s"][n]
        long = results["scalar"]["long_50_uniform"][n]
        props[n] = {
            "bounded_0_1_long_run": all(0 <= x <= 1 for x in long),
            "finite": finite(long) and finite(hi) and finite(lo),
            "long_run_range": [round(min(long), 3), round(max(long), 3)],
            "after_10_high": round(hi[-1], 3),
            "after_10_low": round(lo[-1], 3),
            "after_10_average(3)": round(avg[-1], 3),
            "missing_changes_state": round(miss[-1] - miss[1], 3) != 0,
            "outlier_drop": round(out[2] - out[3], 3),
            "first_rating_5_moves_from_0.5_to": round(hi[0], 3),
        }
    # z-score: session-relative
    zs_equal = zscore_session([4.0, 4.0, 4.0, 4.0])
    zs_one = zscore_session([4.0, None, None, None])
    props["zscore"] = {
        "finite_when_all_equal": finite(zs_equal),
        "finite_with_single_rating": finite([x for x in zs_one[:1]]),
        "note": "session-relative: a question's value depends on the other questions rated that session; "
                "undefined when all ratings are equal or only one is present",
    }
    # identity / reordering: legacy positional alignment vs id-keyed state
    sc = ClinicianHistoryScorer()
    for qid in ("A", "B"):
        sc.init_question(qid)
    sc.update("A", "s1", fb(5.0))
    sc.update("B", "s1", fb(1.0))
    before = {q: sc.score(q) for q in ("A", "B")}
    positional_after_swap = [before["B"], before["A"]]   # rows swapped, weights aligned by position
    id_keyed_after_swap = [sc.score("B"), sc.score("A")]
    sc.init_question("C")  # replacement gets a NEW id
    props["identity"] = {
        "id_keyed_invariant_to_reorder": id_keyed_after_swap == [before["B"], before["A"]],
        "positional_alignment_moves_history": positional_after_swap != [before["A"], before["B"]],
        "replacement_starts_at_prior": sc.score("C") == 0.5 and sc.n_ratings("C") == 0,
        "legacy_new_question_weight": 0.5,
        "legacy_new_vs_well_rated_retained_after_5_sessions": [0.5, round(0.5 + 5 * 0.2, 3)],
    }
    results["properties"] = props

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weighting_simulation.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    rows = ["| Representation | bounded [0,1] (50 sessions) | finite | 50-session range | after 10×5 | after 10×1 | after 10×3 | missing changes state | drop from one outlier | first rating of 5 → |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = props[n]
        rows.append(f"| `{n}` | {p['bounded_0_1_long_run']} | {p['finite']} | {p['long_run_range']} | "
                    f"{p['after_10_high']} | {p['after_10_low']} | {p['after_10_average(3)']} | "
                    f"{p['missing_changes_state']} | {p['outlier_drop']} | {p['first_rating_5_moves_from_0.5_to']} |")
    md = ["# Clinician-score representation simulation", "",
          "Generated by `experiments/simulate_weighting.py` (synthetic sequences, seed 0). "
          "Composite ratings are on the 1–5 Likert scale.", "", *rows, "",
          "## Session-relative z-score", "",
          f"- finite when all ratings are equal: **{props['zscore']['finite_when_all_equal']}**",
          f"- finite with a single rating in the session: **{props['zscore']['finite_with_single_rating']}**",
          f"- {props['zscore']['note']}", "",
          "## Identity", "",
          f"- id-keyed state invariant to row reordering: **{props['identity']['id_keyed_invariant_to_reorder']}**",
          f"- positional alignment moves history when rows are reordered: **{props['identity']['positional_alignment_moves_history']}**",
          f"- replaced question starts at prior with 0 ratings: **{props['identity']['replacement_starts_at_prior']}**",
          f"- legacy: new question 0.5 vs a retained question rated 5 for 5 sessions: "
          f"{props['identity']['legacy_new_vs_well_rated_retained_after_5_sessions']} (unbounded scale)", "",
          "## Reading", "",
          "- `legacy_additive` is unbounded and drifts (it goes negative under uniform random ratings), "
          "treats missing as 0, and falls under a steady *average* rating of 3 because μ = 4.",
          "- `clipped_additive` is bounded but saturates at 0 or 1 and then loses information. Its μ-asymmetry remains.",
          "- `running_mean` is bounded but becomes increasingly insensitive to change in long sequences.",
          "- `zscore` is undefined (NaN) in common cases and is not comparable across sessions.",
          "- `ema` is bounded and responsive, but a single first rating moves an unproven question far from neutral.",
          "- `ema_shrunk` (V2) is bounded, finite, skips missing ratings, is responsive, and damps the first ratings of new questions. "
          "Trade-offs: 2 pseudo-ratings slow early learning; it reacts more to a single outlier than "
          "`running_mean` does; and once n is large the shrinkage fades, so it behaves like an α = 0.5 EMA, "
          "which tracks rating noise (see the 50-session range). α and k are design choices, not validated parameters."]
    (OUT / "weighting_simulation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Smoke and sanity tests for the synthetic evaluation harness."""
import math

import pytest

from adaptive_questionnaires.evaluation.harness import compute_metrics, paired_bootstrap, run_v1, run_v2
from adaptive_questionnaires.evaluation.synthetic import INTENT_TEXTS, World, WorldConfig, intent_of
from adaptive_questionnaires.evaluation.synthetic_bank import BANK
from adaptive_questionnaires.v2.config import V2Config
from adaptive_questionnaires.v2.engine import EngineOptions
from adaptive_questionnaires.v2.redundancy import LexicalSimilarity


@pytest.fixture(scope="module")
def small():
    w = World(WorldConfig(n_patients=3, n_sessions=3, seed=7))
    recs = run_v1(w) + run_v2(w, V2Config(redundancy_backend="lexical"), LexicalSimilarity(), "V2")
    return w, recs, compute_metrics(w, recs, None)


def test_every_bank_text_maps_to_exactly_one_intent():
    texts = [t for forms in INTENT_TEXTS.values() for t in forms]
    assert len(texts) == len(set(texts))
    assert all(intent_of(t) is not None for t in texts)


def test_bank_has_no_screening_or_risk_items():
    import re
    blob = " ".join(t for forms in INTENT_TEXTS.values() for t in forms).lower()
    for pat in (r"suicid", r"\bkill", r"\bdie\b", r"\bdeath", r"self-harm", r"harm yourself", r"\brisk"):
        assert not re.search(pat, blob), pat


def test_world_is_deterministic():
    a, b = World(WorldConfig(n_patients=2, seed=1)), World(WorldConfig(n_patients=2, seed=1))
    assert [p.onsets for p in a.patients] == [p.onsets for p in b.patients]
    assert a.stream(a.patients[0], 1, "Mood", 5) == b.stream(b.patients[0], 1, "Mood", 5)


def test_ratings_are_independent_of_change_status():
    w = World(WorldConfig(n_patients=1, seed=3))
    p = w.patients[0]
    text = BANK["Mood"]["background"]["md_general"][0]
    # same (session, text) -> same rating regardless of which categories changed
    p.changed = [set(), {"Mood"}, set()] + p.changed[3:]
    r1 = w.rate(p, 1, text)
    p.changed[1] = set()
    assert w.rate(p, 1, text) == r1


def test_v1_simulation_uses_legacy_quota(small):
    w, recs, df = small
    v1 = df[(df.method == "V1") & (df.session > 0)]
    assert (v1.n_questions == 20).all()
    # upstream replaces exactly 2 per category unless the LLM reply was rejected
    assert (v1.replacement_churn <= 0.5 + 1e-9).all()


def test_metrics_are_finite_and_bounded(small):
    _, _, df = small
    for col in ("intent_duplicate_rate", "exact_duplicate_rate", "categories_covered", "category_entropy"):
        vals = df[col].dropna()
        assert vals.between(0, 1).all()
    assert all(math.isfinite(x) for x in df.synthetic_rating_quality.dropna())


def test_paired_bootstrap_shape(small):
    _, _, df = small
    r = paired_bootstrap(df, "intent_duplicate_rate", "V1", "V2", n_boot=200)
    assert r["ci_low"] <= r["diff"] <= r["ci_high"] and r["n_patients"] == 3


def test_random_control_runs(small):
    w, _, _ = small
    recs = run_v2(w, V2Config(redundancy_backend="lexical"), LexicalSimilarity(), "ctrl",
                  EngineOptions(random_control=True))
    assert {r.method for r in recs} == {"ctrl"} and all(len(r.texts) == 20 for r in recs)

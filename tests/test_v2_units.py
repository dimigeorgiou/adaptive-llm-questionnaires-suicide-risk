"""V2 unit tests: parsing/schemas, scoring, redundancy, coverage, longitudinal, ranker."""
import json
import math
import os
import random

import pytest

from adaptive_questionnaires.v2.candidate_generator import (
    CandidateGenerator, FixtureBackend, build_messages, candidate_schema, parse_candidates,
)
from adaptive_questionnaires.v2.config import V2Config
from adaptive_questionnaires.v2.coverage import CoverageState
from adaptive_questionnaires.v2.longitudinal import build_longitudinal_state, longitudinal_value
from adaptive_questionnaires.v2.models import CLINICIAN_DIMENSIONS, CandidateQuestion, ClinicianFeedback
from adaptive_questionnaires.v2.ranker import (
    ExpectedUtilityProxy, InformationGainEstimator, RankerWeights, RankingContext,
)
from adaptive_questionnaires.v2.redundancy import LexicalSimilarity, dedupe_batch, redundancy_metrics
from adaptive_questionnaires.v2.scoring import ClinicianHistoryScorer

CATS = ["Routine", "Mood", "Connection", "Coping", "Goals"]
LEX = LexicalSimilarity()


def fb(v):
    return ClinicianFeedback(scores={d: v for d in CLINICIAN_DIMENSIONS})


# ---------------------------------------------------------------- parsing / schemas
def test_structured_candidates_parse():
    raw = json.dumps({"candidates": [{"category": "Mood", "text": "How has your mood been?"},
                                     {"category": "routine", "text": "  How are your   days?  "}]})
    r = parse_candidates(raw, CATS, "S1-s1")
    assert r.response_valid and [c.category for c in r.candidates] == ["Mood", "Routine"]
    assert r.candidates[1].text == "How are your days?"
    assert len({c.candidate_id for c in r.candidates}) == 2


@pytest.mark.parametrize("raw,err", [
    (None, "empty"), ("", "empty"), ("not json", "invalid JSON"), ("[1,2]", "missing 'candidates'"),
    ('{"candidates": {}}', "missing 'candidates'"),
])
def test_malformed_responses(raw, err):
    r = parse_candidates(raw, CATS, "s")
    assert not r.response_valid and err in r.errors[0] and r.candidates == []


def test_invalid_items_are_rejected_individually():
    raw = json.dumps({"candidates": [
        {"category": "Mood"},                                        # missing text
        {"text": "Q?"},                                              # missing category
        {"category": "Diagnosis", "text": "Q?"},                     # invalid category
        {"category": "Mood", "text": "   "},                         # empty
        {"category": "Mood", "text": "x" * 301},                     # too long
        {"category": "Mood", "text": "Same?"}, {"category": "Mood", "text": "same"},  # exact dup
        {"category": "Mood", "text": "A?", "id": 1}, {"category": "Mood", "text": "B?", "id": 1},  # dup id
        "not an object",
        {"category": "Mood", "text": "Valid?", "self_rating": 10, "information_gain": 0.9},
    ]})
    r = parse_candidates(raw, CATS, "s")
    assert [c.text for c in r.candidates] == ["Same?", "A?", "Valid?"]
    assert r.invalid_items == 8
    assert r.ignored_fields == 2                                   # self-ratings are ignored, not used


def test_schema_enforces_category_enum_and_required_fields():
    s = candidate_schema(CATS)
    item = s["properties"]["candidates"]["items"]
    assert item["properties"]["category"]["enum"] == CATS and item["required"] == ["category", "text"]
    assert item["additionalProperties"] is False


def test_generator_retries_until_valid_output():
    good = json.dumps({"candidates": [{"category": "Mood", "text": "Q?"}]})
    gen = CandidateGenerator(FixtureBackend(["garbage", '{"candidates": []}', good]), max_attempts=3)
    pool, m = gen.generate("s", CATS, {"Mood": 1}, {}, {}, 3)
    assert len(pool) == 1 and m.attempts == 3 and m.invalid_responses == 1


def test_generator_gives_up_after_max_attempts():
    gen = CandidateGenerator(FixtureBackend(["garbage"]), max_attempts=2)
    pool, m = gen.generate("s", CATS, {"Mood": 1}, {}, {}, 3)
    assert pool == [] and m.attempts == 2 and m.invalid_generation_rate == 1.0


def test_prompt_contains_summaries_not_history_dump_and_no_self_rating_request():
    class St:
        current_summary, changed, unresolved = "summary text", True, False
    msgs = build_messages({"Mood": 2, "Goals": 0}, {"Mood": St()}, {"Mood": ["Old q?"]}, 3)
    user = json.loads(msgs[1]["content"].split("\n", 1)[1])
    assert [b["category"] for b in user["categories"]] == ["Mood"]
    assert user["categories"][0]["candidates_requested"] == 6
    assert "rating" not in json.dumps(user).lower()
    assert "no ratings" in msgs[0]["content"]


# ---------------------------------------------------------------- scoring
def test_scoring_deterministic_bounded_and_finite():
    rng = random.Random(0)
    for _ in range(50):
        sc = ClinicianHistoryScorer(ema_alpha=rng.uniform(0.05, 1), prior_strength=rng.uniform(0, 5))
        sc.init_question("q")
        for t in range(60):
            v = rng.choice([None, 1, 2, 3, 4, 5, 1.5, 4.5])
            sc.update("q", f"s{t}", None if v is None else fb(v))
            s = sc.score("q")
            assert math.isfinite(s) and 0.0 <= s <= 1.0


def test_missing_ratings_do_not_change_score():
    sc = ClinicianHistoryScorer()
    sc.init_question("q")
    sc.update("q", "s1", fb(5))
    before = sc.score("q")
    sc.update("q", "s2", None)
    sc.update("q", "s3", ClinicianFeedback(scores={}))
    sc.update("q", "s4", ClinicianFeedback(scores={d: None for d in CLINICIAN_DIMENSIONS}))
    assert sc.score("q") == before and sc.n_ratings("q") == 1


def test_partial_rating_uses_rated_dimensions_only():
    f = ClinicianFeedback(scores={"Coherence": 5, "Engagement": None, "Motivational Impact": float("nan")})
    assert f.composite() == 5 and not f.is_complete()


@pytest.mark.parametrize("bad", [0, 6, -1, float("inf")])
def test_out_of_range_scores_rejected(bad):
    sc = ClinicianHistoryScorer()
    sc.init_question("q")
    with pytest.raises(ValueError):
        sc.update("q", "s", ClinicianFeedback(scores={"Coherence": bad}))


def test_new_question_initialisation_is_explicit():
    sc = ClinicianHistoryScorer(prior=0.5)
    sc.init_question("new")
    assert sc.score("new") == 0.5 and sc.n_ratings("new") == 0
    with pytest.raises(ValueError):
        sc.init_question("new")                     # ids are never reused
    with pytest.raises(KeyError):
        sc.update("never-initialised", "s", fb(3))


def test_normalisation_and_shrinkage():
    sc = ClinicianHistoryScorer(ema_alpha=1.0, prior_strength=2.0, prior=0.5)
    sc.init_question("q")
    sc.update("q", "s", fb(5))
    assert sc.score("q") == pytest.approx((1 * 1.0 + 2 * 0.5) / 3)
    for t in range(200):
        sc.update("q", f"s{t}", fb(1))
    assert sc.score("q") == pytest.approx(0.0, abs=0.02)


# ---------------------------------------------------------------- redundancy
def test_exact_duplicates_detected():
    assert LEX.sim("Who have you spoken to?", "who have you spoken to") == 1.0
    m = redundancy_metrics(["A b c?", "a b c", "Different question here?"], LEX, 0.8)
    assert m["exact_duplicate_rate"] == pytest.approx(1 / 3)


def test_unrelated_questions_not_duplicates_lexical():
    assert LEX.sim("How has your sleep changed?", "Who have you spoken to when feeling overwhelmed?") < 0.3


def test_lexical_reordering_detected():
    assert LEX.sim("Who do you talk to when you feel overwhelmed?",
                   "When you feel overwhelmed, who do you talk to?") >= 0.8


@pytest.mark.xfail(strict=True, reason="documented limitation: lexical fallback cannot see paraphrases "
                                       "without word overlap; use the semantic backend")
def test_lexical_fallback_paraphrase_limitation():
    assert LEX.sim("Have you been sleeping badly lately?", "Has your sleep been poor recently?") >= 0.8


def _st():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pytest.importorskip("sentence_transformers")
    from adaptive_questionnaires.v2.redundancy import SentenceTransformerSimilarity
    try:
        return SentenceTransformerSimilarity("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:  # model not cached locally
        pytest.skip(f"embedding model unavailable offline: {type(e).__name__}")


def test_semantic_backend_paraphrase_and_unrelated():
    st = _st()
    assert st.sim("Have you been sleeping badly lately?", "Has your sleep been poor recently?") >= 0.8
    assert st.sim("How has your sleep changed?", "Who have you spoken to when feeling overwhelmed?") < 0.5


def test_dedupe_batch_keeps_first():
    kept, dropped = dedupe_batch(["A b c?", "Q two?", "a b c"], LEX, 0.8)
    assert kept == [0, 1] and dropped[0][:2] == (2, 0)


# ---------------------------------------------------------------- coverage
def _cov(**kw):
    d = dict(categories=CATS, max_questions=20, min_per_category=2, max_per_category=6)
    d.update(kw)
    return CoverageState(**d)


def test_undercovered_domain_has_higher_need():
    cov = _cov()
    for _ in range(4):
        cov.add("Mood")
    assert cov.coverage_need("Goals") == 1.0 and cov.coverage_need("Mood") == 0.0
    assert cov.coverage_need("Goals") > cov.coverage_need("Mood")


def test_saturated_domain_and_floors():
    cov = _cov(max_per_category=3)
    for _ in range(3):
        cov.add("Mood")
    assert cov.is_saturated("Mood")
    assert set(cov.unmet_floor()) == set(CATS) - {"Mood"}


def test_priority_and_required_validation():
    cov = _cov(priority={"Mood": 3.0})
    assert cov.target("Mood") > cov.target("Goals") >= cov.floor("Goals")
    with pytest.raises(ValueError):
        _cov(min_per_category=5)                     # 5 × 5 > 20
    with pytest.raises(ValueError):
        _cov(required=["Unknown"])


# ---------------------------------------------------------------- longitudinal
def test_changed_information_creates_signal_and_value():
    states, signals = build_longitudinal_state(
        CATS, {"Routine": "Sleep described as normal and regular."},
        {"Routine": "Reports barely sleeping this week."}, LEX, change_threshold=0.6)
    assert states["Routine"].changed and signals[0].changed_state_possible
    assert "not" not in signals[0].rationale.split("differs")[0]   # neutral wording
    assert not hasattr(signals[0], "patient_is_inconsistent")
    assert longitudinal_value("How has your sleeping changed this week?", states["Routine"], LEX) > 0


def test_stable_information_creates_no_signal_and_no_longitudinal_value():
    note = "Weekly contact with a sibling."
    states, signals = build_longitudinal_state(CATS, {"Connection": note}, {"Connection": note}, LEX, 0.6)
    assert not states["Connection"].changed and not signals[0].changed_state_possible
    assert longitudinal_value("Who have you spoken to this week?", states["Connection"], LEX) == 0.0


def test_first_session_has_no_change_signal():
    states, signals = build_longitudinal_state(CATS, {}, {"Mood": "x"}, LEX, 0.6)
    assert signals == [] and not states["Mood"].changed


# ---------------------------------------------------------------- ranker
def _ctx(**kw):
    sc = ClinicianHistoryScorer()
    d = dict(similarity=LEX, coverage=_cov(), states={}, scorer=sc, history_texts=[])
    d.update(kw)
    return RankingContext(**d)


def test_ranker_is_deterministic_and_components_bounded():
    r = ExpectedUtilityProxy(RankerWeights())
    c = CandidateQuestion("c1", "How has your week been?", "Mood")
    a = r.score(c, _ctx(), ["How was your week?"])
    b = r.score(c, _ctx(), ["How was your week?"])
    assert a == b
    for v in (c.relevance_score, c.novelty_score, c.coverage_score, c.longitudinal_score,
              c.clinician_prior_score, c.redundancy_score):
        assert 0.0 <= v <= 1.0


def test_redundancy_lowers_score_and_ablation_disables_it():
    c = CandidateQuestion("c1", "Who have you spoken to this week?", "Connection")
    full = ExpectedUtilityProxy(RankerWeights())
    no_red = ExpectedUtilityProxy(RankerWeights(disabled={"redundancy"}))
    sel = ["Who have you spoken to this week?"]
    assert full.score(c, _ctx(), sel) < full.score(c, _ctx(), [])
    assert no_red.score(c, _ctx(), sel) == no_red.score(c, _ctx(), [])
    with pytest.raises(ValueError):
        RankerWeights.from_config(V2Config(), disabled=["accuracy"])


def test_information_gain_is_not_implemented():
    est = InformationGainEstimator()
    assert est.enabled is False
    with pytest.raises(NotImplementedError, match="not implemented/calibrated"):
        est.expected_information_gain(CandidateQuestion("c", "t", "Mood"), None)


def test_config_validation():
    V2Config().validate()
    with pytest.raises(ValueError):
        V2Config(w_novelty=float("nan")).validate()
    with pytest.raises(ValueError):
        V2Config(min_questions=30).validate()
    with pytest.raises(ValueError):
        V2Config.from_mapping({"not_a_key": "1"})
    cfg = V2Config.from_mapping({"max_questions": "12", "adaptive_length": "true",
                                 "category_priority": "Mood:2, Goals:0.5", "required_categories": "Mood,Goals"})
    assert cfg.max_questions == 12 and cfg.adaptive_length and cfg.category_priority["Mood"] == 2.0

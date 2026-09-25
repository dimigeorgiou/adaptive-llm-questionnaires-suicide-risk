"""V2 engine: identity/replacement lifecycle, coverage constraints, state integrity, feedback."""
import json

import pytest

from adaptive_questionnaires.v2.candidate_generator import CandidateGenerator, FixtureBackend
from adaptive_questionnaires.v2.config import V2Config
from adaptive_questionnaires.v2.engine import (
    AdaptiveSessionEngine, EngineOptions, JSONStateStore, SessionContext, StateIntegrityError,
)
from adaptive_questionnaires.v2.models import CLINICIAN_DIMENSIONS
from adaptive_questionnaires.v2.redundancy import LexicalSimilarity

CATS = ["Routine", "Mood", "Connection", "Coping", "Goals"]
TOPICS = {"Routine": ["breakfast", "bedtime", "chores", "weekends", "mornings", "evenings", "errands", "walks"],
          "Mood": ["cheerfulness", "irritation", "calmness", "worry", "boredom", "contentment", "tension", "relief"],
          "Connection": ["siblings", "neighbours", "friends", "colleagues", "cousins", "classmates", "parents", "partner"],
          "Coping": ["breathing", "journaling", "music", "gardening", "painting", "stretching", "reading", "cooking"],
          "Goals": ["studies", "savings", "fitness", "hobbies", "volunteering", "travel", "career", "languages"]}


def cand_json(session_idx, per_cat=4, cats=CATS):
    """Distinct synthetic candidates (lexically dissimilar by construction)."""
    items = []
    for c in cats:
        for k in range(per_cat):
            topic = TOPICS[c][(session_idx * per_cat + k) % len(TOPICS[c])]
            items.append({"category": c, "text": f"What about {topic} s{session_idx}k{k} for you?"})
    return json.dumps({"candidates": items})


def engine(tmp_path, cfg=None, responses=None, **opt):
    cfg = cfg or V2Config(max_questions=10, min_per_category=2, max_per_category=4,
                          max_replacements_per_session=4, near_duplicate_threshold=0.9)
    gen = CandidateGenerator(FixtureBackend(responses or [cand_json(0)]))
    return AdaptiveSessionEngine(cfg, LexicalSimilarity(), gen, JSONStateStore(str(tmp_path)),
                                 options=EngineOptions(**opt))


def ctx(i, **kw):
    return SessionContext(anon_subject_id="S1", session_id=f"S1-s{i}", categories=CATS, **kw)


def rate(eng, i, value_by_qid=None, default=4):
    st = eng.store.load("S1")
    sess = st["sessions"][-1]
    rows = [{"question_id": q, **{d: (value_by_qid or {}).get(q, default) for d in CLINICIAN_DIMENSIONS}}
            for q in sess["selected_question_ids"]]
    return eng.import_feedback("S1", f"S1-s{i}", rows)


def test_first_session_fills_to_size_with_coverage_floor(tmp_path):
    eng = engine(tmp_path)
    r = eng.run_session(ctx(1))
    cats = [c.category for c in r.selected]
    assert len(r.selected) == 10 and all(cats.count(c) >= 2 for c in CATS)
    assert r.requires_clinician_review is True
    assert len({q.question_id for q in r.new_questions}) == 10


def test_replacement_gets_new_id_and_does_not_inherit_history(tmp_path):
    eng = engine(tmp_path, responses=[cand_json(0), cand_json(1)])
    r1 = eng.run_session(ctx(1))
    ids1 = [q.question_id for q in r1.new_questions]
    low = ids1[0]
    rate(eng, 1, {low: 1}, default=5)                     # one clearly low-rated question
    r2 = eng.run_session(ctx(2))
    st = eng.store.load("S1")
    assert low in r2.retired_question_ids
    for q in r2.new_questions:
        assert q.question_id not in ids1
        h = st["clinician_history"][q.question_id]
        assert h["n_ratings"] == 0 and h["history"] == []  # nothing inherited
        assert q.parent_question_id is None or q.parent_question_id in ids1
    assert st["questions"][low]["status"] == "retired"


def test_retained_question_preserves_history(tmp_path):
    eng = engine(tmp_path, responses=[cand_json(0), cand_json(1)])
    eng.run_session(ctx(1))
    rate(eng, 1, default=5)
    r2 = eng.run_session(ctx(2))
    st = eng.store.load("S1")
    retained = [c.carried_question_id for c in r2.selected if c.is_carryover]
    assert retained
    for qid in retained:
        assert st["clinician_history"][qid]["n_ratings"] == 1
        assert st["questions"][qid]["status"] == "active"


def test_replacement_cap_and_churn_bound(tmp_path):
    eng = engine(tmp_path, responses=[cand_json(0), cand_json(1)])
    eng.run_session(ctx(1))
    rate(eng, 1, default=1)                               # everything rated low
    r2 = eng.run_session(ctx(2))
    assert len(r2.new_questions) <= 4 and len(r2.selected) == 10


def test_locked_and_clinician_retired_questions(tmp_path):
    eng = engine(tmp_path, responses=[cand_json(0), cand_json(1)])
    r1 = eng.run_session(ctx(1))
    ids = [q.question_id for q in r1.new_questions]
    rate(eng, 1, {ids[0]: 1}, default=5)
    r2 = eng.run_session(ctx(2, locked_question_ids=[ids[0]], retire_question_ids=[ids[1]]))
    sel = {c.carried_question_id for c in r2.selected if c.is_carryover}
    assert ids[0] in sel and ids[1] not in sel
    with pytest.raises(ValueError):
        eng.run_session(ctx(3, locked_question_ids=["q_unknown"]))


def test_saturated_domain_not_repeated_and_undercovered_preferred(tmp_path):
    cfg = V2Config(max_questions=10, min_per_category=1, max_per_category=3, near_duplicate_threshold=0.9)
    only_mood = json.dumps({"candidates": [{"category": "Mood", "text": f"Mood item {w}?"}
                                           for w in TOPICS["Mood"]]
                                          + [{"category": "Goals", "text": "What about your studies?"}]})
    r = engine(tmp_path, cfg=cfg, responses=[only_mood]).run_session(ctx(1))
    cats = [c.category for c in r.selected]
    assert cats.count("Mood") == 3 and "Goals" in cats
    assert any("coverage floor not met" in w for w in r.metrics["warnings"])  # reported, not invented


def test_near_duplicates_filtered(tmp_path):
    dup = json.dumps({"candidates": [{"category": c, "text": "Who have you spoken to this week?"} for c in CATS[:2]]
                      + [{"category": c, "text": f"Who have you spoken to this week {c}?"} for c in CATS]})
    r = engine(tmp_path, responses=[dup]).run_session(ctx(1))
    texts = [c.text for c in r.selected]
    sim = LexicalSimilarity()
    assert all(sim.sim(a, b) < 0.9 for i, a in enumerate(texts) for b in texts[i + 1:])


def test_redundancy_ablation_allows_duplicates(tmp_path):
    # near-duplicates (not exact: exact duplicates are always rejected by schema validation)
    dup = json.dumps({"candidates": [{"category": c, "text": "Who have you spoken to this week?"} for c in CATS]
                      + [{"category": c, "text": "Who have you spoken to during this week?"} for c in CATS]})
    cfg = V2Config(max_questions=10, min_per_category=2, max_per_category=4)
    r = engine(tmp_path, cfg=cfg, responses=[dup], use_redundancy_filter=False,
               disabled_components=("redundancy",)).run_session(ctx(1))
    assert len(r.selected) == 10                                # ablated: near-duplicates admitted
    full = engine(tmp_path / "full", cfg=cfg, responses=[dup]).run_session(ctx(1))
    assert len(full.selected) < 10                              # full V2 refuses them


def test_adaptive_length_is_off_by_default_and_engineering_only(tmp_path):
    assert V2Config().adaptive_length is False
    cfg = V2Config(max_questions=10, min_questions=5, min_per_category=1, max_per_category=4,
                   adaptive_length=True, stop_min_utility=10.0)   # unreachable utility -> stop at min
    r = engine(tmp_path, cfg=cfg).run_session(ctx(1))
    assert len(r.selected) == 5
    assert r.stop_reason.startswith("adaptive follow-up generation complete")
    assert "safe" not in r.stop_reason.lower()


def test_duplicate_session_and_corrupted_state_refused(tmp_path):
    eng = engine(tmp_path, responses=[cand_json(0), cand_json(1)])
    eng.run_session(ctx(1))
    with pytest.raises(StateIntegrityError):
        eng.run_session(ctx(1))
    p = tmp_path / "S1.json"
    p.write_text(p.read_text()[:-20])                      # truncated file
    with pytest.raises(StateIntegrityError):
        eng.run_session(ctx(2))


def test_state_with_dangling_reference_refused(tmp_path):
    eng = engine(tmp_path)
    eng.run_session(ctx(1))
    p = tmp_path / "S1.json"
    st = json.loads(p.read_text())
    st["sessions"][0]["selected_question_ids"].append("q_missing")
    p.write_text(json.dumps(st))
    with pytest.raises(StateIntegrityError, match="unknown question"):
        eng.store.load("S1")


def test_feedback_import_is_atomic_and_keyed_by_id(tmp_path):
    eng = engine(tmp_path)
    r = eng.run_session(ctx(1))
    ids = [q.question_id for q in r.new_questions]
    bad = [{"question_id": ids[0], "Coherence": "5"}, {"question_id": ids[1], "Coherence": "7"}]
    with pytest.raises(StateIntegrityError):
        eng.import_feedback("S1", "S1-s1", bad)
    assert all(h["n_ratings"] == 0 for h in eng.store.load("S1")["clinician_history"].values())
    with pytest.raises(StateIntegrityError):
        eng.import_feedback("S1", "S1-s1", [{"question_id": "q_x", "Coherence": "5"}])
    with pytest.raises(StateIntegrityError):
        eng.import_feedback("S1", "S1-s1", [{"question_id": ids[0]}, {"question_id": ids[0]}])
    # reordered rows + blanks: fine
    rows = [{"question_id": q, "Coherence": "4", "Engagement": ""} for q in reversed(ids)]
    summary = eng.import_feedback("S1", "S1-s1", rows)
    assert summary == {"rated": 0, "partially_rated": 10, "unrated": 0}
    with pytest.raises(StateIntegrityError, match="already imported"):
        eng.import_feedback("S1", "S1-s1", rows)


def test_question_ids_persist_across_sessions(tmp_path):
    eng = engine(tmp_path, responses=[cand_json(0), cand_json(1), cand_json(2)])
    r1 = eng.run_session(ctx(1))
    rate(eng, 1, default=5)
    r2 = eng.run_session(ctx(2))
    kept = {c.carried_question_id for c in r2.selected if c.is_carryover}
    assert kept <= {q.question_id for q in r1.new_questions}
    st = eng.store.load("S1")
    assert set(st["sessions"][1]["selected_question_ids"]) >= kept


def test_session_context_rejects_identifying_ids_and_unknown_keys():
    with pytest.raises(ValueError):
        SessionContext(anon_subject_id="Jane Doe", session_id="s1", categories=CATS)
    with pytest.raises(ValueError):
        SessionContext.from_dict({"anon_subject_id": "S1", "session_id": "s1", "categories": CATS,
                                  "patient_name": "x"})

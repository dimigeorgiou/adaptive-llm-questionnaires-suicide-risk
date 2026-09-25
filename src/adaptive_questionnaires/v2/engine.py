"""V2 session orchestration and local state (JSON, one file per anonymous subject).

State files contain question text and clinician-approved summaries, which is
clinical content. They live under a git-ignored directory (default ``./v2_state``)
and must stay on authorised storage. Writes are atomic (tmp file + ``os.replace``).
If a state file cannot be read or validated, the engine raises and does not
start from an empty state.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from adaptive_questionnaires.v2.candidate_generator import CandidateGenerator, GenerationMetrics
from adaptive_questionnaires.v2.config import V2Config
from adaptive_questionnaires.v2.coverage import CoverageState
from adaptive_questionnaires.v2.instruments import NoInstrumentAdapter, ValidatedInstrumentAdapter
from adaptive_questionnaires.v2.longitudinal import build_longitudinal_state
from adaptive_questionnaires.v2.models import (
    CLINICIAN_DIMENSIONS, CandidateQuestion, ClinicianFeedback, Question, QuestionSource, QuestionStatus,
    SelectionResult, SessionQuestion, new_question_id, utcnow_iso,
)
from adaptive_questionnaires.v2.ranker import ExpectedUtilityProxy, RankerWeights, RankingContext
from adaptive_questionnaires.v2.redundancy import Similarity
from adaptive_questionnaires.v2.scoring import ClinicianHistoryScorer, ClinicianHistoryState
from adaptive_questionnaires.v2.selector import select

STATE_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class StateIntegrityError(RuntimeError):
    pass


@dataclass
class SessionContext:
    """Input for one session. ``notes_by_category`` should be clinician-written or approved summaries."""
    anon_subject_id: str
    session_id: str
    categories: List[str]
    notes_by_category: Dict[str, str] = field(default_factory=dict)
    unresolved_categories: List[str] = field(default_factory=list)
    locked_question_ids: List[str] = field(default_factory=list)
    retire_question_ids: List[str] = field(default_factory=list)
    category_priority: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("anon_subject_id", "session_id"):
            v = getattr(self, name)
            if not isinstance(v, str) or not _SAFE_ID.match(v):
                raise ValueError(f"{name} must match {_SAFE_ID.pattern} (opaque research id, no PII)")
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must be a non-empty list without duplicates")
        unknown = (set(self.notes_by_category) | set(self.unresolved_categories)) - set(self.categories)
        if unknown:
            raise ValueError(f"unknown categories in session context: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SessionContext":
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown session-context keys: {sorted(extra)}")
        return cls(**dict(d))


# ------------------------------------------------------------------ state store
class JSONStateStore:
    def __init__(self, root: str = "./v2_state"):
        self.root = root

    def path(self, subject: str) -> str:
        if not _SAFE_ID.match(subject):
            raise ValueError("invalid subject id")
        return os.path.join(self.root, f"{subject}.json")

    def load(self, subject: str) -> Dict[str, Any]:
        p = self.path(subject)
        if not os.path.exists(p):
            return {"version": STATE_VERSION, "anon_subject_id": subject, "questions": {},
                    "clinician_history": {}, "sessions": []}
        try:
            with open(p, encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise StateIntegrityError(f"state file for {subject} unreadable ({e.__class__.__name__}); "
                                      f"refusing to continue with an empty state") from e
        validate_state(st, subject)
        return st

    def save(self, subject: str, state: Dict[str, Any]) -> None:
        validate_state(state, subject)
        os.makedirs(self.root, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{subject}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp, self.path(subject))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def validate_state(st: Mapping[str, Any], subject: str) -> None:
    problems = []
    if st.get("version") != STATE_VERSION:
        problems.append(f"unsupported state version {st.get('version')!r}")
    if st.get("anon_subject_id") != subject:
        problems.append("subject id mismatch")
    qs = st.get("questions", {})
    for qid, q in qs.items():
        if q.get("question_id") != qid:
            problems.append(f"question key/id mismatch for {qid}")
        pq = q.get("parent_question_id")
        if pq is not None and pq not in qs:
            problems.append(f"{qid}: parent {pq} missing")
    for qid in st.get("clinician_history", {}):
        if qid not in qs:
            problems.append(f"clinician history for unknown question {qid}")
    seen_sessions = set()
    for s in st.get("sessions", []):
        sid = s.get("session_id")
        if sid in seen_sessions:
            problems.append(f"duplicate session {sid}")
        seen_sessions.add(sid)
        for qid in s.get("selected_question_ids", []):
            if qid not in qs:
                problems.append(f"session {sid} references unknown question {qid}")
    if problems:
        raise StateIntegrityError("; ".join(problems[:10]))


# ------------------------------------------------------------------ engine
@dataclass
class EngineOptions:
    disabled_components: Sequence[str] = ()       # ablations (see ranker.COMPONENTS)
    use_redundancy_filter: bool = True
    deterministic_ids: bool = False               # experiments: ids derived from content
    random_control: bool = False                  # evaluation control: no ranking signal, hashed order


class AdaptiveSessionEngine:
    def __init__(self, cfg: V2Config, similarity: Similarity, generator: Optional[CandidateGenerator],
                 store: JSONStateStore, instrument: ValidatedInstrumentAdapter = None,
                 options: EngineOptions = None):
        self.cfg = cfg.validate()
        self.sim = similarity
        self.generator = generator
        self.store = store
        self.instrument = instrument or NoInstrumentAdapter()
        self.options = options or EngineOptions()
        disabled = set(self.options.disabled_components)
        if self.options.random_control:
            from adaptive_questionnaires.v2.ranker import COMPONENTS
            disabled |= set(COMPONENTS)
            self.cfg = self.cfg.with_overrides(replacement_margin=0.0)
        self.ranker = ExpectedUtilityProxy(RankerWeights.from_config(self.cfg, disabled))

    # -- helpers ---------------------------------------------------------------
    def _scorer(self, state) -> ClinicianHistoryScorer:
        sc = ClinicianHistoryScorer(self.cfg.ema_alpha, self.cfg.prior_strength, self.cfg.new_question_prior)
        for qid, d in state["clinician_history"].items():
            sc.states[qid] = ClinicianHistoryState.from_dict(d)
        return sc

    def _qid(self, session_id: str, text: str, category: str) -> str:
        if self.options.deterministic_ids:
            return new_question_id(f"{session_id}|{category}|{text}".encode("utf-8"))
        return new_question_id()

    @staticmethod
    def _recent_sessions(state, n):
        return state["sessions"][-n:] if n > 0 else []

    def plan_slots(self, categories: Sequence[str], carried: Sequence[Question], first_session: bool) -> Dict[str, int]:
        """How many new questions to request per category (before candidates_per_slot)."""
        cfg = self.cfg
        cov = CoverageState(list(categories), cfg.max_questions, cfg.min_per_category, cfg.max_per_category,
                            dict(cfg.category_priority), cfg.required_categories or None)
        if first_session:
            return {c: max(1, int(round(cov.target(c)))) for c in categories}
        per_cat = {c: sum(1 for q in carried if q.category == c) for c in categories}
        budget = cfg.max_replacements_per_session
        slots = {c: max(0, cov.floor(c) - per_cat[c]) for c in categories}  # floors first
        # distribute the rest evenly so every category gets alternatives to compare against
        remaining = max(0, budget - sum(slots.values()))
        order = sorted(categories, key=lambda c: (per_cat[c] / max(cov.target(c), 1e-9), c))
        i = 0
        while remaining > 0 and order:
            slots[order[i % len(order)]] += 1
            remaining -= 1
            i += 1
        return slots

    # -- main entry ------------------------------------------------------------
    def run_session(self, ctx: SessionContext, candidates: Optional[List[CandidateQuestion]] = None) -> SelectionResult:
        cfg = self.cfg
        state = self.store.load(ctx.anon_subject_id)
        if any(s["session_id"] == ctx.session_id for s in state["sessions"]):
            raise StateIntegrityError(f"session {ctx.session_id} already recorded; refusing to overwrite")
        questions = {qid: Question.from_dict(q) for qid, q in state["questions"].items()}
        scorer = self._scorer(state)
        prev = state["sessions"][-1] if state["sessions"] else None
        first_session = prev is None

        unknown_ctl = (set(ctx.locked_question_ids) | set(ctx.retire_question_ids)) - set(questions)
        if unknown_ctl:
            raise ValueError(f"locked/retire ids not found: {sorted(unknown_ctl)}")

        carried_q: List[Question] = []
        if prev:
            for qid in prev["selected_question_ids"]:
                q = questions[qid]
                if qid in ctx.retire_question_ids or q.status == QuestionStatus.REJECTED.value:
                    continue
                if q.category not in ctx.categories:
                    continue
                carried_q.append(q)

        # recent history texts (novelty / repetition) and per-category last-asked
        recent = self._recent_sessions(state, cfg.history_sessions)
        history_ids = [qid for s in recent for qid in s["selected_question_ids"]]
        history_texts = list(dict.fromkeys(questions[q].text for q in history_ids))
        last_asked: Dict[str, str] = {}
        times: Dict[str, int] = {}
        for s in recent:
            for qid in s["selected_question_ids"]:
                c = questions[qid].category
                last_asked[c] = s["session_id"]
                times[c] = times.get(c, 0) + 1
        prior_notes = prev.get("notes_by_category", {}) if prev else {}
        priority = {**cfg.category_priority, **ctx.category_priority}
        lstates, signals = build_longitudinal_state(
            ctx.categories, prior_notes, ctx.notes_by_category, self.sim, cfg.change_similarity_threshold,
            ctx.unresolved_categories, last_asked, times, priority)

        # candidate generation (or fixture replay)
        gen_metrics = GenerationMetrics()
        if candidates is None:
            if self.generator is None:
                raise ValueError("no candidates supplied and no generator configured")
            slots = self.plan_slots(ctx.categories, carried_q, first_session)
            recent_by_cat: Dict[str, List[str]] = {}
            for qid in history_ids:
                recent_by_cat.setdefault(questions[qid].category, []).append(questions[qid].text)
            candidates, gen_metrics = self.generator.generate(
                ctx.session_id, ctx.categories, slots, lstates, recent_by_cat, cfg.candidates_per_slot)
        new_cands = [c for c in candidates if c.category in ctx.categories]

        # batch de-duplication of new candidates (keep first occurrence)
        from adaptive_questionnaires.v2.redundancy import dedupe_batch
        batch_dupes = 0
        if self.options.use_redundancy_filter and new_cands:
            kept, dropped = dedupe_batch([c.text for c in new_cands], self.sim, cfg.near_duplicate_threshold)
            batch_dupes = len(dropped)
            new_cands = [new_cands[i] for i in kept]

        carried_cands = [CandidateQuestion(candidate_id=f"carry_{q.question_id}", text=q.text, category=q.category,
                                           carried_question_id=q.question_id, intent_id=q.intent_id)
                         for q in carried_q]
        locked = [c for c in carried_cands if c.carried_question_id in ctx.locked_question_ids
                  or questions[c.carried_question_id].status == QuestionStatus.LOCKED.value]
        pool = [c for c in carried_cands if c not in locked] + new_cands

        coverage = CoverageState(list(ctx.categories), cfg.max_questions, cfg.min_per_category,
                                 cfg.max_per_category, priority, cfg.required_categories or None)
        rctx = RankingContext(self.sim, coverage, lstates, scorer, history_texts,
                              cfg.carryover_novelty, cfg.new_question_prior)
        outcome = select(pool, locked, rctx, self.ranker, cfg, history_texts,
                         use_redundancy_filter=self.options.use_redundancy_filter,
                         tie_break="hash" if self.options.random_control else "carryover_first")

        # materialise: new questions get NEW ids; no history is inherited
        now = utcnow_iso()
        selected_ids: List[str] = []
        new_questions: List[Question] = []
        carried_selected = {c.carried_question_id for c in outcome.selected if c.is_carryover}
        retired = [q.question_id for q in carried_q if q.question_id not in carried_selected]
        for c in outcome.selected:
            if c.is_carryover:
                selected_ids.append(c.carried_question_id)
                continue
            qid = self._qid(ctx.session_id, c.text, c.category)
            if qid in questions:
                raise StateIntegrityError(f"question id collision {qid}")
            parent = self._lineage_parent(c, [questions[r] for r in retired])
            q = Question(question_id=qid, text=c.text, category=c.category,
                         source=QuestionSource.LLM_CANDIDATE.value if self.generator else QuestionSource.FIXTURE.value,
                         created_at=now, session_id=ctx.session_id, parent_question_id=parent,
                         generation_model=getattr(getattr(self.generator, "backend", None), "model_name", None),
                         prompt_version=cfg.prompt_version, initial_weight=cfg.new_question_prior,
                         current_weight=None, intent_id=c.intent_id)
            questions[qid] = q
            scorer.init_question(qid)
            new_questions.append(q)
            selected_ids.append(qid)
            c.selection_metadata["question_id"] = qid
        for qid in retired:
            questions[qid].status = QuestionStatus.RETIRED.value
        for qid in ctx.locked_question_ids:
            questions[qid].status = QuestionStatus.LOCKED.value

        session_rows = []
        for order, c in enumerate(outcome.selected):
            qid = c.carried_question_id or c.selection_metadata["question_id"]
            session_rows.append(asdict(SessionQuestion(
                anon_subject_id=ctx.anon_subject_id, session_id=ctx.session_id, question_id=qid, selected=True,
                displayed_order=order,
                replacement_status=("locked" if c.selection_metadata.get("locked") else
                                    "retained" if c.is_carryover else "new"),
                selection_score=c.total_selection_score,
                selection_components={"relevance": c.relevance_score, "novelty": c.novelty_score,
                                      "coverage": c.coverage_score, "longitudinal": c.longitudinal_score,
                                      "clinician_history": c.clinician_prior_score,
                                      "redundancy": c.redundancy_score})))
        state["questions"] = {qid: q.to_dict() for qid, q in questions.items()}
        state["clinician_history"] = {qid: st.to_dict() for qid, st in scorer.states.items()}
        state["sessions"].append({
            "session_id": ctx.session_id, "created_at": now, "selected_question_ids": selected_ids,
            "retired_question_ids": retired, "notes_by_category": dict(ctx.notes_by_category),
            "session_questions": session_rows, "stop_reason": outcome.stop_reason,
            "warnings": outcome.warnings, "feedback_imported": False,
        })
        self.store.save(ctx.anon_subject_id, state)

        metrics = {
            "generation": asdict(gen_metrics) | {"invalid_generation_rate": gen_metrics.invalid_generation_rate},
            "batch_near_duplicates_dropped": batch_dupes, "filtered": outcome.filtered,
            "n_selected": len(outcome.selected), "n_new": len(new_questions), "n_retired": len(retired),
            "warnings": outcome.warnings,
        }
        return SelectionResult(ctx.session_id, ctx.anon_subject_id, outcome.selected, outcome.rejected, retired,
                               new_questions, outcome.stop_reason, signals, metrics)

    def _lineage_parent(self, cand: CandidateQuestion, retired: Sequence[Question]) -> Optional[str]:
        """Closest retired question in the same category. Lineage only; nothing is inherited."""
        same = [q for q in retired if q.category == cand.category]
        if not same:
            return None
        return max(same, key=lambda q: (self.sim.sim(cand.text, q.text), q.question_id)).question_id

    # -- clinician feedback ------------------------------------------------------
    def import_feedback(self, subject: str, session_id: str, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """Apply clinician ratings for a recorded session, keyed by question_id (never by row).

        The import is atomic: any unknown id, duplicate row, question not in the session, or
        out-of-range score aborts it with no changes. Blank cells mean "not rated".
        """
        state = self.store.load(subject)
        sess = next((s for s in state["sessions"] if s["session_id"] == session_id), None)
        if sess is None:
            raise StateIntegrityError(f"unknown session {session_id}")
        if sess.get("feedback_imported"):
            raise StateIntegrityError(f"feedback for {session_id} already imported")
        scorer = self._scorer(state)
        in_session = set(sess["selected_question_ids"])
        parsed: Dict[str, ClinicianFeedback] = {}
        errors = []
        for i, r in enumerate(rows):
            qid = (r.get("question_id") or "").strip()
            if qid not in in_session:
                errors.append(f"row {i}: question_id not in session")
                continue
            if qid in parsed:
                errors.append(f"row {i}: duplicate question_id")
                continue
            scores = {}
            for d in CLINICIAN_DIMENSIONS:
                v = r.get(d)
                if v is None or str(v).strip() == "":
                    continue
                try:
                    scores[d] = float(v)
                except ValueError:
                    errors.append(f"row {i}: non-numeric {d}")
            fbk = ClinicianFeedback(scores=scores, notes=r.get("clinician_notes") or None)
            try:
                fbk.validated_scores()
            except ValueError as e:
                errors.append(f"row {i}: {e}")
                continue
            parsed[qid] = fbk
        if errors:
            raise StateIntegrityError("feedback rejected, nothing applied: " + "; ".join(errors[:10]))
        summary = {"rated": 0, "partially_rated": 0, "unrated": 0}
        for qid in sess["selected_question_ids"]:
            fbk = parsed.get(qid)
            scorer.update(qid, session_id, fbk)
            if fbk is None or not fbk.validated_scores():
                summary["unrated"] += 1
            elif fbk.is_complete():
                summary["rated"] += 1
            else:
                summary["partially_rated"] += 1
            for row in sess["session_questions"]:
                if row["question_id"] == qid and fbk is not None:
                    row["clinician_scores"] = fbk.validated_scores()
                    row["clinician_notes"] = fbk.notes
        for qid, st in scorer.states.items():
            state["clinician_history"][qid] = st.to_dict()
            if qid in state["questions"]:
                state["questions"][qid]["current_weight"] = None if st.n_ratings == 0 else scorer.score(qid)
        sess["feedback_imported"] = True
        self.store.save(subject, state)
        return summary


# ------------------------------------------------------------------ I/O helpers
PROPOSAL_COLUMNS = ["question_id", "category", "question_text", "replacement_status",
                    "selection_score"] + CLINICIAN_DIMENSIONS + ["clinician_notes"]


def write_proposal_csv(result: SelectionResult, state: Mapping[str, Any], path: str) -> None:
    """Clinician-review sheet keyed by question_id (clinical content; keep local)."""
    sess = next(s for s in state["sessions"] if s["session_id"] == result.session_id)
    qs = state["questions"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROPOSAL_COLUMNS)
        w.writeheader()
        for row in sess["session_questions"]:
            q = qs[row["question_id"]]
            w.writerow({"question_id": q["question_id"], "category": q["category"], "question_text": q["text"],
                        "replacement_status": row["replacement_status"],
                        "selection_score": f"{row['selection_score']:.4f}" if row["selection_score"] is not None else "",
                        **{d: "" for d in CLINICIAN_DIMENSIONS}, "clinician_notes": ""})


def read_feedback_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

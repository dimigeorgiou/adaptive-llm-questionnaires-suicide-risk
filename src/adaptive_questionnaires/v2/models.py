"""Structured data model for Adaptive Questioning V2.

A question is an entity with a stable identity and provenance, not a string in a
spreadsheet row. Identity rules:

* ``question_id`` is assigned once, when the question is first created, and never
  reused. A replacement always gets a NEW id and records the replaced question in
  ``parent_question_id``. It never inherits the parent's clinician history.
* ``anon_subject_id`` is an opaque research identifier. These models must never
  hold names, contact details or other direct identifiers.

Every numeric field on candidates is a question-selection heuristic. None of them
is a suicide-risk estimate, a clinical score, or a measure of accuracy.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

CLINICIAN_DIMENSIONS: List[str] = [
    "Coherence",
    "Emotional Resonance",
    "Perceived Helpfulness",
    "Motivational Impact",
    "Engagement",
]
LIKERT_MIN, LIKERT_MAX = 1, 5


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_question_id(rng_bytes: Optional[bytes] = None) -> str:
    """Opaque, never-reused id. Pass ``rng_bytes`` for deterministic ids in experiments."""
    if rng_bytes is not None:
        return "q_" + hashlib.sha256(rng_bytes).hexdigest()[:16]
    return "q_" + uuid.uuid4().hex[:16]


def normalize_text(text: str) -> str:
    """Case/whitespace/punctuation-insensitive form used for exact-duplicate checks."""
    import unicodedata

    t = unicodedata.normalize("NFKC", str(text)).casefold()
    t = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in t)
    return " ".join(t.split())


class QuestionStatus(str, Enum):
    ACTIVE = "active"            # in the current questionnaire
    RETIRED = "retired"          # removed by the selector or a clinician
    LOCKED = "locked"            # clinician-pinned; never removed automatically
    REJECTED = "rejected"        # clinician rejected before use


class QuestionSource(str, Enum):
    LLM_CANDIDATE = "llm_candidate"
    LEGACY_IMPORT = "legacy_import"
    CLINICIAN = "clinician"
    FIXTURE = "fixture"
    VALIDATED_INSTRUMENT = "validated_instrument"  # reserved: adapter only, never generated


@dataclass
class Question:
    question_id: str
    text: str
    category: str
    source: str
    created_at: str
    session_id: str                       # session in which the question was created
    parent_question_id: Optional[str] = None
    generation_model: Optional[str] = None
    prompt_version: Optional[str] = None
    initial_weight: Optional[float] = None
    current_weight: Optional[float] = None   # clinician-history score (see v2.scoring); None = no ratings
    status: str = QuestionStatus.ACTIVE.value
    intent_id: Optional[str] = None       # synthetic-fixture ground truth only; never set in production

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Question":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ClinicianFeedback:
    """Raw clinician ratings for one question in one session. Missing = not rated (never 0)."""
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: Optional[str] = None
    retire_requested: bool = False
    lock_requested: bool = False

    def validated_scores(self) -> Dict[str, float]:
        """Scores that are present and within the Likert range; raise on out-of-range values."""
        out: Dict[str, float] = {}
        for dim, v in self.scores.items():
            if dim not in CLINICIAN_DIMENSIONS:
                raise ValueError(f"unknown clinician dimension {dim!r}")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            v = float(v)
            if not (LIKERT_MIN <= v <= LIKERT_MAX) or math.isinf(v):
                raise ValueError(f"score {v} for {dim!r} outside {LIKERT_MIN}-{LIKERT_MAX}")
            out[dim] = v
        return out

    def is_complete(self) -> bool:
        return set(self.validated_scores()) == set(CLINICIAN_DIMENSIONS)

    def composite(self) -> Optional[float]:
        """Mean over *rated* dimensions only; None when nothing is rated."""
        s = self.validated_scores()
        return sum(s.values()) / len(s) if s else None


@dataclass
class CandidateQuestion:
    candidate_id: str
    text: str
    category: str
    # --- selection heuristics, all in [0, 1]; NOT clinical scores ---
    relevance_score: float = 0.0
    novelty_score: float = 0.0
    redundancy_score: float = 0.0
    coverage_score: float = 0.0
    longitudinal_score: float = 0.0
    clinician_prior_score: float = 0.0
    total_selection_score: float = 0.0
    selection_metadata: Dict[str, Any] = field(default_factory=dict)
    carried_question_id: Optional[str] = None   # set when the candidate is an existing question
    intent_id: Optional[str] = None             # synthetic ground truth (fixtures only)

    @property
    def is_carryover(self) -> bool:
        return self.carried_question_id is not None


@dataclass
class SessionQuestion:
    anon_subject_id: str
    session_id: str
    question_id: str
    selected: bool
    displayed_order: Optional[int] = None
    clinician_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    clinician_notes: Optional[str] = None
    replacement_status: str = "new"   # new | retained | retired | locked
    selection_score: Optional[float] = None
    selection_components: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChangeConsistencySignal:
    """Conservative *review* flag: text about a domain differs across sessions.

    This may reflect genuine change, different context or phrasing, or measurement
    noise. It is NOT a judgement that the person is inconsistent and must never be
    presented as one. Its only use is to make a clarifying follow-up more relevant.
    """
    category: str
    changed_state_possible: bool
    similarity_prior_current: Optional[float]
    last_asked_session: Optional[str] = None
    rationale: str = ("Session text for this domain differs from the prior session; this may reflect "
                      "genuine change, context, or measurement noise. For clinician review only.")


@dataclass
class CategoryLongitudinalState:
    category: str
    prior_summary: Optional[str] = None
    current_summary: Optional[str] = None
    changed: bool = False
    unresolved: bool = False              # clinician-marked
    last_asked_session: Optional[str] = None
    times_asked_recent: int = 0
    clinician_priority: float = 1.0


@dataclass
class SelectionResult:
    session_id: str
    anon_subject_id: str
    selected: List[CandidateQuestion]
    rejected: List[CandidateQuestion]
    retired_question_ids: List[str]
    new_questions: List[Question]
    stop_reason: str
    change_signals: List[ChangeConsistencySignal]
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Output is a proposal. A clinician reviews it before it is used.
    requires_clinician_review: bool = True

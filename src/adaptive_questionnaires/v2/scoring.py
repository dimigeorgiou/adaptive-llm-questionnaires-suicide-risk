"""Clinician-history score for V2 (a question-selection feature, not a clinical score).

Representation: an exponential moving average of the normalised composite rating,
shrunk toward a neutral prior by the number of ratings received:

    x_t     = (mean of rated Likert dimensions − 1) / 4          ∈ [0, 1]
    ema_t   = (1 − α)·ema_{t−1} + α·x_t      (ema_0 = prior)
    score   = (n·ema + k·prior) / (n + k)                          ∈ [0, 1]

α = ``ema_alpha``, k = ``prior_strength``, prior = ``new_question_prior``.

It was chosen after comparing it with legacy additive, clipped additive, running
mean, per-session z-score and plain EMA (``experiments/simulate_weighting.py``,
results in outputs/evaluation/weighting_simulation.md). Properties that hold by
construction and are tested:

* bounded in [0, 1]; never NaN or inf (inputs are validated, NaN or inf is rejected)
* missing ratings leave the state unchanged (they are never treated as 0)
* a new or replaced question starts at the prior with n = 0 (explicit initialisation)
* state is keyed by ``question_id``, so reordering rows cannot move history
* shrinkage keeps a single rating from dominating an unproven question

The five raw dimensions are kept in ``history`` and are never overwritten. The
composite is derived from them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from adaptive_questionnaires.v2.models import LIKERT_MAX, LIKERT_MIN, ClinicianFeedback


def normalise_composite(composite: float) -> float:
    if composite is None or not math.isfinite(composite):
        raise ValueError(f"composite must be finite, got {composite!r}")
    x = (composite - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)
    return min(1.0, max(0.0, x))


@dataclass
class ClinicianHistoryState:
    question_id: str
    prior: float = 0.5
    ema: Optional[float] = None
    n_ratings: int = 0
    history: List[Dict] = field(default_factory=list)  # [{"session_id", "scores", "composite"}]

    def score(self, prior_strength: float) -> float:
        if self.n_ratings == 0 or self.ema is None:
            return self.prior
        n, k = float(self.n_ratings), float(prior_strength)
        return (n * self.ema + k * self.prior) / (n + k)

    def to_dict(self) -> Dict:
        return {"question_id": self.question_id, "prior": self.prior, "ema": self.ema,
                "n_ratings": self.n_ratings, "history": self.history}

    @classmethod
    def from_dict(cls, d: Dict) -> "ClinicianHistoryState":
        return cls(question_id=d["question_id"], prior=d.get("prior", 0.5), ema=d.get("ema"),
                   n_ratings=int(d.get("n_ratings", 0)), history=list(d.get("history", [])))


class ClinicianHistoryScorer:
    def __init__(self, ema_alpha: float = 0.5, prior_strength: float = 2.0, prior: float = 0.5):
        if not (0 < ema_alpha <= 1):
            raise ValueError("ema_alpha must be in (0, 1]")
        if prior_strength < 0 or not (0 <= prior <= 1):
            raise ValueError("prior_strength >= 0 and prior in [0, 1] required")
        self.alpha, self.k, self.prior = ema_alpha, prior_strength, prior
        self.states: Dict[str, ClinicianHistoryState] = {}

    def init_question(self, question_id: str) -> ClinicianHistoryState:
        """Explicit initialisation for new questions. Re-initialising an existing id is an error."""
        if question_id in self.states:
            raise ValueError(f"question {question_id} already has history; ids are never reused")
        st = ClinicianHistoryState(question_id=question_id, prior=self.prior)
        self.states[question_id] = st
        return st

    def update(self, question_id: str, session_id: str, feedback: Optional[ClinicianFeedback]) -> ClinicianHistoryState:
        st = self.states.get(question_id)
        if st is None:
            raise KeyError(f"unknown question_id {question_id}; call init_question first")
        if feedback is None:
            return st
        scores = feedback.validated_scores()   # raises on out-of-range / unknown dims
        if not scores:
            return st                           # missing rating: no update, no zero
        composite = sum(scores.values()) / len(scores)
        x = normalise_composite(composite)
        st.ema = x if st.ema is None else (1 - self.alpha) * st.ema + self.alpha * x
        st.n_ratings += 1
        st.history.append({"session_id": session_id, "scores": dict(scores), "composite": composite,
                           "complete": len(scores) == 5})
        return st

    def score(self, question_id: str) -> float:
        st = self.states.get(question_id)
        s = self.prior if st is None else st.score(self.k)
        if not math.isfinite(s):  # defensive; unreachable with validated inputs
            raise FloatingPointError(f"non-finite score for {question_id}")
        return s

    def n_ratings(self, question_id: str) -> int:
        st = self.states.get(question_id)
        return 0 if st is None else st.n_ratings

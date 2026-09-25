"""Candidate ranking: an expected-utility PROXY for question selection.

    selection_score(q) =  w_r·relevance(q) + w_n·novelty(q) + w_c·coverage_need(q)
                        + w_l·longitudinal_value(q) + w_h·clinician_history(q)
                        − w_d·redundancy(q)

Every component is a heuristic in [0, 1]. The score orders candidate questions for
clinician review. It is NOT a probability, a suicide-risk estimate, a clinical
score, or a measure of accuracy. The coefficients are experimental configuration
(``[adaptive_v2] w_*``) and are not validated.

``InformationGainEstimator`` exists as an interface only. True expected information
gain needs a calibrated probabilistic model of the latent construct and of the
response distribution for each item (as in IRT-based CAT). This repository has no
such model or data, so the estimator is **not implemented/calibrated in V2** and is
disabled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Set

from adaptive_questionnaires.v2.coverage import CoverageState
from adaptive_questionnaires.v2.longitudinal import longitudinal_value, relevance
from adaptive_questionnaires.v2.models import CandidateQuestion, CategoryLongitudinalState
from adaptive_questionnaires.v2.redundancy import Similarity
from adaptive_questionnaires.v2.scoring import ClinicianHistoryScorer

COMPONENTS = ("relevance", "novelty", "coverage", "longitudinal", "clinician_history", "redundancy")


class InformationGainEstimator:
    """Interface for a future, calibrated expected-information-gain estimator.

    Not implemented/calibrated in V2. It would require a validated measurement model
    (for example IRT item parameters estimated on appropriate data) and a
    distribution over possible responses. It must not be approximated by an
    LLM self-rating.
    """
    enabled = False

    def expected_information_gain(self, candidate: CandidateQuestion, state: object) -> float:
        raise NotImplementedError("Information gain is not implemented/calibrated in V2.")


@dataclass
class RankerWeights:
    relevance: float = 0.25
    novelty: float = 0.20
    coverage: float = 0.20
    longitudinal: float = 0.15
    clinician_history: float = 0.30
    redundancy: float = 0.40
    disabled: Set[str] = field(default_factory=set)   # ablations

    @classmethod
    def from_config(cls, cfg, disabled: Iterable[str] = ()) -> "RankerWeights":
        disabled = set(disabled)
        unknown = disabled - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown components to disable: {sorted(unknown)}")
        return cls(cfg.w_relevance, cfg.w_novelty, cfg.w_coverage, cfg.w_longitudinal,
                   cfg.w_clinician_history, cfg.w_redundancy, disabled)

    def get(self, name: str) -> float:
        return 0.0 if name in self.disabled else float(getattr(self, name))


@dataclass
class RankingContext:
    similarity: Similarity
    coverage: CoverageState
    states: Mapping[str, CategoryLongitudinalState]
    scorer: ClinicianHistoryScorer
    history_texts: List[str]                 # questions asked in recent previous sessions
    carryover_novelty: float = 0.5
    new_question_prior: float = 0.5


class ExpectedUtilityProxy:
    def __init__(self, weights: RankerWeights):
        self.weights = weights

    def components(self, cand: CandidateQuestion, ctx: RankingContext, selected_texts: List[str]) -> Dict[str, float]:
        st = ctx.states.get(cand.category)
        sim = ctx.similarity
        if cand.is_carryover:
            novelty = ctx.carryover_novelty
            hist = ctx.scorer.score(cand.carried_question_id)
        else:
            novelty = 1.0 - sim.max_sim(cand.text, ctx.history_texts) if ctx.history_texts else 1.0
            hist = ctx.new_question_prior
        comp = {
            "relevance": relevance(cand.text, st, sim),
            "novelty": novelty,
            "coverage": ctx.coverage.coverage_need(cand.category),
            "longitudinal": longitudinal_value(cand.text, st, sim),
            "clinician_history": hist,
            "redundancy": sim.max_sim(cand.text, selected_texts) if selected_texts else 0.0,
        }
        for k, v in comp.items():
            if not math.isfinite(v):
                raise FloatingPointError(f"non-finite {k} for {cand.candidate_id}")
            comp[k] = min(1.0, max(0.0, float(v)))
        return comp

    def total(self, comp: Mapping[str, float]) -> float:
        w = self.weights
        return (w.get("relevance") * comp["relevance"] + w.get("novelty") * comp["novelty"]
                + w.get("coverage") * comp["coverage"] + w.get("longitudinal") * comp["longitudinal"]
                + w.get("clinician_history") * comp["clinician_history"]
                - w.get("redundancy") * comp["redundancy"])

    def score(self, cand: CandidateQuestion, ctx: RankingContext, selected_texts: List[str]) -> float:
        comp = self.components(cand, ctx, selected_texts)
        cand.relevance_score = comp["relevance"]
        cand.novelty_score = comp["novelty"]
        cand.coverage_score = comp["coverage"]
        cand.longitudinal_score = comp["longitudinal"]
        cand.clinician_prior_score = comp["clinician_history"]
        cand.redundancy_score = comp["redundancy"]
        cand.total_selection_score = self.total(comp)
        return cand.total_selection_score

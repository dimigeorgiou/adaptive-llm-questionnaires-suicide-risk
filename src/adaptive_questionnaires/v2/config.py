"""V2 configuration (``[adaptive_v2]`` in config.ini).

Every default below is an ENGINEERING default chosen to make the system run and be
comparable with V1 (e.g. 20 questions, at most 10 replacements per session, which
mirrors V1's fixed 2-per-category quota). None of them is a validated clinical
parameter, and none should be read as clinically optimal.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class V2Config:
    # --- candidate pool / questionnaire size -------------------------------------
    candidates_per_slot: int = 3            # LLM candidates requested per open slot
    max_questions: int = 20                 # questionnaire size cap (V1: fixed 20)
    min_questions: int = 10                 # floor when adaptive_length is enabled
    min_per_category: int = 2               # clinician-controlled coverage floor
    max_per_category: int = 6
    max_replacements_per_session: int = 10  # V1 always replaces exactly 10 of 20
    replacement_margin: float = 0.05        # a new candidate must beat a carried item by this
    # --- selection-score coefficients (experimental heuristics) --------------------
    w_relevance: float = 0.25
    w_novelty: float = 0.20
    w_coverage: float = 0.20
    w_longitudinal: float = 0.15
    w_clinician_history: float = 0.30
    w_redundancy: float = 0.40
    # --- similarity thresholds (engineering, not clinical) -----------------------
    near_duplicate_threshold: float = 0.80  # >= : treated as a near-duplicate (hard filter)
    change_similarity_threshold: float = 0.60  # prior/current domain text below this -> change signal
    redundancy_backend: str = "auto"        # auto | lexical | sentence_transformers | openai
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    history_sessions: int = 3               # previous sessions used for novelty/repetition
    # --- clinician-history score (see v2.scoring and docs/V2_EVALUATION.md) ------
    ema_alpha: float = 0.5
    prior_strength: float = 2.0             # pseudo-ratings pulling a new question toward the prior
    new_question_prior: float = 0.5         # neutral prior on the [0,1] scale for unrated questions
    carryover_novelty: float = 0.5          # novelty assigned to carried-over questions (neutral)
    # --- experimental adaptive length (disabled by default) -----------------------
    adaptive_length: bool = False
    stop_min_utility: float = 0.0
    # --- generation --------------------------------------------------------------
    prompt_version: str = "v2-candidates-1"
    generation_model: Optional[str] = None
    max_generation_attempts: int = 3
    max_question_chars: int = 300
    # --- categories --------------------------------------------------------------
    categories: List[str] = field(default_factory=list)          # empty = take from session
    category_priority: Dict[str, float] = field(default_factory=dict)  # clinician-set, default 1.0
    required_categories: List[str] = field(default_factory=list)  # empty = all categories

    COEFFICIENTS = ("w_relevance", "w_novelty", "w_coverage", "w_longitudinal",
                    "w_clinician_history", "w_redundancy")

    def validate(self) -> "V2Config":
        errs = []
        for name in self.COEFFICIENTS + ("replacement_margin", "stop_min_utility", "ema_alpha",
                                         "prior_strength", "new_question_prior", "carryover_novelty"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                errs.append(f"{name} must be a finite number >= 0 (got {v!r})")
        for name in ("near_duplicate_threshold", "change_similarity_threshold", "new_question_prior",
                     "carryover_novelty"):
            v = getattr(self, name)
            if not (0.0 <= float(v) <= 1.0):
                errs.append(f"{name} must be in [0, 1] (got {v!r})")
        if not (0.0 < self.ema_alpha <= 1.0):
            errs.append("ema_alpha must be in (0, 1]")
        if self.min_questions > self.max_questions:
            errs.append("min_questions must be <= max_questions")
        if self.min_per_category > self.max_per_category:
            errs.append("min_per_category must be <= max_per_category")
        for name in ("candidates_per_slot", "max_questions", "max_generation_attempts", "history_sessions"):
            if int(getattr(self, name)) < 1:
                errs.append(f"{name} must be >= 1")
        if self.redundancy_backend not in ("auto", "lexical", "sentence_transformers", "openai"):
            errs.append(f"unknown redundancy_backend {self.redundancy_backend!r}")
        for c, p in self.category_priority.items():
            if not math.isfinite(float(p)) or float(p) < 0:
                errs.append(f"category_priority[{c!r}] must be >= 0")
        if errs:
            raise ValueError("invalid [adaptive_v2] config: " + "; ".join(errs))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kw) -> "V2Config":
        d = self.to_dict()
        d.update(kw)
        return V2Config(**d).validate()

    @classmethod
    def from_mapping(cls, m: Dict[str, str]) -> "V2Config":
        kwargs: Dict[str, Any] = {}
        types = {f.name: f.type for f in fields(cls)}
        for k, raw in m.items():
            if k not in types:
                raise ValueError(f"unknown [adaptive_v2] key {k!r}")
            t = str(types[k])
            if t in ("int",):
                kwargs[k] = int(raw)
            elif t in ("float",):
                kwargs[k] = float(raw)
            elif t in ("bool",):
                kwargs[k] = str(raw).strip().lower() in ("1", "true", "yes", "on")
            elif t.startswith("List"):
                kwargs[k] = [x.strip() for x in str(raw).split(",") if x.strip()]
            elif t.startswith("Dict"):
                pairs = [x.split(":", 1) for x in str(raw).split(",") if ":" in x]
                kwargs[k] = {a.strip(): float(b) for a, b in pairs}
            elif t.startswith("Optional"):
                kwargs[k] = raw or None
            else:
                kwargs[k] = raw
        return cls(**kwargs).validate()

    @classmethod
    def from_config(cls, config) -> "V2Config":
        """Read from a RawConfigParser-like object; missing section -> defaults."""
        try:
            has = config.has_section("adaptive_v2")
        except Exception:
            has = False
        if not has:
            return cls().validate()
        return cls.from_mapping(dict(config.items("adaptive_v2")))

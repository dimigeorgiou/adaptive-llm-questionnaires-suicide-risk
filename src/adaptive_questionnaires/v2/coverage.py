"""Domain coverage state for V2 selection.

The coverage floor (``min_per_category``), ceiling (``max_per_category``), required
categories and per-category priority are clinician-controlled configuration. The
selector can never drop a required category below its floor while eligible
questions exist for it. If too few exist, it reports the shortfall and does not
fabricate questions.

``coverage_need(c)`` is a heuristic in [0, 1]: the fraction of the category's
target still unfilled in the current selection. It is not a clinical quantity.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class CoverageState:
    categories: List[str]
    max_questions: int
    min_per_category: int
    max_per_category: int
    priority: Dict[str, float] = field(default_factory=dict)
    required: Optional[List[str]] = None       # None/empty = every category is required
    counts: Counter = field(default_factory=Counter)

    def __post_init__(self):
        if not self.categories:
            raise ValueError("coverage requires at least one category")
        req = self.required or list(self.categories)
        unknown = set(req) - set(self.categories)
        if unknown:
            raise ValueError(f"required categories not in category set: {sorted(unknown)}")
        self.required = req
        if len(self.required) * self.min_per_category > self.max_questions:
            raise ValueError("min_per_category × required categories exceeds max_questions")

    # -- targets ---------------------------------------------------------------
    def target(self, category: str) -> float:
        pr = {c: max(0.0, float(self.priority.get(c, 1.0))) for c in self.categories}
        total = sum(pr.values()) or 1.0
        raw = self.max_questions * pr[category] / total
        floor = self.min_per_category if category in self.required else 0
        return float(min(self.max_per_category, max(floor, raw)))

    def floor(self, category: str) -> int:
        return self.min_per_category if category in (self.required or []) else 0

    # -- state -----------------------------------------------------------------
    def add(self, category: str) -> None:
        self.counts[category] += 1

    def count(self, category: str) -> int:
        return self.counts.get(category, 0)

    def is_saturated(self, category: str) -> bool:
        return self.count(category) >= self.max_per_category

    def unmet_floor(self) -> Dict[str, int]:
        return {c: self.floor(c) - self.count(c) for c in self.categories if self.count(c) < self.floor(c)}

    def coverage_need(self, category: str) -> float:
        t = self.target(category)
        if t <= 0:
            return 0.0
        return max(0.0, t - self.count(category)) / t

    def summary(self, selected_categories: Iterable[str]) -> Dict[str, object]:
        c = Counter(selected_categories)
        return {
            "per_category": {k: c.get(k, 0) for k in self.categories},
            "categories_covered": sum(1 for k in self.categories if c.get(k, 0) > 0),
            "floors_met": all(c.get(k, 0) >= self.floor(k) for k in self.categories),
        }

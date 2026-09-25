"""Parsing of the legacy Greek markdown questionnaire (``task0`` output).

Upstream format::

    ### Κατηγορία 1: <name>
    1. <question>
    ...

``parse_questionnaire_markdown`` keeps the upstream regexes (same accepted
inputs, same output). ``QuestionnaireParseReport`` adds diagnostics so a caller
can refuse to write an empty or partial questionnaire instead of doing so silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PATTERN_CATEGORY = r"### Κατηγορία (\d+): (.+)"
PATTERN_QUESTION = r"(\d+)\.\s(.+)"


@dataclass
class QuestionnaireParseReport:
    items: List[Dict] = field(default_factory=list)
    numbered_lines_total: int = 0          # lines that look like "n. text"
    numbered_lines_without_category: int = 0
    categories: Dict[int, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.items) and self.numbered_lines_without_category == 0

    def problems(self, expected_categories: Optional[int] = None,
                 expected_per_category: Optional[int] = None) -> List[str]:
        out = []
        if not self.items:
            out.append("no questions parsed (heading format not recognised?)")
        if self.numbered_lines_without_category:
            out.append(f"{self.numbered_lines_without_category} numbered line(s) appear before any category heading")
        if expected_categories is not None and len(self.categories) != expected_categories:
            out.append(f"expected {expected_categories} categories, parsed {len(self.categories)}")
        if expected_per_category is not None:
            counts: Dict[int, int] = {}
            for it in self.items:
                counts[it["category"]] = counts.get(it["category"], 0) + 1
            bad = {c: n for c, n in counts.items() if n != expected_per_category}
            if bad:
                out.append(f"categories with != {expected_per_category} questions: {bad}")
        return out


def parse_questionnaire_markdown(raw_text: str) -> QuestionnaireParseReport:
    report = QuestionnaireParseReport()
    current_category = None
    current_category_name = None
    for line in str(raw_text).splitlines():
        cat_match = re.match(PATTERN_CATEGORY, line.strip())
        q_match = re.match(PATTERN_QUESTION, line.strip())
        if cat_match:
            current_category = int(cat_match.group(1))
            current_category_name = cat_match.group(2).strip()
            report.categories[current_category] = current_category_name
        elif q_match:
            report.numbered_lines_total += 1
            if current_category:
                report.items.append({
                    "category": current_category,
                    "category_name": current_category_name,
                    "question_text": q_match.group(2).strip(),
                    "q": int(q_match.group(1)),
                })
            else:
                report.numbered_lines_without_category += 1
    return report

"""Constrained greedy selection of the next questionnaire (proposal for clinician review).

Pool = carried-over questions (previous session, not retired) + validated new candidates.

Constraints, in priority order:
1. clinician-locked questions are always kept
2. no near-duplicates in the selection (``near_duplicate_threshold``), and a new
   candidate may not be a near-duplicate of a recently asked question
3. per-category ceiling; required categories reach their floor whenever eligible
   items exist (a shortfall is reported, never filled with invented items)
4. at most ``max_replacements_per_session`` new questions; a new candidate must beat
   the alternatives by ``replacement_margin`` (switching cost against churn)

Stopping (engineering conditions, never clinical ones):
* the size target is reached, or
* no eligible non-redundant items remain, or
* (experimental, ``adaptive_length=true``) the best remaining utility is below
  ``stop_min_utility`` and ``min_questions`` is met. This reports
  "adaptive follow-up generation complete". It says nothing about the person.
The clinician can always add, remove or stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from adaptive_questionnaires.v2.config import V2Config
from adaptive_questionnaires.v2.models import CandidateQuestion
from adaptive_questionnaires.v2.ranker import ExpectedUtilityProxy, RankingContext


@dataclass
class SelectionOutcome:
    selected: List[CandidateQuestion]
    rejected: List[CandidateQuestion]
    stop_reason: str
    warnings: List[str] = field(default_factory=list)
    filtered: Dict[str, int] = field(default_factory=dict)


def select(pool: Sequence[CandidateQuestion], locked: Sequence[CandidateQuestion], ctx: RankingContext,
           ranker: ExpectedUtilityProxy, cfg: V2Config, recent_history_texts: Sequence[str],
           use_redundancy_filter: bool = True, max_new: Optional[int] = None,
           tie_break: str = "carryover_first") -> SelectionOutcome:
    """``max_new`` caps new questions (default: ``max_replacements_per_session`` when any
    carried-over question exists, otherwise uncapped, e.g. for a first session)."""
    sim = ctx.similarity
    if max_new is None:
        has_carry = any(c.is_carryover for c in pool) or any(c.is_carryover for c in locked)
        max_new = cfg.max_replacements_per_session if has_carry else cfg.max_questions
    thr = cfg.near_duplicate_threshold
    filtered = {"near_duplicate_of_recent_question": 0, "near_duplicate_of_selected": 0,
                "category_ceiling": 0, "replacement_cap": 0}
    warnings: List[str] = []

    # new candidates that merely re-ask a recently asked question are not "new"
    eligible: List[CandidateQuestion] = []
    for c in pool:
        if (use_redundancy_filter and not c.is_carryover and recent_history_texts
                and sim.max_sim(c.text, recent_history_texts) >= thr):
            filtered["near_duplicate_of_recent_question"] += 1
            c.selection_metadata["rejected_reason"] = "near_duplicate_of_recent_question"
            continue
        eligible.append(c)

    selected: List[CandidateQuestion] = []
    for c in locked:
        c.selection_metadata["locked"] = True
        selected.append(c)
        ctx.coverage.add(c.category)
    target = max(cfg.max_questions, len(selected))
    replacements = sum(1 for c in selected if not c.is_carryover)
    remaining = [c for c in eligible if c.candidate_id not in {s.candidate_id for s in selected}]
    stop_reason = "size target reached"

    while len(selected) < target:
        sel_texts = [s.text for s in selected]
        unmet = ctx.coverage.unmet_floor()
        slots_left = target - len(selected)
        must_fill = sum(unmet.values()) >= slots_left and bool(unmet)

        best: Optional[Tuple[float, int, str, CandidateQuestion]] = None
        capped = False
        for c in remaining:
            if ctx.coverage.is_saturated(c.category):
                continue
            if must_fill and c.category not in unmet:
                continue
            if not c.is_carryover and replacements >= max_new:
                capped = True
                continue
            if use_redundancy_filter and sel_texts and sim.max_sim(c.text, sel_texts) >= thr:
                continue
            u = ranker.score(c, ctx, sel_texts)
            if not c.is_carryover:
                u -= cfg.replacement_margin
            c.selection_metadata["effective_utility"] = u
            if tie_break == "hash":   # evaluation control: order independent of carry-over status
                import hashlib
                key = (u, 0, hashlib.sha256(c.candidate_id.encode()).hexdigest())
            else:
                key = (u, 1 if c.is_carryover else 0, c.candidate_id)
            if best is None or key > best[:3]:
                best = (u, key[1], c.candidate_id, c)
        if best is None:
            stop_reason = ("replacement cap reached and no eligible carried-over questions remain" if capped
                           else "no eligible non-redundant candidates remaining")
            break
        u, _, _, chosen = best
        if cfg.adaptive_length and len(selected) >= cfg.min_questions and u < cfg.stop_min_utility \
                and not ctx.coverage.unmet_floor():
            stop_reason = "adaptive follow-up generation complete (utility below configured floor)"
            break
        chosen.selection_metadata["selected_rank"] = len(selected)
        selected.append(chosen)
        ctx.coverage.add(chosen.category)
        if not chosen.is_carryover:
            replacements += 1
        remaining.remove(chosen)

    # accounting for items that were never chosen
    sel_ids = {s.candidate_id for s in selected}
    sel_texts = [s.text for s in selected]
    rejected = []
    for c in eligible:
        if c.candidate_id in sel_ids:
            continue
        if ctx.coverage.is_saturated(c.category):
            reason = "category_ceiling"
        elif not c.is_carryover and replacements >= max_new:
            reason = "replacement_cap"
        elif use_redundancy_filter and sel_texts and sim.max_sim(c.text, sel_texts) >= thr:
            reason = "near_duplicate_of_selected"
        else:
            reason = "lower_utility"
        filtered[reason] = filtered.get(reason, 0) + 1
        c.selection_metadata["rejected_reason"] = reason
        rejected.append(c)
    rejected.extend(c for c in pool if c not in eligible)

    for cat, deficit in ctx.coverage.unmet_floor().items():
        warnings.append(f"coverage floor not met for category {cat!r}: {deficit} short "
                        f"(not enough eligible questions); clinician review needed")
    if len(selected) < cfg.max_questions and not cfg.adaptive_length:
        warnings.append(f"only {len(selected)} of {cfg.max_questions} questions selected: {stop_reason}")
    return SelectionOutcome(selected, rejected, stop_reason, warnings, filtered)

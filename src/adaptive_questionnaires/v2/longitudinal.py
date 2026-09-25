"""Structured longitudinal context for V2.

Earlier notes are not dumped into the prompt. Each category gets a compact state::

    category:
        prior_summary        (previous session's clinician-approved summary)
        current_summary      (this session's summary)
        changed              (ChangeConsistencySignal.changed_state_possible)
        unresolved           (clinician-marked)
        last_asked_session
        clinician_priority

A change signal means only that the text for a domain differs between sessions. It
is a conservative review flag that makes a clarifying follow-up more relevant. It
does NOT mean the person is inconsistent, and it is never used to infer a diagnosis
or risk. Summaries are expected to be written or approved by the clinician. This
module does not summarise raw notes with an LLM.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from adaptive_questionnaires.v2.models import CategoryLongitudinalState, ChangeConsistencySignal
from adaptive_questionnaires.v2.redundancy import Similarity


def _clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    return s or None


def build_longitudinal_state(
    categories: Sequence[str],
    prior_notes: Mapping[str, str],
    current_notes: Mapping[str, str],
    similarity: Similarity,
    change_threshold: float,
    unresolved: Iterable[str] = (),
    last_asked: Optional[Mapping[str, str]] = None,
    times_asked_recent: Optional[Mapping[str, int]] = None,
    priority: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, CategoryLongitudinalState], List[ChangeConsistencySignal]]:
    unresolved = set(unresolved)
    states: Dict[str, CategoryLongitudinalState] = {}
    signals: List[ChangeConsistencySignal] = []
    for c in categories:
        prior, current = _clean(prior_notes.get(c)), _clean(current_notes.get(c))
        sim = similarity.sim(prior, current) if (prior and current) else None
        changed = sim is not None and sim < change_threshold
        st = CategoryLongitudinalState(
            category=c, prior_summary=prior, current_summary=current, changed=changed,
            unresolved=c in unresolved, last_asked_session=(last_asked or {}).get(c),
            times_asked_recent=int((times_asked_recent or {}).get(c, 0)),
            clinician_priority=float((priority or {}).get(c, 1.0)),
        )
        states[c] = st
        if sim is not None:
            signals.append(ChangeConsistencySignal(category=c, changed_state_possible=changed,
                                                   similarity_prior_current=round(sim, 4),
                                                   last_asked_session=st.last_asked_session))
    return states, signals


def relevance(text: str, state: Optional[CategoryLongitudinalState], similarity: Similarity) -> float:
    """Similarity of a question to the current session's summary for its category (0 if none)."""
    if state is None or not state.current_summary:
        return 0.0
    return similarity.sim(text, state.current_summary)


def longitudinal_value(text: str, state: Optional[CategoryLongitudinalState], similarity: Similarity) -> float:
    """Favour questions that clarify a possible change or a clinician-marked unresolved domain.

    Stable, resolved domains contribute 0, so they do not attract more questions just
    because they were discussed before. Repetition is handled by novelty/redundancy.
    """
    if state is None or not (state.changed or state.unresolved) or not state.current_summary:
        return 0.0
    return similarity.sim(text, state.current_summary)

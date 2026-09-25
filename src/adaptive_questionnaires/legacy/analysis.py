"""Longitudinal analysis of legacy patient tabs, with explicit question provenance.

Upstream (audit issue B) aligned weights across meetings by row position::

    current_df["w"] = previous_df["w_new"].values

Replacements reuse the spreadsheet row, so a newly generated question inherited
the updated weight of the question it replaced. Upstream also keyed the registry
with salted ``hash()`` (issue H), which is not reproducible across processes.

Here:

* Each meeting block's own ``w`` column is authoritative. ``run_task1`` writes it
  explicitly (retained rows: previous ``w_new``; new rows: 0.5).
* Question identity is reconstructed explicitly. Slot ``(category, q)`` in meeting
  *k* is the same question as in meeting *k-1* only if the text is identical.
  Otherwise it is a NEW question whose ``parent_question_id`` is the previous occupant.
* IDs are deterministic and readable: ``{patient}-c{category}-q{slot}-m{first_meeting}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from adaptive_questionnaires.legacy.algorithm import BLOCK_COLS, COLS_PER_MEETING, LIKERT_COLS

METRIC_COLS = {  # timeseries column prefix -> block column
    "w": "w",
    "Coherence": "Coherence",
    "Emotional_Resonance": "Emotional Resonance",
    "Perceived_Helpfulness": "Perceived Helpfulness",
    "Motivational_Impact": "Motivational Impact",
    "Engagement": "Engagement",
}


@dataclass
class TrackedQuestion:
    question_id: str
    category: object
    category_name: str
    q_number: object
    question_text: str
    first_meeting: int
    parent_question_id: Optional[str] = None
    last_meeting: int = 0
    timeseries: Dict[int, Dict[str, object]] = field(default_factory=dict)


def split_meeting_blocks(patient_df: pd.DataFrame, cols_per_meeting: int = COLS_PER_MEETING) -> List[pd.DataFrame]:
    num_meetings = patient_df.shape[1] // cols_per_meeting
    blocks = []
    for m in range(num_meetings):
        block = patient_df.iloc[:, m * cols_per_meeting:(m + 1) * cols_per_meeting].copy()
        block.columns = BLOCK_COLS[:block.shape[1]] if list(block.columns) != BLOCK_COLS else block.columns
        block["meeting"] = m
        blocks.append(block)
    return blocks


def _slot(row) -> Tuple[str, str]:
    return (str(row["category"]), str(row["q"]))


def build_question_registry(patient_id: str, blocks: List[pd.DataFrame]):
    """Return (registry: question_id -> TrackedQuestion, changes: list of dicts)."""
    registry: Dict[str, TrackedQuestion] = {}
    occupant: Dict[Tuple[str, str], str] = {}  # slot -> question_id in previous meeting
    changes: List[dict] = []
    for m, block in enumerate(blocks):
        current: Dict[Tuple[str, str], str] = {}
        for _, row in block.iterrows():
            slot = _slot(row)
            text = row["question_text"]
            prev_id = occupant.get(slot)
            if prev_id is not None and registry[prev_id].question_text == text:
                qid = prev_id
            else:
                qid = f"{patient_id}-c{slot[0]}-q{slot[1]}-m{m}"
                if qid in registry:  # defensive: duplicate slot within a block
                    raise ValueError(f"duplicate slot {slot} in meeting {m} of {patient_id}")
                registry[qid] = TrackedQuestion(
                    question_id=qid, category=row["category"], category_name=row["category_name"],
                    q_number=row["q"], question_text=text, first_meeting=m, parent_question_id=prev_id,
                )
                if prev_id is not None:
                    changes.append({
                        "meeting_transition": f"{m - 1} -> {m}", "category": row["category"],
                        "q_number": row["q"], "old_question_id": prev_id, "new_question_id": qid,
                    })
            tq = registry[qid]
            tq.last_meeting = m
            tq.timeseries[m] = {k: row[col] for k, col in METRIC_COLS.items()}
            tq.timeseries[m]["w_new"] = row["w_new"]
            current[slot] = qid
        occupant = current
    return registry, changes


def registry_to_timeseries(patient_id: str, registry: Dict[str, TrackedQuestion], n_meetings: int) -> pd.DataFrame:
    rows = []
    for tq in registry.values():
        r = {
            "patient_id": patient_id, "question_id": tq.question_id, "parent_question_id": tq.parent_question_id,
            "first_meeting": tq.first_meeting, "category": tq.category, "category_name": tq.category_name,
            "q_number": tq.q_number, "question_text": tq.question_text,
        }
        for m in range(n_meetings):
            metrics = tq.timeseries.get(m)
            for k in METRIC_COLS:
                r[f"{k}_{m}"] = None if metrics is None else metrics[k]
        rows.append(r)
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if any(c.startswith(f"{k}_") for k in METRIC_COLS)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def registry_to_json(registry: Dict[str, TrackedQuestion]) -> Dict[str, dict]:
    out = {}
    for qid, tq in registry.items():
        out[qid] = {
            "question_id": qid, "parent_question_id": tq.parent_question_id, "category": tq.category,
            "category_name": tq.category_name, "q_number": tq.q_number, "question_text": tq.question_text,
            "first_meeting": tq.first_meeting, "last_meeting": tq.last_meeting,
            "timeseries": {str(m): v for m, v in tq.timeseries.items()},
        }
    return out


def assert_finite_weights(df: pd.DataFrame) -> None:
    w_cols = [c for c in df.columns if c.startswith("w_") and c[2:].isdigit()]
    vals = df[w_cols].to_numpy(dtype=float)
    if np.isinf(vals).any():
        raise ValueError("infinite weight in timeseries")

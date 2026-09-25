"""Legacy (V1) adaptation algorithm, extracted verbatim from ``pipeline.py``.

These functions reproduce the upstream behaviour of commit 29752f3
(Georgiou et al., NICE TEAS Europe 2026) and are the reference for
``adaptive_mode = legacy``. The arithmetic must not change: parity is checked
against a golden fixture generated from the unmodified upstream code
(``tests/fixtures/legacy_golden.json``).

    f_i  = mean of the five clinician Likert dimensions
    w'_i = w_i + alpha * (f_i - mu)          alpha = 0.2, mu = 4
    per category, the 2 lowest w'_i are replaced by new LLM questions (w = 0.5)
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

LIKERT_COLS: List[str] = [
    "Coherence",
    "Emotional Resonance",
    "Perceived Helpfulness",
    "Motivational Impact",
    "Engagement",
]
BLOCK_COLS: List[str] = ["meeting", "category", "category_name", "question_text", "q", "w", "w_new"] + LIKERT_COLS
COLS_PER_MEETING = len(BLOCK_COLS)  # 12

DEFAULT_ALPHA = 0.2
DEFAULT_MU = 4.0
INITIAL_WEIGHT = 0.5
REPLACEMENTS_PER_CATEGORY = 2


def calculate_composite_and_update_weights(df: pd.DataFrame, likert_cols: Iterable[str] = LIKERT_COLS,
                                           alpha: float = DEFAULT_ALPHA, mu: float = DEFAULT_MU) -> pd.DataFrame:
    """w_new = w + alpha * (mean(likert) - mu). Mirrors upstream ``_calculate_composite_and_update_weights``."""
    likert_cols = list(likert_cols)
    df["w"] = df["w"].apply(pd.to_numeric, errors="coerce")
    df[likert_cols] = df[likert_cols].apply(pd.to_numeric, errors="coerce")
    df["composite_score"] = df[likert_cols].mean(axis=1)
    df["w_new"] = df["w"] + alpha * (df["composite_score"] - mu)
    return df.drop("composite_score", axis=1)


def mark_lowest_for_replacement(df: pd.DataFrame, n_per_category: int = REPLACEMENTS_PER_CATEGORY) -> pd.Series:
    """Boolean mask of the ``n`` lowest-``w_new`` rows per category (``nsmallest`` tie-breaking)."""
    to_drop = pd.Series(False, index=df.index)
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        to_drop.loc[cat_df.nsmallest(n_per_category, "w_new").index] = True
    return to_drop


def replace_lowest_scoring_questions(
    df: pd.DataFrame,
    generate: Callable[[object, List[str]], Optional[str]],
    n_per_category: int = REPLACEMENTS_PER_CATEGORY,
    initial_weight: float = INITIAL_WEIGHT,
    likert_cols: Iterable[str] = LIKERT_COLS,
    on_count_mismatch: Optional[Callable[[object, int, int], None]] = None,
) -> pd.DataFrame:
    """Replace the lowest-weight questions in place (same row, same ``q`` slot).

    ``generate(category, existing_questions)`` returns the raw LLM text; it is split
    on newlines exactly as upstream. If the count differs from the number of
    dropped rows, the category is left unchanged (upstream behaviour).
    """
    likert_cols = list(likert_cols)
    # Values re-read from Sheets are strings; pandas>=3's strict string dtype rejects the
    # float resets below. advance_meeting() zeroes every score afterwards, so coercing
    # here does not change any written value.
    df[likert_cols] = df[likert_cols].apply(pd.to_numeric, errors="coerce")
    df["To_Drop"] = mark_lowest_for_replacement(df, n_per_category)
    for cat in df["category"].unique():
        drop_idx = df[(df["category"] == cat) & (df["To_Drop"])].index
        if drop_idx.empty:
            continue
        existing_questions = df[(df["category"] == cat) & (~df["To_Drop"])]["question_text"].tolist()
        raw = generate(cat, existing_questions)
        new_questions = split_llm_lines(raw)
        if len(new_questions) != len(drop_idx):
            if on_count_mismatch is not None:
                on_count_mismatch(cat, len(new_questions), len(drop_idx))
            continue
        for i, idx in enumerate(drop_idx):
            df.at[idx, "question_text"] = new_questions[i]
            df.at[idx, "w_new"] = initial_weight
            for col in likert_cols:
                df.at[idx, col] = 0.0
    return df.drop("To_Drop", axis=1)


def split_llm_lines(raw: Optional[str]) -> List[str]:
    """Upstream parsing of replacement output: one question per non-blank line."""
    return [q for q in (raw or "").split("\n") if q.strip()]


def advance_meeting(df: pd.DataFrame, current_meeting_idx: int,
                    likert_cols: Iterable[str] = LIKERT_COLS) -> pd.DataFrame:
    """Build the next meeting block: w <- w_new, reset w_new and scores (upstream Step 7)."""
    df["meeting"] = current_meeting_idx + 1
    df[list(likert_cols)] = 0.0
    df["w"] = df["w_new"]
    df["w_new"] = 0.0
    return df


def merge_categories(df: pd.DataFrame) -> str:
    """Render a block back to the Greek markdown questionnaire format."""
    merged_text = ""
    for cat in sorted(df["category"].unique()):
        cat_name = df[df["category"] == cat]["category_name"].iloc[0]
        merged_text += f"### Κατηγορία {cat}: {cat_name}\n"
        for _, row in df[df["category"] == cat].sort_values("q").iterrows():
            merged_text += f"{row['q']}. {row['question_text']}\n"
        merged_text += "\n"
    return merged_text.strip()


def get_last_populated_columns(df: pd.DataFrame, num_cols: int = COLS_PER_MEETING):
    """Return the last ``num_cols`` populated columns and their start index."""
    last_col_with_data = 0
    for col_idx in range(len(df.columns) - 1, -1, -1):
        if not df.iloc[:, col_idx].isnull().all():
            last_col_with_data = col_idx
            break
    start_col = max(0, last_col_with_data - num_cols + 1)
    return df.iloc[:, start_col:last_col_with_data + 1], start_col


def column_number_to_letter(col_num: int) -> str:
    """1 -> A, 27 -> AA."""
    result = ""
    while col_num > 0:
        col_num -= 1
        result = chr(col_num % 26 + ord("A")) + result
        col_num //= 26
    return result


def initial_block_rows(parsed: List[Dict]) -> pd.DataFrame:
    """Meeting-0 block for a parsed questionnaire (upstream task1_preparation)."""
    data = [{
        "meeting": 0,
        "category": item["category"],
        "category_name": item["category_name"],
        "question_text": item["question_text"],
        "q": item["q"],
        "w": INITIAL_WEIGHT,
        "w_new": 0.0,
        **{col: 0.0 for col in LIKERT_COLS},
    } for item in parsed]
    return pd.DataFrame(data).fillna("")


def validate_scored_block(df: pd.DataFrame, likert_cols: Iterable[str] = LIKERT_COLS,
                          scale=(1.0, 5.0), sentinel: float = 0.0) -> List[str]:
    """Integrity checks before the legacy weight update (audit issues E, K).

    Upstream treats the ``0.0`` "not scored" sentinel as a real score and crashes on
    blank cells. Returns human-readable problems (empty list = OK). Question text is
    never included, only category/slot identifiers.
    """
    likert_cols = list(likert_cols)
    problems: List[str] = []
    raw = df[likert_cols]
    num = raw.apply(pd.to_numeric, errors="coerce")
    blank = num.isna()
    lo, hi = scale
    for idx in df.index:
        slot = f"category={df.at[idx, 'category']} q={df.at[idx, 'q']}"
        if blank.loc[idx].any():
            dims = [c for c in likert_cols if blank.at[idx, c]]
            problems.append(f"{slot}: blank/non-numeric score(s) in {dims}")
            continue
        vals = num.loc[idx]
        unscored = vals == sentinel
        if unscored.all():
            problems.append(f"{slot}: not scored at all (would be averaged as 0)")
        elif unscored.any():
            dims = [c for c in likert_cols if unscored[c]]
            problems.append(f"{slot}: partially scored; unscored {dims} would be averaged as 0")
        bad = vals[(~unscored) & ((vals < lo) | (vals > hi))]
        if len(bad):
            problems.append(f"{slot}: score(s) outside {lo:g}-{hi:g}: {bad.to_dict()}")
    return problems


def output_tab_data_start_row(range_name: str) -> int:
    """Sheet row (1-based) of the first *data* row for an output-tab range.

    ``output`` or ``output!A1:...`` → header on row 1, data from row 2.
    ``output!A2:...`` → header on row 2, data from row 3 (the upstream hard-coded
    ``row_idx + 3`` assumption, audit issue F).
    """
    import re

    _, _, rng = str(range_name).partition("!")
    m = re.match(r"\$?[A-Za-z]+\$?(\d+)", rng) if rng else None
    header_row = int(m.group(1)) if m else 1
    return header_row + 1

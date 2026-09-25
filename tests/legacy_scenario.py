"""Synthetic end-to-end scenario for the legacy (V1) pipeline.

Runs task1_preparation → (scores) → task1 → (scores) → task1 → analysis against
an in-memory FakeSheets, for either the upstream code or the refactored fork.
All text is synthetic and non-clinical; no real patient content.
"""
from __future__ import annotations

import contextlib
import io
import os
import random
import sys
import tempfile

LIKERT = ["Coherence", "Emotional Resonance", "Perceived Helpfulness", "Motivational Impact", "Engagement"]
CATEGORY_NAMES = ["Routine", "Mood", "Connection", "Coping", "Goals"]
N_PATIENTS = 2


def initial_markdown(patient: int) -> str:
    lines = []
    for c, name in enumerate(CATEGORY_NAMES, start=1):
        lines.append(f"### Κατηγορία {c}: {name}")
        for q in range(1, 5):
            lines.append(f"{q}. Synthetic P{patient} C{c} initial question {q}?")
        lines.append("")
    return "\n".join(lines)


def build_sheets(FakeSheets):
    header = ["history", "task0", "meeting1_notes", "meeting1_task1", "meeting2_notes", "meeting2_task1"]
    output = [["(title row above header — matches output!A2 range)"], header]
    for p in range(N_PATIENTS):
        output.append([f"synthetic history {p}", initial_markdown(p), f"synthetic notes m1 p{p}", "",
                       f"synthetic notes m2 p{p}", ""])
    prompts = [["task", "prompt", "system_role", "user_role", "requirements", "examples"],
               ["task_0", "History: {history}", "sys0", "user0", "r0a\nr0b", "e0"],
               ["task_1", "Notes: {meeting_notes}\nKeep: {existing_questions}\nCategory: {category}",
                "sys1", "user1", "Return exactly the number of questions requested", "e1"]]
    return FakeSheets({"output": output, "prompts": prompts, "input": [["x"], ["1"]]})


def score_last_block(sheets, seed: int):
    """Clinician stand-in: fill the 5 Likert columns of the newest block with seeded integers."""
    rng = random.Random(seed)
    for p in range(N_PATIENTS):
        tab = f"patient{p + 3}"
        grid = sheets.tabs[tab]
        header = grid[0]
        # last block = last 12 columns that have a header
        width = max(i for i, h in enumerate(header) if h != "") + 1
        start = width - 12
        for r in range(1, len(grid)):
            if len(grid[r]) <= start or grid[r][start + 3] == "":
                continue
            for d in range(5):
                grid[r][start + 7 + d] = str(rng.choice([1, 2, 3, 3, 4, 4, 5]))


def llm_responder(call):
    v = call["variables"]
    n = 2  # V1 always drops 2 per category
    cat = v.get("category")
    notes = v.get("meeting_notes", "")
    return "\n".join(f"Synthetic replacement for C{cat} [{notes[-5:]}] #{i}?" for i in range(1, n + 1))


def run(controller_cls, FakeSheets, FakeLLM, fake_mk1, argv_patch=True):
    """Run the scenario. Returns dict with final tab grids and analysis CSV text."""
    sheets = build_sheets(FakeSheets)
    llm = FakeLLM(llm_responder)
    old_argv = sys.argv
    sys.argv = ["main.py", "-o", "task1"]
    try:
        ctrl = controller_cls(fake_mk1())
    finally:
        sys.argv = old_argv
    ctrl.google_sheets_api = sheets
    ctrl.openai_api = llm
    out = {}
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        cwd = os.getcwd()
        os.chdir(d)
        try:
            ctrl.run_task1_preparation()
            score_last_block(sheets, seed=11)
            ctrl.run_task1()
            score_last_block(sheets, seed=22)
            ctrl.run_task1()
            ctrl.run_task1_analyze_results()
            out["analysis"] = {}
            for p in range(N_PATIENTS):
                with open(os.path.join(d, "data", f"timeseries_patient{p + 3}.csv"), encoding="utf-8") as f:
                    out["analysis"][f"patient{p + 3}"] = f.read()
        finally:
            os.chdir(cwd)
    out["tabs"] = {name: sheets.grid(name) for name in sorted(sheets.tabs)}
    out["llm_calls"] = [{k: (str(v) if k == "variables" else v) for k, v in c.items()} for c in llm.calls]
    return out

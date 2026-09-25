"""Regression tests for reproduced legacy-pipeline defects (docs/BASELINE_AUDIT.md)."""
import contextlib
import io
import json
import os
import pathlib

import pandas as pd
import pytest

import fakes
import legacy_scenario
from adaptive_questionnaires.legacy import algorithm as legacy
from adaptive_questionnaires.legacy import analysis
from adaptive_questionnaires.legacy.parsing import parse_questionnaire_markdown
from adaptive_questionnaires.pipeline import Controller

GOLDEN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "legacy_golden.json").read_text(encoding="utf-8"))


def _ctrl(overrides=None, sheets=None, llm=None):
    c = Controller(fakes.fake_mk1(overrides), argv=["-o", "task1"])
    c.google_sheets_api = sheets
    c.openai_api = llm
    return c


# ---------------------------------------------------------------- B / H provenance
@pytest.fixture(scope="module")
def run():
    return legacy_scenario.run(Controller, fakes.FakeSheets, fakes.FakeLLM, fakes.fake_mk1)


def test_new_question_does_not_inherit_replaced_weight(run):
    ts = pd.read_csv(io.StringIO(run["analysis"]["patient3"]))
    new = ts[ts.first_meeting == 1]
    assert len(new) == 10                              # 2 per category × 5 replaced at meeting 1
    assert (new["w_1"] == 0.5).all()                   # the weight task1 actually wrote
    assert new["parent_question_id"].notna().all()
    # upstream (positional alignment) reported the replaced question's updated weight instead
    up = pd.read_csv(io.StringIO(GOLDEN["analysis"]["patient3"]))
    up_new = up[up.question_text.isin(new.question_text)]
    assert (up_new["w_1"] != 0.5).any()


def test_retained_question_keeps_history_and_id(run):
    ts = pd.read_csv(io.StringIO(run["analysis"]["patient3"]))
    kept = ts[(ts.first_meeting == 0) & ts["w_1"].notna()]
    assert len(kept) == 10
    assert kept["question_id"].str.endswith("-m0").all()
    assert kept["w_0"].notna().all()


def test_registry_ids_are_deterministic():
    a = legacy_scenario.run(Controller, fakes.FakeSheets, fakes.FakeLLM, fakes.fake_mk1)
    b = legacy_scenario.run(Controller, fakes.FakeSheets, fakes.FakeLLM, fakes.fake_mk1)
    ids = lambda r: pd.read_csv(io.StringIO(r["analysis"]["patient3"]))["question_id"].tolist()
    assert ids(a) == ids(b)


def test_reordered_rows_keep_identity():
    block0 = pd.DataFrame([[0, 1, "c", t, q, 0.5, 0.6] + [4] * 5 for q, t in [(1, "A?"), (2, "B?")]],
                          columns=legacy.BLOCK_COLS)
    block1 = block0.iloc[::-1].reset_index(drop=True).copy()   # same questions, rows swapped
    block1["meeting"], block1["w"] = 1, [0.6, 0.6]
    reg, changes = analysis.build_question_registry("p", [block0, block1])
    assert len(reg) == 2 and changes == []


# ---------------------------------------------------------------- E / K validation
def _scored_sheet(scores_by_row):
    grid = [legacy.BLOCK_COLS]
    for i, s in enumerate(scores_by_row):
        grid.append([0, 1, "c", f"Q{i}?", i + 1, 0.5, 0.0] + list(s))
    header = ["history", "task0", "meeting1_notes", "meeting1_task1"]
    return fakes.FakeSheets({"patient3": grid, "output": [["title"], header, ["h", "x", "n", ""]],
                             "prompts": [["task", "prompt", "system_role", "user_role", "requirements", "examples"],
                                         ["task_1", "p", "s", "u", "r", ""]]})


def _run_task1(sheets, overrides=None):
    llm = fakes.FakeLLM(lambda call: "N1?\nN2?")
    c = _ctrl(overrides, sheets, llm)
    with contextlib.redirect_stdout(io.StringIO()):
        c.run_task1()
    return c


@pytest.mark.parametrize("row", [[5, 5, 5, 5, 0], [5, 5, "", 5, 5], [5, 5, 5, 5, 9]])
def test_invalid_scores_block_weight_update(row):
    sheets = _scored_sheet([[4] * 5, row])
    before = [r[:] for r in sheets.tabs["patient3"]]
    c = _run_task1(sheets)
    assert c.run_issues and "weights NOT updated" in c.run_issues[0]["message"]
    assert "Q1?" not in c.run_issues[0]["message"]       # no question text in reports
    assert sheets.tabs["patient3"][:3] == before[:3]     # nothing written for this patient
    assert c.report() == 2


def test_non_strict_mode_reproduces_upstream_arithmetic():
    sheets = _scored_sheet([[5, 5, 5, 5, 0]])
    c = _run_task1(sheets, {("adaptive", "legacy_strict_validation"): "false"})
    w_new = float(sheets.grid("patient3")[1][6])
    assert c.run_issues == [] and w_new == pytest.approx(0.5 + 0.2 * (4.0 - 4.0))


def test_valid_scores_update_weights_with_legacy_rule():
    sheets = _scored_sheet([[5] * 5, [3] * 5])
    _run_task1(sheets)
    g = sheets.grid("patient3")
    assert float(g[1][6]) == pytest.approx(0.7) and float(g[2][6]) == pytest.approx(0.3)


def test_validate_scored_block_messages():
    df = pd.DataFrame([[0, 1, "c", "t", 1, 0.5, 0.0, 4, 4, 4, 4, 4],
                       [0, 1, "c", "t", 2, 0.5, 0.0, 0, 0, 0, 0, 0]], columns=legacy.BLOCK_COLS)
    probs = legacy.validate_scored_block(df)
    assert len(probs) == 1 and "not scored at all" in probs[0]


# ---------------------------------------------------------------- F, G, L, D
@pytest.mark.parametrize("rng,row", [("output", 2), ("output!A1:Z", 2), ("output!A2:AH", 3), ("output!$B$3:Z", 4)])
def test_output_data_start_row(rng, row):
    assert legacy.output_tab_data_start_row(rng) == row


def test_output_cell_written_to_the_right_patient_row_for_header_on_row1():
    # config.example.ini layout: header on row 1
    sheets = legacy_scenario.build_sheets(fakes.FakeSheets)
    sheets.tabs["output"] = sheets.tabs["output"][1:]           # drop the title row
    c = _ctrl({("google_sheets", "reporter_tab_output"): "output"}, sheets,
              fakes.FakeLLM(legacy_scenario.llm_responder))
    with contextlib.redirect_stdout(io.StringIO()):
        c.run_task1_preparation()
        legacy_scenario.score_last_block(sheets, seed=1)
        c.run_task1()
    out = sheets.grid("output")
    col = out[0].index("meeting1_task1")
    assert "P0" in out[1][col] and "P1" in out[2][col]         # each patient's own row


def test_preparation_refuses_unparseable_questionnaire():
    sheets = fakes.FakeSheets({"output": [["title"], ["history", "task0"],
                                          ["h", "**Κατηγορία 1: X**\n1. q?\n2. r?"]]})
    c = _ctrl(sheets=sheets)
    with contextlib.redirect_stdout(io.StringIO()):
        c.run_task1_preparation()
    assert "patient3" not in sheets.tabs and "tab NOT created" in c.run_issues[0]["message"]


def test_parse_report_accepts_upstream_format():
    rep = parse_questionnaire_markdown(legacy_scenario.initial_markdown(0))
    assert rep.ok and len(rep.items) == 20 and rep.problems(5, 4) == []


def test_logger_creates_missing_directory(tmp_path):
    from adaptive_questionnaires.core.mark_i import Logger
    vals = {"level": "INFO", "format": "%(message)s", "asctime": "%H", "name": "t-logger",
            "fn_path": str(tmp_path / "new" / "logs.log"), "official_name": "x"}

    class C:
        def get(self, s, k):
            return vals.get(k, "x")

    Logger(C())
    assert (tmp_path / "new" / "logs.log").exists()


def test_config_path_env_override(tmp_path, monkeypatch):
    from adaptive_questionnaires.core.mark_i import MkI
    ini = tmp_path / "c.ini"
    ini.write_text("[app]\nname = x\n")
    monkeypatch.setenv("ADAPTIVE_QUESTIONNAIRES_CONFIG", str(ini))
    monkeypatch.setattr(MkI, "instance", None)
    assert MkI.get_instance().config.get("app", "name") == "x"
    monkeypatch.setattr(MkI, "instance", None)

"""Legacy compatibility: the refactored V1 path reproduces upstream outputs.

The golden fixture was produced by running the UNMODIFIED upstream Controller
(commit 29752f3) on the synthetic scenario in tests/legacy_scenario.py
(see experiments/make_legacy_golden.py).
"""
import json
import pathlib

import pytest

import fakes
import legacy_scenario

GOLDEN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "legacy_golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fork_run():
    from adaptive_questionnaires.pipeline import Controller
    return legacy_scenario.run(Controller, fakes.FakeSheets, fakes.FakeLLM, fakes.fake_mk1)


@pytest.mark.parametrize("tab", ["output", "patient3", "patient4"])
def test_sheet_state_matches_upstream(fork_run, tab):
    assert fork_run["tabs"][tab] == GOLDEN["tabs"][tab]


def test_llm_prompts_match_upstream(fork_run):
    assert fork_run["llm_calls"] == GOLDEN["llm_calls"]

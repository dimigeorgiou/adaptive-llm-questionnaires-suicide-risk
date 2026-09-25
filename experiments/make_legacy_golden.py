#!/usr/bin/env python3
"""Generate tests/fixtures/legacy_golden.json by running the UPSTREAM pipeline.

    git worktree add /tmp/upstream-29752f3 29752f3
    python experiments/make_legacy_golden.py /tmp/upstream-29752f3/src

The upstream Controller runs task1_preparation → task1 → task1 → analysis on a
synthetic in-memory spreadsheet (tests/fakes.py, tests/legacy_scenario.py).
The resulting sheet state is the parity target for the refactored legacy path.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
upstream_src = sys.argv[1]
sys.path.insert(0, upstream_src)
sys.path.insert(0, str(ROOT / "tests"))

from adaptive_questionnaires.pipeline import Controller  # noqa: E402  (upstream code)
import fakes, legacy_scenario  # noqa: E402

result = legacy_scenario.run(Controller, fakes.FakeSheets, fakes.FakeLLM, fakes.fake_mk1)
result["_provenance"] = {"generated_from": "upstream commit 29752f3", "script": "experiments/make_legacy_golden.py"}
out = ROOT / "tests" / "fixtures" / "legacy_golden.json"
out.write_text(json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
print(f"wrote {out}")

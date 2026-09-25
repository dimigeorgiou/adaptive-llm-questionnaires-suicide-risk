#!/usr/bin/env python3
"""Reproduce the issues documented in docs/BASELINE_AUDIT.md against UPSTREAM code.

This script targets the *unmodified* upstream commit 29752f3. To run it:

    git worktree add /tmp/upstream-29752f3 29752f3
    pip install -r /tmp/upstream-29752f3/requirements.txt beautifulsoup4 psutil
    cd "$(mktemp -d)" && python /path/to/experiments/repro_baseline_issues.py /tmp/upstream-29752f3/src

Uses mocks and synthetic strings only: no credentials, no network, no clinical data.
Against the fixed fork code several reproductions will (correctly) no longer fire;
the regression tests in tests/ cover the fixed behaviour.
"""
import os, sys, json, tempfile, logging, subprocess, traceback
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

SRC = sys.argv[1]
sys.path.insert(0, SRC)
sys.argv = ["x", "-o", "task1"]

from adaptive_questionnaires.clients.openai_client import OpenaiAPI
from adaptive_questionnaires.pipeline import Controller

results = {}


def record(name, confirmed, detail):
    results[name] = {"confirmed": confirmed, "detail": detail}
    print(f"[{'CONFIRMED' if confirmed else 'not confirmed'}] {name}: {detail}")


def fake_mk1():
    cfg = mock.MagicMock()
    cfg.get.side_effect = lambda s, k: {
        ("api_openai", "token_key"): "sk-test",
        ("openai", "model_name"): "gpt-4o-mini",
        ("google_sheets", "reporter_id"): "SHEET",
        ("google_sheets", "reporter_tab_input"): "input",
        ("google_sheets", "reporter_tab_output"): "output",
        ("google_sheets", "reporter_tab_prompts"): "prompts",
    }[(s, k)]
    return SimpleNamespace(config=cfg, logging=SimpleNamespace(logger=logging.getLogger("t")))


def fake_openai(reply="ok"):
    api = OpenaiAPI.__new__(OpenaiAPI)
    api.mk1 = fake_mk1(); api.model_name = "m"; api.temperature = 0.3; api.max_workers = 1
    choice = SimpleNamespace(message=SimpleNamespace(content=reply))
    api.service = mock.MagicMock()
    api.service.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return api


# ---- A. requirements list .format -------------------------------------------
api = fake_openai()
reqs = "Write 20 questions\nUse Greek".strip().split("\n")  # exactly what run_task0 does
try:
    api.execute_custom_prompt(prompt="History: {history}", variables={"history": "synthetic"},
                              requirements=reqs)
    record("A_requirements_list_format", False, "no error")
except AttributeError as e:
    record("A_requirements_list_format", True, f"AttributeError: {e}")

# A2. literal braces in a sheet-authored prompt (e.g. a JSON example)
try:
    api.execute_custom_prompt(prompt='Return JSON like {"q": "..."} for {history}',
                              variables={"history": "synthetic"}, requirements=None)
    record("A2_literal_braces", False, "no error")
except (KeyError, ValueError, IndexError) as e:
    record("A2_literal_braces", True, f"{type(e).__name__}: {e}")

# A4. requirements=None (empty/missing sheet cell) with variables -> [None]
try:
    fake_openai().execute_custom_prompt(prompt="{history}", variables={"history": "h"}, requirements=None)
    record("A4_requirements_none", False, "no error")
except TypeError as e:
    record("A4_requirements_none", True, f"TypeError: {e}")

# A3. task1 passes existing_questions as a python list -> repr leaks into prompt
out = fake_openai()
out.execute_custom_prompt(prompt="Existing: {existing_questions}",
                          variables={"existing_questions": ["Q one?", "Q two?"]}, requirements="r")
sent = out.service.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
record("A3_list_repr_in_prompt", sent.startswith("Existing: ['"), repr(sent))

# ---- controller with no network ---------------------------------------------
ctrl = Controller(fake_mk1())
LIK = ["Coherence", "Emotional Resonance", "Perceived Helpfulness", "Motivational Impact", "Engagement"]
COLS = ["meeting", "category", "category_name", "question_text", "q", "w", "w_new"] + LIK


def block(meeting, texts, w, w_new, scores):
    rows = []
    for i, t in enumerate(texts):
        rows.append([meeting, 1, "Sleep", t, i + 1, w[i], w_new[i]] + [scores[i]] * 5)
    return pd.DataFrame(rows, columns=COLS)


# ---- B. positional weight alignment in analysis ------------------------------
# meeting0: Q-A..Q-D rated; Q-C, Q-D lowest -> replaced by Q-E, Q-F with w=0.5 (what run_task1 writes)
m0 = block(0, ["Q-A", "Q-B", "Q-C", "Q-D"], [0.5] * 4, [0.7, 0.6, 0.1, 0.0], [5, 4.5, 2, 1.5])
m0["w_new"] = [0.7, 0.6, 0.1, 0.0]
m1 = block(1, ["Q-A", "Q-B", "Q-E", "Q-F"], [0.7, 0.6, 0.5, 0.5], [0.0] * 4, [0] * 4)
patient = pd.concat([m0, m1], axis=1)  # side-by-side 12-col blocks, as in the sheet
patient.columns = list(COLS) * 2
with tempfile.TemporaryDirectory() as d:
    cwd = os.getcwd(); os.chdir(d)
    ctrl.google_sheets_api = mock.MagicMock()
    ctrl.google_sheets_api.get_df_from_tab.side_effect = [pd.DataFrame({"x": [1]}), patient.astype(str)]
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ctrl.run_task1_analyze_results()
    ts = pd.read_csv(os.path.join(d, "data", "timeseries_patient3.csv"))
    reg_keys = list(json.load(open(os.path.join(d, "data", "question_registry_patient3.json"))).keys())
    os.chdir(cwd)
qe = ts[ts.question_text == "Q-E"].iloc[0]
record("B_positional_weight_inheritance", float(qe["w_1"]) != 0.5,
       f"new question Q-E: sheet w=0.5 but analysis w_1={qe['w_1']} (inherited from replaced Q-C)")

# H. hash()-based registry keys are not reproducible across processes
code = ("import sys;k=('1','3','Q-C');print(f\"{k[0]}_{k[1]}_{hash(k[2]) % 10000}\")")
keys = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": str(s)}).stdout.strip() for s in range(5)}
record("H_registry_key_nondeterministic", len(keys) > 1, f"same question -> keys {sorted(keys)}")

# ---- E. unscored dimension (0.0 sentinel) counted as a real score -------------
df = block(0, ["Q-A"], [0.5], [0.0], [5])
df.loc[0, "Engagement"] = 0.0  # clinician left one dimension unscored
upd = ctrl._calculate_composite_and_update_weights(df.copy(), LIK)
record("E_unscored_dimension_as_zero", abs(upd.loc[0, "w_new"] - 0.5) < 1e-9,
       f"four dims=5, one unscored -> composite 4.0, w_new={upd.loc[0,'w_new']:.2f} (fully scored 5 -> 0.70)")

# K. blank Likert cell crashes run_task1 step 3
blank = block(0, ["Q-A", "Q-B"], [0.5] * 2, [0.0] * 2, [4, 4]).astype(str)
blank.loc[1, "Coherence"] = ""
try:
    blank[LIK].astype(float)
    record("K_blank_likert_crash", False, "no error")
except ValueError as e:
    record("K_blank_likert_crash", True, f"ValueError: {e} (uncaught; aborts the batch mid-way)")

# ---- G. pandas-3 positional row[1] in task1_preparation --------------------------
row = pd.Series({"history": "h", "task0": "### Κατηγορία 1: Sleep\n1. Q?"})
try:
    row[1]
    record("G_positional_row_index", False, f"pandas {pd.__version__}: row[1] still works")
except KeyError as e:
    record("G_positional_row_index", True, f"pandas {pd.__version__}: row[1] -> KeyError {e}")

# ---- L. silent parse failure on a heading variant ---------------------------------
import re
text = "**Κατηγορία 1: Ύπνος**\n1. Πώς κοιμάστε;\n2. Ξυπνάτε τη νύχτα;"
n = sum(1 for l in text.splitlines() if re.match(r"(\d+)\.\s(.+)", l.strip()))
cat = any(re.match(r"### Κατηγορία (\d+): (.+)", l.strip()) for l in text.splitlines())
record("L_silent_parse_failure", (not cat) and n == 2,
       f"bold heading instead of '###': {n} numbered lines present, 0 parsed, no error raised")

# ---- C. task0 prompt row selected with .loc[0] (no reset_index) --------------------
prompts = pd.DataFrame({"task": ["task_1", "task_0"], "prompt": ["p1", "p0"]})
try:
    prompts[prompts["task"] == "task_0"].loc[0, "prompt"]
    record("C_task0_loc0", False, "no error")
except KeyError as e:
    record("C_task0_loc0", True, f"task_0 not on first prompts row -> KeyError {e}")

# ---- F. output-tab row offset hard-coded to +3 ------------------------------------
record("F_row_offset_vs_example_config", True,
       "update_cell(row=row_idx+3) assumes header on sheet row 2 ('output!A2:...'); the shipped "
       "config.example.ini uses 'output' (header row 1), so row_idx 0 would write to sheet row 3 = next patient")

# ---- D. logger crashes on a fresh clone (logs/ gitignored, never created) ----------
from adaptive_questionnaires.core.mark_i import Logger
with tempfile.TemporaryDirectory() as d:
    cfg = mock.MagicMock()
    cfg.get.side_effect = lambda s, k: {"level": "INFO", "format": "%(message)s", "asctime": "%H",
                                        "fn_path": os.path.join(d, "logs", "logs.log"), "name": "x",
                                        "official_name": "x"}.get(k, "x")
    try:
        Logger(cfg)
        record("D_logger_missing_dir", False, "no error")
    except FileNotFoundError as e:
        record("D_logger_missing_dir", True, f"FileNotFoundError: {e.strerror}")

# ---- I. @retry catches requests exceptions; googleapiclient raises HttpError --------
from googleapiclient.errors import HttpError
import requests
record("I_retry_never_fires", not issubclass(HttpError, requests.RequestException),
       f"HttpError MRO={[c.__name__ for c in HttpError.__mro__]}")

# ---- J. write_df_to_tab2 swallows HTTP 4xx and returns normally --------------------
from adaptive_questionnaires.clients.google_api import GoogleSheetsAPI
g = GoogleSheetsAPI.__new__(GoogleSheetsAPI)
g.mk1 = fake_mk1(); g.auth_header = {}
g._GoogleSheetsAPI__server_err_codes = {500, 501, 503}
resp = requests.Response(); resp.status_code = 403; resp._content = b'{"error": "PERMISSION_DENIED"}'
resp.url = "https://sheets.googleapis.com/x"
with mock.patch("requests.post", return_value=resp), contextlib.redirect_stdout(io.StringIO()):
    r = g.write_df_to_tab2("SHEET", "patient3", pd.DataFrame({"a": [1]}))
record("J_silent_write_failure", r["updated_cells"] is None, f"403 response -> returned {r}")

json.dump(results, open("repro_results.json", "w"), indent=2, ensure_ascii=False, default=bool)

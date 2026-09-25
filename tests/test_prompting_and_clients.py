"""Regression tests for reproduced baseline defects in prompting and API clients."""
import logging
from types import SimpleNamespace
from unittest import mock

import httplib2
import pandas as pd
import pytest
import requests
from googleapiclient.errors import HttpError

import fakes
from adaptive_questionnaires import privacy
from adaptive_questionnaires.clients import google_api
from adaptive_questionnaires.clients.openai_client import OpenaiAPI
from adaptive_questionnaires.prompting import (
    fill_requirements, fill_template, normalize_requirements, select_prompt_row, unknown_placeholders,
)


# ---------------------------------------------------------------- prompting (A, A2, A4, C)
def test_fill_template_matches_str_format_for_valid_templates():
    tpl = "Notes: {meeting_notes}\nKeep: {existing_questions}\nCategory: {category} {{literal}}"
    v = {"meeting_notes": "n", "existing_questions": ["a?", "b?"], "category": 3}
    assert fill_template(tpl, v) == tpl.format(**v)


def test_fill_template_leaves_literal_braces_and_unknown_placeholders():
    tpl = 'Return JSON like {"q": "..."} for {history} and {unknown}'
    out = fill_template(tpl, {"history": "H"})
    assert out == 'Return JSON like {"q": "..."} for H and {unknown}'
    assert unknown_placeholders(tpl, {"history": "H"}) == ["unknown"]


@pytest.mark.parametrize("req,expected", [
    (None, []), (float("nan"), []), ("one\ntwo", ["one\ntwo"]), (["a {x}", None, "b"], ["a X", "b"]),
])
def test_requirements_normalisation(req, expected):
    assert fill_requirements(req, {"x": "X"}) == expected


def _api(content="ok", side_effect=None):
    service = mock.MagicMock()
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    service.chat.completions.create.side_effect = side_effect
    service.chat.completions.create.return_value = resp
    api = OpenaiAPI(fakes.fake_mk1(), service=service, backoff_base=0.0)
    api._sleep = lambda s: None
    return api, service


def test_task0_list_requirements_no_longer_crash():
    api, service = _api()
    reqs = "Write 20 questions\nUse Greek {history}".strip().split("\n")   # exactly what task0 passes
    assert api.execute_custom_prompt("History: {history}", {"history": "H"}, requirements=reqs) == "ok"
    system = service.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Write 20 questions\nUse Greek H" in system


def test_requirements_none_with_variables():
    api, _ = _api()
    assert api.execute_custom_prompt("{history}", {"history": "h"}, requirements=None) == "ok"


def test_select_prompt_row_not_first_row():
    df = pd.DataFrame({"task": ["task_1", "task_0"], "prompt": ["p1", "p0"]})
    assert select_prompt_row(df, "task_0")["prompt"] == "p0"
    with pytest.raises(ValueError):
        select_prompt_row(df, "task_9")
    with pytest.raises(ValueError):
        select_prompt_row(pd.concat([df, df]), "task_0")


# ---------------------------------------------------------------- OpenAI retries (Q)
def _openai_error(cls):
    import openai
    req = mock.MagicMock()
    if cls is openai.APIConnectionError:
        return cls(request=req)
    return cls("boom", response=mock.MagicMock(status_code=429, request=req), body=None)


def test_openai_transient_errors_are_retried_then_succeed():
    import openai
    api, service = _api()
    ok = service.chat.completions.create.return_value
    service.chat.completions.create.side_effect = [_openai_error(openai.RateLimitError),
                                                   _openai_error(openai.APIConnectionError), ok]
    assert api.execute_custom_prompt("x") == "ok"
    assert api.usage["retries"] == 2 and api.usage["calls"] == 1 and api.usage["prompt_tokens"] == 10


def test_openai_gives_up_after_max_retries():
    import openai
    api, service = _api()
    service.chat.completions.create.side_effect = _openai_error(openai.RateLimitError)
    with pytest.raises(openai.RateLimitError):
        api.execute_custom_prompt("x")
    assert service.chat.completions.create.call_count == api.max_retries


def test_openai_permanent_errors_are_not_retried():
    import openai
    api, service = _api()
    err = openai.AuthenticationError("bad key", response=mock.MagicMock(status_code=401), body=None)
    service.chat.completions.create.side_effect = err
    with pytest.raises(openai.AuthenticationError):
        api.execute_custom_prompt("x")
    assert service.chat.completions.create.call_count == 1


# ---------------------------------------------------------------- Google (I, J, S, scopes, logging)
def _http_error(status):
    return HttpError(httplib2.Response({"status": status}), b"{}")


def test_retry_transient_retries_429_and_5xx():
    calls = []

    @google_api.retry_transient(tries=3, sleep=lambda s: None)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(503 if len(calls) == 1 else 429)
        return "ok"

    assert f() == "ok" and len(calls) == 3


def test_retry_transient_does_not_retry_permanent_errors():
    calls = []

    @google_api.retry_transient(tries=5, sleep=lambda s: None)
    def f():
        calls.append(1)
        raise _http_error(403)

    with pytest.raises(HttpError):
        f()
    assert len(calls) == 1


def _sheets():
    g = google_api.GoogleSheetsAPI.__new__(google_api.GoogleSheetsAPI)
    g.mk1 = fakes.fake_mk1()
    g.auth_header = {}
    g._GoogleSheetsAPI__server_err_codes = {500, 501, 503}
    return g


def test_write_df_to_tab2_raises_on_4xx():
    resp = requests.Response()
    resp.status_code, resp._content, resp.url = 403, b'{"error": "PERMISSION_DENIED"}', "https://x"
    with mock.patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="HTTP 403"):
            _sheets().write_df_to_tab2("S", "patient3", pd.DataFrame({"a": [1]}))


def test_format_sheet_tab_refuses_missing_tab():
    g = _sheets()
    g.sheet_exists = lambda sid, name: False
    with pytest.raises(ValueError, match="not found"):
        g.format_sheet_tab("S", "patient99", pd.DataFrame({"a": [1]}))


def test_result_to_df_pads_rows_shorter_than_header():
    df = _sheets().result_to_df({"values": [["h", "t0", "n1", "t1"], ["a", "b", "c"], ["d", "e"]]},
                                has_index=False)
    assert df.shape == (2, 4) and df.iloc[1].tolist() == ["d", "e", "", ""]


def test_default_scopes_are_sheets_only_and_configurable():
    assert google_api.DEFAULT_SCOPES == ["https://www.googleapis.com/auth/spreadsheets"]
    assert google_api.configured_scopes(fakes.fake_mk1().config) == google_api.DEFAULT_SCOPES
    cfg = fakes.fake_mk1({("api_google", "scopes"): "a b,c"}).config
    assert google_api.configured_scopes(cfg) == ["a", "b", "c"]
    assert not any("gmail" in s or "drive" in s or "documents" in s for s in google_api.GoogleAPI.SCOPES)


def test_update_cell_does_not_log_clinical_text(caplog):
    privacy.configure(enabled=False)
    g = _sheets()
    g.service = mock.MagicMock()
    secret = "### Κατηγορία 1: synthetic\n1. very private synthetic question?"
    with caplog.at_level(logging.INFO, logger="adaptive_questionnaires.tests"):
        g.update_cell("S", "output", 3, 4, secret)
    assert "private" not in caplog.text and "<redacted len=" in caplog.text
    privacy.configure(enabled=None)


def test_redaction_opt_in():
    privacy.configure(enabled=True)
    try:
        assert privacy.redact("abc") == "abc"
    finally:
        privacy.configure(enabled=False)
    assert privacy.redact("abc").startswith("<redacted len=3")

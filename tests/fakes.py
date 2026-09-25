"""In-memory fakes for external services (no credentials, no network).

``FakeSheets`` emulates the subset of ``GoogleSheetsAPI`` used by the legacy
pipeline: a grid of strings per tab, header-row handling for ranges such as
``output!A2:Z``, trailing-empty-cell trimming, and A1-range writes. It is used
both to generate the legacy golden fixture from the *upstream* code and to
check the refactored legacy path against it.

``FakeLLM`` returns scripted replies and records every call.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import pandas as pd


def _col_to_num(letters: str) -> int:
    n = 0
    for c in letters.upper():
        n = n * 26 + (ord(c) - 64)
    return n


def _parse_a1(cell: str):
    m = re.match(r"([A-Za-z]+)(\d+)", cell)
    return int(m.group(2)), _col_to_num(m.group(1))


def _cell_str(v) -> str:
    """Approximate how Sheets returns a USER_ENTERED value on read."""
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return repr(v).rstrip("0").rstrip(".") if "." in repr(v) else repr(v)
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v)


class FakeSheets:
    def __init__(self, tabs: Optional[Dict[str, List[List]]] = None):
        self.tabs: Dict[str, List[List[str]]] = {}
        self.calls: List[tuple] = []
        for name, grid in (tabs or {}).items():
            self.tabs[name] = [[_cell_str(v) for v in row] for row in grid]

    # -- helpers -----------------------------------------------------------
    def _ensure(self, tab, rows, cols):
        grid = self.tabs.setdefault(tab, [])
        while len(grid) < rows:
            grid.append([])
        for r in grid:
            while len(r) < cols:
                r.append("")

    def _write(self, tab, row1, col1, values):
        rows = row1 - 1 + len(values)
        cols = col1 - 1 + max((len(v) for v in values), default=0)
        self._ensure(tab, rows, cols)
        for i, vals in enumerate(values):
            for j, v in enumerate(vals):
                self.tabs[tab][row1 - 1 + i][col1 - 1 + j] = _cell_str(v)

    def grid(self, tab) -> List[List[str]]:
        """Grid with trailing empty cells/rows trimmed (as the API returns)."""
        out = []
        for r in self.tabs.get(tab, []):
            r = list(r)
            while r and r[-1] == "":
                r.pop()
            out.append(r)
        while out and not out[-1]:
            out.pop()
        return out

    # -- API subset --------------------------------------------------------
    def get_df_from_tab(self, spreadsheet_id, spreadsheet_range_name, spreadsheet_has_index=True,
                        spreadsheet_has_headers=True, spreadsheet_empty_value=""):
        self.calls.append(("get", spreadsheet_range_name))
        tab, _, rng = spreadsheet_range_name.partition("!")
        if tab not in self.tabs:
            raise KeyError(f"Unable to parse range: {spreadsheet_range_name}")
        values = self.grid(tab)
        if rng:
            start_row, _ = _parse_a1(rng.split(":")[0])
            values = values[start_row - 1:]
        if not values:
            return pd.DataFrame()
        headers, data = values[0], values[1:]
        width = len(headers)
        # The real API trims trailing empty cells; pad so the fake behaves like a
        # well-formed read (upstream result_to_df fails if *all* rows are short).
        data = [(r + [""] * width)[:width] for r in data]
        df = pd.DataFrame(data, columns=headers)
        if spreadsheet_has_index:
            df = df.set_index(df.columns[0])
        return df

    def update_range(self, spreadsheet_id, tab_name, range_name, values):
        self.calls.append(("update_range", tab_name, range_name))
        start = range_name.split(":")[0]
        r, c = _parse_a1(start)
        self._write(tab_name, r, c, values)
        return True

    def update_cell(self, spreadsheet_id, tab_name, row, col, value):
        self.calls.append(("update_cell", tab_name, row, col))
        self._write(tab_name, row, col, [[value]])

    def sheet_exists(self, spreadsheet_id, sheet_name):
        return sheet_name in self.tabs

    def ensure_sheet_exists(self, spreadsheet_id, sheet_name):
        self.tabs.setdefault(sheet_name, [])

    def write_df_to_tab2(self, spreadsheet_id, tab_name, df):
        self.calls.append(("write_df_to_tab2", tab_name))
        values = [df.columns.tolist()] + df.astype(str).values.tolist()
        self._write(tab_name, 1, 1, values)
        return {"updated_cells": sum(len(v) for v in values)}

    def write_df_to_tab(self, df, spreadsheet_id, spreadsheet_range_name, **kw):
        self.calls.append(("write_df_to_tab", spreadsheet_range_name))
        tab, _, rng = spreadsheet_range_name.partition("!")
        r, c = _parse_a1(rng.split(":")[0]) if rng else (1, 1)
        values = [df.columns.tolist()] + df.values.tolist()
        self._write(tab, r, c, values)
        return {"updated_cells": 1}

    def format_sheet_tab(self, spreadsheet_id, tab_name, df, start_col_idx=0):
        self.calls.append(("format", tab_name, start_col_idx))
        return True


class FakeLLM:
    """Scripted stand-in for ``OpenaiAPI.execute_custom_prompt``."""

    def __init__(self, responder: Callable[[dict], Optional[str]]):
        self.responder = responder
        self.calls: List[dict] = []

    def execute_custom_prompt(self, prompt, variables=None, user_role=None, system_role=None,
                              requirements=None, examples=None, max_tokens=6000):
        call = dict(prompt=prompt, variables=variables or {}, user_role=user_role,
                    system_role=system_role, requirements=requirements, examples=examples)
        self.calls.append(call)
        return self.responder(call)


def fake_mk1(overrides: Optional[Dict[tuple, str]] = None):
    """Minimal MkI stand-in with a dict-backed config and a null logger."""
    import logging

    values = {
        ("google_sheets", "reporter_id"): "SYNTHETIC-SHEET",
        ("google_sheets", "reporter_tab_input"): "input",
        ("google_sheets", "reporter_tab_output"): "output!A2:ZZ",
        ("google_sheets", "reporter_tab_prompts"): "prompts",
        ("api_openai", "token_key"): "sk-test-not-a-key",
        ("openai", "model_name"): "gpt-4o-mini",
    }
    values.update(overrides or {})

    class _Cfg:
        def get(self, section, key, fallback=None, **kw):
            if (section, key) in values:
                return values[(section, key)]
            if fallback is not None:
                return fallback
            import configparser
            raise configparser.NoOptionError(key, section)

        def has_option(self, section, key):
            return (section, key) in values

        def getboolean(self, section, key, fallback=False):
            v = values.get((section, key))
            return fallback if v is None else str(v).lower() in ("1", "true", "yes", "on")

        def getfloat(self, section, key, fallback=None):
            v = values.get((section, key))
            return fallback if v is None else float(v)

        def getint(self, section, key, fallback=None):
            v = values.get((section, key))
            return fallback if v is None else int(v)

        def has_section(self, section):
            return any(s == section for s, _ in values)

        def items(self, section):
            return [(k, v) for (s, k), v in values.items() if s == section]

    log = logging.getLogger("adaptive_questionnaires.tests")
    return SimpleNamespace(config=_Cfg(), logging=SimpleNamespace(logger=log))

"""Safe filling of sheet-authored prompt templates.

Upstream filled templates with ``str.format``, which failed in three reproduced ways
(docs/BASELINE_AUDIT.md, issues A/A2/A4):

* ``requirements`` split into a list and then ``.format``-ed → AttributeError
* any literal ``{`` / ``}`` in a prompt (e.g. a JSON example) → KeyError
* ``requirements=None`` → ``[None]`` → TypeError in ``"\\n".join``

``fill_template`` replaces only ``{name}`` placeholders whose name is a supplied
variable. ``{{``/``}}`` escapes behave as in ``str.format``. Everything else is
left verbatim. For templates that ``str.format`` accepts, the output is identical
(``str(value)`` substitution), so legacy prompts are unchanged.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

_TOKEN = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fill_template(text: Optional[str], variables: Optional[Mapping[str, Any]]) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    variables = variables or {}

    def _sub(m: "re.Match[str]") -> str:
        tok = m.group(0)
        if tok == "{{":
            return "{"
        if tok == "}}":
            return "}"
        name = m.group(1)
        if name in variables:
            return str(variables[name])
        return tok  # unknown placeholder: leave untouched rather than crash

    return _TOKEN.sub(_sub, text)


def unknown_placeholders(text: Optional[str], variables: Optional[Mapping[str, Any]]) -> List[str]:
    """Placeholder names in ``text`` that have no value (useful for warnings)."""
    if not text:
        return []
    variables = variables or {}
    return sorted({m.group(1) for m in _TOKEN.finditer(str(text)) if m.group(1) and m.group(1) not in variables})


def normalize_requirements(requirements: Union[None, str, Sequence[Any]]) -> List[str]:
    """Accept None, a single (possibly multi-line) string, or a list; return list of str.

    NaN / None entries are dropped. A single string stays one entry, matching how
    upstream task1 passed it; lists (task0) keep one entry per line.
    """
    if requirements is None:
        return []
    if isinstance(requirements, float) and requirements != requirements:  # NaN from pandas
        return []
    if isinstance(requirements, str):
        return [requirements]
    out: List[str] = []
    for r in requirements:
        if r is None or (isinstance(r, float) and r != r):
            continue
        out.append(str(r))
    return out


def fill_requirements(requirements: Union[None, str, Sequence[Any]],
                      variables: Optional[Mapping[str, Any]]) -> List[str]:
    return [fill_template(r, variables) for r in normalize_requirements(requirements)]


def select_prompt_row(prompts_df, task: str):
    """Return the single prompts-tab row for ``task`` (upstream used ``.loc[0]``, issue C)."""
    rows = prompts_df[prompts_df["task"] == task].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"prompts tab has no row with task == {task!r}")
    if len(rows) > 1:
        raise ValueError(f"prompts tab has {len(rows)} rows with task == {task!r}; expected exactly one")
    return rows.loc[0]


def split_cell_lines(value: Any) -> List[str]:
    """Split a sheet cell into lines; empty/NaN → []."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    s = str(value).strip()
    return s.split("\n") if s else []


def iter_nonempty(xs: Iterable[str]) -> List[str]:
    return [x for x in xs if x and x.strip()]

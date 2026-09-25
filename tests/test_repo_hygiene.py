"""Safety boundaries, dependency declarations, credential-free CLI, and secrets hygiene."""
import ast
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "adaptive_questionnaires"


def _py_files():
    return sorted(SRC.rglob("*.py"))


# ---------------------------------------------------------------- clinical safety boundary
FORBIDDEN = re.compile(r"\b(risk_probability|suicide_risk|patient_is_safe|discharge_patient|"
                       r"hospitalization_required|patient_is_inconsistent|clinical_truth_score|"
                       r"patient_risk_accuracy)\b")


def test_no_autonomous_clinical_decision_identifiers():
    hits = []
    for p in _py_files():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            name = getattr(node, "id", None) or getattr(node, "attr", None) or getattr(node, "arg", None)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
            if name and FORBIDDEN.fullmatch(name):
                hits.append(f"{p.relative_to(ROOT)}:{getattr(node, 'lineno', '?')} {name}")
    assert hits == []


# ---------------------------------------------------------------- dependencies
IMPORT_TO_DIST = {"yaml": "PyYAML", "googleapiclient": "google-api-python-client", "google_auth_oauthlib":
                  "google-auth-oauthlib", "google": "google-auth", "oauth2client": "oauth2client",
                  "bs4": "beautifulsoup4", "sentence_transformers": "sentence-transformers"}


def _declared():
    import tomllib
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    name = lambda s: re.split(r"[<>=!~\[ ]", s, maxsplit=1)[0].lower()
    core = {name(d) for d in pp["dependencies"]}
    optional = {name(d) for ds in pp["optional-dependencies"].values() for d in ds}
    return core, optional


def test_requirements_txt_matches_pyproject():
    core, _ = _declared()
    req = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            req.add(re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower())
    assert req == core


def test_every_third_party_import_is_declared():
    core, optional = _declared()
    stdlib = set(sys.stdlib_module_names)
    missing = set()
    for p in _py_files():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                if top in stdlib or top == "adaptive_questionnaires" or top in ("M3Client",):
                    continue
                dist = IMPORT_TO_DIST.get(top, top).lower()
                if dist not in core and dist not in optional:
                    missing.add(f"{p.name}: {top}")
    assert missing == set()


def test_core_import_paths_do_not_load_optional_dependencies():
    """Pipeline and V2 must import without the optional extras (dropbox, dataset, bs4, ...)."""
    code = ("import sys; import adaptive_questionnaires.pipeline, adaptive_questionnaires.v2.engine, "
            "adaptive_questionnaires.v2.cli, adaptive_questionnaires.v2.redundancy; "
            "print(','.join(m for m in ('dropbox','dataset','bs4','sentence_transformers','psutil') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=120)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == ""


# ---------------------------------------------------------------- credential-free V2 CLI
def test_v2_cli_runs_without_credentials(tmp_path):
    env = {**os.environ, "ADAPTIVE_QUESTIONNAIRES_CONFIG": str(ROOT / "config" / "config.example.ini"),
           "OPENAI_API_KEY": "", "HF_HUB_OFFLINE": "1"}
    ini = (ROOT / "config" / "config.example.ini").read_text().replace("redundancy_backend         = auto",
                                                                       "redundancy_backend         = lexical")
    (tmp_path / "c.ini").write_text(ini)
    env["ADAPTIVE_QUESTIONNAIRES_CONFIG"] = str(tmp_path / "c.ini")
    ex = ROOT / "examples" / "v2"
    cmd = [sys.executable, str(ROOT / "main.py"), "-o", "v2_select", "--session", str(ex / "session_S001-s1.json"),
           "--candidates", str(ex / "candidates_S001-s1.json"), "--state-dir", str(tmp_path / "state")]
    out = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "20 questions" in out.stdout and "not a clinical decision" in out.stdout
    assert "How have your days" not in out.stdout            # no question text on the console
    assert (tmp_path / "state" / "S001" / "S001-s1_proposal.csv").exists()


# ---------------------------------------------------------------- secrets / data hygiene
SECRET_PATTERNS = [r"sk-[A-Za-z0-9]{32,}", r"AIza[0-9A-Za-z_\-]{35}", r"ya29\.[0-9A-Za-z_\-]+",
                   r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", r'"refresh_token"\s*:\s*"[^"]{20,}"']


def test_no_secrets_in_tracked_files():
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    hits = []
    for f in files:
        p = ROOT / f
        if not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            if re.search(pat, text):
                hits.append(f"{f}: {pat}")
    assert hits == []


def test_gitignore_covers_secrets_and_clinical_exports():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("config/config.ini", "config/secrets/", "data/", "logs/", "v2_state/", ".env"):
        assert pat in gi

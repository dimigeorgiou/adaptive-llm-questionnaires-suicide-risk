"""Redaction of clinical free text in logs and console output.

Upstream logged the full regenerated questionnaire at INFO (``update_cell``) and
printed question text to stdout (audit: "Clinical text in logs"). By default,
clinical text is now replaced by a length and a short, non-reversible digest.
Set ``[privacy] log_clinical_text = true`` (or env
``ADAPTIVE_QUESTIONNAIRES_LOG_CLINICAL_TEXT=1``) only for local debugging on an
authorised machine.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Optional

_ENV = "ADAPTIVE_QUESTIONNAIRES_LOG_CLINICAL_TEXT"
_override: Optional[bool] = None


def configure(mk1_config: Any = None, enabled: Optional[bool] = None) -> None:
    """Set the process-wide policy from config (``[privacy] log_clinical_text``) or explicitly."""
    global _override
    if enabled is not None:
        _override = bool(enabled)
        return
    if mk1_config is not None:
        try:
            _override = mk1_config.getboolean("privacy", "log_clinical_text", fallback=False)
        except Exception:
            _override = False


def clinical_text_logging_enabled() -> bool:
    if _override is not None:
        return _override
    return os.environ.get(_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def redact(value: Any, max_chars: int = 80) -> str:
    """Return ``value`` if clinical-text logging is enabled, else ``<redacted len=N sha=xxxxxxxx>``."""
    text = "" if value is None else str(value)
    if clinical_text_logging_enabled():
        return text if len(text) <= max_chars else text[:max_chars] + "…"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"<redacted len={len(text)} sha={digest}>"

"""Adapter interface for validated instruments (e.g. ASQ, C-SSRS). No content is shipped.

Validated screening instruments are administered according to the service's
approved protocol, by trained staff, with their exact validated wording and usage
terms. V2 never generates, rewords, re-orders or de-selects their items, and never
interprets their results. If a deployment has the approved instrument content and
the right to use it, an adapter can expose the items as a fixed, locked block that
sits outside the adaptive selection.

A positive or acute screen must follow the established clinical protocol (for the
ASQ, NIMH specifies a brief suicide safety assessment and, for an acute positive,
an immediate full mental-health evaluation). That escalation is a clinical workflow
outside this software. This repository neither implements it nor replaces it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InstrumentItem:
    instrument: str        # e.g. "ASQ"
    item_id: str           # instrument's own item identifier
    text: str              # exact approved wording (supplied by the deployment)
    language: str
    source_reference: str  # citation / licence reference supplied by the deployment


class ValidatedInstrumentAdapter:
    """Supplies a fixed block of approved items. Implementations must not alter wording."""
    name = "none"

    def items(self) -> List[InstrumentItem]:
        raise NotImplementedError


class NoInstrumentAdapter(ValidatedInstrumentAdapter):
    """Default: no validated instrument content is bundled with this repository."""
    name = "none"

    def items(self) -> List[InstrumentItem]:
        return []

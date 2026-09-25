"""Synthetic longitudinal world for comparing V1 and V2 (no real data, no API calls).

Assumptions, stated because the results depend on them:

* The simulated LLM draws background intents with Zipf popularity, so common
  questions recur, as LLMs tend to do. It picks a random surface form, so
  paraphrases recur too. It avoids exact repeats of the questions it was shown,
  and V1 and V2 get the same treatment here.
* When a category's note changes, each draw is a follow-up intent with
  probability ``follow_prob``. V1 and V2 see the same candidate stream: V1 takes
  the first items (as upstream uses the model's output directly), V2 ranks a
  larger pool.
* Simulated clinician ratings = latent per-(patient, intent) quality + noise. They
  are independent of every V2 feature (relevance, novelty, change status, ...),
  so they cannot reward V2's heuristics by construction.
* Malformed output: V1 replies occasionally carry a preamble line (upstream then
  rejects the category on a count mismatch). V2 JSON occasionally contains invalid
  items.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from adaptive_questionnaires.evaluation.synthetic_bank import BANK, CATEGORIES
from adaptive_questionnaires.v2.models import CLINICIAN_DIMENSIONS, normalize_text


def _rng(*key) -> random.Random:
    h = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


@dataclass
class WorldConfig:
    seed: int = 20260925
    n_patients: int = 30
    n_sessions: int = 6
    p_change_per_transition: float = 0.5
    follow_prob: float = 0.35
    zipf_s: float = 1.1
    rating_noise_sd: float = 0.8
    v1_preamble_rate: float = 0.08
    v2_malformed_item_rate: float = 0.05


@dataclass
class Patient:
    pid: str
    quality: Dict[str, float]                       # latent per-intent quality (1..5)
    notes: List[Dict[str, str]]                     # per session, per category
    changed: List[Set[str]]                         # categories whose change note is active per session
    onsets: List[Tuple[int, str]]                   # (session, category) change onsets


INTENT_TEXTS: Dict[str, List[str]] = {}
INTENT_CATEGORY: Dict[str, str] = {}
FOLLOWUP_EVENT: Dict[str, str] = {}                 # follow-up intent -> category of the change event
TEXT_TO_INTENT: Dict[str, str] = {}
for _c, spec in BANK.items():
    for _i, forms in spec["background"].items():
        INTENT_TEXTS[_i], INTENT_CATEGORY[_i] = forms, _c
    for _i, forms in spec["change"]["followups"].items():
        INTENT_TEXTS[_i], INTENT_CATEGORY[_i] = forms, _c
        FOLLOWUP_EVENT[_i] = _c
for _i, forms in INTENT_TEXTS.items():
    for _t in forms:
        TEXT_TO_INTENT[normalize_text(_t)] = _i


def intent_of(text: str) -> Optional[str]:
    return TEXT_TO_INTENT.get(normalize_text(text))


class World:
    def __init__(self, cfg: WorldConfig = WorldConfig()):
        self.cfg = cfg
        self.patients = [self._make_patient(i) for i in range(cfg.n_patients)]

    def _make_patient(self, i: int) -> Patient:
        cfg = self.cfg
        r = _rng(cfg.seed, "patient", i)
        quality = {k: r.uniform(1.5, 4.5) for k in INTENT_TEXTS}
        changed_now: Set[str] = set()
        notes, changed, onsets = [], [], []
        for s in range(cfg.n_sessions):
            if s > 0 and r.random() < cfg.p_change_per_transition:
                options = [c for c in CATEGORIES if c not in changed_now]
                if options:
                    c = r.choice(options)
                    changed_now.add(c)
                    onsets.append((s, c))
            notes.append({c: (BANK[c]["change"]["note"] if c in changed_now
                              else r.choice(BANK[c]["baseline_notes"])) for c in CATEGORIES})
            changed.append(set(changed_now))
        return Patient(f"P{i:03d}", quality, notes, changed, onsets)

    # ------------------------------------------------------------ simulated LLM
    def stream(self, p: Patient, session: int, category: str, n: int, avoid: Sequence[str] = (),
               call: int = 0) -> List[Tuple[str, str]]:
        """First ``n`` (intent, text) items the simulated LLM would produce for this call."""
        cfg = self.cfg
        r = _rng(cfg.seed, "stream", p.pid, session, category, call)
        bg = list(BANK[category]["background"])
        weights = [1.0 / (k + 1) ** cfg.zipf_s for k in range(len(bg))]
        fu = list(BANK[category]["change"]["followups"])
        avoid_n = {normalize_text(t) for t in avoid}
        out: List[Tuple[str, str]] = []
        guard = 0
        while len(out) < n and guard < 50 * max(n, 1):
            guard += 1
            if category in p.changed[session] and r.random() < cfg.follow_prob:
                intent = r.choice(fu)
            else:
                intent = r.choices(bg, weights)[0]
            text = r.choice(INTENT_TEXTS[intent])
            if normalize_text(text) in avoid_n or any(normalize_text(text) == normalize_text(t) for _, t in out):
                continue
            out.append((intent, text))
        return out

    def rate(self, p: Patient, session: int, text: str) -> Dict[str, float]:
        intent = intent_of(text)
        q = p.quality.get(intent, 2.5) if intent else 1.5      # unknown text (e.g. a preamble) rates low
        r = _rng(self.cfg.seed, "rate", p.pid, session, normalize_text(text))
        return {d: float(min(5, max(1, round(q + r.gauss(0, self.cfg.rating_noise_sd))))) for d in CLINICIAN_DIMENSIONS}

    def latent_quality(self, p: Patient, text: str) -> Optional[float]:
        i = intent_of(text)
        return p.quality.get(i) if i else None

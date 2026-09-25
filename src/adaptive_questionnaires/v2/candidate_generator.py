"""LLM candidate generation with structured (JSON-schema) output and strict parsing.

V1 asked the model for final questions and split the reply on newlines. V2 asks for
a larger candidate pool in a fixed JSON schema, validates every item
deterministically, and hands the valid items to the ranker. The model is not asked
to rate its own questions. Any self-rating fields it adds anyway are ignored and
counted (``ignored_fields``).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from adaptive_questionnaires.v2.models import CandidateQuestion, normalize_text

SCHEMA_NAME = "candidate_questions"


def candidate_schema(categories: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["category", "text"],
                    "properties": {
                        "category": {"type": "string", "enum": list(categories)},
                        "text": {"type": "string"},
                    },
                },
            }
        },
    }


SYSTEM_PROMPT = (
    "You assist a licensed clinician who prepares a supervised, clinician-reviewed follow-up "
    "questionnaire in a preventive psychosocial-care context. Propose CANDIDATE follow-up "
    "questions only; the clinician decides what is used.\n"
    "Rules:\n"
    "- Do not diagnose, estimate risk, give advice, or write crisis instructions.\n"
    "- Do not reproduce, reword, or replace items of validated screening instruments; "
    "those are administered separately under the service's protocol.\n"
    "- One question per candidate, open and respectful, in the language of the session notes.\n"
    "- Prefer questions that clarify what has CHANGED or is marked unresolved; avoid "
    "repeating or paraphrasing the listed recent questions.\n"
    "- Output JSON matching the schema; no commentary and no ratings."
)


def build_messages(slots_by_category: Mapping[str, int], states: Mapping[str, Any],
                   recent_questions: Mapping[str, Sequence[str]], candidates_per_slot: int) -> List[Dict[str, str]]:
    """Structured, category-scoped context. It contains summaries, not raw note dumps."""
    blocks = []
    for c, slots in slots_by_category.items():
        if slots <= 0:
            continue
        st = states.get(c)
        blocks.append({
            "category": c,
            "candidates_requested": int(slots * candidates_per_slot),
            "current_summary": getattr(st, "current_summary", None),
            "changed_since_last_session": bool(getattr(st, "changed", False)),
            "unresolved": bool(getattr(st, "unresolved", False)),
            "recent_questions_do_not_repeat": list(recent_questions.get(c, []))[:12],
        })
    user = ("Generate candidate follow-up questions per category as requested below.\n"
            + json.dumps({"categories": blocks}, ensure_ascii=False, indent=1))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


@dataclass
class ParseResult:
    candidates: List[CandidateQuestion] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    invalid_items: int = 0
    ignored_fields: int = 0
    response_valid: bool = True


def _cid(session_id: str, attempt: int, idx: int, text: str) -> str:
    h = hashlib.sha256(f"{session_id}|{attempt}|{idx}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"c_{h}"


def parse_candidates(raw: Optional[str], allowed_categories: Sequence[str], session_id: str,
                     attempt: int = 0, max_chars: int = 300, fixture_labels: bool = False) -> ParseResult:
    """Validate a model response. Never raises on malformed content; errors are reported.

    ``fixture_labels=True`` (synthetic evaluation only) reads the ground-truth
    ``_intent_id`` field; production parsing ignores it like any other extra field.
    """
    res = ParseResult()
    allowed = {c.casefold(): c for c in allowed_categories}
    if raw is None or not str(raw).strip():
        res.response_valid = False
        res.errors.append("empty response")
        return res
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        res.response_valid = False
        res.errors.append(f"invalid JSON: {e.__class__.__name__}")
        return res
    items = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(items, list):
        res.response_valid = False
        res.errors.append("missing 'candidates' array")
        return res
    seen_text, seen_ids = set(), set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            res.invalid_items += 1
            res.errors.append(f"item {i}: not an object")
            continue
        extra = set(it) - {"category", "text", "id"} - ({"_intent_id"} if fixture_labels else set())
        res.ignored_fields += len(extra)            # e.g. model self-ratings: never used
        if "id" in it:
            if it["id"] in seen_ids:
                res.invalid_items += 1
                res.errors.append(f"item {i}: duplicate id")
                continue
            seen_ids.add(it["id"])
        cat, text = it.get("category"), it.get("text")
        if not isinstance(cat, str) or not isinstance(text, str):
            res.invalid_items += 1
            res.errors.append(f"item {i}: missing/invalid category or text")
            continue
        cat_n = allowed.get(cat.strip().casefold())
        if cat_n is None:
            res.invalid_items += 1
            res.errors.append(f"item {i}: unknown category")
            continue
        text = " ".join(text.split())
        if not text:
            res.invalid_items += 1
            res.errors.append(f"item {i}: empty text")
            continue
        if len(text) > max_chars:
            res.invalid_items += 1
            res.errors.append(f"item {i}: text longer than {max_chars} chars")
            continue
        key = (cat_n, normalize_text(text))
        if key in seen_text:
            res.invalid_items += 1
            res.errors.append(f"item {i}: exact duplicate in batch")
            continue
        seen_text.add(key)
        res.candidates.append(CandidateQuestion(candidate_id=_cid(session_id, attempt, i, text),
                                                text=text, category=cat_n,
                                                intent_id=it.get("_intent_id") if fixture_labels else None))
    return res


# ---------------------------------------------------------------- backends
class GenerationBackend:
    model_name: str = "unknown"

    def complete_json(self, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError


class OpenAIJSONBackend(GenerationBackend):
    """Uses an ``OpenaiAPI`` instance (retries/backoff/usage accounting live there)."""

    def __init__(self, openai_api, max_tokens: int = 2000):
        self.api, self.max_tokens = openai_api, max_tokens
        self.model_name = openai_api.model_name

    def complete_json(self, messages, schema):
        resp = self.api._create_with_retry(
            model=self.api.model_name, messages=messages, temperature=self.api.temperature,
            max_tokens=self.max_tokens, n=1,
            response_format={"type": "json_schema",
                             "json_schema": {"name": SCHEMA_NAME, "schema": schema, "strict": True}},
        )
        return resp.choices[0].message.content


class FixtureBackend(GenerationBackend):
    """Replays pre-recorded/sanitised responses (no network). ``responses`` is a list or callable."""

    def __init__(self, responses, model_name: str = "fixture-replay"):
        self._responses = responses
        self.model_name = model_name
        self.calls: List[Dict[str, Any]] = []

    def complete_json(self, messages, schema):
        self.calls.append({"messages": messages, "schema": schema})
        if callable(self._responses):
            return self._responses(messages, schema, len(self.calls) - 1)
        i = len(self.calls) - 1
        return self._responses[min(i, len(self._responses) - 1)]


@dataclass
class GenerationMetrics:
    attempts: int = 0
    invalid_responses: int = 0
    invalid_items: int = 0
    valid_items: int = 0
    ignored_fields: int = 0
    latency_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def invalid_generation_rate(self) -> float:
        total = self.valid_items + self.invalid_items
        return (self.invalid_items / total) if total else (1.0 if self.invalid_responses else 0.0)


class CandidateGenerator:
    def __init__(self, backend: GenerationBackend, max_attempts: int = 3, max_chars: int = 300,
                 clock: Callable[[], float] = time.monotonic, fixture_labels: bool = False):
        self.backend, self.max_attempts, self.max_chars, self.clock = backend, max_attempts, max_chars, clock
        self.fixture_labels = fixture_labels

    def generate(self, session_id: str, categories: Sequence[str], slots_by_category: Mapping[str, int],
                 states: Mapping[str, Any], recent_questions: Mapping[str, Sequence[str]],
                 candidates_per_slot: int, min_valid: int = 1):
        """Return (candidates, metrics). Retries while fewer than ``min_valid`` valid items."""
        schema = candidate_schema(categories)
        messages = build_messages(slots_by_category, states, recent_questions, candidates_per_slot)
        metrics = GenerationMetrics()
        pool: List[CandidateQuestion] = []
        seen = set()
        for attempt in range(self.max_attempts):
            metrics.attempts += 1
            t0 = self.clock()
            raw = self.backend.complete_json(messages, schema)
            metrics.latency_s += self.clock() - t0
            res = parse_candidates(raw, categories, session_id, attempt, self.max_chars, self.fixture_labels)
            metrics.invalid_items += res.invalid_items
            metrics.ignored_fields += res.ignored_fields
            metrics.errors.extend(res.errors[:20])
            if not res.response_valid:
                metrics.invalid_responses += 1
            for c in res.candidates:
                key = (c.category, normalize_text(c.text))
                if key not in seen:
                    seen.add(key)
                    pool.append(c)
            metrics.valid_items = len(pool)
            if len(pool) >= min_valid:
                break
        return pool, metrics

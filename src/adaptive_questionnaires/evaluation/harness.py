"""Runners and metrics for the synthetic V1-vs-V2 comparison.

V1 is simulated with the parity-tested legacy functions (``legacy.algorithm``).
V2 is the real ``AdaptiveSessionEngine`` fed by a fixture backend that answers its
actual prompts from the synthetic world. Question-selection metrics only:
nothing here measures, or can measure, clinical accuracy.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from adaptive_questionnaires.evaluation.synthetic import (
    FOLLOWUP_EVENT, World, _rng, intent_of,
)
from adaptive_questionnaires.evaluation.synthetic_bank import CATEGORIES
from adaptive_questionnaires.legacy import algorithm as legacy
from adaptive_questionnaires.v2.candidate_generator import CandidateGenerator, FixtureBackend
from adaptive_questionnaires.v2.config import V2Config
from adaptive_questionnaires.v2.engine import AdaptiveSessionEngine, EngineOptions, JSONStateStore, SessionContext
from adaptive_questionnaires.v2.models import normalize_text
from adaptive_questionnaires.v2.redundancy import Similarity


@dataclass
class SessionRecord:
    method: str
    pid: str
    session: int
    texts: List[str]
    categories: List[str]
    new_texts: List[str]
    llm_calls: int
    invalid_outputs: int        # V1: category calls rejected; V2: invalid items
    generated_items: int
    retries: int
    selection_ms: float


# =================================================================== V1
def run_v1(world: World) -> List[SessionRecord]:
    cfg = world.cfg
    recs: List[SessionRecord] = []
    for p in world.patients:
        rows = []
        for c_idx, c in enumerate(CATEGORIES, start=1):
            for q, (_, t) in enumerate(world.stream(p, 0, c, 4), start=1):
                rows.append({"category": c_idx, "category_name": c, "question_text": t, "q": q})
        df = legacy.initial_block_rows(rows)
        recs.append(SessionRecord("V1", p.pid, 0, df.question_text.tolist(), df.category_name.tolist(),
                                  df.question_text.tolist(), 1, 0, 20, 0, 0.0))
        for s in range(1, cfg.n_sessions):
            prev = df.copy()
            for i, row in df.iterrows():
                for d, v in world.rate(p, s - 1, row["question_text"]).items():
                    df.at[i, d] = v
            t0 = time.perf_counter()
            df = legacy.calculate_composite_and_update_weights(df)
            calls = {"n": 0, "bad": 0, "items": 0}

            def generate(cat, existing, _s=s):
                calls["n"] += 1
                name = CATEGORIES[int(cat) - 1]
                items = world.stream(p, _s, name, 2, avoid=existing, call=calls["n"])
                lines = [t for _, t in items]
                r = _rng(cfg.seed, "preamble", p.pid, _s, name)
                if r.random() < cfg.v1_preamble_rate:
                    lines = ["Here are two new questions:"] + lines
                calls["items"] += len(lines)
                return "\n".join(lines)

            def mismatch(cat, got, expected):
                calls["bad"] += 1

            df = legacy.replace_lowest_scoring_questions(df, generate, on_count_mismatch=mismatch)
            df = legacy.advance_meeting(df, s - 1)
            ms = (time.perf_counter() - t0) * 1000
            prev_texts = set(prev.question_text)
            new = [t for t in df.question_text if t not in prev_texts]
            recs.append(SessionRecord("V1", p.pid, s, df.question_text.tolist(), df.category_name.tolist(), new,
                                      calls["n"], calls["bad"], calls["items"], 0, ms))
    return recs


# =================================================================== V2
def _fixture_responder(world: World, p, session: int, malformed_rate: float):
    def respond(messages, schema, call_idx):
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        items = []
        for block in payload["categories"]:
            c = block["category"]
            avoid = block.get("recent_questions_do_not_repeat", [])
            for k, (intent, text) in enumerate(world.stream(p, session, c, block["candidates_requested"],
                                                            avoid=avoid, call=100 + call_idx)):
                r = _rng(world.cfg.seed, "malformed", p.pid, session, c, k, call_idx)
                if r.random() < malformed_rate:
                    items.append(r.choice([{"category": "Unknown", "text": text}, {"category": c, "text": ""},
                                           {"category": c}]))
                else:
                    items.append({"category": c, "text": text, "_intent_id": intent})
        return json.dumps({"candidates": items})
    return respond


def run_v2(world: World, cfg: V2Config, similarity: Similarity, label: str,
           options: Optional[EngineOptions] = None, rate_fn: Optional[Callable] = None) -> List[SessionRecord]:
    recs: List[SessionRecord] = []
    opts = options or EngineOptions()
    opts.deterministic_ids = True
    with tempfile.TemporaryDirectory() as d:
        for p in world.patients:
            backend = FixtureBackend(None, model_name="synthetic-fixture")
            store = JSONStateStore(d)
            gen = CandidateGenerator(backend, cfg.max_generation_attempts, cfg.max_question_chars,
                                     fixture_labels=True)
            eng = AdaptiveSessionEngine(cfg, similarity, gen, store, options=opts)
            prev_texts: set = set()
            for s in range(world.cfg.n_sessions):
                backend._responses = _fixture_responder(world, p, s, world.cfg.v2_malformed_item_rate)
                n_calls_before = len(backend.calls)
                ctx = SessionContext(anon_subject_id=p.pid, session_id=f"{p.pid}-s{s}", categories=CATEGORIES,
                                     notes_by_category=p.notes[s])
                t0 = time.perf_counter()
                res = eng.run_session(ctx)
                ms = (time.perf_counter() - t0) * 1000
                texts = [c.text for c in res.selected]
                gm = res.metrics["generation"]
                recs.append(SessionRecord(label, p.pid, s, texts, [c.category for c in res.selected],
                                          [t for t in texts if t not in prev_texts],
                                          len(backend.calls) - n_calls_before, gm["invalid_items"],
                                          gm["valid_items"] + gm["invalid_items"], gm["attempts"] - 1, ms))
                prev_texts = set(texts)
                st = store.load(p.pid)
                rows = [{"question_id": q["question_id"], **(rate_fn or world.rate)(p, s, q["text"])}
                        for qid in st["sessions"][-1]["selected_question_ids"] for q in [st["questions"][qid]]]
                eng.import_feedback(p.pid, ctx.session_id, rows)
    return recs


# =================================================================== metrics
def session_metrics(world: World, rec: SessionRecord, history: Dict[str, set], eval_sim: Optional[Similarity],
                    near_thr: float) -> Dict[str, object]:
    p = next(x for x in world.patients if x.pid == rec.pid)
    intents = [intent_of(t) for t in rec.texts]
    n = len(rec.texts)
    ic = Counter(i for i in intents if i)
    norm = Counter(normalize_text(t) for t in rec.texts)
    cat_counts = Counter(rec.categories)
    probs = [v / n for v in cat_counts.values()] if n else []
    entropy = -sum(q * math.log(q) for q in probs) / math.log(len(CATEGORIES)) if n else 0.0
    entropy = min(1.0, max(0.0, entropy))  # float rounding
    seen = history.setdefault(rec.pid, set())
    new_intents = [intent_of(t) for t in rec.new_texts]
    repeated = sum(1 for i in new_intents if i and i in seen)
    onsets = [c for (s, c) in p.onsets if s == rec.session]
    followed = sum(1 for c in onsets if any(FOLLOWUP_EVENT.get(i) == c for i in intents if i))
    qual = [world.latent_quality(p, t) for t in rec.texts]
    qual = [q for q in qual if q is not None]
    m = {
        "method": rec.method, "pid": rec.pid, "session": rec.session, "n_questions": n,
        "exact_duplicate_rate": (n - len(norm)) / n if n else 0.0,
        "intent_duplicate_rate": sum(v for v in ic.values() if v > 1) / n if n else 0.0,
        "categories_covered": sum(1 for c in CATEGORIES if cat_counts.get(c, 0) > 0) / len(CATEGORIES),
        "min_per_category": min(cat_counts.get(c, 0) for c in CATEGORIES),
        "category_entropy": entropy,
        "replacement_churn": (len(rec.new_texts) / n) if (n and rec.session > 0) else None,
        "longitudinal_repetition": (repeated / len(rec.new_texts)) if (rec.new_texts and rec.session > 0) else None,
        "change_onsets": len(onsets),
        "change_followup_rate": (followed / len(onsets)) if onsets else None,
        "mean_words": sum(len(t.split()) for t in rec.texts) / n if n else 0.0,
        "non_question_items": sum(1 for i in intents if i is None) / n if n else 0.0,
        "invalid_output_rate": (rec.invalid_outputs / max(rec.llm_calls if rec.method == "V1" else rec.generated_items, 1)),
        "llm_calls": rec.llm_calls, "retries": rec.retries, "generated_items": rec.generated_items,
        "selection_ms": rec.selection_ms,
        "synthetic_rating_quality": sum(qual) / len(qual) if qual else None,
    }
    if eval_sim is not None and n > 1:
        sims = [eval_sim.sim(a, b) for a, b in combinations(rec.texts, 2)]
        m["semantic_near_dup_pair_rate"] = sum(1 for x in sims if x >= near_thr) / len(sims)
        m["semantic_diversity"] = 1 - sum(sims) / len(sims)
        cur = p.notes[rec.session]
        rel = [eval_sim.sim(t, cur[c]) for t, c in zip(rec.texts, rec.categories) if t in set(rec.new_texts)]
        m["relevance_proxy_new"] = sum(rel) / len(rel) if rel else None
    seen.update(i for i in intents if i)
    return m


def compute_metrics(world: World, records: Sequence[SessionRecord], eval_sim: Optional[Similarity],
                    near_thr: float = 0.8) -> pd.DataFrame:
    rows = []
    by_method: Dict[str, List[SessionRecord]] = {}
    for r in records:
        by_method.setdefault(r.method, []).append(r)
    for method, recs in by_method.items():
        history: Dict[str, set] = {}
        for r in sorted(recs, key=lambda x: (x.pid, x.session)):
            rows.append(session_metrics(world, r, history, eval_sim, near_thr))
    return pd.DataFrame(rows)


def paired_bootstrap(df: pd.DataFrame, metric: str, a: str, b: str, n_boot: int = 2000, seed: int = 0,
                     sessions_from: int = 0):
    """Mean difference b − a over patients (patient-level means), with a 95% bootstrap CI."""
    d = df[(df.session >= sessions_from) & df[metric].notna()]
    pa = d[d.method == a].groupby("pid")[metric].mean()
    pb = d[d.method == b].groupby("pid")[metric].mean()
    common = pa.index.intersection(pb.index)
    if len(common) == 0:
        return None
    diff = (pb[common] - pa[common]).to_numpy(dtype=float)
    r = _rng(seed, "boot", metric, a, b)
    boots = []
    for _ in range(n_boot):
        sample = [diff[r.randrange(len(diff))] for _ in range(len(diff))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return {"a_mean": float(pa[common].mean()), "b_mean": float(pb[common].mean()),
            "diff": float(diff.mean()), "ci_low": boots[int(0.025 * n_boot)], "ci_high": boots[int(0.975 * n_boot) - 1],
            "n_patients": int(len(common))}

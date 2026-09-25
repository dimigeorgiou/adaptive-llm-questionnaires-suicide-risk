"""CLI operations for V2 (``python main.py -o v2_select|v2_import_feedback``).

They never touch the legacy Google Sheets layout. OpenAI is used only when no
``--candidates`` fixture is given. Console output contains ids and counts, never
question text (the proposal CSV holds the text and is written locally).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from adaptive_questionnaires.v2.candidate_generator import CandidateGenerator, FixtureBackend, OpenAIJSONBackend
from adaptive_questionnaires.v2.config import V2Config
from adaptive_questionnaires.v2.engine import (
    AdaptiveSessionEngine, JSONStateStore, SessionContext, read_feedback_csv, write_proposal_csv,
)
from adaptive_questionnaires.v2.redundancy import make_similarity


def _engine(mk1, state_dir: Optional[str], candidates_path: Optional[str]):
    cfg = V2Config.from_config(mk1.config)
    openai_api = None
    if candidates_path:
        with open(candidates_path, encoding="utf-8") as f:
            raw = f.read()
        backend = FixtureBackend([raw])
    else:
        from adaptive_questionnaires.clients.openai_client import OpenaiAPI
        openai_api = OpenaiAPI(mk1=mk1)
        backend = OpenAIJSONBackend(openai_api)
    sim = make_similarity(cfg.redundancy_backend, cfg.embedding_model,
                          openai_client=getattr(openai_api, "service", None))
    gen = CandidateGenerator(backend, cfg.max_generation_attempts, cfg.max_question_chars)
    root = state_dir or _cfg(mk1, "app", "dir_v2_state", "./v2_state")
    return AdaptiveSessionEngine(cfg, sim, gen, JSONStateStore(root)), openai_api


def _cfg(mk1, section, key, default):
    try:
        return mk1.config.get(section, key, fallback=default)
    except Exception:
        return default


def v2_select(mk1, args) -> int:
    if not args.session:
        raise SystemExit("v2_select requires --session <session_context.json>")
    with open(args.session, encoding="utf-8") as f:
        ctx = SessionContext.from_dict(json.load(f))
    engine, api = _engine(mk1, args.state_dir, args.candidates)
    result = engine.run_session(ctx)
    state = engine.store.load(ctx.anon_subject_id)
    out = os.path.join(engine.store.root, ctx.anon_subject_id, f"{ctx.session_id}_proposal.csv")
    write_proposal_csv(result, state, out)
    m = result.metrics
    print(f"V2 proposal for {ctx.anon_subject_id}/{ctx.session_id}: {m['n_selected']} questions "
          f"({m['n_new']} new, {m['n_retired']} retired) — stop: {result.stop_reason}")
    changed = [s.category for s in result.change_signals if s.changed_state_possible]
    if changed:
        print(f"  change signals (review flags, not judgements): {changed}")
    for w in m["warnings"]:
        print(f"  ⚠️ {w}")
    gm = m["generation"]
    print(f"  generation: attempts={gm['attempts']} valid={gm['valid_items']} invalid={gm['invalid_items']}")
    if api is not None:
        print(f"  api usage: {api.usage}")
    print(f"  clinician review sheet: {out}  (contains clinical text — keep local)")
    print("  This is a proposal for clinician review, not a clinical decision.")
    return 0


def v2_import_feedback(mk1, args) -> int:
    if not (args.subject and args.session and args.feedback):
        raise SystemExit("v2_import_feedback requires --subject, --session <session_id>, --feedback <csv>")
    cfg = V2Config.from_config(mk1.config)
    from adaptive_questionnaires.v2.redundancy import LexicalSimilarity
    root = args.state_dir or _cfg(mk1, "app", "dir_v2_state", "./v2_state")
    engine = AdaptiveSessionEngine(cfg, LexicalSimilarity(), None, JSONStateStore(root))
    summary = engine.import_feedback(args.subject, args.session, read_feedback_csv(args.feedback))
    print(f"feedback imported for {args.subject}/{args.session}: {summary}")
    return 0

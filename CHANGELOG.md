# Changelog

All changes below are fork modifications on branch `adaptive-questioning-v2`, based on upstream
`dimigeorgiou/adaptive-llm-questionnaires-suicide-risk` at `29752f3`. The original work, method and
clinical pilot belong to Georgiou, Makrigiannis, Gerogiannis and Kanavos (NICE TEAS Europe 2026).
Issue IDs (A, B, …) refer to `docs/BASELINE_AUDIT.md`. Every "Fixed" entry was reproduced against
the unmodified upstream code before it was fixed.

# Adaptive Questioning V2

### Added

- **V2 package** `adaptive_questionnaires.v2` (experimental, clinician-reviewed proposals only):
  - structured data model with stable, never-reused `question_id` and `parent_question_id`
    provenance (`models.py`);
  - JSON-schema candidate generation with strict parsing and bounded retries, where model
    self-ratings are ignored (`candidate_generator.py`);
  - similarity backends: deterministic lexical, local multilingual embeddings (`[semantic]` extra),
    and OpenAI embeddings, plus redundancy metrics (`redundancy.py`);
  - clinician-controlled coverage floor, ceiling, required domains and priority (`coverage.py`);
  - structured per-domain longitudinal state and the conservative `ChangeConsistencySignal`
    (`longitudinal.py`);
  - clinician-history score: shrunk EMA, bounded, missing ≠ 0 (`scoring.py`);
  - `ExpectedUtilityProxy` ranker with ablatable coefficients, and an `InformationGainEstimator`
    interface (not implemented/calibrated) (`ranker.py`);
  - constrained greedy selector (locks, floors, near-duplicate filter, replacement cap, switching
    margin, and an experimental adaptive length that is off by default) (`selector.py`);
  - engine with an atomic, validated JSON state store and atomic `question_id`-keyed feedback import
    (`engine.py`);
  - `ValidatedInstrumentAdapter` interface; no instrument content is shipped (`instruments.py`);
  - CLI operations `v2_select` and `v2_import_feedback`, which run without Google credentials, and
    without OpenAI when a candidate fixture is supplied.
- `[adaptive]`, `[adaptive_v2]` and `[privacy]` config sections, and `[api_google] scopes`.
- `adaptive_questionnaires.legacy`: the V1 algorithm extracted into pure functions (`algorithm.py`,
  `parsing.py`) and provenance-based analysis (`analysis.py`).
- `prompting.py` (safe template filling) and `privacy.py` (redaction).
- Synthetic examples in `examples/v2/`.
- Test suite (`tests/`, 108 tests), including golden parity with the unmodified upstream pipeline.

### Changed

- `pipeline.py` delegates to `legacy/` with unchanged V1 arithmetic. Parity was verified on pandas
  2.3 and 3.0.
- `Controller(mk1, argv=None)`; `main()` exits with code 2 when integrity issues were found.
- `legacy_strict_validation = true` (default): blank, partial or out-of-range clinician scores
  block the weight update for that patient and are reported. `false` restores exact upstream
  arithmetic.
- `requirements.txt` and `pyproject.toml` are synchronised with lower bounds. `dropbox`, `dataset`
  and `beautifulsoup4` become optional extras, imported lazily.
- Analysis exports carry `question_id`, `parent_question_id` and `first_meeting`. Registry JSON is
  keyed by deterministic question ids.
- Visualisation reads only `timeseries_*.csv`.

### Fixed

- **A / A2 / A4**: `task0` always crashed (`'list' object has no attribute 'format'`). Literal braces
  in prompts raised `KeyError`. `requirements=None` raised `TypeError`.
- **B**: analysis gave a new question the replaced question's weight (positional alignment).
- **C**: `task0` required the `task_0` prompt row to be the first row.
- **D**: a fresh clone crashed because `logs/` did not exist.
- **E**: the unscored sentinel 0.0 was averaged as a real score (strict mode now refuses; see
  Changed).
- **F**: the output-tab row is derived from the configured range (upstream hard-coded `+3`, wrong for
  the shipped example config).
- **G / G2**: pandas ≥ 3 broke `row[1]` and the float resets of string-typed Likert columns.
- **H**: `hash()`-based registry keys were not reproducible across processes.
- **I / Q**: retries never fired for Google `HttpError`, and OpenAI had no retry or backoff. Both now
  retry transient errors only.
- **J / P**: failed Sheets writes and formatting returned normally, and formatting could hit sheet
  id 0.
- **K**: a blank Likert cell aborted the batch mid-way.
- **L**: an unparseable questionnaire silently created an empty patient tab.
- **S**: `result_to_df` raised when every row was shorter than the header.
- Clean install: missing `google-auth` declared; unused `psutil` import removed; `bs4` import made
  lazy.
- `NOTICE`: corrected references to non-existent `lib/`, root `run.sh` and `docs/ETHICS.md`.

Not fixed (documented in the audit): M (a wrong LLM line count leaves that category unreplaced),
N (preamble lines accepted when the count matches), O (non-atomic multi-step Sheets writes), R (dead
M3 code).

### Security

- Google OAuth scopes reduced to `spreadsheets` by default. Upstream also requested `drive.readonly`,
  `drive.file`, `gmail.readonly`, `gmail.modify`, `documents` and `spreadsheets.readonly`. **Existing
  tokens keep their old grant until `config/secrets/*_accessed.json` is deleted and consent is
  repeated.**
- Clinical text is redacted in logs and console output by default: the full questionnaire used to be
  logged at INFO. Opt in with `[privacy] log_clinical_text`.
- `.env.example` (placeholders only). `.gitignore` extended for credentials, token files, V2 state
  and caches. Tests check tracked files for secret patterns.
- V2 ids are validated as opaque research ids, and unknown session-context keys are rejected.
- A history scan of upstream found no committed secrets or clinical data.

### Evaluation

- `experiments/evaluate_v1_vs_v2.py`: synthetic, deterministic, credential-free V1-vs-V2
  comparison. Primary proxies and the decision rule were fixed in code before running. Includes a
  random control, a semantic-backend variant, ablations, and one labelled post-hoc sensitivity run.
- **Result: "V2 produces mixed results."** V2 was better on intent duplication, longitudinal
  repetition and change follow-up, worse on per-category minimum coverage, and showed no clear
  difference on synthetic rating quality. See `docs/V2_EVALUATION.md`.
- **Clinical predictive accuracy: NOT EVALUATED.**
- `experiments/simulate_weighting.py`: simulation behind the choice of clinician-score
  representation.
- `experiments/repro_baseline_issues.py` and `experiments/make_legacy_golden.py`: reproduction of
  the baseline defects and generation of the golden fixture from upstream.

### Documentation

- `docs/BASELINE_AUDIT.md`, `docs/ADAPTIVE_V2_DESIGN.md`, `docs/EVIDENCE_REVIEW.md`,
  `docs/V2_EVALUATION.md`.
- README: an "Experimental Adaptive Questioning V2" section added below the unchanged original
  content, plus fork provenance.

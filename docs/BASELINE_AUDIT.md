# Baseline Audit — upstream `dimigeorgiou/adaptive-llm-questionnaires-suicide-risk`

| | |
|---|---|
| Audited commit | `29752f3` ("Restructure to professional src/ package layout.") on upstream `main` |
| Audit date | 2026-09-25 |
| Scope | Every tracked file (≈4.4k lines): `README.md`, `NOTICE`, `CITATION.cff`, `LICENSE`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `config/config.example.ini`, `main.py`, `scripts/run.sh`, `src/adaptive_questionnaires/**` |
| Method | Static reading plus **executed reproductions** against the unmodified upstream code, using mocks and synthetic data only (`experiments/repro_baseline_issues.py`). No Google or OpenAI credentials were used, and no clinical data was accessed. |
| Auditor role | Fork contributor. The original work belongs to Georgiou, Makrigiannis, Gerogiannis and Kanavos (NICE TEAS Europe 2026). |

This document describes the upstream code as it was. Fixes live on the fork branch `adaptive-questioning-v2` and are listed in `CHANGELOG.md`.

Each issue carries one of these labels:

- **CONFIRMED (executed)**: reproduced by running the upstream code.
- **CONFIRMED (inspection)**: follows directly from the code and needs no runtime state. Impact may depend on the deployed configuration.
- **NOT CONFIRMED**: investigated, and no defect was found.

---

## Existing architecture

```
main.py ──► adaptive_questionnaires.pipeline.main()
              └─ Controller(MkI singleton)
                   ├─ MkI (core/mark_i.py): INI config at ./config/config.ini + file logger
                   │     (also DataSet [sqlite via `dataset`] and M3 [Uber-internal metrics] — unused)
                   ├─ GoogleAPI / GoogleSheetsAPI (clients/google_api.py, oauth2client flow)
                   │     (also GoogleEmailAPI, GoogleDocsAPI, GoogleDriveAPI — unused by the pipeline)
                   ├─ OpenaiAPI (clients/openai_client.py, Chat Completions)
                   └─ DropboxAPI (clients/dropbox_client.py; only used by a disabled token-sync helper)
```

One CLI flag (`-o/--operation`) selects a stage: `task0`, `task1_preparation`, `task1`, `task1_analyze_results`, `task1_visualize_results`. `scripts/run.sh` runs all five in order.

All clinical state lives in **one Google Spreadsheet**:

| Tab | Content |
|---|---|
| `input` | Read by `task0` but not used afterwards. |
| `output` | One row per patient: `history`, `task0` (initial questionnaire as markdown text), `meeting{k}_notes`, `meeting{k}_task1` (regenerated questionnaire text). |
| `prompts` | Rows keyed by `task` (`task_0`, `task_1`) with `prompt`, `system_role`, `user_role`, `requirements`, `examples`. **The prompts are clinician- or researcher-authored sheet content and are not in the repository.** |
| `patient{N}` | One tab per patient. Each meeting is a block of **12 columns** placed side by side (`meeting, category, category_name, question_text, q, w, w_new, Coherence, Emotional Resonance, Perceived Helpfulness, Motivational Impact, Engagement`). The block has 20 rows (5 categories × 4 questions). `N = output-row-index + 3`. |

## Data flow

```
output.history ──task0──► LLM ──► output.task0  (Greek markdown: "### Κατηγορία k: name" + "n. question")
                                     │
                          task1_preparation (regex parse)
                                     ▼
                       patient{N} block 0  (w = 0.5, w_new = 0, Likert = 0)
                                     │
                  clinician fills the 5 Likert columns in Sheets (HITL)
                                     ▼
 task1, loop 1:  w_new = w + 0.2·(mean(Likert) − 4)  → written into the same block
 task1, loop 2:  if output.meeting{k+1}_notes present:
                   per category mark the 2 lowest w_new → LLM(prompt task_1, notes, kept questions, category)
                   → split reply on newlines → overwrite text in the same rows
                   → new block k+1 (w ← w_new, new rows w = 0.5; w_new, Likert ← 0) appended to the right
                   → merged markdown written to output.meeting{k+1}_task1
 task1_analyze_results: split patient tab into 12-col blocks → question registry + wide time series → ./data/*.csv/json
 task1_visualize_results: ./data/*.csv → 4-panel PNG per patient in ./outputs/figures
```

## Current question-generation method

- **Initial questionnaire (`task0`)** is a single-shot Chat Completion per patient (default `gpt-4o-mini`, `temperature=0.3`, `max_tokens=6000`). The prompt, system role, user role, requirements and examples come from the `prompts` tab, and the `{history}` variable is filled in with `str.format`. Output is free-form markdown, and the Greek headings are the only structure.
- **Replacement questions (`task1`)** use one call per category, with variables `meeting_notes`, `existing_questions` (a Python list interpolated as its `repr`) and `category` (the integer category number). The reply is split on `\n`. If the number of non-empty lines is not exactly the number of dropped rows, that category is skipped with a `print` warning.
- The model does not produce structured output, and nothing validates the language, length, category, or overlap with existing items.

## Current adaptation formula

`pipeline.py:488-506`:

$$f_i = \tfrac{1}{5}\sum_{d=1}^{5} s_{i,d},\qquad w'_i = w_i + \alpha\,(f_i-\mu),\quad \alpha = 0.2,\ \mu = 4$$

- The constants are hard-coded and not configurable.
- New questions start at `w = 0.5`.
- The update is unbounded and asymmetric on a 1–5 scale. A perfect 5 adds `+0.2` and a 1 subtracts `−0.6`. Because μ = 4 sits above the scale midpoint, an average rating of 3 still lowers the weight (by 0.2 per session).
- `0.0` is the "not yet scored" sentinel. Nothing distinguishes it from a real score when it is averaged (issue E).

## Current replacement policy

`pipeline.py:523-582`: per category, `nsmallest(2, "w_new")` is marked for replacement. **Exactly two items per category, ten of twenty overall, are replaced every session**, whatever the absolute scores are. If all four items in a category are rated 5, two are still replaced. Ties go to the earlier row. A replacement overwrites `question_text` **in the same spreadsheet row and `q` slot**, so the new question keeps the old row's position and number. There is no question identifier.

## Current clinician feedback mechanism

Five Likert dimensions are entered per question per meeting in the patient tab: Coherence, Emotional Resonance, Perceived Helpfulness, Motivational Impact, Engagement. The README says clinicians gate every replacement cycle. In code, that gating is implicit: `task1` runs when the scores are non-zero and the next meeting's notes exist, and its output is written straight to the sheet for the clinician to review. No explicit approve/reject flag exists.

## Current analytics

`task1_analyze_results` builds a registry keyed by `(category, q, question_text)`, a wide time series (`w_k` and the five dimensions per meeting), and a list of text changes per slot. `task1_visualize_results` draws a weight heatmap (NaN → 0), a median/IQR weight trajectory (zeros → NA), a per-question weight standard-deviation heatmap, and a violin plot (zeros dropped). Nothing measures redundancy, coverage, or repetition.

## External services

| Service | Use | Data sent |
|---|---|---|
| OpenAI Chat Completions | `task0` and `task1` generation | **Patient history and session notes (clinical free text)**, plus existing questions |
| Google Sheets API v4 | All state | Everything |
| Dropbox (optional, disabled) | Syncing a Google OAuth token file | OAuth token |
| Gmail / Docs / Drive | Classes exist and are **never called** by the pipeline | – |

## Data/security assumptions

- Secrets and local clinical exports are git-ignored (`config/config.ini`, `config/secrets/`, `data/`, `outputs/*`, `logs/`, `.env*`, `*.pickle`). **A history scan found no committed secrets or clinical data** (checked patterns: OpenAI/Google key formats, OAuth fields, long tokens).
- **OAuth over-privilege (CONFIRMED, inspection)**: `GoogleAPI.SCOPES` requests `drive.file`, `drive.readonly` (read access to *all* the user's Drive files), `spreadsheets`, `spreadsheets.readonly`, `gmail.readonly`, `gmail.modify` (read, label and trash all mail), and `documents`. The pipeline only calls Sheets endpoints, so `https://www.googleapis.com/auth/spreadsheets` alone is sufficient.
- **Clinical text in logs (CONFIRMED, inspection)**: `GoogleSheetsAPI.update_cell` logs the full written value at INFO, and it is called with the whole merged questionnaire. `task1_preparation` prints `df.head()` (question text), and analysis prints the first 60 characters of old and new questions. These go to the persistent `./logs/logs.log` and to stdout.
- Clinical free text is sent to OpenAI. Whether that is acceptable depends on the ethics approval (KL-2025-07/ETH) and any data-processing agreement. The code cannot establish either, and neither is documented in the repository.
- `GoogleAPI.get_credentials` can `pickle.load` a token file. That is only safe while the file is fully trusted (it is in the unused `oauth()` path).
- The `NOTICE` file refers to `lib/`, `run.sh` at the root, and `docs/ETHICS.md`. None of these exist after the restructure.

## Reproducibility issues

1. **Unpinned dependencies with pandas 3 breakage (CONFIRMED, executed — G)**: `task1_preparation` reads `row[1]` on a label-indexed Series. On a fresh install today (`pandas==3.0.6`), this raises `KeyError: 1`.
2. **Clean install cannot import the pipeline (CONFIRMED, executed)**: `pip install -r requirements.txt` into a fresh Python 3.13 venv, then `import adaptive_questionnaires.pipeline`, fails with `ModuleNotFoundError: No module named 'bs4'`. After adding `bs4`, `psutil` is also missing (imported, never used). The pinned `google-auth-oauthlib==0.4.1` dates from 2019, and `oauth2client` has been deprecated since 2017.
3. **Registry keys are not reproducible (CONFIRMED, executed — H)**: `question_registry_*.json` keys use `hash(question_text) % 10000`. Python salts string hashes per process, so across five `PYTHONHASHSEED` values the same question got five different keys (`1_3_236`, `1_3_3090`, …). Collisions are also possible.
4. No random seeds, prompt versions, model parameters or git SHA are recorded with any output. LLM outputs are not cached, so a run cannot be replayed.
5. The prompts live only in the spreadsheet and are not versioned.
6. **Fresh clone crashes at start-up (CONFIRMED, executed — D)**: the logger opens `./logs/logs.log`. `logs/` is git-ignored and never created, so the result is `FileNotFoundError`. The config path is relative to the working directory (`./config/config.ini`).

## Potential software bugs

| ID | Location | Finding | Status |
|---|---|---|---|
| **A** | `pipeline.py:118` → `openai_client.py:210` | `task0` splits `requirements` into a **list**, and `execute_custom_prompt` then calls `requirements.format(**variables)`. Result: `AttributeError: 'list' object has no attribute 'format'`. `variables` is always non-empty in `task0`, so **`task0` fails every time a requirements cell is present**. | CONFIRMED (executed) |
| **A2** | `openai_client.py:207-210` | Prompts are filled with `str.format`. Any literal brace in sheet-authored text, such as a JSON example, raises `KeyError`. | CONFIRMED (executed) |
| **A3** | `pipeline.py:547-551` | `existing_questions` is interpolated as a Python list `repr` (`"['Q one?', 'Q two?']"`). | CONFIRMED (executed). This is cosmetic, but it is part of the V1 prompt and is **kept for legacy fidelity**. |
| **A4** | `openai_client.py:210` | If `requirements` is `None` (empty or missing cell) and variables are given, it becomes `[None]`, and `"\n".join` raises `TypeError`. | CONFIRMED (executed) |
| **B** | `pipeline.py:667-673` | Analysis sets `df['w'] = previous_df['w_new'].values`, aligning by **row position**. Replacements reuse the row, so in the reproduction a new question **Q-E** showed `w_1 = 0.1`, the updated weight of the question it replaced (Q-C). The sheet itself correctly held `w = 0.5`. Every exported time series and plot after the first replacement is affected. | CONFIRMED (executed) |
| **C** | `pipeline.py:112-119` | `task0` filters the prompts tab and reads `.loc[0, …]` without `reset_index`. If the `task_0` row is not the first row, the result is `KeyError: 0`. (`task1` does reset the index.) | CONFIRMED (executed) |
| **D** | `core/mark_i.py:464` | Logger directory is never created (see Reproducibility 6). | CONFIRMED (executed) |
| **E** | `pipeline.py:325-333, 495` | The "not scored" sentinel `0.0` is averaged as a real score. With four dimensions at 5 and one left unscored, the composite is 4.0 and `w_new = 0.50`. Fully scored at 5, it would be 0.70. A question left entirely unscored gets `f = 0` and loses 0.8, which makes its replacement almost certain. The gate only checks that *some* score is non-zero. | CONFIRMED (executed) |
| **F** | `pipeline.py:435`, `191`, `298` | The output-tab row is hard-coded as `row_idx + 3`, which assumes the data range starts at sheet row 2 (`output!A2:…`, per the code comment). The shipped `config.example.ini` sets `reporter_tab_output = output`, which puts the header on row 1. Under that config, patient *i*'s regenerated questionnaire would be written to patient *i+1*'s row. | CONFIRMED (inspection). Impact depends on the deployed config. The production config is unknown and probably uses `A2`. |
| **G** | `pipeline.py:185` | `row[1]` positional access fails under pandas ≥ 3. | CONFIRMED (executed) |
| **H** | `pipeline.py:821` | `hash()`-based registry keys (see Reproducibility 3). | CONFIRMED (executed) |
| **I** | `google_api.py` `@retry(...)` decorators | Retries are declared for `requests` exceptions, but calls made through `googleapiclient` raise `googleapiclient.errors.HttpError`, which is **not** a `requests.RequestException` (MRO: `HttpError → Error → Exception`). **Retries never fire** for `get_df_from_tab`, `update_range`, and `update_cell`. | CONFIRMED (executed) |
| **J** | `google_api.py:437-461` (`write_df_to_tab2`) | The return value of `request_check` is discarded. A 403 response still **returns normally** (`updated_cells: None`), so a failed write to a patient tab is silent. `format_sheet_tab` behaves the same way. | CONFIRMED (executed) |
| **K** | `pipeline.py:322` | A blank Likert cell (`""`) raises an uncaught `ValueError` in `astype(float)`. That aborts the whole batch after earlier patients were already written (partial update). | CONFIRMED (executed) |
| **L** | `pipeline.py:180-230` | The markdown parser requires exactly `### Κατηγορία k: name`. A trivially different LLM heading, such as `**Κατηγορία 1: …**`, parses to **zero questions without an error**, and an empty patient tab is then created. | CONFIRMED (executed) |
| M | `pipeline.py:565-567` | If the LLM returns the wrong number of lines, the category is silently left unreplaced. Other categories proceed and the meeting still advances. | CONFIRMED (inspection) |
| N | `pipeline.py:562` | LLM lines are inserted verbatim. Preambles ("Here are two questions:"), numbering and markdown become question text if the line count happens to match. | CONFIRMED (inspection) |
| O | `pipeline.py` `task1` loop 2 | Non-atomic multi-step writes (new block, then formatting, then the output cell). If a failure occurs after the new block is written, a re-run skips that patient: the new block has `w_new` all zeros, and the `meeting{k+1}_task1` cell is never filled. | CONFIRMED (inspection) |
| P | `google_api.py` `get_tab_gid` | Returns `0` when a tab is not found. `format_sheet_tab` would then format sheet id 0, which is usually the first tab. | CONFIRMED (inspection) |
| Q | `openai_client.py` | Every exception is re-raised (`raise e` followed by unreachable logging). There is no retry or backoff for rate limits or timeouts. `RateLimiter` is defined but never used. | CONFIRMED (inspection) |
| R | `mark_i.py` `M3` | References `M3Client` / `ReadM3Client`, which are never imported (Uber-internal). This is dead code and harmless, because `_m3` is never enabled. | CONFIRMED (inspection) |

## Potential methodological weaknesses

1. **The weight is a single opaque number.** Five clinician dimensions are collapsed into one scalar before anything is stored. The raw scores survive in the sheet, but the adaptation uses only the mean.
2. **Replacement is quota-driven, not evidence-driven.** Always replacing 2 of 4 per category creates 50% churn per session, whatever the ratings say. A question needs only one session of relatively low rating to be removed, and there is no uncertainty handling, even though n = 1 rating at a time.
3. **An asymmetric, unbounded update** means weights drift downward under average ratings. They are not comparable across questions with different ages, because new questions reset to 0.5 regardless of the current weight distribution.
4. **Nothing controls redundancy.** Replacement prompts see the kept questions of that category only. There is no check against earlier sessions, retired questions, or other categories, and no semantic comparison.
5. **Fixed 5 × 4 structure.** Coverage cannot adapt to what changed, and no minimum-coverage concept exists beyond the fixed grid.
6. **Longitudinal context is the raw notes of the next meeting only.** Earlier sessions are not summarised, and nothing states what has changed or what has already been explored.
7. **Engagement-type ratings are the only feedback.** The five dimensions measure the clinician's perception of a question's quality. They are not outcomes. Treating rising weights as "improvement" would conflate question acceptability with clinical value.
8. **Generation is unvalidated.** Output language, length, category fidelity and safety are not checked. The LLM's output is trusted structurally (bugs L, M, N).
9. Category identity is only the integer position in the LLM output. Names can drift between sessions.

## Claims that are supported

- The software orchestrates clinician-supervised LLM generation of questionnaire items from patient history. Items are stored in Sheets for clinician scoring, and a weight-based rule selects items for replacement (code present, with the defects above).
- The adaptation rule described in the README ($w' = w + 0.2(f-4)$, two lowest per category) matches the code.
- The LLM is not used for diagnosis or risk scoring. No code computes a risk estimate.
- Longitudinal CSV/JSON/PNG exports exist (but see B and H for their correctness).

## Claims that are NOT supported by the current implementation/data

- **"Fuzzy logic–based adaptation"** (README abstract, CITATION): the code contains no fuzzy sets, membership functions or fuzzy inference. The adaptation is a linear additive update with a fixed-quota argmin.
- **"Clinicians … gate every replacement cycle"**: there is no approve/reject mechanism in code. Gating depends on clinicians acting on the sheet before `task1` is run.
- **Any statement about improved suicide-risk assessment accuracy** cannot be evaluated with this repository. It has no outcome labels, reference-standard assessments, or evaluation code, and none were available to this audit. Clinical accuracy is **not evaluated** here or upstream.
- "Longitudinal analytics … for transparency": the exported weight trajectories are corrupted by positional alignment (B) after the first replacement.
- The README's `pip install -r requirements.txt` quick start does not produce a working install today (Reproducibility 1–2, 6).

## Addendum — issues found during implementation

Found after the initial audit and reproduced against the unmodified upstream code (29752f3) before
being fixed:

| ID | Location | Finding | Status |
|---|---|---|---|
| **S** | `google_api.py` `result_to_df` | The Sheets API trims trailing empty cells per row. If **every** data row is shorter than the header (for example, an all-empty last column such as a not-yet-filled `meeting2_task1`), `pd.DataFrame(data, columns=headers)` raises `ValueError: 4 columns passed, passed data had 3 columns`. | CONFIRMED (executed, upstream `result_to_df`) |
| **G2** | `pipeline.py` `_replace_lowest_scoring_questions` | Under pandas ≥ 3, columns re-read from Sheets use the strict string dtype. Writing `0.0` into a Likert cell of a replaced row raises `TypeError: Invalid value '0.0' for dtype 'str'`. A sibling of G. | CONFIRMED (executed: the legacy scenario fails on pandas 3.0.6 before the fix; parity passes on pandas 2.3.3) |

Environment note: the golden parity fixture had to be generated under pandas 2.3.3, because the
unmodified upstream `task1_preparation` cannot run under pandas 3 (issue G).

## Investigated but not confirmed

- **Duplicate column names** in patient tabs (each 12-column block repeats the same headers): `result_to_df` keeps duplicate column names, and the pipeline always slices by position before selecting by name, so no defect was found.
- **`_column_number_to_letter` / `_get_column_letter`**: both are correct for multi-letter columns (checked A, Z, AA, AZ, BA).
- **Secrets in history**: none found.

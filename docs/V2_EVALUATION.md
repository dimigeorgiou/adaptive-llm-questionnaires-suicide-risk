# Result

**V2 produces mixed results.**

**Clinical predictive accuracy: NOT EVALUATED.** No outcome labels, reference-standard assessments or
clinical data exist in this repository. Every number below measures *question selection* in a
**synthetic** world. None of them says anything about detecting, predicting or classifying suicide risk.

| | |
|---|---|
| Harness | `experiments/evaluate_v1_vs_v2.py` (code in `src/adaptive_questionnaires/evaluation/`) |
| Code commit of the run | `f136e85` (clean tree; recorded in `outputs/evaluation/run_manifest.json`) |
| Data | 30 synthetic patients × 6 sessions, seed `20260925`; no LLM calls (deterministic replay) |
| Reproducibility | two consecutive runs produced identical outputs |
| Raw outputs | `outputs/evaluation/{per_session_metrics.csv, comparison.csv, ablations.csv, summary.json, results.md}` |
| Statistics | patient = unit of analysis (mean over sessions); paired bootstrap of V2 − V1, 2,000 resamples, 95% CI |

## Decision rule (fixed in code before the first run)

Primary proxies: `intent_duplicate_rate`, `longitudinal_repetition`, `change_followup_rate`,
`min_per_category`, `synthetic_rating_quality`. A proxy is *better* or *worse* when the 95% CI of
(V2 − V1) excludes 0 in that direction. Verdicts:

- "improves": at least one primary proxy is better and none is worse
- "mixed": some are better and some worse
- "does not outperform": none is better

## Primary comparison: V2 (default install, lexical redundancy) vs V1

| Metric | V1 | V2 | Difference [95% CI] | |
|---|--:|--:|--:|---|
| Intent duplicate rate (share of items whose intent occurs twice or more in one questionnaire) ↓ | 0.377 | 0.282 | −0.095 [−0.137, −0.053] | better |
| Longitudinal repetition (share of *new* items re-asking an intent from an earlier session) ↓ | 0.551 | 0.430 | −0.122 [−0.170, −0.076] | better |
| Change follow-up rate (share of change onsets that got at least one clarifying follow-up) ↑ | 0.628 | 0.844 | +0.217 [+0.053, +0.372] | better |
| Minimum questions in any category ↑ | 4.000 | 3.761 | −0.239 [−0.317, −0.167] | **worse** |
| Synthetic rating quality (mean latent quality of asked items, 1–5) ↑ | 3.212 | 3.258 | +0.046 [−0.005, +0.097] | no clear difference |

Secondary and descriptive (full table in `outputs/evaluation/results.md`):

| Metric | V1 | V2 | Difference [95% CI] |
|---|--:|--:|--:|
| Semantic diversity (1 − mean pairwise embedding cosine) ↑ | 0.679 | 0.727 | +0.048 [+0.042, +0.055] |
| Relevance proxy of new items to current notes ↑ | 0.301 | 0.346 | +0.046 [+0.030, +0.060] |
| Category entropy (normalised) ↑ | 1.000 | 0.998 | −0.002 [−0.003, −0.001] (worse) |
| Exact duplicate rate ↓ | 0.000 | 0.000 | 0 |
| Semantic near-duplicate pair rate (cosine ≥ 0.8) ↓ | 0.002 | 0.002 | no clear difference |
| Replacement churn (share of items new each session) | 0.394 | 0.258 | −0.136 [−0.157, −0.113] |
| LLM calls per session | 4.33 | 1.00 | −3.33 |
| Candidate items generated per session (cost proxy) | 11.95 | 35.00 | +23.05 |
| Invalid-output rate (units differ; see below) | 0.057 | 0.056 | no clear difference |
| Local selection time (ms) | 5.0 | 13.1 | +8.1 |

Per session, V2 generates about **3× more candidate text** than V1, in a single call instead of about
4 (V1 averages 4.33 because its initial session is one call). Tokens, API latency and cost were **not
measured**, because the evaluation makes no API calls.

## Controls, variants and ablations

**Random-selection control.** Same candidate pool and constraints, no ranking signal, hashed order.
V2 beats it on longitudinal repetition (−0.054 [−0.111, −0.005]), change follow-up (+0.457),
per-category minimum (+1.49) and synthetic rating quality (+0.295 [+0.244, +0.347]). It does **not**
clearly beat it on intent duplication (−0.019 [−0.057, +0.016]). The control keeps V2's schema
validation and near-duplicate filters and turns off only the ranking weights. So most of V2's
duplication advantage over V1 comes from the larger pool plus those filters, not from the ranking
weights. The redundancy component as a whole (weight plus filter) does matter (see ablations).

**Semantic redundancy backend (V2-semantic, optional `[semantic]` extra).**

- Much lower duplication (−0.236) and repetition (−0.371) than V1.
- *Worse* synthetic rating quality (−0.079 [−0.142, −0.015]), *worse* per-category minimum (−0.617),
  and churn of only 0.097.
- Mechanism (checked on 8 patients): embedding cosines between related questions run higher than
  lexical scores, so the mean novelty of new candidates falls from 0.664 to 0.392. At
  `w_novelty = 0.2` that costs as much as the 0.05 switching margin, so carried-over items, including
  poorly rated ones, win more often. The near-duplicate filter plays a smaller part (67 vs 2 filtered
  candidates).
- **The coefficients are not calibrated across similarity backends.** This is an open problem, not a
  tuned result.
- The semantic evaluation metrics use the same embedding model as this variant's selector, so they
  are partly circular *for this row only*. Intent metrics are not.

**Ablations (each vs full V2).**

| Removed component | Clear effect(s) when removed | Reading |
|---|---|---|
| redundancy (weight + hard filter) | intent duplication +0.097 [+0.059, +0.136]; semantic near-dup pairs +0.006; floor slightly *better* (+0.117) | does what it is meant to do, at a small coverage cost |
| coverage | minimum per category −1.49; entropy −0.036; synthetic rating quality slightly *better* (+0.042) | the only component enforcing balance; it trades a little rating quality |
| longitudinal | longitudinal repetition +0.030 [+0.005, +0.056]; change follow-up −0.031 [−0.072, +0.008] (no clear difference) | **marginal**: overlaps heavily with `relevance`. Under the lexical backend, change signals fire on 13 of 15 *unchanged* re-phrased summaries (semantic: 0 of 15), so the feature is nearly always on |
| clinician_history | synthetic rating quality −0.268 [−0.310, −0.232]; floor slightly *better* (+0.183) | retaining well-rated items drives the quality proxy |

**Post-hoc sensitivity (not part of the verdict).** This was added *after* the results were seen,
because V2's default floor (2) differs from V1's fixed grid (4). With `min_per_category = 4`, V2 is
better than V1 on intent duplication (−0.085), repetition (−0.103) and change follow-up (+0.208). It
shows no clear difference on the floor (4.0 vs 4.0) or on rating quality (+0.040 [−0.016, +0.091]).
Treat this as a hypothesis for a new pre-specified evaluation, not as a result.

## What can be concluded

- In this synthetic setting, the V2 pipeline (larger structured candidate pool, filtering and ranking)
  produces questionnaires with **less within-session and across-session intent repetition**, and
  **follows up on changed domains more often**, than the V1 algorithm run on the same simulated model
  outputs.
- The ranking components (not just the pool size) matter for follow-up, coverage and retention of
  well-rated items (vs the random control).
- V2's default adaptive distribution **reduces per-category minimum coverage** relative to V1's
  fixed 4 × 5 grid. Whether that is acceptable is a **clinical** decision. The floor is configurable.
- The semantic backend is not a free upgrade under the current coefficients.
- The lexical backend's change signal has low specificity (13 of 15 false flags on unchanged
  re-phrased summaries). Part of V2-lexical's follow-up advantage may come from reacting to text
  differences in general rather than to real changes. Real changes were still flagged 15 of 15.

## What cannot be concluded

- **Anything about clinical accuracy, safety or benefit.** No sensitivity, specificity, PPV, NPV,
  AUROC, AUPRC, calibration or Brier score was computed, because no valid reference outcomes exist.
- Anything about real clinicians' judgement. `synthetic_rating_quality` is a simulated construct
  (random latent quality plus noise). It is not a clinician rating.
- Anything about real LLM behaviour. The simulated model's paraphrase rate, Zipf popularity and
  follow-up probability are **assumptions** that drive the size of every difference. Real outputs
  are non-deterministic and were not evaluated.
- Anything about Greek. The synthetic bank is English, while the deployment questionnaires are Greek.
- Generalisation beyond this synthetic design. The same fork contributor wrote both the synthetic
  world and V2. The ratings were built to be independent of V2's features, and the intent labels are
  independent of V2's similarity function, but design bias cannot be ruled out.

## Units and caveats

- `invalid_output_rate`: for V1, the share of category calls rejected by the upstream line-count
  check (from a simulated preamble line). For V2, invalid JSON items as a share of generated items.
  The units differ and are not directly comparable.
- The legacy weight rule's handling of missing ratings (issue E) is not exercised here: all simulated
  ratings are complete.

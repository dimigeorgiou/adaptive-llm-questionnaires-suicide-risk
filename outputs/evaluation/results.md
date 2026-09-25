# V1 vs V2 — synthetic question-selection evaluation

**Result: V2 produces mixed results.**

**Clinical predictive accuracy: NOT EVALUATED.**

Synthetic world: 30 patients × 6 sessions, seed 20260925. Unit of analysis: patient (mean over sessions); paired bootstrap, 2000 resamples. Evaluation similarity model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Git: `f136e8534180489ac8dfd46ce3bd00a2a479219c`.

Decision rule (fixed before the run): Primary proxies: intent_duplicate_rate, longitudinal_repetition, change_followup_rate, min_per_category, synthetic_rating_quality. 'Better'/'worse' = the 95% paired-bootstrap CI of (V2 − V1) excludes 0 in the favourable/unfavourable direction. 'V2 improves the measured question-selection proxies' if >=1 primary proxy is better and none is worse; 'V2 produces mixed results' if some are better and some worse; 'V2 does not outperform the baseline on the current evaluation' if none is better.

## V2 (default install, lexical redundancy) vs V1

| Metric | V1 | V2 | Difference (V2 − V1) [95% CI] | Direction | Assessment |
|---|--:|--:|--:|---|---|
| `intent_duplicate_rate` **(primary)** | 0.377 | 0.282 | -0.095 [-0.137, -0.053] | lower better | better |
| `longitudinal_repetition` **(primary)** | 0.551 | 0.430 | -0.122 [-0.170, -0.076] | lower better | better |
| `change_followup_rate` **(primary)** | 0.628 | 0.844 | 0.217 [0.053, 0.372] | higher better | better |
| `min_per_category` **(primary)** | 4.000 | 3.761 | -0.239 [-0.317, -0.167] | higher better | worse |
| `synthetic_rating_quality` **(primary)** | 3.212 | 3.258 | 0.046 [-0.005, 0.097] | higher better | no clear difference |
| `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `semantic_near_dup_pair_rate` | 0.002 | 0.002 | -0.000 [-0.001, 0.001] | lower better | no clear difference |
| `semantic_diversity` | 0.679 | 0.727 | 0.048 [0.042, 0.055] | higher better | better |
| `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | higher better | no clear difference |
| `category_entropy` | 1.000 | 0.998 | -0.002 [-0.003, -0.001] | higher better | worse |
| `relevance_proxy_new` | 0.301 | 0.346 | 0.046 [0.030, 0.060] | higher better | better |
| `replacement_churn` | 0.394 | 0.258 | -0.136 [-0.157, -0.113] | descriptive | descriptive |
| `mean_words` | 8.710 | 8.765 | 0.055 [-0.025, 0.124] | descriptive | descriptive |
| `invalid_output_rate` | 0.057 | 0.056 | -0.001 [-0.020, 0.017] | lower better | no clear difference |
| `llm_calls` | 4.333 | 1.000 | -3.333 [-3.333, -3.333] | descriptive | descriptive |
| `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `generated_items` | 11.950 | 35.000 | 23.050 [22.961, 23.133] | descriptive | descriptive |
| `selection_ms` | 4.688 | 12.742 | 8.054 [7.642, 8.565] | descriptive | descriptive |

## V2 vs random-selection control (same pool, same constraints, no ranking signal)

| Metric | control | V2 | Difference (V2 − control) [95% CI] | Direction | Assessment |
|---|--:|--:|--:|---|---|
| `intent_duplicate_rate` **(primary)** | 0.301 | 0.282 | -0.019 [-0.057, 0.016] | lower better | no clear difference |
| `longitudinal_repetition` **(primary)** | 0.484 | 0.430 | -0.054 [-0.111, -0.005] | lower better | better |
| `change_followup_rate` **(primary)** | 0.387 | 0.844 | 0.457 [0.325, 0.575] | higher better | better |
| `min_per_category` **(primary)** | 2.272 | 3.761 | 1.489 [1.383, 1.583] | higher better | better |
| `synthetic_rating_quality` **(primary)** | 2.963 | 3.258 | 0.295 [0.244, 0.347] | higher better | better |
| `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `semantic_near_dup_pair_rate` | 0.002 | 0.002 | -0.000 [-0.001, 0.001] | lower better | no clear difference |
| `semantic_diversity` | 0.682 | 0.727 | 0.045 [0.038, 0.052] | higher better | better |
| `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | higher better | no clear difference |
| `category_entropy` | 0.961 | 0.998 | 0.037 [0.033, 0.040] | higher better | better |
| `relevance_proxy_new` | 0.295 | 0.346 | 0.051 [0.035, 0.068] | higher better | better |
| `replacement_churn` | 0.299 | 0.258 | -0.041 [-0.062, -0.020] | descriptive | descriptive |
| `mean_words` | 8.685 | 8.765 | 0.080 [-0.019, 0.174] | descriptive | descriptive |
| `invalid_output_rate` | 0.056 | 0.056 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `llm_calls` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | descriptive | descriptive |
| `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `generated_items` | 35.000 | 35.000 | 0.000 [0.000, 0.000] | descriptive | descriptive |
| `selection_ms` | 11.965 | 12.742 | 0.777 [0.298, 1.458] | descriptive | descriptive |

## V2 with semantic redundancy backend vs V1

Caution: the semantic evaluation metrics use the same embedding model that this variant uses for selection, so they are partly circular for this row. Intent-based metrics are not.

| Metric | V1 | V2-semantic | Difference (V2-semantic − V1) [95% CI] | Direction | Assessment |
|---|--:|--:|--:|---|---|
| `intent_duplicate_rate` **(primary)** | 0.377 | 0.141 | -0.236 [-0.279, -0.193] | lower better | better |
| `longitudinal_repetition` **(primary)** | 0.551 | 0.180 | -0.371 [-0.429, -0.304] | lower better | better |
| `change_followup_rate` **(primary)** | 0.628 | 0.783 | 0.156 [-0.011, 0.331] | higher better | no clear difference |
| `min_per_category` **(primary)** | 4.000 | 3.383 | -0.617 [-0.722, -0.511] | higher better | worse |
| `synthetic_rating_quality` **(primary)** | 3.212 | 3.133 | -0.079 [-0.142, -0.015] | higher better | worse |
| `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `semantic_near_dup_pair_rate` | 0.002 | 0.000 | -0.002 [-0.002, -0.001] | lower better | better |
| `semantic_diversity` | 0.679 | 0.767 | 0.089 [0.081, 0.096] | higher better | better |
| `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | higher better | no clear difference |
| `category_entropy` | 1.000 | 0.994 | -0.006 [-0.007, -0.005] | higher better | worse |
| `relevance_proxy_new` | 0.301 | 0.423 | 0.122 [0.106, 0.138] | higher better | better |
| `replacement_churn` | 0.394 | 0.097 | -0.296 [-0.312, -0.282] | descriptive | descriptive |
| `mean_words` | 8.710 | 8.481 | -0.228 [-0.319, -0.142] | descriptive | descriptive |
| `invalid_output_rate` | 0.057 | 0.056 | -0.001 [-0.020, 0.018] | lower better | no clear difference |
| `llm_calls` | 4.333 | 1.000 | -3.333 [-3.333, -3.333] | descriptive | descriptive |
| `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `generated_items` | 11.950 | 35.000 | 23.050 [22.961, 23.133] | descriptive | descriptive |
| `selection_ms` | 4.688 | 28.305 | 23.617 [10.219, 45.781] | descriptive | descriptive |

## Post-hoc sensitivity: V2 with min_per_category = 4 vs V1

Added AFTER the pre-specified results were seen, because V2's default floor (2) differs from V1's fixed grid (4). It is **not** part of the decision rule and must not be read as the primary result.

| Metric | V1 | V2 floor=4 | Difference (V2 floor=4 − V1) [95% CI] | Direction | Assessment |
|---|--:|--:|--:|---|---|
| `intent_duplicate_rate` **(primary)** | 0.377 | 0.292 | -0.085 [-0.122, -0.046] | lower better | better |
| `longitudinal_repetition` **(primary)** | 0.551 | 0.448 | -0.103 [-0.147, -0.059] | lower better | better |
| `change_followup_rate` **(primary)** | 0.628 | 0.836 | 0.208 [0.050, 0.378] | higher better | better |
| `min_per_category` **(primary)** | 4.000 | 4.000 | 0.000 [0.000, 0.000] | higher better | no clear difference |
| `synthetic_rating_quality` **(primary)** | 3.212 | 3.252 | 0.040 [-0.016, 0.091] | higher better | no clear difference |
| `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `semantic_near_dup_pair_rate` | 0.002 | 0.002 | 0.000 [-0.001, 0.001] | lower better | no clear difference |
| `semantic_diversity` | 0.679 | 0.727 | 0.048 [0.042, 0.054] | higher better | better |
| `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | higher better | no clear difference |
| `category_entropy` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | higher better | no clear difference |
| `relevance_proxy_new` | 0.301 | 0.344 | 0.043 [0.026, 0.059] | higher better | better |
| `replacement_churn` | 0.394 | 0.261 | -0.132 [-0.153, -0.110] | descriptive | descriptive |
| `mean_words` | 8.710 | 8.769 | 0.060 [-0.024, 0.136] | descriptive | descriptive |
| `invalid_output_rate` | 0.057 | 0.056 | -0.001 [-0.020, 0.017] | lower better | no clear difference |
| `llm_calls` | 4.333 | 1.000 | -3.333 [-3.333, -3.333] | descriptive | descriptive |
| `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | lower better | no clear difference |
| `generated_items` | 11.950 | 35.000 | 23.050 [22.961, 23.133] | descriptive | descriptive |
| `selection_ms` | 4.688 | 11.866 | 7.178 [6.840, 7.746] | descriptive | descriptive |

## Ablations (each vs full V2)

| Ablation | Metric | V2 | Ablated | Difference [95% CI] | Assessment (of the ablated variant) |
|---|---|--:|--:|--:|---|
| V2 −redundancy | `intent_duplicate_rate` | 0.282 | 0.378 | 0.097 [0.059, 0.136] | worse |
| V2 −redundancy | `longitudinal_repetition` | 0.430 | 0.405 | -0.025 [-0.066, 0.015] | no clear difference |
| V2 −redundancy | `change_followup_rate` | 0.844 | 0.869 | 0.025 [-0.036, 0.097] | no clear difference |
| V2 −redundancy | `min_per_category` | 3.761 | 3.878 | 0.117 [0.028, 0.217] | better |
| V2 −redundancy | `synthetic_rating_quality` | 3.258 | 3.267 | 0.009 [-0.034, 0.049] | no clear difference |
| V2 −redundancy | `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −redundancy | `semantic_near_dup_pair_rate` | 0.002 | 0.007 | 0.006 [0.004, 0.007] | worse |
| V2 −redundancy | `semantic_diversity` | 0.727 | 0.720 | -0.007 [-0.013, -0.002] | worse |
| V2 −redundancy | `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −redundancy | `category_entropy` | 0.998 | 0.999 | 0.001 [0.000, 0.002] | better |
| V2 −redundancy | `relevance_proxy_new` | 0.346 | 0.380 | 0.034 [0.021, 0.048] | better |
| V2 −redundancy | `replacement_churn` | 0.258 | 0.265 | 0.007 [-0.009, 0.024] | descriptive |
| V2 −redundancy | `invalid_output_rate` | 0.056 | 0.056 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −redundancy | `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −coverage | `intent_duplicate_rate` | 0.282 | 0.283 | 0.002 [-0.031, 0.034] | no clear difference |
| V2 −coverage | `longitudinal_repetition` | 0.430 | 0.418 | -0.012 [-0.046, 0.028] | no clear difference |
| V2 −coverage | `change_followup_rate` | 0.844 | 0.861 | 0.017 [-0.025, 0.058] | no clear difference |
| V2 −coverage | `min_per_category` | 3.761 | 2.272 | -1.489 [-1.583, -1.389] | worse |
| V2 −coverage | `synthetic_rating_quality` | 3.258 | 3.299 | 0.042 [0.010, 0.075] | better |
| V2 −coverage | `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −coverage | `semantic_near_dup_pair_rate` | 0.002 | 0.001 | -0.000 [-0.001, 0.000] | no clear difference |
| V2 −coverage | `semantic_diversity` | 0.727 | 0.728 | 0.001 [-0.003, 0.004] | no clear difference |
| V2 −coverage | `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −coverage | `category_entropy` | 0.998 | 0.962 | -0.036 [-0.039, -0.033] | worse |
| V2 −coverage | `relevance_proxy_new` | 0.346 | 0.360 | 0.014 [0.003, 0.027] | better |
| V2 −coverage | `replacement_churn` | 0.258 | 0.252 | -0.006 [-0.020, 0.008] | descriptive |
| V2 −coverage | `invalid_output_rate` | 0.056 | 0.056 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −coverage | `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −longitudinal | `intent_duplicate_rate` | 0.282 | 0.276 | -0.006 [-0.022, 0.009] | no clear difference |
| V2 −longitudinal | `longitudinal_repetition` | 0.430 | 0.460 | 0.030 [0.005, 0.056] | worse |
| V2 −longitudinal | `change_followup_rate` | 0.844 | 0.814 | -0.031 [-0.072, 0.008] | no clear difference |
| V2 −longitudinal | `min_per_category` | 3.761 | 3.750 | -0.011 [-0.083, 0.061] | no clear difference |
| V2 −longitudinal | `synthetic_rating_quality` | 3.258 | 3.260 | 0.002 [-0.025, 0.024] | no clear difference |
| V2 −longitudinal | `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −longitudinal | `semantic_near_dup_pair_rate` | 0.002 | 0.002 | 0.000 [-0.001, 0.001] | no clear difference |
| V2 −longitudinal | `semantic_diversity` | 0.727 | 0.726 | -0.001 [-0.003, 0.001] | no clear difference |
| V2 −longitudinal | `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −longitudinal | `category_entropy` | 0.998 | 0.998 | 0.000 [-0.001, 0.001] | no clear difference |
| V2 −longitudinal | `relevance_proxy_new` | 0.346 | 0.347 | 0.000 [-0.009, 0.010] | no clear difference |
| V2 −longitudinal | `replacement_churn` | 0.258 | 0.254 | -0.004 [-0.015, 0.006] | descriptive |
| V2 −longitudinal | `invalid_output_rate` | 0.056 | 0.056 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −longitudinal | `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −clinician_history | `intent_duplicate_rate` | 0.282 | 0.260 | -0.022 [-0.052, 0.008] | no clear difference |
| V2 −clinician_history | `longitudinal_repetition` | 0.430 | 0.405 | -0.025 [-0.059, 0.012] | no clear difference |
| V2 −clinician_history | `change_followup_rate` | 0.844 | 0.867 | 0.022 [-0.042, 0.081] | no clear difference |
| V2 −clinician_history | `min_per_category` | 3.761 | 3.944 | 0.183 [0.111, 0.261] | better |
| V2 −clinician_history | `synthetic_rating_quality` | 3.258 | 2.989 | -0.268 [-0.310, -0.232] | worse |
| V2 −clinician_history | `exact_duplicate_rate` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −clinician_history | `semantic_near_dup_pair_rate` | 0.002 | 0.002 | 0.000 [-0.000, 0.001] | no clear difference |
| V2 −clinician_history | `semantic_diversity` | 0.727 | 0.730 | 0.003 [-0.001, 0.008] | no clear difference |
| V2 −clinician_history | `categories_covered` | 1.000 | 1.000 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −clinician_history | `category_entropy` | 0.998 | 1.000 | 0.002 [0.001, 0.002] | better |
| V2 −clinician_history | `relevance_proxy_new` | 0.346 | 0.349 | 0.002 [-0.011, 0.015] | no clear difference |
| V2 −clinician_history | `replacement_churn` | 0.258 | 0.264 | 0.006 [-0.014, 0.028] | descriptive |
| V2 −clinician_history | `invalid_output_rate` | 0.056 | 0.056 | 0.000 [0.000, 0.000] | no clear difference |
| V2 −clinician_history | `retries` | 0.000 | 0.000 | 0.000 [0.000, 0.000] | no clear difference |

Units: `invalid_output_rate` is rejected category calls / calls for V1 and invalid items / generated items for V2 (different units; not directly comparable). Tokens, cost and API latency were not measured (no API calls); `selection_ms` is local compute only.

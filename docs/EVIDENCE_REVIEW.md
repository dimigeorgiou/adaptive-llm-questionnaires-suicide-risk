# Evidence Review — adaptive questioning in suicide-risk assessment contexts

Purpose: separate what is **established** from what this fork **assumes**. The review covers
authoritative sources (official bodies, original peer-reviewed papers, systematic or large reviews)
relevant to the upstream project and to V2. It is not a clinical guideline. Nothing here licenses
automated risk assessment.

Sources were checked on 2026-09-25 (bibliographic details against PubMed, publisher pages or the
issuing body). Where only part of a source could be checked, the entry says so.

## Evidence-status scale

| Status | Meaning |
|---|---|
| **established** | supported by official guidance or consistent validation evidence in the relevant population |
| **supported** | supported by peer-reviewed evidence for the *general* concept, not for this implementation |
| **experimental** | a design choice of this fork; plausible, but not validated |
| **unsupported** | no evidence located; must not be relied on |

## 1. Sources reviewed

### 1.1 Validated screening and assessment instruments

- **NIMH Ask Suicide-Screening Questions (ASQ) Toolkit.** National Institute of Mental Health.
  https://www.nimh.nih.gov/research/research-conducted-at-nimh/asq-toolkit-materials
  - The ASQ information sheet describes a brief screen "for use by non-psychiatric clinicians" with
    "sound psychometric properties for youth and adult medical patients".
  - The toolkit pairs screening with a clinical pathway. A positive screen requires a brief suicide
    safety assessment (a Brief Suicide Safety Assessment guide and worksheets are provided), which
    determines whether a full mental-health evaluation is needed.
  - For an acute positive screen, the information sheet states that the "patient requires a STAT
    safety/full mental health evaluation" and "cannot leave until evaluated for safety".
  - The information sheet itself does not state licensing or modification terms. Deployments must
    confirm usage terms and approved translations with NIMH materials.
- **Horowitz LM, Bridge JA, Teach SJ, et al.** Ask Suicide-Screening Questions (ASQ): a brief
  instrument for the pediatric emergency department. *Arch Pediatr Adolesc Med.* 2012;166(12).
  PMID 23027429. In a pediatric ED sample, four items reached sensitivity 96.9% (95% CI
  91.3–99.4) and specificity 87.6% (84.0–90.5) against a reference-standard questionnaire.
- **Horowitz LM, et al.** Validation of the Ask Suicide-Screening Questions for adult medical
  inpatients: a brief tool for all ages. *Psychosomatics.* 2020. PMID 32487323. 727 adult medical
  inpatients across 4 hospitals, with the Adult Suicidal Ideation Questionnaire as the criterion
  standard.
- **Posner K, Brown GK, Stanley B, et al.** The Columbia–Suicide Severity Rating Scale: initial
  validity and internal consistency findings from three multisite studies with adolescents and
  adults. *Am J Psychiatry.* 2011;168(12):1266–1277. doi:10.1176/appi.ajp.2011.10111704.

### 1.2 Computerized adaptive testing (CAT) for suicide-related constructs

- **Gibbons RD, Weiss DJ, Frank E, Kupfer D.** Computerized adaptive diagnosis and testing of
  mental health disorders. *Annu Rev Clin Psychol.* 2016;12:83–104. PMID 26651865.
  Review: adaptive testing is built on multidimensional item response theory (severity) and random
  forests (diagnosis), using **item banks calibrated on large samples**.
- **Gibbons RD, Kupfer D, Frank E, Moore T, Beiser DG, Boudreaux ED.** Development of a
  Computerized Adaptive Test Suicide Scale — the CAT-SS. *J Clin Psychiatry.* 2017;78(9):1376–1382.
  doi:10.4088/JCP.16m10922. Measures the latent suicide dimension with a mean of about 10
  adaptively selected items, drawn from a psychometrically calibrated bank harmonised across
  suicide, depression and anxiety items.
- **Validation of a Computerized Adaptive Test Suicide Scale (CAT-SS) among United States Military
  Veterans.** *PLOS ONE* (e0261920). doi:10.1371/journal.pone.0261920. External validation in a
  different population. Only the title and journal were checked; see the source for methods.
- **King CA, Brent D, Grupp-Phelan J, et al.** Prospective development and validation of the
  Computerized Adaptive Screen for Suicidal Youth. *JAMA Psychiatry.* 2021;78(5):540–549.
  doi:10.1001/jamapsychiatry.2020.4576. Developed and prospectively validated in pediatric EDs
  (PECARN). At 80% specificity, sensitivity for predicting a suicide attempt was about 82–83%, and
  the AUC in the independent validation cohort was 0.87.

### 1.3 Prediction limits and guidance on risk tools

- **Franklin JC, Ribeiro JD, Fox KR, et al.** Risk factors for suicidal thoughts and behaviors: a
  meta-analysis of 50 years of research. *Psychol Bull.* 2017;143(2):187–232. 365 studies. Prediction
  was only slightly better than chance for all outcomes and had not improved over 50 years.
- **NICE guideline NG225** (2022). Self-harm: assessment, management and preventing recurrence.
  https://www.nice.org.uk/guidance/ng225
  - Do not use risk-assessment tools and scales to predict future suicide or repetition of
    self-harm.
  - Do not use them to decide who is offered treatment or who is discharged.
  - Do not use global low/medium/high risk stratification for those purposes.
  - Focus instead on needs and safety, with a risk *formulation* as part of every psychosocial
    assessment.

### 1.4 Human-in-the-loop, clinical AI, and LLM behaviour

- **WHO.** *Ethics and governance of artificial intelligence for health: WHO guidance.* 2021. Six
  principles: protect autonomy; promote well-being, safety and the public interest; ensure
  transparency, explainability and intelligibility; foster responsibility and accountability;
  ensure inclusiveness and equity; promote responsive and sustainable AI.
- **US FDA.** *Clinical Decision Support Software: Guidance for Industry and FDA Staff*, final,
  September 2022 (Federal Register 2022-20993). Interprets the 21st Century Cures Act criteria for
  non-device CDS, including that a health-care professional must be able to review the basis of a
  recommendation independently. Cited for principle only: regulatory status depends on
  jurisdiction and intended use, and this fork makes no regulatory claim.
- **McBain RK, et al.** Evaluation of alignment between large language models and expert clinicians
  in suicide risk assessment. *Psychiatric Services*, 2025. doi:10.1176/appi.ps.20250086.
  Three chatbots answered 30 clinician-graded suicide-related queries 100 times each (9,000
  responses) to test how their responses aligned with expert risk levels. Motivates keeping LLMs out
  of risk judgements. See the paper for the exact findings.

### 1.5 Engineering methods used by V2 (not clinical evidence)

- **Carbonell J, Goldstein J.** The use of MMR, diversity-based reranking for reordering documents
  and producing summaries. *SIGIR '98* (reprinted in *ACM SIGIR Forum*,
  doi:10.1145/3130348.3130369). Combines relevance with
  novelty to reduce redundancy.
- **Reimers N, Gurevych I.** Sentence-BERT (EMNLP 2019) and Making monolingual sentence embeddings
  multilingual using knowledge distillation (EMNLP 2020). These are the basis of the optional
  `paraphrase-multilingual-MiniLM-L12-v2` backend.

## 2. Design concepts and their evidence status

| Concept (upstream or V2) | Evidence status | Basis / caveat |
|---|---|---|
| Validated brief screening (e.g. ASQ) with a defined follow-up pathway run by trained clinicians | **established** | NIMH toolkit; Horowitz 2012, 2020. *Not implemented here*; V2 only provides an adapter interface and never alters instrument items |
| Positive or acute screens handled by an established clinical protocol, not software | **established** | NIMH pathway; NICE NG225. *This software implements no escalation workflow and must not be used as one* |
| Not using scores or tools to predict suicide or to decide treatment or discharge | **established** (guidance) | NICE NG225; Franklin 2017. V2 outputs no risk estimate or stratification (tested) |
| Adaptive item selection *in general* | **supported** | CAT literature (Gibbons 2016, 2017; King 2021). There, it relies on calibrated item banks and IRT models, which this project does not have |
| Expected information gain for item selection | **unsupported here** | Needs a calibrated measurement model; `InformationGainEstimator` is not implemented/calibrated in V2 |
| LLM-generated questionnaire items (upstream and V2) | **experimental** | No validation of LLM-generated items as measurement instruments was located; items are unvalidated prompts for clinical conversation |
| Clinician review of every generated item (human in the loop) | **supported** (principle) | WHO 2021; FDA CDS 2022 (independent review). The upstream pilot's operational gating is not verifiable from the code |
| Upstream five-dimension Likert rating of items (Coherence … Engagement) | **experimental** | Face-valid perceptions of item quality; no psychometric validation located; not outcomes |
| Upstream adaptation rule $w' = w + 0.2(f-4)$, 2 lowest replaced | **experimental** | Engineering rule; also, the "fuzzy logic" described upstream is not present in the code (BASELINE_AUDIT.md) |
| V2 ranking equation and coefficients | **experimental** | Not clinically validated; not backend-invariant (V2_EVALUATION.md) |
| Semantic or lexical redundancy penalty | **experimental** (engineering) | Carbonell & Goldstein 1998 for the general idea; not a clinical construct; thresholds not validated |
| Domain coverage floors and priorities | **experimental** | Must be set by clinicians; defaults are engineering values |
| ChangeConsistencySignal | **experimental** | A review flag only; low specificity under the lexical backend (13 of 15 false flags on the synthetic bank) |
| Shrunk-EMA clinician-history score | **experimental** | Chosen by simulation (weighting_simulation.md), not by clinical data |
| Adaptive session length (disabled by default) | **experimental** | Engineering stop conditions only; never a clinical stopping rule |
| "V2 improves clinical accuracy" | **unsupported** | Clinical accuracy was not evaluated; no outcome labels exist |

## 3. Implications for this repository

1. The adaptive system sits *around* validated practice. It proposes conversation prompts for
   clinicians and must not replace a validated screen, its wording, or its follow-up pathway.
2. Any future claim of predictive or diagnostic value needs appropriate reference outcomes, a
   pre-registered protocol and ethics approval. Given NICE NG225 and Franklin 2017, it would also
   need a very strong justification for using any score predictively at all.
3. If adaptive *measurement* (rather than prompt selection) is ever the goal, the literature points
   to calibrated item banks and IRT-based CAT. Those are validated instruments, and an LLM-generated
   item stream is not one.

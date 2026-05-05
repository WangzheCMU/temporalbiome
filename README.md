# temporalbiome — Reference implementation of MicrobiomeRiskScore (MRS)

Companion code repository for **"Gut microbiome temporal dynamics predict
healthcare-associated infections days before clinical detection: a multi-database
deep learning study in cancer patients"** (Ma *et al.*, *npj Digital Medicine*).
The package re-implements the **MicrobiomeRiskScore (MRS)** architecture and the
statistical apparatus that surrounds its evaluation, framed as a strictly
computational retrospective pipeline that operates on publicly released,
de-identified resources only.

---

## Synopsis

Healthcare-associated infections (HAI) remain the leading complication of
oncology-directed surgical and cytotoxic therapy, yet every fielded prediction
system to date has reasoned exclusively over electronic-health-record (EHR)
features measured **after** physiological breakdown. MRS inverts that ordering by
operating on the **causally upstream substrate**—the gut microbiome—through a
treatment-aware Transformer that ingests irregularly-sampled longitudinal 16S
rRNA profiles, fuses their representation with hourly clinical time-series via
bidirectional cross-attention, and emits a temperature-calibrated probabilistic
risk score with attribution. On the Memorial Sloan Kettering allogeneic
haematopoietic-cell-transplantation cohort the model attains AUROC
$0.921 \pm 0.009$, transfers to MIMIC-IV ($0.908 \pm 0.010$) and eICU-CRD
($0.886 \pm 0.013$) with cross-treatment-modality concordance of $85.6\%$, and
crosses the high-risk alert threshold a median of **35.8 hours** before the
clinical team first suspects infection. The paper articulates this as a
*surveillance-of-the-source* principle: whenever a predictable biological cascade
separates substrate disruption from clinical manifestation, temporal monitoring
of the substrate dominates downstream clinical-feature aggregation.

This repository ships the deterministic foundations and the neural architecture
that realise that principle. Training loops, the 19-baseline benchmark, SHAP
attribution, the three-tier validation pipeline, and the credentialed PhysioNet
adapters are sequenced for subsequent iterations on the established roadmap
(§ 11). All source operates on synthetic NumPy / PyTorch tensors; no protected
patient information is materialised or distributed by this package.

---

## 1 · Architecture

The MRS forward pass implements

$$
r(t) \;=\; \sigma\!\bigl(f_{\theta}(\mathbf{z})\,/\,\tau\bigr)
\quad\text{with}\quad
\mathbf{z} \;=\; \mathrm{Pool}\!\bigl(\mathbf{H}^{(f)}\bigr),
$$

where $\mathbf{H}^{(f)} \in \mathbb{R}^{B \times T \times 2d}$ is the gated
fusion of the bi-directional cross-attention streams and $\tau > 0$ is a
calibration temperature fitted on the validation set after end-to-end training.
The three composing modules are summarised below; equation numbers refer to the
companion paper.

### 1.1 Composite positional encoding

For sample index $t_i$ (in absolute hospital hours since admission) the encoder
adds three additive components,

$$
\mathbf{p}_{t_i} \;=\; \mathbf{p}_{t_i}^{\text{time}} \,+\, \mathbf{p}_{t_i}^{\text{treat}} \,+\, \mathbf{p}_{t_i}^{\text{phase}}, \qquad \text{(Eq.\ 1)}
$$

with continuous-time sinusoidal entries

$$
\mathbf{p}_{t_i}^{\text{time}}[2j] \;=\; \sin\!\bigl(t_i / 10000^{\,2j/d}\bigr), \qquad
\mathbf{p}_{t_i}^{\text{time}}[2j+1] \;=\; \cos\!\bigl(t_i / 10000^{\,2j/d}\bigr).
$$

The treatment-phase component is a 32-state learned table indexing chemotherapy
cycle, days-since-surgery bin, or radiation fraction; the disease-phase component
is a 4-state table indexing pre-treatment / active / recovery / maintenance.
Tables are orthogonally initialised so that the three components do not collide
at $t_i = 0$.

### 1.2 Temporal Microbiome Encoder (TME)

| Hyper-parameter | Symbol | Default | Source |
|---|---|---|---|
| Hidden dimension | $d$ | 256 | Methods § 4.4.1 |
| Number of layers | $L$ | 4 | Methods § 4.4.1 |
| Attention heads | $K$ | 8 | Methods § 4.4.1 |
| Feed-forward dimension | $d_{\text{ff}}$ | 1024 | Inferred ($4d$) |
| Dropout | $p$ | 0.20 | Table 5 |
| Activation | — | GELU | Methods § 4.4.1 |
| Microbiome panel size | $d_m$ | 274 | Methods § 4.2 |

The microbiome time-series $\mathbf{X}^{(m)} \in \mathbb{R}^{T \times d_m}$ is
linearly projected into the embedding space, summed with $\mathbf{p}_{t_i}$, and
processed by an $L$-layer scaled-dot-product Transformer encoder

$$
\mathrm{Attention}(Q, K, V) \;=\; \mathrm{softmax}\!\Bigl(\tfrac{Q K^{\top}}{\sqrt{d/K}}\Bigr) V, \qquad \text{(Eq.\ 3)}
$$

with `src_key_padding_mask` semantics propagated end-to-end so that
variable-length cohorts are batched without information leakage from padded
positions.

### 1.3 Clinical Feature Fusion (CFF)

Clinical features $\mathbf{X}^{(c)} \in \mathbb{R}^{T_c \times d_c}$ ($d_c = 87$;
six-hour-aggregated vital signs, laboratory values, drug-class flags, and
procedure indicators) are encoded by a two-layer Transformer that re-uses the
same composite positional encoding. The fusion stage applies symmetric
cross-attention,

$$
\mathbf{H}^{(m \to c)} \;=\; \mathrm{CrossAttn}\!\bigl(\mathbf{H}^{(m)}, \mathbf{H}^{(c)}, \mathbf{H}^{(c)}\bigr),
\qquad
\mathbf{H}^{(c \to m)} \;=\; \mathrm{CrossAttn}\!\bigl(\mathbf{H}^{(c)}, \mathbf{H}^{(m)}, \mathbf{H}^{(m)}\bigr), \qquad \text{(Eq.\ 4–5)}
$$

followed by a learned sigmoid gate that dynamically rebalances the modalities,

$$
\mathbf{H}^{(f)} \;=\; \sigma\!\Bigl(\mathbf{W}_g \bigl[\mathbf{H}^{(m \to c)};\, \mathbf{H}^{(c \to m)}\bigr]\Bigr) \,\odot\, \bigl[\mathbf{H}^{(m \to c)};\, \mathbf{H}^{(c \to m)}\bigr]. \qquad \text{(Eq.\ 6)}
$$

Each direction wraps the multi-head attention in a pre-/post-LayerNorm residual
block with its own GELU feed-forward sub-layer, mirroring `nn.TransformerDecoderLayer`
without the unused causal-mask path. Misaligned microbiome and clinical sequence
lengths are reconciled by truncating to the shorter dimension before gating.

### 1.4 Real-Time Risk Score Generator (RSG)

The fused stream is collapsed to a single vector by attention-weighted temporal
pooling,

$$
\mathbf{z} \;=\; \sum_{t=1}^{T} \alpha_{t}\,\mathbf{H}^{(f)}_{t}, \qquad
\alpha_{t} \;=\; \mathrm{softmax}\!\bigl(\mathbf{w}^{\top} \mathbf{H}^{(f)}_{t}\bigr),
$$

with padded positions deterministically excluded via a `-∞` mask before the
softmax. A two-layer feed-forward network with $p = 0.2$ dropout produces the
raw logit $f_{\theta}(\mathbf{z})$, which is calibrated by a single learnable
temperature $\tau$ on the held-out validation set:

$$
r(t) \;=\; \sigma\!\bigl(f_{\theta}(\mathbf{z})\,/\,\tau\bigr) \in [0, 1]. \qquad \text{(Eq.\ 7)}
$$

The temperature is implemented as a non-trainable module buffer that takes
effect only in `eval()` mode, leaving the loss surface untouched during
optimisation.

### 1.5 Training objective

End-to-end optimisation minimises a focal cross-entropy that addresses the
8–15 % class imbalance characteristic of HAI cohorts,

$$
\mathcal{L}(\theta) \;=\; -\alpha (1-\hat{y})^{\gamma} \log(\hat{y})\,y \;-\; (1-\alpha)\,\hat{y}^{\gamma} \log(1-\hat{y})\,(1-y), \qquad \text{(Eq.\ 8)}
$$

with $\alpha = 0.75$ and $\gamma = 2$. The closed-form analytic gradient with
respect to the logit is exposed as `focal_cross_entropy_gradient` so that
non-PyTorch consumers (e.g. boosting baselines in the upcoming Turn 3) can
back-propagate without a second autograd dependency.

---

## 2 · Statistical inference toolkit

The package implements the full statistical envelope reported in Table 1 of the
manuscript and exposes each test as a pure-NumPy primitive:

| Procedure | Public symbol | Reference |
|---|---|---|
| Empirical AUROC (mid-rank) | `auroc` | Sun & Xu (2014) |
| Paired DeLong test | `paired_delong` | DeLong *et al.* (1988); Methods § 4.6 |
| Family-wise Bonferroni adjustment | `bonferroni_adjust` | $\alpha_{\text{adj}} = 0.05/19 = 0.0026$ |
| Patient-clustered bootstrap | `patient_clustered_bootstrap` | 1000 resamples; Methods § 4.6 |
| BCa percentile interval | `bca_interval` | Efron & Tibshirani (1993) |
| Cohen's $h$ (proportion effect size) | `cohens_h` | $h = 2\arcsin\sqrt{p_1} - 2\arcsin\sqrt{p_2}$ |
| Brier score | `brier_score` | Brier (1950) |
| Expected calibration error | `expected_calibration_error` | Naeini *et al.* (2015) |
| Reliability bins | `reliability_bins` | Quantile-binned predictive vs. empirical |
| Calibration slope | `calibration_slope` | Platt (1999) |
| Temperature scaling | `fit_temperature`, `apply_temperature` | Guo *et al.* (2017) |
| Decision-curve net benefit | `decision_curve` | Vickers & Elkin (2006) |
| Number-needed-to-screen / treat | `number_needed_to_screen`, `number_needed_to_treat` | Pocock (2013) |

The DeLong implementation uses the $O((m+n)\log(m+n))$ Sun–Xu mid-rank
algorithm; `paired_delong` returns both the unadjusted two-sided $p$-value and
the Bonferroni-19 family-wise rate, matching the reporting convention adopted in
the manuscript ("primary comparisons (MSK) were significant after adjusting for
multiple comparisons"). `patient_clustered_bootstrap` resamples patient
identifiers with replacement and recomputes the user-supplied statistic on each
draw, addressing the dependence violation that ordinary i.i.d. bootstraps would
otherwise commit on longitudinal data.

---

## 3 · Software architecture

The package follows a single-namespace, leading-underscore module layout. The
public API is re-exported from `temporalbiome/__init__.py`; module names are
deliberately spelled out for unambiguous mapping to Methods sub-sections of the
manuscript.

```
src/temporalbiome/
├── _types.py                            value-records, Final[…] hyperparameters,
│                                        domain-specific exception classes
│
├── statistical foundations  ─────────  Methods § 4.6
│   ├── _delong.py                       Sun–Xu mid-rank paired DeLong + Bonferroni
│   ├── _bootstrap.py                    patient-clustered bootstrap, BCa, Cohen's h
│   ├── _calibration.py                  Newton temperature solve, Platt, Brier, ECE
│   ├── _decision_curve.py               net benefit, NNS, NNT
│   └── _risk_tiers.py                   four-tier mapping (low / mod / high / critical)
│
├── microbiome processing  ────────────  Methods § 4.2
│   ├── _taxonomy.py                     ASV→genus rollup, species exceptions
│   ├── _diversity.py                    Shannon, Pielou, observed richness, ratios
│   └── _harmonise.py                    CLR z-score and centre-aware harmonisation
│
├── clinical processing  ──────────────  Methods § 4.3
│   └── _clinical_panel.py               87-feature schema, 6-h aggregation, imputation
│
├── temporal alignment & cohorting  ───  Methods § 4.1, § 4.5
│   ├── _alignment.py                    Δt utilities, simulated-integration matching
│   ├── _composite_pe.py                 NumPy reference for Eq. 1–2
│   └── _cohort_split.py                 stratified, patient-disjoint 60/20/20
│
├── learning objective  ───────────────  Methods § 4.5
│   └── _focal_loss.py                   forward + closed-form gradient (Eq. 8)
│
└── neural architecture  ──────────────  Methods § 4.4
    ├── _torch_pe.py                     CompositePositionalEncoding nn.Module
    ├── _temporal_microbiome_encoder.py  TME: 4-layer Transformer encoder
    ├── _clinical_feature_fusion.py      CFF: clinical encoder + bidir cross-attn + gating
    ├── _risk_score_generator.py         RSG: pooling + FFN + temperature buffer (Eq. 7)
    └── _mrs_network.py                  composed MicrobiomeRiskScoreNetwork

tests/
├── property_based/      Hypothesis: simplex axioms, rank preservation, spread compression
├── invariants/          mathematical equalities, mask isolation, eval determinism
├── shape/               tensor-dimension contracts (TME, CFF, RSG, MRS, panel, PE)
├── regression/          Cohen's h(0.921, 0.854), parameter-count envelope, BCa textbook
├── integration/         cohort partition + full MRS forward / backward / tier classification
└── overfit_one_batch/   MRS drives loss to <10 % of initial on an 8-sample batch
```

The implementation deliberately avoids in-line documentation strings so that the
audit surface is the identifier system itself; module names, function names,
named exceptions, and the public `__all__` lists carry the semantic load that
docstrings would normally occupy. The full public surface comprises **111
symbols** (99 foundational + 12 neural) and is enumerated in
`temporalbiome/__init__.py`.

---

## 4 · Installation

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Three install profiles are available:

| Extra | Packages added | When to use |
|---|---|---|
| *(none)* | NumPy + SciPy | foundations only — statistical / preprocessing primitives |
| `[neural]` | `torch>=2.1` | deploy the MRS architecture without running the suite |
| `[test]` | `pytest>=8.0`, `hypothesis>=6.100`, `torch>=2.1` | development and verification |

`requires-python = ">=3.9"`. The reference environment is documented in § 10.

---

## 5 · Reproducibility envelope

The table below contrasts paper-reported quantities with what is verified end-to-end
on the current commit. Cells marked *(roadmap)* are intentionally deferred to the
turns enumerated in § 11 — they require components (training loop, baselines, real
data, attribution) that the present package does not yet ship.

| Manuscript quantity | Paper claim | Status in this commit |
|---|---|---|
| MRS AUROC on MSK HCT (primary) | $0.921 \pm 0.009$ | *(roadmap — Turn 2 training)* |
| MRS AUROC on MIMIC-IV (transfer) | $0.908 \pm 0.010$ | *(roadmap — Turn 6 adapters)* |
| MRS AUROC on eICU-CRD (transfer) | $0.886 \pm 0.013$ | *(roadmap — Turn 6 adapters)* |
| Lead-time before clinical suspicion | $35.8\,\text{h}$ | helper `lead_time_threshold_crossing` ready |
| Cross-modality concordance (HCT→CRC) | $85.6\,\%$ | *(roadmap — Turn 3)* |
| Total trainable parameters | $\sim 8.2\,\text{M}$ | $6.83\,\text{M}$ at default config (envelope $4\text{–}12\,\text{M}$) |
| Cohen's $h$(MRS, XGBoost) | $0.48$ (manuscript) | $0.214$ (canonical formula on the same proportions) |
| TME default config (d, L, K) | $(256, 4, 8)$ | matched |
| Focal loss $(\alpha, \gamma)$ | $(0.75, 2)$ | matched |
| Optimiser, schedule | AdamW, cosine, lr $10^{-4}$, wd $10^{-2}$ | constants exposed; loop in Turn 2 |
| Random seeds reported | 15 fixed seeds | exposed as `PAPER_RANDOM_SEEDS` |

**Note on Cohen's $h$**: applying the canonical formula
$h = 2\arcsin\sqrt{0.921} - 2\arcsin\sqrt{0.854}$ yields $0.21432$, not the
$0.48$ quoted in the manuscript. The discrepancy is preserved transparently in
`tests/regression/test_cohens_h_paper_value.py`; the test asserts agreement with
the closed-form value rather than with the published numeral.

---

## 6 · Verification suite

```bash
pytest -q tests/property_based      # 9 tests   — ~1.5 s
pytest -q tests/invariants          # 27 tests  — ~0.5 s
pytest -q tests/shape               # 41 tests  — ~1.5 s
pytest -q tests/regression          # 27 tests  — ~0.5 s
pytest -q tests/integration         # 11 tests  — ~1.0 s
pytest -q tests/overfit_one_batch   # 3 tests   — ~2.5 s
pytest -q                           # 118 tests — ~7 s end-to-end on CPU
```

Six test families cover orthogonal failure modes:

- **property-based** — Hypothesis-driven axioms over simplex draws (Shannon
  $\geq 0$ and $\leq \log S$, Pielou $\in [0, 1]$, observed richness $\leq S$),
  rank preservation under temperature scaling on well-separated logits, and
  monotonic spread compression with rising temperature;
- **invariants** — bitwise reductions ($\gamma = 0$ recovers standard CE, $\tau$
  applied to identical scorers yields $z = 0$), centre-aware ComBat zeros
  per-centre means while preserving the biological signal, attention pooling
  with `-∞` masking is provably invariant to perturbations of padded positions;
- **shape** — tensor-dimension contracts on TME ($B \times T \times d$), CFF
  ($B \times \min(T_m, T_c) \times 2d$), RSG (per-sample logit and probability
  in $[0, 1]$), the 87-feature clinical panel, and orthonormal embedding tables;
- **regression** — Cohen's $h$ closed-form on the manuscript's AUROC pair,
  parameter-count envelope and depth-monotonicity, BCa containment of the point
  estimate on textbook data, decision-curve net benefit at the prevalence
  threshold matching the closed-form definition;
- **integration** — outcome-stratified, patient-disjoint 60/20/20 split with
  $|\Delta\,\text{prevalence}| < 0.5\,\%$ across folds, full MRS forward +
  backward + tier classification on synthetic four-sample batches, temperature
  buffer activation in eval mode;
- **overfit-one-batch** — the canonical neural-correctness probe: with dropout
  disabled, AdamW must drive focal cross-entropy below ten percent of the
  initial value on an eight-sample batch and produce per-sample predictions on
  the correct side of $0.5$, demonstrating that the gradient flow from RSG → CFF
  → TME → composite positional encoding is connected end to end.

The package's own continuous integration only runs the local CPU suite; GPU
benchmarks and 15-seed sweeps are deferred to Turn 2.

---

## 7 · Reporting standards compliance

The original manuscript follows three reporting frameworks; this implementation
preserves the items that fall within a reproducibility-code repository's remit:

- **TRIPOD+AI** (Collins *et al.*, 2024): the 27-item checklist is documented in
  the manuscript's supplementary files; the items concerning code availability,
  random-seed disclosure, hyper-parameter grids, and pre-specified outcome
  measures are honoured by `PAPER_RANDOM_SEEDS`, the explicit
  `Final[…]` hyper-parameter constants in `_types.py`, and the
  `parameter_summary()` introspection on `MicrobiomeRiskScoreNetwork`.
- **MI-CLAIM** (Norgeot *et al.*, 2020): six-component checklist for clinical AI
  reporting; the data-handling, technical, and evaluation components are
  addressed at the code level. Components requiring prospective deployment do
  not apply to a retrospective computational analysis.
- **STORMS** (Mirzayi *et al.*, 2021): microbiome-specific reporting standards;
  the relevant code-level items — bioinformatic pipeline version (QIIME2
  `2024.5`), reference database (`SILVA 138.2`), batch-effect correction
  (`neuroCombat 0.2.12`-equivalent in `_harmonise.py`), and minimum read count
  per sample (`MINIMUM_READS_PER_SAMPLE = 1000`) — are exposed as named
  constants in `_types.py` and reflected in `_taxonomy.filter_low_read_samples`.

In keeping with U.S. Common Rule 45 CFR 46.104(d)(4) and equivalent
international guidance, the underlying study constitutes a **secondary
computational analysis of previously published, de-identified data**, and no
separate institutional review board process or additional informed consent is
applicable.

---

## 8 · Pre-registered analysis plan (paraphrased)

The manuscript pre-registered the following hypotheses, which the present
package's downstream turns are required to evaluate without modification once
trained weights are produced:

1. **Primary hypothesis**: AUROC of MRS on the MSK HCT test set will exceed
   the strongest clinical-only baseline by at least $0.05$ AUROC points.
2. **Primary metric**: AUROC compared via the DeLong test with Bonferroni
   correction across all 19 baseline comparisons ($\alpha_{\text{adj}} = 0.0026$).
3. **Secondary metrics** (pre-specified): AUPRC, sensitivity at $90\,\%$
   specificity, Brier score, and lead time.
4. **Pre-specified subgroup analyses**: by treatment type (chemotherapy /
   surgery / combined), infection type (BSI / SSI / CDI / UTI), age stratum, and
   inter-centre transfer direction.
5. **Statistical power**: planned MSK test set ($N_{\text{pos}} = 20$,
   $N_{\text{neg}} = 150$) yields $\sim 84\,\%$ power to detect an AUROC
   difference of $0.067$ at $\alpha = 0.05$ via two-sided DeLong; the Tier-2
   test set ($N_{\text{pos}} = 736$, $N_{\text{neg}} = 4{,}164$) yields
   $> 99\,\%$ power for the same effect.

Items (1)–(3) become enforceable once Turn 2 (training) and Turn 3 (baselines)
land. Items (4) and (5) are addressed by the existing
`stratified_patient_disjoint_split`, `paired_delong`, `bonferroni_adjust`, and
`patient_clustered_bootstrap` primitives.

---

## 9 · Ethics and data governance

This repository performs **no new human-subject data collection, no new
biological-specimen sequencing, and no wet-laboratory experimentation**. All
test inputs are synthetic NumPy and PyTorch tensors generated from seeded
pseudorandom sources. The downstream turns that integrate real-world data will
do so exclusively through opt-in adapters to the following publicly released
resources, each governed by its originating consent framework:

- **Memorial Sloan Kettering allogeneic-HCT microbiome cohort** — accessible by
  application to the originating institution; raw 16S V4–V5 sequence data
  released under the original publication's accession number.
- **MIMIC-IV v3.1** — released by the Beth Israel Deaconess / MIT
  Laboratory for Computational Physiology via PhysioNet; access requires the
  CITI human-subjects research training programme and a credentialed
  PhysioNet account.
- **eICU Collaborative Research Database** — released by the Philips eICU
  programme via PhysioNet under the same credentialed access regime.
- **curatedMetagenomicData** — released through Bioconductor.
- **CRCbiome / NORCCAP** — accessible by application to the NORCCAP study group.

Within the scope of this code repository the present analysis is best
characterised as a **computational retrospective analysis on publicly available,
de-identified datasets**, exempt from prospective IRB oversight under U.S.
Common Rule 45 CFR 46.104(d)(4) and equivalent international guidance.

---

## 10 · Computational environment

Reference environment for the verification suite (paper-reported environment
indicated where it differs):

| Component | Reference (this package) | Paper environment |
|---|---|---|
| Operating system | Darwin (macOS) | Ubuntu 22.04 LTS |
| Python | 3.9.6 | 3.10.12 |
| NumPy | 2.0.x | 1.26.3 |
| SciPy | 1.13.x | 1.12.0 |
| PyTorch | 2.8.x | 2.2.1 |
| pytest | 8.4.x | — |
| Hypothesis | 6.141.x | — |
| Hardware | CPU only (suite < 15 s) | NVIDIA A100 80 GB HBM2e |

The full pytest suite is bit-deterministic across re-runs at fixed
`torch.manual_seed`, `numpy.random.default_rng(seed)`, and `hypothesis.seed`
under the registered settings.

---

## 11 · Implementation roadmap

| Turn | Scope | Status |
|:---:|---|:---:|
| 0 | Statistical / microbiome / clinical foundations | shipped |
| 1 | Neural architecture (TME, CFF, RSG, composed MRS) | shipped |
| 2 | AdamW + cosine schedule + 15-seed sweep + early stopping + Bayesian hyper-parameter search | queued |
| 3 | 19-baseline benchmark (LACE / NEWS2 / SOFA / APACHE-II; LR / RF / SVM / XGBoost / LightGBM; DeepMicro / PopPhy-CNN / GDmicro / MetaNN; LSTM / TCN; phyLoSTM / MicroProphet) | queued |
| 4 | SHAP + attention-based attribution; reliability diagrams; Top-20 feature reporting (Table 6) | queued |
| 5 | Three-tier validation pipeline (gold-standard / transfer / clinical-only ablation); FGSM adversarial robustness; 50 % missingness perturbation analysis (Tables 4, 7, 10, 11, 12) | queued |
| 6 | Credentialed PhysioNet adapters (MIMIC-IV, eICU-CRD); QIIME2 / DADA2 wrappers; 250-trial random-search + Bayesian-optimisation hyper-parameter pipeline | queued |

Each subsequent turn extends the existing package with new modules under the
flat `temporalbiome` namespace and adds matching test categories (regression
tables, training-curve smoke checks, baseline parity, attribution stability)
without altering the foundations established here.

---

## 12 · License

Released for research purposes accompanying the manuscript. The source includes
no proprietary patient data, no third-party code, and no model weights derived
from restricted data sources.

# Datasheet: SynthPGx-DPYD-SAS v1

> **SYNTHETIC. In silico cohort validation. Not clinical evidence.**

Written in the style of *Datasheets for Datasets* (Gebru et al.). Every figure
here is produced by `generate.py` and `validate.py` at seed 42 and can be
re-derived from `qc/qc_metrics.json` and `qc/generation_manifest.json`.

---

## Motivation

**Why was this dataset created?**

Fluoropyrimidines (5-fluorouracil, capecitabine) are broken down by the enzyme
DPD, encoded by *DPYD*. Carriers of reduced-function variants accumulate the
drug and can suffer severe or fatal toxicity. The CPIC guideline that tells
clinicians what to do was built largely from European-ancestry data, and the
variants it recommends testing were characterised in that population.

South Asian patients carry a different distribution of *DPYD* variation. A
CPIC-only screen reads some of those patients as normal. There is no public
dataset anywhere linking South Asian genotypes to fluoropyrimidine outcomes, so
there is nothing to test an engine against. This dataset is a stand-in for that
missing test set: fully labelled, population-aware, and reproducible from a
seed.

**Who created it and who funded it?** Anukriti AI Private Limited, Kerala,
India, internally funded.

**What task does it support?** Validating a deterministic pharmacogenomics
engine (regression, ground-truth correctness, refusal behaviour, population
equity), and training and evaluating the ML components beside it. It does not
support any clinical or regulatory claim.

---

## Composition

**What do the instances represent?** Virtual patients. Each has a *DPYD*
genotype, clinical covariates, a deterministic rule label, and a simulated drug
outcome. No instance corresponds to any real person.

### Cohorts

| Cohort | n | Purpose |
|---|---|---|
| Equity panel | 10,000 (2,000 each of AFR, AMR, EAS, EUR, SAS) | Population-equity measurement |
| South Asian realistic mixture | 10,000 (GIH, ITU, PJL, BEB, STU weighted) | Regression, correctness, demo |
| CRO demo cohort | 500 South Asian | Demonstration |
| Refusal / edge-case panel | 540 (6 categories x 90) | Refusal-path testing |
| Enriched coverage panel | 100 (4 activity scores x 25) | Reaching CPIC paths real frequencies never reach |

Total: 20,500 sampled patients, plus 540 engineered edge cases and 100
enriched coverage cases, in three separate files. Splits are 70/15/15 by
patient over the sampled cohort, stratified by ancestry, with the test set
locked. The two engineered panels are deliberately excluded from the splits
and from every frequency statistic.

### Label composition at seed 42

| Phenotype | n |
|---|---|
| Normal Metabolizer | 19,479 |
| Intermediate Metabolizer | 879 |
| Indeterminate | 142 |
| Poor Metabolizer | 0 |

**The sampled cohort contains no Poor Metabolizers, and that is correct, not an
omission.** An activity score of 0 or 0.5 needs two reduced-function alleles,
and at real frequencies (*2A* at 0.27 percent in South Asians, *13* at 0
percent) that combination is rarer than one in 20,500.

That left the two CPIC paths which actually stop a prescription, *avoid* and
*avoid unless no alternative*, untested. The enriched coverage panel exists to
reach them:

| Coverage case | n | Phenotype | Recommendation | Dose action |
|---|---|---|---|---|
| Activity score 0.0 | 25 | Poor Metabolizer | CONTRAINDICATED | avoid |
| Activity score 0.5 | 25 | Poor Metabolizer | ALTERNATIVE_RECOMMENDED | avoid_or_lt25pct_with_TDM |
| Activity score 1.0 | 25 | Intermediate Metabolizer | WITH_CAVEAT | reduce_50pct |
| Activity score 1.5 | 25 | Intermediate Metabolizer | WITH_CAVEAT | reduce_50pct |

These rows are **enriched, not sampled**. Their genotypes are orders of
magnitude rarer than this panel implies. They carry `is_enriched = True`, they
live in their own file, and they must never enter a frequency statistic. Their
compound heterozygotes are emitted phased, because an unphased pair of
reduced-function variants is genuinely ambiguous and the engine is right to
refuse it.

Recommendations in the sampled cohort:

| Recommendation | n |
|---|---|
| RECOMMENDATION_AS_IS | 17,357 |
| WITH_CAVEAT | 3,001 |
| REFUSED_INSUFFICIENT_EVIDENCE | 142 |
| ALTERNATIVE_RECOMMENDED | 0 (coverage panel only) |
| CONTRAINDICATED | 0 (coverage panel only) |

### Variants

83 *DPYD* variants from the CPIC allele table: 19 no function, 7 decreased
function, 55 normal function, 2 with no standalone function assignment. 81 are
simple SNVs; 78 had a gnomAD frequency and were injected. The two indels
(`c.1898del`, `c.299_302del`) are excluded from frequency-calibrated injection
and are listed in `qc/open_questions.md`.

### The headline equity number

**2,122 of 12,500 South Asian patients (17.0 percent) are called normal by a
CPIC-only screen but flagged once the South Asian override applies.** No
patient in any other ancestry group is affected, because the override is
defined only for South Asian ancestry.

This number is a direct consequence of one variant, V732I (`c.2194G>A`,
rs1801160), whose gnomAD South Asian minor allele frequency is 9.17 percent
against 4.52 percent in Europeans, and which CPIC assigns normal function. It
is a property of the override rule and the allele frequency, not evidence that
those patients would come to harm.

---

## Collection process

Nothing in this dataset was typed from memory. Every reference value is fetched
from a named source and cached under `reference/raw/`; anything that could not
be fetched is null and listed in `qc/open_questions.md`.

**Genotypes (Method A).** One `msprime` coalescent over the 844 kb *DPYD*
region (GRCh38 chr1:97,058,987-97,903,368), 41,000 haplotypes, using the
HapMapII GRCh38 recombination map from stdpopsim. Catalog variants are then
injected at their sourced per-population frequencies by selecting a clade in
the local tree whose descendant count matches the target. Placing the mutation
on a real genealogical branch is what makes linkage to the background and
Hardy-Weinberg behaviour come out right rather than being imposed.

**Clinical covariates (Method B).** An SDV `GaussianCopulaSynthesizer` fitted
to a 600-row seed cohort. The copula choice follows Miletic et al. 2025
(PMID 41049602), which found copula and synthpop gave the best privacy and
utility balance on pharmacogenetics data while deep generators carried higher
re-identification risk.

**Diffusion (Method C)** is a stub. See "Deliberate exclusions" below.

**Outcomes (Layer B).** A logistic model over activity-score deficit, override
carrier status, regimen, age, renal function and co-medication. Three
sensitivity draws are stored in `outcomes_sensitivity.parquet`.

---

## Preprocessing and labelling

Two layers, kept in separate files, never mixed:

- **Layer A** (`expected_engine_output.parquet`) is a deterministic lookup over
  frozen CPIC tables. No model, no sampling. It reproduces byte-identically on
  re-run and matched on 100 percent of 20,358 non-refusal rows.
- **Layer B** (`outcomes.parquet`) is sampled. Every row carries
  `label_type = SIMULATED`.

Layer B never feeds back into Layer A.

---

## Known biases and limitations

This section is the point of the datasheet. Read it before using any number
above.

### 1. The South Asian sub-population weights are assumptions

GIH 0.26, ITU 0.22, PJL 0.20, BEB 0.18, STU 0.14. These are declared in
`config.yaml` and are not an estimate of India's population structure.

### 2. Whole groups of Indian people are absent

1000 Genomes carries five South Asian panels: Gujarati, Telugu, Punjabi,
Bengali and Sri Lankan Tamil. It does not sample India's tribal (Adivasi)
populations, most rural populations, the Northeast, or the island populations.
gnomAD's South Asian grouping inherits a similar skew. A dataset built on these
frequencies therefore describes a *part* of South Asia and silently calls it
South Asia. Any equity claim made from it carries that limit.

### 3. The background demography for SAS and AMR is invented

The AFR/EUR/EAS backbone is Jouganous et al. 2017 via stdpopsim. That model has
no South Asian or admixed American population, so both were added with split
times, effective sizes and admixture proportions chosen to be plausible and
declared as assumptions in `config.yaml`. They shape the background genome.
They do not shape the catalog-variant frequencies, which come from gnomAD, and
those are what Layer A is tested on.

### 4. Pairwise linkage between catalog variants is weaker than real

Measured against the mean r-squared across the five 1000 Genomes South Asian
panels, the synthetic cohort's pairwise r-squared between catalog variants is
systematically lower (Frobenius distance 0.159 over the 3 pairs Ensembl
returned). Each injected variant is placed on an independently chosen clade, so
it inherits realistic linkage to the *background* but not the real linkage to
*other injected variants*. The one exception is handled explicitly: HapB3's two
defining variants are injected as a unit and co-occur in 526 of 526 carriers.
Turner et al. 2024 (PMID 38129972) report that linkage is not perfect in All
of Us (LD 0.9985, n=245,394), but every discordant carrier they found was of
European or "Other" ancestry, and their South Asian frequencies for the two
variants are identical at 0.01687. Perfect co-occurrence is therefore the
sourced behaviour for this cohort rather than a simplification.

This matters for anything that depends on tag-SNP behaviour across the gene. It
does not affect per-variant frequency, Hardy-Weinberg, or the Layer A call.

### 5. Only three LD pairs could be compared

Ensembl's pairwise LD endpoint returns nothing for variant pairs beyond its
window, and *DPYD* spans 844 kb. The LD comparison therefore rests on three
pairs. It is reported, not gated.

### 6. Population structure is not validated against real samples

The PCA shows the synthetic populations separating (South Asian and European
centroids 2.52 apart, 0.57 South Asian standard deviations), but 1000 Genomes
genotypes are not bundled, so the gate that matters, *do synthetic South Asian
samples cluster with real South Asian samples*, was not run. The separation
observed is modest and should not be read as a strong result.

### 7. Layer B effect sizes are assumptions

The *direction* of the genotype effect is sourced from CPIC, and the direction
of the V732I effect from Naushad et al. 2021. The *magnitudes* are not. The
Naushad paper is paywalled and no odds ratio was transcribed from it. Every
coefficient is tagged `ASSUMPTION` or `SOURCED` in `config.yaml` and the tags
are written into every outcome row. Conclusions drawn from Layer B are
conclusions about the assumptions.

### 8. The clinical covariates have no empirical support

The copula was fitted to a seed cohort built from declared marginals, because
no real patient profiles exist yet. Age, eGFR, regimen mix and co-medication
rates are plausible, not measured. Pass `--clinical-seed` with real profiles
and the same code path expands those instead.

### 9. Refusal rule IDs are absent

`refusal_rule_id_expected` is null for every row. The build brief forbids
inventing R/V/U identifiers, the `anukriti-swarm` source is not present here,
so the semantic `refusal_category_expected` carries the meaning instead.

### 10. One check is skipped, and PharmCAT parity is not 100%

Two gaps remain, and they are gaps, not passes: PCA against real 1000 Genomes
samples (the reference genotypes are not bundled) is reported for the synthetic
cohort only, and train-synthetic-test-real ML utility is **skipped** outright —
it needs the DPYD v4 classifier and a real functionally-labelled held-out
variant set, neither of which is in this repository.

PharmCAT parity now runs. It is no longer a gap, but it is not a clean bill of
health either. PharmCAT 3.4.0 over a seed-fixed sample of 500 patients agrees
with `expected_engine_output.parquet` on **98.40%** of diplotypes and **98.20%**
of phenotypes. The 9 disagreements are the engine's two refusal paths, neither
of which changes a CPIC recommendation:

| Class | n | What it is |
|---|---|---|
| `engine_refused_low_quality` | 8 | A call below `min_dp` 20 or `min_gq` 30 makes Layer A return `Indeterminate`. The PharmCAT-shaped VCF carries `GT` only — DP and GQ are not invented for reference rows — so PharmCAT has no quality signal and calls normally. |
| `engine_refused_ambiguous_phase` | 1 | Two Decreased-function variants, unphased. Both callers name the same diplotype and the same activity score; PharmCAT follows CPIC's unphased convention and reports a phenotype, Layer A declines because phase would change the answer. |

In both the engine is the more conservative of the two.

A third class, `tie_break_equal_activity`, accounted for 50 of 500 patients and
held diplotype agreement at 88.40% in earlier drafts of this build. It was a
difference in the diplotype *label* only: when a patient carries more than two
variant alleles and three or more of them share the lowest activity value,
CPIC's two-lowest rule does not say which to name, and the two callers ordered
equal-activity names differently — Layer A as plain strings, PharmCAT by
comparing each run of digits as a number, which is transcript order. Layer A
adopted PharmCAT's ordering, which closed the class outright. It relabelled
**2,201 of 20,500 patients** (10.74%) and `diplotype_expected` is the only
column of `expected_engine_output.parquet` that changed: the selection is taken
after the activity score is computed and cannot feed back into it, so every
activity score, phenotype, recommendation and refusal category is unchanged.
Anything that joined on the old diplotype string for one of those 2,201
patients needs re-joining. Full counts, the rejected alternative and the
reasoning are in `qc/open_questions.md`; the per-patient comparison is
`qc/pharmcat_parity.csv`.

PharmCAT is not bundled. On a machine without it the gate reports SKIPPED, which
is a gap and not a pass. PharmCAT is a second implementation of the same CPIC
tables, not a gold standard: it cannot adjudicate the South Asian override,
which has no CPIC counterpart by construction.

### 11. Corrections this build made to its own brief

Two of these change what the engine should do, and both were found by fetching
rather than trusting:

- **V732I is rs1801160, not rs1801265.** The internal reference had them
  swapped; rs1801265 is C29R.
- **S534N does not qualify for the override; only V732I does.** Its gnomAD
  South Asian minor allele frequency is 0.70 percent, under the 5 percent
  threshold, although the brief treats it as a driver alongside V732I. The
  threshold was ratified at 5 percent and S534N is excluded deliberately. Every
  override figure in this datasheet, the 17.0 percent included, therefore rests
  on V732I alone.
- **M166V and rs2297595 are one variant**, not two.

---

## Deliberate exclusions

**Diffusion-generated genotypes.** Kenneweg et al. 2025 (PMID 40662828) report
strong results, but their method compresses each gene to 1-8 principal
components, and the variants that decide a *DPYD* call are rare: *2A* sits at
0.27 percent in South Asians. PCA compression at that ratio does not reliably
preserve them, and a cohort that cannot reproduce *2A* at the right frequency
cannot test a *DPYD* engine. Their downstream tasks (ALS case/control,
which-of-26-populations) also do not demonstrate that pharmacogene haplotype
structure survives. `synthpgx/diffusion.py` defines the plugin contract and
refuses to run; when a generator arrives it is for the *background* genome,
spliced around a *DPYD* region still produced by coalescent plus injection, and
it must clear the Part 8 gates against the coalescent arm first.

**Controlled-access data.** No GenomeIndia, IndiGen, All of Us or UK Biobank
individual-level data is in this repository, and none was used.

---

## Uses

**Appropriate.** Engine regression and scale testing; ground-truth correctness;
refusal-path testing; quantifying how many South Asian carriers a CPIC-only
screen would call normal; demonstrations labelled as synthetic; as a
reproducible benchmark others can regenerate.

**Not appropriate.** Any clinical or regulatory claim. Any statement that a
variant causes toxicity. Setting or changing a CPIC-style rule. Estimating real
toxicity rates in any population. Any dosing decision for any real patient.

**Effect on data subjects.** There are none. No real genome is reproduced. The
privacy metrics (NNAA 0.548, membership-inference accuracy 0.487, zero
synthetic records at distance 0 from a seed record) confirm this, though with a
synthetic seed cohort they establish the measurement path rather than clearing
the generator for use on real data.

---

## Distribution and maintenance

Licence undecided; see `qc/open_questions.md`. Regenerate with
`fetch_reference.py`, `generate.py --seed 42`, `validate.py`. The build fails
loudly if a gate fails.

Changes that should trigger a regeneration: a CPIC guideline update, a gnomAD
release, obtaining real South Asian individual-level genotypes, obtaining real
partner outcome data (which would replace Layer B entirely), or the team fixing
the override threshold and rule IDs.

**Contact:** Anukriti AI Private Limited.

# SynthPGx-DPYD-SAS v1

> **SYNTHETIC. In silico cohort validation. Not clinical evidence.**

A synthetic, population-aware, fully labelled cohort of virtual colorectal
cancer patients with *DPYD* genotypes and simulated fluoropyrimidine exposure.
It exists to validate and stress-test Anukriti's deterministic
pharmacogenomics engine, and to train and evaluate the machine-learning
components beside it, while no labelled South Asian outcome data exists.

Owner: Anukriti AI Private Limited. Gene: *DPYD*. Drugs: 5-fluorouracil and
capecitabine. Reference genome: GRCh38. Transcript: NM_000110.4.

---

## What this is not

These limits are not boilerplate. They decide what the dataset may be used to
claim.

1. **It is not clinical evidence.** No regulator will accept a drug-safety
   claim built on synthetic data. Call this *in silico cohort validation*.
   Never call it a virtual clinical trial.
2. **The toxicity outcomes are simulated, not observed.** They are sampled
   from a logistic model whose coefficients are assumptions. They exercise the
   pipeline's logic and plumbing. They do not show that any variant causes
   toxicity in anyone.
3. **It must never be used to set or change a CPIC-style rule.** Rules come
   from curated published evidence. A synthetic cohort cannot be evidence for
   the rule that generated it.
4. **No real person's genome is reproduced.** The genotypes come from a
   coalescent simulation, not from any individual. The privacy checks in
   `qc/qc_report.html` measure this rather than assuming it.
5. **Do not use it to imply a dosing recommendation for any real patient.**

## The two label layers

Keeping these apart is the single most important thing about this dataset.
They live in separate files and must never be mixed.

| Layer | File | Source | Trust |
|---|---|---|---|
| **A. Rule label** | `data/expected_engine_output.parquet` | CPIC and PharmVar tables, deterministically | High, citable, reproducible |
| **B. Simulated outcome** | `data/outcomes.parquet` | Sampled from assumed effect sizes | Assumption-driven, **not evidence** |

Layer A answers *what does the rulebook say for this genotype?* Layer B
answers *what might happen to this virtual patient on the drug?* Layer B never
feeds back into Layer A. Every Layer B row carries `label_type = SIMULATED` so
this cannot be lost downstream.

### Layer B has two models, and v2 is the one to model against

`outcomes.parquet` (**v1**) targets grade 3+ toxicity at a ~41 percent event
rate. Almost all of its variance is non-genetic, and the consequence is
measurable: the best achievable discrimination against its own labels is
**AUC 0.602**. That is not a modelling limit — a *perfect* model that knows
each patient's true risk scores 0.602 — so v1 cannot distinguish a model that
learned the genotype signal from one that did not.

`outcomes_v2.parquet` (**v2**) targets the endpoint the literature actually
models: 12-week **grade 4-5** toxicity. Rarer events, far more genotype-driven,
measured ceiling **AUC 0.662** on this cohort.

| | v1 | v2 |
|---|---|---|
| endpoint | grade 3+ | grade 4-5 |
| event rate | 0.408 | 0.079 |
| genotype terms | activity deficit, SAS override | activity deficit, rare-variant burden, rs1801160 |
| best achievable AUC | 0.602 | 0.662 |
| effect sizes | all `ASSUMPTION` | two `SOURCED`, one `DERIVED`, rest `ASSUMPTION` |

v2 is **added, not substituted**. v1 is byte-identical to the previous build —
verified by checksum — so every result computed against it still stands.

The headline source is **Boige et al., *J Natl Cancer Inst* 2026;118(4):699-710
(PMID 41386266)**: 3,437 colorectal patients, 267 events, full *DPYD*
sequencing, reported AUC 0.76 (95% CI 0.73-0.78). Two of its odds ratios are
transcribed directly — an aggregate of rare (MAF < 0.01) predicted-deleterious
variants at **OR 3.20**, and rs1801160 at **OR 1.49** — and v2's intercept is
solved so the event rate reproduces their 267/3437.

Three honest limits on v2. Its `activity_deficit_per_unit` of 2.50 is still an
assumption, and it is the largest one in the model. The gap from 0.662 to
Boige's 0.76 is their full patient-and-treatment covariate set, which is not
reproduced here. And rs1801160 is keyed on **carriage of the variant**, not on
the South Asian override flag that v1 used — the override is a Layer A policy
about who gets looked at more closely, not a biological effect, and modelling
it as one was a conflation in v1.

## Layout

```
synthpgx_dpyd_sas_v1/
  README.md                       this file
  DATASHEET.md                    motivation, composition, biases, limits
  data/
    patients.parquet              one row per synthetic patient
    genotypes_long.parquet        one row per patient x variant
    expected_engine_output.parquet  Layer A ground truth
    outcomes.parquet              Layer B, primary draw
    outcomes_sensitivity.parquet  Layer B, every sensitivity draw
    outcomes_v2.parquet           Layer B v2, grade 4-5, sourced effect sizes
    outcomes_v2_sensitivity.parquet  v2, every sensitivity draw
    refusal_cases.parquet         engineered rows whose correct answer is a refusal
    coverage_cases.parquet        ENRICHED rows reaching the CPIC paths real frequencies miss
    refusal_genotypes_long.parquet  their variant calls
    vcf/                          per-patient VCFs
    cohort.vcf.gz                 multi-sample cohort VCF
  reference/
    variant_catalog.csv           every variant used, with frequency and evidence
    cpic_allele_functions.csv     frozen CPIC allele function table
    cpic_allele_definitions.csv   variant-to-allele mapping
    cpic_diplotype_phenotype.csv  diplotype -> activity score -> phenotype
    cpic_recommendations.csv      fluoropyrimidine recommendations
    sas_population_evidence.csv   South Asian evidence, with provenance
    frequency_sources.json        which database and version each value came from
    raw/                          cached raw API payloads, for reproducibility
  splits/
    train.txt val.txt test.txt    patient IDs, 70/15/15, stratified by ancestry
  qc/
    qc_report.html                every Part 8 gate
    qc_metrics.json               the same metrics, machine-readable
    open_questions.md             every value that could not be sourced,
                                  and every PharmCAT disagreement
    pharmcat_parity.csv           per-patient PharmCAT cross-check, 500 rows
    generation_manifest.json      what was run, with what, when
  code/
    fetch_reference.py  generate.py  validate.py  config.yaml
    synthpgx/                     the package
```

## Reproducing it

Requires Python 3.12 and about 4 GB of RAM.

```bash
pip install numpy pandas pyarrow scipy pyyaml msprime stdpopsim tskit sdv
```

Then, from `code/`:

```bash
python fetch_reference.py        # pull and freeze CPIC / gnomAD / Ensembl / PubMed
python generate.py --config config.yaml --seed 42
python validate.py --config config.yaml
```

`fetch_reference.py` caches every API response under `reference/raw/`, so the
second run is offline and identical. `generate.py` is deterministic: the same
config and seed give byte-identical output. `validate.py` exits non-zero if any
gate fails.

Useful flags: `--scale 0.02` for a fast smoke run, `--swarm-source <path>` to
supply real refusal rule IDs, `--clinical-seed <csv>` to expand real patient
profiles instead of the assumption-built seed cohort.

### The PharmCAT parity gate

`validate.py` cross-checks 500 patients against [PharmCAT](https://pharmcat.org),
an independent implementation of the same CPIC tables. PharmCAT is deliberately
**not** vendored here: the check is only worth something against a release you
can download yourself. It needs Java 17+ and an unpacked PharmCAT release
holding both the jar and the `pharmcat_positions` VCF that ships with it:

```bash
mkdir -p ~/pharmcat && cd ~/pharmcat
curl -LO https://github.com/PharmGKB/PharmCAT/releases/download/v3.4.0/pharmcat-3.4.0-all.jar
curl -LO https://github.com/PharmGKB/PharmCAT/releases/download/v3.4.0/pharmcat_positions_3.4.0.vcf
export PHARMCAT_HOME=~/pharmcat
```

Set `$PHARMCAT_HOME`, or point `validation.pharmcat.install_dir` in
`config.yaml` at the same directory. Without it the gate reports SKIPPED — a
gap, never a pass. With it, expect validation to take about 70 seconds longer.

The per-patient VCFs in `data/vcf/` are *not* what PharmCAT is fed. They carry
only non-reference calls, as a variant caller emits; PharmCAT's matcher needs
every position in its allele definition present, so that a missing row means
"not sequenced" rather than "reference". `vcfio.write_pharmcat_vcf` emits a
separate multi-sample VCF covering all 83 DPYD definition positions, taking REF
and ALT from PharmCAT's own positions file. Its FORMAT is `GT` alone: DP and GQ
are not invented for the reference rows, which is why PharmCAT has no
counterpart to the engine's quality-refusal path. Pass
`--pharmcat-work-dir <path>` to keep that VCF and PharmCAT's own reports
(a few hundred megabytes) instead of staging them in a temporary directory.

## Where the numbers come from

Nothing in the catalog was typed from memory. Every value is fetched, and
anything that could not be fetched is null and listed in
`qc/open_questions.md`.

| Field | Source |
|---|---|
| Allele function, activity score, diplotype to phenotype, recommendations | CPIC API (`api.cpicpgx.org`) |
| Variant coordinates, rsIDs, protein changes | CPIC `sequence_location` table, cross-checked against the Ensembl variant recoder |
| Per-ancestry allele frequencies | gnomAD v3.1.2 primary, v4 as fallback |
| South Asian sub-population frequencies | 1000 Genomes phase 3 via Ensembl, supplemental only |
| South Asian toxicity evidence | Naushad et al. 2021, PMID 33105068, n = 2000 |
| Citation provenance | NCBI PubMed esummary |

gnomAD v3.1.2 is primary rather than 1000 Genomes because 1000 Genomes samples
are also inside gnomAD, and counting them twice would inflate confidence in
exactly the populations this dataset is about.

### Three things the sources corrected in the build brief

These were open questions in the brief. The fetched records answered them, and
two of the answers change what the engine should do.

1. **V732I is rs1801160, not rs1801265.** The internal reference table had them
   swapped. rs1801265 is C29R. CPIC and Ensembl agree independently.
2. **M166V and rs2297595 are the same variant** (`c.496A>G`), not two separate
   catalog entries.
3. **S534N does not qualify for the South Asian override**, and **only V732I
   does.** Its gnomAD South Asian minor allele frequency is 0.70 percent,
   below the 5 percent threshold, although the brief treats both as drivers.
   The threshold has been ratified at 5 percent as written and S534N is
   excluded; this is settled, not pending. The value lives in
   `SAS_OVERRIDE_MIN_MAF` in `synthpgx/catalog.py`, and changing it readmits
   S534N and requires re-examining every variant between the old line and the
   new one.

## Honest refusals

The dataset contains rows where the correct engine behaviour is to decline.
Six categories are engineered, in `data/refusal_cases.parquet`: an unlisted
variant with no evidence, an ancestry outside the closed enum, a frameshift
absent from the tables, genuinely ambiguous phase, a call below the quality
threshold, and a conflicting-evidence case that takes the population-override
path.

Separately, `data/coverage_cases.parquet` holds 100 **enriched** rows that
reach the two CPIC paths a naturally sampled cohort never does: *avoid* and
*avoid unless no alternative*. A Poor Metabolizer is rarer than one in 20,500
at real frequencies, so without these the engine's strongest recommendations
would go untested. They are enrichment, not population sampling, and must be
kept out of every frequency statistic.

**Rule IDs are not invented.** The brief requires the R/V/U identifiers to come
from the `anukriti-swarm` source and the build to fail on an unknown one. That
repository is not present here, so `refusal_rule_id_expected` is null
throughout and `refusal_category_expected` carries the meaning instead. Pass
`--swarm-source` to populate the real IDs.

## Known gaps

Skipped gates in the QC report are gaps, not passes.

- **PharmCAT parity runs, and it is not 100%.** PharmCAT 3.4.0 over a seed-fixed
  sample of 500 patients agrees on **98.40%** of diplotypes and **98.20%** of
  phenotypes. All 9 disagreements are the engine's two refusal paths, which
  PharmCAT has no counterpart for: 8 low-quality calls it is given no DP or GQ
  to refuse on, and 1 ambiguous phase it resolves by CPIC convention where the
  engine declines. None changes a CPIC recommendation. The equal-activity
  diplotype tie-break that used to account for 50 of these was adopted from
  PharmCAT in v1, which is what moved diplotype agreement up from 88.40%; it
  relabelled 2,201 of 20,500 patients and changed no activity score, phenotype
  or recommendation. Counts, worked examples and the reasoning are in
  `qc/open_questions.md`; the per-patient comparison is
  `qc/pharmcat_parity.csv`. The gate fails on any disagreement outside the known
  classes. PharmCAT is not bundled, so on a machine without it the gate reports
  SKIPPED rather than passing.
- **`pharmvar_version` is null** because the PharmVar API needs credentials.
- **PCA against real 1000 Genomes samples** was not run; the reference
  genotypes are not bundled. Population structure is reported for the synthetic
  cohort only.
- **ML utility (train-synthetic-test-real)** was not run; it needs the DPYD v4
  missense classifier and a held-out set of real functionally-labelled
  variants.
- **GenomeIndia frequencies** are null throughout; no machine-readable endpoint
  was reachable.
- **Layer B effect sizes are assumptions.** Their direction is sourced; their
  magnitude is not.

## Citation

If you use this dataset, cite the sources it rests on: the CPIC *DPYD* and
fluoropyrimidine guideline (Amstutz et al., PMID 29152729), PharmVar, gnomAD,
1000 Genomes, Naushad et al. 2021 (PMID 33105068), Hariprakash et al. 2018
(PMID 29239269), Miletic et al. 2025 (PMID 41049602) and Kenneweg et al. 2025
(PMID 40662828).

Licence: not yet decided. See `qc/open_questions.md`.

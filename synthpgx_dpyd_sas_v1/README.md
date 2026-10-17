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
    open_questions.md             every value that could not be sourced
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

- **PharmCAT parity** was not run; PharmCAT is not installed. `pharmvar_version`
  is null because the PharmVar API needs credentials.
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

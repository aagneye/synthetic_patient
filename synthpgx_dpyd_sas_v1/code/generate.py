#!/usr/bin/env python
"""Generate the SynthPGx-DPYD-SAS cohort.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Usage::

    python generate.py --config config.yaml --seed 42

Deterministic: the same config and seed produce byte-identical outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthpgx import covariates, genotypes, layer_a, outcomes, refusals, vcfio  # noqa: E402

SUPER_POPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SAS_SUB_POPS = ["GIH", "ITU", "PJL", "BEB", "STU"]
LABEL = "SYNTHETIC. In silico cohort validation. Not clinical evidence."


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------


def plan_cohorts(config: dict) -> pd.DataFrame:
    """Decide who exists before any genotype is drawn."""
    rows = []
    index = 1
    for name, spec in config["cohorts"].items():
        if "per_population" in spec:
            for pop in spec["populations"]:
                for _ in range(int(spec["per_population"])):
                    rows.append({"cohort": name, "ancestry_super_pop": pop,
                                 "patient_index": index})
                    index += 1
        else:
            pops = spec["populations"]
            for i in range(int(spec["total"])):
                rows.append({"cohort": name,
                             "ancestry_super_pop": pops[i % len(pops)],
                             "patient_index": index})
                index += 1
    frame = pd.DataFrame(rows)
    frame["patient_id"] = frame["patient_index"].map(lambda i: f"SYN-{i:06d}")
    return frame


def assign_sub_populations(frame: pd.DataFrame, config: dict, seed: int) -> pd.DataFrame:
    """Give every SAS patient a sub-population, per that cohort's weights."""
    rng = np.random.default_rng(seed + 11)
    frame = frame.copy()
    frame["ancestry_sub_pop"] = None
    for cohort, spec in config["cohorts"].items():
        mask = (frame["cohort"] == cohort) & (frame["ancestry_super_pop"] == "SAS")
        n = int(mask.sum())
        if n == 0:
            continue
        weights = spec.get("sas_sub_pop_weights")
        if weights:
            labels = list(weights)
            probabilities = np.array([float(weights[k]) for k in labels])
            probabilities = probabilities / probabilities.sum()
        else:
            labels = SAS_SUB_POPS
            probabilities = np.full(len(labels), 1.0 / len(labels))
        frame.loc[mask, "ancestry_sub_pop"] = rng.choice(
            labels, size=n, p=probabilities
        )
    return frame


# ----------------------------------------------------------------------


def frequency_targets(catalog: pd.DataFrame) -> tuple[dict, list[str]]:
    """Per-population target allele frequencies, from the sourced catalog."""
    targets: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for _, row in catalog.iterrows():
        if not bool(row.get("is_snv")):
            continue
        per_pop: dict[str, float] = {}
        for pop in SUPER_POPS:
            value = row.get(f"freq_gnomad_{pop.lower()}")
            if pd.notna(value):
                per_pop[pop] = float(value)
        if not per_pop:
            missing.append(str(row["variant_id"]))
            continue
        targets[str(row["variant_id"])] = per_pop
    return targets, missing


def read_haplotype_groups(reference_dir: Path) -> dict[str, list[str]]:
    """Compound CPIC alleles, whose variants sit together on one haplotype.

    HapB3 is the case that matters for DPYD: it is defined by c.1129-5923C>G
    and c.1236G>A, which are in near-complete linkage. Injecting them
    independently would hand some patients one without the other, which is not
    a genotype that occurs at any meaningful rate in real data.

    Linking them is right for this cohort specifically, not merely convenient.
    Turner et al. 2024 (PMID 38129972) found 14 carriers of c.1236G>A without
    c.1129-5923C>G in All of Us (n=245,394), reporting LD 0.9985 -- but 13 of
    the 14 were of European genetic ancestry and one was "Other". Their
    per-ancestry frequencies are identical in South Asians (0.01687 for both
    variants) where they differ in Europeans (0.02091 against 0.02085). At the
    resolution of that cohort the South Asian discordance rate is zero, so a
    unit injection is the sourced choice for a South Asian cohort.
    """
    path = reference_dir / "cpic_allele_definitions.csv"
    if not path.exists():
        return {}
    definitions = pd.read_csv(path)
    groups: dict[str, list[str]] = {}
    for allele_name, group in definitions.groupby("allele_name"):
        if allele_name == "Reference":
            continue
        members = sorted(set(group["variant_id"].astype(str)))
        if len(members) > 1:
            groups[str(allele_name)] = members
    return groups


def simulate_quality(n: int, spec: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dp = np.clip(
        rng.normal(float(spec["dp_mean"]), float(spec["dp_sd"]), size=n),
        float(spec["dp_min"]),
        None,
    ).round().astype(int)
    gq = np.clip(
        rng.normal(float(spec["gq_mean"]), float(spec["gq_sd"]), size=n),
        1,
        float(spec["gq_max"]),
    ).round().astype(int)
    return dp, gq


# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=str(here.parent))
    parser.add_argument("--clinical-seed", default=None,
                        help="CSV of real patient profiles for the copula to expand.")
    parser.add_argument("--swarm-source", default=None,
                        help="Path to an anukriti-swarm checkout, for refusal rule IDs.")
    parser.add_argument("--max-patient-vcfs", type=int, default=1000,
                        help="Cap on per-patient VCF files written (0 = all).")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale every cohort size, for quick smoke runs.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    root = Path(args.out)
    data_dir = root / "data"
    reference_dir = root / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "qc").mkdir(parents=True, exist_ok=True)

    if args.scale != 1.0:
        for spec in config["cohorts"].values():
            if "per_population" in spec:
                spec["per_population"] = max(10, int(spec["per_population"] * args.scale))
            else:
                spec["total"] = max(10, int(spec["total"] * args.scale))
        config["refusal_panel"]["per_category"] = max(
            3, int(config["refusal_panel"]["per_category"] * args.scale)
        )

    log(f"seed={seed}  config={args.config}")
    catalog = pd.read_csv(reference_dir / "variant_catalog.csv")
    log(f"catalog: {len(catalog)} variants, {int(catalog['is_snv'].sum())} SNVs")

    # --- who exists -------------------------------------------------
    patients = plan_cohorts(config)
    patients = assign_sub_populations(patients, config, seed)
    counts = patients["ancestry_super_pop"].value_counts().to_dict()
    log(f"planned {len(patients)} patients: {counts}")

    # --- Method A: background coalescent ----------------------------
    region = genotypes.region_from_catalog(
        catalog, flank=int(config["genotypes"]["region_flank_bp"])
    )
    log(f"DPYD region {region.chrom}:{region.start}-{region.end} "
        f"({region.length / 1000:.0f} kb)")

    demography, demography_provenance = genotypes.build_demography(
        config["genotypes"]["demography"]
    )
    samples_per_pop = {
        pop: int((patients["ancestry_super_pop"] == pop).sum()) for pop in SUPER_POPS
    }
    log(f"simulating coalescent for {sum(samples_per_pop.values())} individuals ...")
    started = time.time()
    ts, hap_index = genotypes.simulate_background(
        demography,
        samples_per_pop,
        region,
        seed=seed,
        recombination_map=config["genotypes"]["recombination_map"],
        mutation_rate=config["genotypes"]["mutation_rate"],
    )
    log(f"  tree sequence: {ts.num_trees} trees, {ts.num_sites} background sites, "
        f"{ts.num_samples} haplotypes ({time.time() - started:.1f}s)")

    # --- injection ---------------------------------------------------
    targets, missing_freq = frequency_targets(catalog)
    haplotype_groups = read_haplotype_groups(reference_dir)
    if haplotype_groups:
        log("  linked haplotypes kept together: "
            + "; ".join(f"{k} ({len(v)} variants)" for k, v in haplotype_groups.items()))
    log(f"injecting {len(targets)} catalog variants at sourced frequencies ...")
    started = time.time()
    carriers = genotypes.inject_catalog(
        ts, hap_index, catalog.to_dict("records"), targets, region, seed=seed + 7,
        haplotype_groups=haplotype_groups,
    )
    log(f"  injection done ({time.time() - started:.1f}s)")

    pairs = genotypes.pair_haplotypes(hap_index, seed=seed + 13)

    # --- map patients onto diploids ----------------------------------
    patients = patients.sort_values(["ancestry_super_pop", "patient_index"]).reset_index(drop=True)
    patient_pairs = np.zeros((len(patients), 2), dtype=np.int64)
    for pop in SUPER_POPS:
        mask = (patients["ancestry_super_pop"] == pop).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        available = pairs[pop]
        if len(available) < n:
            raise RuntimeError(f"not enough simulated individuals for {pop}")
        patient_pairs[mask] = available[:n]
    patients = patients.sort_values("patient_index").reset_index(drop=True)
    order = np.argsort(patients["patient_index"].to_numpy())
    del order

    # Rebuild pair assignment in patient_index order.
    pair_lookup = {}
    tmp = patients.copy()
    tmp_sorted = tmp.sort_values(["ancestry_super_pop", "patient_index"]).reset_index(drop=True)
    for i, pid in enumerate(tmp_sorted["patient_id"]):
        pair_lookup[pid] = patient_pairs[i]
    patient_pairs = np.array([pair_lookup[p] for p in patients["patient_id"]])

    # --- genotypes_long ----------------------------------------------
    log("building genotype table ...")
    rng = np.random.default_rng(seed + 17)
    quality_spec = config["genotypes"]["quality"]
    snv_catalog = catalog[catalog["is_snv"].fillna(False).astype(bool)]

    long_rows = []
    per_patient_calls: dict[str, dict] = {p: {} for p in patients["patient_id"]}
    patient_ids = patients["patient_id"].to_numpy()

    for _, variant in snv_catalog.iterrows():
        variant_id = str(variant["variant_id"])
        flags = carriers.get(variant_id)
        if flags is None:
            continue
        dosage = genotypes.genotype_codes(flags, patient_pairs)
        nonzero = np.flatnonzero(dosage > 0)
        if nonzero.size == 0:
            continue
        dp, gq = simulate_quality(nonzero.size, quality_spec, rng)
        for k, i in enumerate(nonzero):
            call = {
                "patient_id": patient_ids[i],
                "variant_id": variant_id,
                "hgvs_c": variant["hgvs_c"],
                "rsid": variant["rsid"] if pd.notna(variant["rsid"]) else None,
                "chrom": "chr1",
                "pos_grch38": int(variant["grch38_pos"]),
                "ref": variant["ref"],
                "alt": variant["alt"],
                "genotype": genotypes.GENOTYPE_STRINGS[int(dosage[i])],
                "phased": None,
                "dp": int(dp[k]),
                "gq": int(gq[k]),
            }
            long_rows.append(call)
            per_patient_calls[call["patient_id"]][variant_id] = call

    genotypes_long = pd.DataFrame(long_rows)
    log(f"  {len(genotypes_long)} non-reference calls across "
        f"{genotypes_long['patient_id'].nunique()} patients")

    # --- Method B: clinical covariates --------------------------------
    log("sampling clinical covariates (Gaussian copula) ...")
    covariate_frame, covariate_provenance = covariates.sample_covariates(
        config["clinical_covariates"], len(patients), seed + 23, args.clinical_seed
    )
    covariate_frame = covariates.coerce_types(covariate_frame)
    covariate_frame = covariates.reconcile_dose(
        covariate_frame, config["clinical_covariates"], seed + 29
    )
    log(f"  generator: {covariate_provenance['generator']}")

    patients = pd.concat(
        [patients.reset_index(drop=True), covariate_frame.reset_index(drop=True)], axis=1
    )
    patients["cancer_type"] = config["dataset"]["cancer_type"]

    # BMI, on its own RNG stream so the copula's draws — and therefore every
    # pre-existing covariate and all of Layer B v1 — are byte-identical
    # whether or not this runs. Marginals are SOURCED from FUSAFE; see the
    # config note for the transposition caveat in the published table.
    bmi_spec = config["clinical_covariates"].get("bmi")
    if bmi_spec:
        bmi_rng = np.random.default_rng(seed + 80_000)
        names = list(bmi_spec["bands"])
        weights = np.array([bmi_spec["bands"][k][0] for k in names], dtype=float)
        weights = weights / weights.sum()
        picked = bmi_rng.choice(len(names), size=len(patients), p=weights)
        lows = np.array([bmi_spec["bands"][k][1] for k in names], dtype=float)
        highs = np.array([bmi_spec["bands"][k][2] for k in names], dtype=float)
        patients["bmi_category"] = [names[i] for i in picked]
        patients["bmi"] = np.round(
            lows[picked] + bmi_rng.random(len(patients)) * (highs[picked] - lows[picked]),
            1,
        )
        log(f"  BMI: {patients['bmi_category'].value_counts(normalize=True).round(3).to_dict()}")
    patients["admixture_json"] = [
        json.dumps({"SAS": 1.0}) if p == "SAS" else None
        for p in patients["ancestry_super_pop"]
    ]
    patients["generation_method"] = "coalescent+copula"
    patients["seed"] = [seed + int(i) for i in patients["patient_index"]]
    patients["is_real_variant_anchor"] = True
    patients["co_medication_flags"] = [
        json.dumps({"interacting_drug": bool(v)})
        for v in patients["co_medication_interacting"]
    ]
    patients["label_type"] = LABEL

    # --- refusal panel -------------------------------------------------
    log("building refusal panel ...")
    refusal_patients, refusal_calls = refusals.build_refusal_panel(
        catalog,
        region,
        int(config["refusal_panel"]["per_category"]),
        seed + 31,
        int(config["engine"]["min_dp"]),
        int(config["engine"]["min_gq"]),
    )
    log(f"  {len(refusal_patients)} engineered cases, {len(refusal_calls)} calls")

    # Enriched coverage panel: the CPIC paths that real frequencies never reach.
    coverage_patients, coverage_calls = refusals.build_coverage_panel(
        catalog,
        int(config["refusal_panel"].get("coverage_per_target", 25)),
        seed + 37,
        int(config["engine"]["min_dp"]),
        int(config["engine"]["min_gq"]),
    )
    log(f"  {len(coverage_patients)} enriched coverage cases "
        f"({len(coverage_calls)} calls)")

    for call in list(refusal_calls) + list(coverage_calls):
        per_patient_calls.setdefault(call["patient_id"], {})[call["variant_id"]] = call

    # --- Layer A --------------------------------------------------------
    log("computing Layer A labels ...")
    tables = layer_a.CpicTables(reference_dir)
    rules = layer_a.RuleRegistry.from_source(
        args.swarm_source or config["engine"].get("swarm_source")
    )
    caller = layer_a.make_caller(
        tables,
        rules=rules,
        outcome_map=config["engine"]["outcome_map"],
        min_dp=int(config["engine"]["min_dp"]),
        min_gq=int(config["engine"]["min_gq"]),
        supported_ancestries=config["engine"]["supported_ancestries"],
    )
    log(f"  caller: {caller.name}; refusal rule IDs "
        f"{'from ' + str(rules.source) if rules.available else 'UNAVAILABLE (null)'}")

    cpic_version = str(tables.gene.get("version", "")) or None
    expected_rows = []
    for _, patient in patients.iterrows():
        calls = list(per_patient_calls[patient["patient_id"]].values())
        drug = "capecitabine" if patient["regimen"] in ("capecitabine", "XELOX") else "fluorouracil"
        call = caller.call(
            calls,
            patient["ancestry_super_pop"],
            drug,
            patient.get("ancestry_sub_pop"),
        )
        row = {"patient_id": patient["patient_id"], "gene": "DPYD", "drug": drug}
        row.update(call.as_row())
        row["cpic_version"] = cpic_version
        row["pharmvar_version"] = None  # PharmVar API requires credentials
        row["pmid_list"] = ";".join(call.pmids)
        row["label_type"] = LABEL
        expected_rows.append(row)
    expected = pd.DataFrame(expected_rows)
    log("  phenotype counts: "
        + str(expected["phenotype_expected"].value_counts().to_dict()))
    log("  recommendation counts: "
        + str(expected["cpic_recommendation_expected"].value_counts().to_dict()))
    log(f"  SAS override flagged: {int(expected['sas_override_expected'].sum())}")

    # --- refusal panel Layer A ------------------------------------------
    refusal_rows = []
    for patient in refusal_patients:
        calls = list(per_patient_calls[patient["patient_id"]].values())
        call = caller.call(calls, patient["ancestry_super_pop"], "fluorouracil",
                           patient.get("ancestry_sub_pop"))
        row = dict(patient)
        row.update(call.as_row())
        row["cpic_version"] = cpic_version
        row["pharmvar_version"] = None
        row["pmid_list"] = ";".join(call.pmids)
        row["label_type"] = LABEL
        refusal_rows.append(row)
    refusal_frame = pd.DataFrame(refusal_rows)
    agreement = (
        refusal_frame["expected_category"] == refusal_frame["refusal_category_expected"]
    ).mean()
    log(f"  refusal category agreement: {agreement:.1%}")

    # --- coverage panel Layer A ------------------------------------------
    coverage_rows = []
    for patient in coverage_patients:
        calls = list(per_patient_calls[patient["patient_id"]].values())
        call = caller.call(calls, patient["ancestry_super_pop"], "fluorouracil",
                           patient.get("ancestry_sub_pop"))
        row = dict(patient)
        row.update(call.as_row())
        row["cpic_version"] = cpic_version
        row["pmid_list"] = ";".join(call.pmids)
        row["label_type"] = LABEL
        coverage_rows.append(row)
    coverage_frame = pd.DataFrame(coverage_rows)
    if not coverage_frame.empty:
        hit = (
            coverage_frame["activity_score_expected"]
            == coverage_frame["target_activity_score"]
        ).mean()
        log(f"  coverage panel hit target activity score: {hit:.1%}")
        log("  coverage recommendations: "
            + str(coverage_frame["cpic_recommendation_expected"].value_counts().to_dict()))

    # --- Layer B ----------------------------------------------------------
    log("simulating Layer B outcomes ...")
    outcome_frames = []
    for draw in range(int(config["outcomes"]["sensitivity_draws"])):
        frame, effects = outcomes.simulate_outcomes(
            patients, expected, config["outcomes"], seed, draw_index=draw
        )
        outcome_frames.append(frame)
        rate = frame["ctcae_grade_max"].ge(3).mean()
        log(f"  draw {draw} ({effects.version}): grade3+ rate {rate:.3f}")
    primary_outcomes = outcome_frames[0]
    all_outcomes = pd.concat(outcome_frames, ignore_index=True)

    # --- Layer B, model v2 -------------------------------------------------
    # Emitted alongside v1, never instead of it, and on its own RNG offset so
    # every v1 output stays byte-identical whether or not v2 runs.
    v2_spec = config.get("outcomes_v2") or {}
    outcomes_v2_primary = None
    outcomes_v2_all = None
    if v2_spec.get("enabled", False):
        log("simulating Layer B v2 (grade 4-5, sourced effect sizes) ...")
        v2_frames = []
        for draw in range(int(v2_spec["sensitivity_draws"])):
            frame, effects = outcomes.simulate_outcomes_v2(
                patients, expected, genotypes_long, catalog, v2_spec, seed,
                draw_index=draw,
            )
            v2_frames.append(frame)
            rate = frame["severe_grade45"].mean()
            target = float(v2_spec["target_event_rate"])
            log(f"  draw {draw} ({effects.version}): grade4-5 rate {rate:.4f} "
                f"(target {target:.4f})")
        outcomes_v2_primary = v2_frames[0]
        outcomes_v2_all = pd.concat(v2_frames, ignore_index=True)
        log(f"  rare-deleterious carriers "
            f"{outcomes_v2_primary['rare_deleterious_carrier'].mean():.4f}, "
            f"rs1801160 carriers "
            f"{outcomes_v2_primary['rs1801160_carrier'].mean():.4f}")

    # --- Layer B, model v3 -------------------------------------------------
    v3_spec = config.get("outcomes_v3") or {}
    outcomes_v3_primary = None
    outcomes_v3_all = None
    if v3_spec.get("enabled", False) and "bmi_category" in patients.columns:
        log("simulating Layer B v3 (FUSAFE clinical + DPYD) ...")
        v3_frames = []
        for draw in range(int(v3_spec["sensitivity_draws"])):
            frame, effects = outcomes.simulate_outcomes_v3(
                patients, genotypes_long, v3_spec, seed, draw_index=draw
            )
            v3_frames.append(frame)
            log(f"  draw {draw} ({effects.version}): grade4-5 rate "
                f"{frame['severe_grade45'].mean():.4f} "
                f"(target {float(v3_spec['target_event_rate']):.4f})")
        outcomes_v3_primary = v3_frames[0]
        outcomes_v3_all = pd.concat(v3_frames, ignore_index=True)
        log(f"  DPYD *2A/*13/D949V carriers "
            f"{outcomes_v3_primary['dpyd_aggregate_carrier'].mean():.4f}, "
            f"HapB3 {outcomes_v3_primary['hapb3_carrier'].mean():.4f}")

    # --- write ------------------------------------------------------------
    log("writing data files ...")
    patient_columns = [
        "patient_id", "ancestry_super_pop", "ancestry_sub_pop", "admixture_json",
        "sex", "age_years", "cancer_type", "regimen", "dose_mg_m2",
        "renal_function_egfr", "co_medication_flags", "generation_method", "seed",
        "is_real_variant_anchor", "cohort", "label_type",
    ]
    if "bmi" in patients.columns:
        patient_columns = patient_columns[:-1] + ["bmi", "bmi_category", "label_type"]
    patients_out = patients[patient_columns]
    patients_out.to_parquet(data_dir / "patients.parquet", index=False)
    genotypes_long.to_parquet(data_dir / "genotypes_long.parquet", index=False)
    expected.to_parquet(data_dir / "expected_engine_output.parquet", index=False)
    primary_outcomes.to_parquet(data_dir / "outcomes.parquet", index=False)
    all_outcomes.to_parquet(data_dir / "outcomes_sensitivity.parquet", index=False)
    if outcomes_v3_primary is not None:
        outcomes_v3_primary.to_parquet(data_dir / "outcomes_v3.parquet", index=False)
        outcomes_v3_all.to_parquet(
            data_dir / "outcomes_v3_sensitivity.parquet", index=False
        )
    if outcomes_v2_primary is not None:
        outcomes_v2_primary.to_parquet(data_dir / "outcomes_v2.parquet", index=False)
        outcomes_v2_all.to_parquet(
            data_dir / "outcomes_v2_sensitivity.parquet", index=False
        )
    refusal_frame.to_parquet(data_dir / "refusal_cases.parquet", index=False)
    if not coverage_frame.empty:
        coverage_frame.to_parquet(data_dir / "coverage_cases.parquet", index=False)
        pd.DataFrame(coverage_calls).to_parquet(
            data_dir / "coverage_genotypes_long.parquet", index=False
        )
    pd.DataFrame(refusal_calls).to_parquet(
        data_dir / "refusal_genotypes_long.parquet", index=False
    )

    # --- VCFs --------------------------------------------------------------
    log("writing VCFs ...")
    vcf_dir = data_dir / "vcf"
    for old in vcf_dir.glob("*.vcf"):
        old.unlink()
    limit = args.max_patient_vcfs or len(patients)
    written = 0
    for patient_id in list(patients["patient_id"])[:limit]:
        calls = list(per_patient_calls[patient_id].values())
        vcfio.write_patient_vcf(
            vcf_dir / f"{patient_id}.vcf", patient_id, calls,
            int(config["engine"]["min_dp"]), int(config["engine"]["min_gq"]),
            meta={"label": LABEL},
        )
        written += 1
    for patient in (refusal_patients + coverage_patients)[: min(len(refusal_patients) + len(coverage_patients), limit)]:
        calls = list(per_patient_calls[patient["patient_id"]].values())
        vcfio.write_patient_vcf(
            vcf_dir / f"{patient['patient_id']}.vcf", patient["patient_id"], calls,
            int(config["engine"]["min_dp"]), int(config["engine"]["min_gq"]),
            meta={"label": LABEL, "case_category": patient["case_category"]},
        )
        written += 1

    sites = snv_catalog.rename(columns={"grch38_pos": "pos_grch38"}).to_dict("records")
    sites = [s for s in sites if s["variant_id"] in carriers]
    cohort_ids = list(patients["patient_id"])
    vcfio.write_cohort_vcf(
        data_dir / "cohort.vcf.gz", cohort_ids, sites, per_patient_calls,
        int(config["engine"]["min_dp"]), int(config["engine"]["min_gq"]),
    )
    log(f"  {written} per-patient VCFs + cohort.vcf.gz")

    # --- splits -------------------------------------------------------------
    write_splits(patients, config, root / "splits", seed)

    # --- QC inputs -----------------------------------------------------------
    log("saving QC inputs ...")
    background, background_positions = genotypes.background_matrix(
        ts, patient_pairs,
        max_sites=int(config["genotypes"]["background_sites_for_qc"]),
        min_maf=float(config["genotypes"]["background_min_maf"]),
    )
    np.savez_compressed(
        root / "qc" / "background_genotypes.npz",
        dosages=background,
        positions=background_positions,
        populations=patients["ancestry_super_pop"].to_numpy().astype("U3"),
        sub_populations=patients["ancestry_sub_pop"].fillna("NA").to_numpy().astype("U3"),
        patient_ids=patients["patient_id"].to_numpy().astype("U16"),
    )

    manifest = {
        "label": LABEL,
        "dataset": config["dataset"],
        "seed": seed,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "region": {"chrom": region.chrom, "start": region.start, "end": region.end},
        "n_patients": int(len(patients)),
        "n_refusal_cases": int(len(refusal_frame)),
        "n_catalog_variants": int(len(catalog)),
        "n_injected_variants": int(len(targets)),
        "variants_without_frequency": missing_freq,
        "layer_a_caller": caller.name,
        "refusal_rule_ids_available": bool(rules.available),
        "refusal_rule_source": rules.source,
        "cpic_version": cpic_version,
        "pharmvar_version": None,
        "covariate_provenance": covariate_provenance,
        "demography_provenance": demography_provenance,
        "coalescent": {
            "trees": int(ts.num_trees),
            "background_sites": int(ts.num_sites),
            "recombination_map": config["genotypes"]["recombination_map"],
        },
        "outcome_model_versions": sorted(all_outcomes["outcome_model_version"].unique()),
        "outcomes_v2": (
            {
                "enabled": True,
                "endpoint": str(v2_spec["endpoint"]),
                "target_event_rate": float(v2_spec["target_event_rate"]),
                "realised_event_rate": float(
                    outcomes_v2_primary["severe_grade45"].mean()
                ),
                "rare_deleterious_carrier_rate": float(
                    outcomes_v2_primary["rare_deleterious_carrier"].mean()
                ),
                "rs1801160_carrier_rate": float(
                    outcomes_v2_primary["rs1801160_carrier"].mean()
                ),
                "versions": sorted(outcomes_v2_all["outcome_model_version"].unique()),
                "source_pmids": sorted(
                    {
                        pmid
                        for entry in v2_spec["coefficients"].values()
                        for pmid in (entry.get("source_pmids") or [])
                    }
                ),
            }
            if outcomes_v2_primary is not None
            else {"enabled": False}
        ),
    }
    (root / "qc" / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    log("done.")
    return 0


def write_splits(patients: pd.DataFrame, config: dict, splits_dir: Path, seed: int) -> None:
    """70/15/15 by patient, stratified by ancestry so no stratum is starved."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 41)
    spec = config["splits"]
    assignment = {}
    for pop, group in patients.groupby("ancestry_super_pop"):
        ids = group["patient_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_train = int(round(len(ids) * float(spec["train"])))
        n_val = int(round(len(ids) * float(spec["val"])))
        for pid in ids[:n_train]:
            assignment[pid] = "train"
        for pid in ids[n_train:n_train + n_val]:
            assignment[pid] = "val"
        for pid in ids[n_train + n_val:]:
            assignment[pid] = "test"
    for split in ("train", "val", "test"):
        members = sorted(p for p, s in assignment.items() if s == split)
        (splits_dir / f"{split}.txt").write_text(
            "\n".join(members) + "\n", encoding="utf-8", newline="\n"
        )
    log(f"  splits: " + ", ".join(
        f"{s}={sum(1 for v in assignment.values() if v == s)}"
        for s in ("train", "val", "test")
    ))


if __name__ == "__main__":
    raise SystemExit(main())

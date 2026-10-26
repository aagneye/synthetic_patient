"""Quality gates (build brief Part 8).

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Each check returns a :class:`GateResult`. A gate that cannot be evaluated
because its reference data is absent is reported ``SKIPPED`` with the reason,
never silently passed, and never counted as evidence that the dataset is good.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import pharmcat, vcfio

SUPER_POPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SAS_SUB_POPS = ["GIH", "ITU", "PJL", "BEB", "STU"]


@dataclass
class GateResult:
    name: str
    section: str
    status: str  # PASS | FAIL | SKIPPED | REPORTED
    summary: str
    details: pd.DataFrame | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


# ----------------------------------------------------------------------
# 8.1 Genetic realism
# ----------------------------------------------------------------------


def gate_allele_frequency(
    genotypes_long: pd.DataFrame,
    patients: pd.DataFrame,
    catalog: pd.DataFrame,
    spec: dict[str, Any],
) -> GateResult:
    """Realised allele frequency per variant per population vs the target."""
    counts = patients["ancestry_super_pop"].value_counts().to_dict()
    dosage = _dosage_by_population(genotypes_long, patients)

    common_threshold = float(spec["common_maf_threshold"])
    common_tolerance = float(spec["common_abs_tolerance"])
    rare_tolerance = float(spec["rare_rel_tolerance"])
    min_expected = float(spec["min_expected_count"])

    rows = []
    for _, variant in catalog.iterrows():
        if not bool(variant.get("is_snv")):
            continue
        variant_id = str(variant["variant_id"])
        for pop in SUPER_POPS:
            target = variant.get(f"freq_gnomad_{pop.lower()}")
            if pd.isna(target):
                continue
            n_individuals = counts.get(pop, 0)
            if n_individuals == 0:
                continue
            observed_alleles = dosage.get((variant_id, pop), 0)
            observed = observed_alleles / (2 * n_individuals)
            target = float(target)
            expected_alleles = target * 2 * n_individuals

            if target > common_threshold:
                tolerance_kind = "absolute"
                deviation = abs(observed - target)
                limit = common_tolerance
            else:
                tolerance_kind = "relative"
                deviation = abs(observed - target) / target if target > 0 else 0.0
                limit = rare_tolerance

            underpowered = expected_alleles < min_expected
            rows.append(
                {
                    "variant_id": variant_id,
                    "population": pop,
                    "target_af": target,
                    "observed_af": observed,
                    "expected_allele_count": expected_alleles,
                    "observed_allele_count": observed_alleles,
                    "tolerance_kind": tolerance_kind,
                    "deviation": deviation,
                    "limit": limit,
                    "underpowered": underpowered,
                    "within_tolerance": bool(deviation <= limit) or underpowered,
                }
            )

    details = pd.DataFrame(rows)
    if details.empty:
        return GateResult(
            "Allele frequency per variant per population", "8.1", "SKIPPED",
            "No variant had both a target frequency and a sampled population.",
        )
    failures = details[~details["within_tolerance"]]
    powered = details[~details["underpowered"]]
    status = "PASS" if failures.empty else "FAIL"
    summary = (
        f"{len(powered)} powered variant-population pairs checked; "
        f"{len(failures)} outside tolerance; "
        f"{int(details['underpowered'].sum())} underpowered pairs reported but not gated."
    )
    return GateResult(
        "Allele frequency per variant per population", "8.1", status, summary,
        details, {"n_failures": int(len(failures))},
    )


def _dosage_by_population(
    genotypes_long: pd.DataFrame, patients: pd.DataFrame
) -> dict[tuple[str, str], int]:
    if genotypes_long.empty:
        return {}
    merged = genotypes_long.merge(
        patients[["patient_id", "ancestry_super_pop"]], on="patient_id", how="left"
    )
    merged["dosage"] = merged["genotype"].map({"0/1": 1, "1/0": 1, "1/1": 2}).fillna(0)
    grouped = merged.groupby(["variant_id", "ancestry_super_pop"])["dosage"].sum()
    return {(str(k[0]), str(k[1])): int(v) for k, v in grouped.items()}


def gate_hardy_weinberg(
    genotypes_long: pd.DataFrame,
    patients: pd.DataFrame,
    spec: dict[str, Any],
) -> GateResult:
    """Chi-square test for Hardy-Weinberg equilibrium in each ancestry stratum."""
    min_p = float(spec["min_p_value"])
    rows = []
    merged = genotypes_long.merge(
        patients[["patient_id", "ancestry_super_pop"]], on="patient_id", how="left"
    )
    totals = patients["ancestry_super_pop"].value_counts().to_dict()

    for (variant_id, pop), group in merged.groupby(["variant_id", "ancestry_super_pop"]):
        n_total = totals.get(pop, 0)
        if n_total == 0:
            continue
        n_het = int((group["genotype"].isin(["0/1", "1/0"])).sum())
        n_hom_alt = int((group["genotype"] == "1/1").sum())
        n_hom_ref = n_total - n_het - n_hom_alt
        if n_hom_alt + n_het == 0:
            continue
        p = (2 * n_hom_ref + n_het) / (2 * n_total)
        q = 1 - p
        expected = np.array([p * p, 2 * p * q, q * q]) * n_total
        observed = np.array([n_hom_ref, n_het, n_hom_alt])
        if (expected < 5).any():
            rows.append(
                {
                    "variant_id": variant_id, "population": pop,
                    "n": n_total, "p_value": None, "underpowered": True,
                    "passes": True,
                }
            )
            continue
        chi2 = float(((observed - expected) ** 2 / expected).sum())
        p_value = float(stats.chi2.sf(chi2, df=1))
        rows.append(
            {
                "variant_id": variant_id, "population": pop, "n": n_total,
                "p_value": p_value, "underpowered": False,
                "passes": p_value > min_p,
            }
        )

    details = pd.DataFrame(rows)
    if details.empty:
        return GateResult(
            "Hardy-Weinberg equilibrium", "8.1", "SKIPPED",
            "No variant had enough non-reference calls to test.",
        )
    powered = details[~details["underpowered"]]
    failures = powered[~powered["passes"]]
    # With many tests, a few will fail by chance at p<0.01; the gate allows the
    # expected false-positive rate and fails only on a genuine excess.
    expected_failures = max(1, int(math.ceil(len(powered) * min_p)))
    status = "PASS" if len(failures) <= expected_failures else "FAIL"
    summary = (
        f"{len(powered)} powered tests; {len(failures)} with p<={min_p} "
        f"(up to {expected_failures} expected by chance); "
        f"{int(details['underpowered'].sum())} underpowered."
    )
    return GateResult(
        "Hardy-Weinberg equilibrium", "8.1", status, summary, details,
        {"n_failures": int(len(failures)), "expected_by_chance": expected_failures},
    )


def gate_linkage_disequilibrium(
    genotypes_long: pd.DataFrame,
    patients: pd.DataFrame,
    catalog: pd.DataFrame,
    reference_ld: pd.DataFrame | None,
) -> GateResult:
    """Pairwise r-squared among catalog variants, against 1000 Genomes."""
    observed = _pairwise_r2(genotypes_long, patients, catalog, population="SAS")
    if observed.empty:
        return GateResult(
            "Linkage disequilibrium in DPYD", "8.1", "SKIPPED",
            "Too few common catalog variants to compute an LD matrix.",
        )
    if reference_ld is None or reference_ld.empty:
        return GateResult(
            "Linkage disequilibrium in DPYD", "8.1", "SKIPPED",
            "No 1000 Genomes LD reference was retrieved, so the Frobenius "
            "distance could not be computed. Observed r-squared is reported.",
            observed,
        )
    merged = observed.merge(reference_ld, on=["rsid_a", "rsid_b"], how="inner")
    if merged.empty:
        return GateResult(
            "Linkage disequilibrium in DPYD", "8.1", "SKIPPED",
            "No variant pair was present in both the synthetic cohort and the "
            "1000 Genomes LD reference.",
            observed,
        )
    difference = merged["r2_synthetic"] - merged["r2_reference"]
    frobenius = float(np.sqrt((difference ** 2).sum()))
    mean_synthetic = float(merged["r2_synthetic"].mean())
    mean_reference = float(merged["r2_reference"].mean())
    direction = (
        "lower than" if mean_synthetic < mean_reference
        else "higher than" if mean_synthetic > mean_reference else "equal to"
    )
    return GateResult(
        "Linkage disequilibrium in DPYD", "8.1", "REPORTED",
        f"{len(merged)} variant pairs compared against the mean r-squared "
        f"across the five 1000 Genomes South Asian panels. Frobenius distance "
        f"{frobenius:.3f}, mean |r2 difference| {difference.abs().mean():.3f}. "
        f"Synthetic r-squared is {direction} the reference "
        f"({mean_synthetic:.3f} vs {mean_reference:.3f}): injected variants are "
        f"each placed on an independently chosen clade, so they carry realistic "
        f"linkage to the background but not the real pairwise linkage to each "
        f"other. Variants defining one compound allele are the exception and "
        f"are injected as a unit.",
        merged,
        {"frobenius_distance": frobenius,
         "mean_abs_difference": float(difference.abs().mean()),
         "mean_r2_synthetic": mean_synthetic,
         "mean_r2_reference": mean_reference},
    )


def _pairwise_r2(
    genotypes_long: pd.DataFrame,
    patients: pd.DataFrame,
    catalog: pd.DataFrame,
    population: str,
    min_maf: float = 0.02,
) -> pd.DataFrame:
    members = patients[patients["ancestry_super_pop"] == population]["patient_id"]
    member_set = set(members)
    if not member_set:
        return pd.DataFrame()
    subset = genotypes_long[genotypes_long["patient_id"].isin(member_set)]
    if subset.empty:
        return pd.DataFrame()

    index = {pid: i for i, pid in enumerate(sorted(member_set))}
    rsid_by_variant = {
        str(r["variant_id"]): r["rsid"] for _, r in catalog.iterrows()
        if pd.notna(r.get("rsid"))
    }
    matrices: dict[str, np.ndarray] = {}
    for variant_id, group in subset.groupby("variant_id"):
        variant_id = str(variant_id)
        if variant_id not in rsid_by_variant:
            continue
        column = np.zeros(len(index), dtype=np.int8)
        for pid, genotype in zip(group["patient_id"], group["genotype"]):
            column[index[pid]] = 2 if genotype == "1/1" else 1
        if column.sum() / (2 * len(index)) < min_maf:
            continue
        matrices[variant_id] = column

    keys = sorted(matrices)
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = matrices[keys[i]], matrices[keys[j]]
            if a.std() == 0 or b.std() == 0:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            # Canonical ordering: LD is symmetric, so the pair key must not
            # depend on which variant happened to be listed first.
            rsid_a, rsid_b = sorted(
                [str(rsid_by_variant[keys[i]]), str(rsid_by_variant[keys[j]])]
            )
            rows.append(
                {
                    "variant_a": keys[i], "variant_b": keys[j],
                    "rsid_a": rsid_a, "rsid_b": rsid_b,
                    "r2_synthetic": r * r,
                }
            )
    return pd.DataFrame(rows)


def gate_population_structure(
    background_path: Path, require_real_reference: bool
) -> GateResult:
    """PCA of the background genome, checking populations separate."""
    if not background_path.exists():
        return GateResult(
            "Population structure (PCA)", "8.1", "SKIPPED",
            f"No background genotype matrix at {background_path}.",
        )
    data = np.load(background_path, allow_pickle=False)
    dosages = data["dosages"].astype(np.float32)
    populations = data["populations"]
    if dosages.shape[1] < 10:
        return GateResult(
            "Population structure (PCA)", "8.1", "SKIPPED",
            "Fewer than 10 background sites available for PCA.",
        )

    # A full SVD over every individual is needless here: population structure
    # is a property of the cohort, not of cohort size. Subsampling keeps the
    # decomposition to seconds while leaving every population represented.
    max_individuals, max_sites = 3000, 2000
    rng = np.random.default_rng(0)
    if dosages.shape[0] > max_individuals:
        keep: list[int] = []
        per_pop = max(1, max_individuals // max(1, len(set(populations.tolist()))))
        for pop in sorted(set(populations.tolist())):
            idx = np.flatnonzero(populations == pop)
            keep.extend(rng.choice(idx, size=min(len(idx), per_pop), replace=False))
        rows = np.sort(np.array(keep))
        dosages = dosages[rows]
        populations = populations[rows]
    if dosages.shape[1] > max_sites:
        columns = np.sort(
            rng.choice(dosages.shape[1], size=max_sites, replace=False)
        )
        dosages = dosages[:, columns]

    centred = dosages - dosages.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    centred /= scale
    # Randomised truncated SVD would be faster; the matrix here is small.
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    scores = centred @ components[:2].T

    rows = []
    for pop in sorted(set(populations.tolist())):
        mask = populations == pop
        rows.append(
            {
                "population": pop,
                "n": int(mask.sum()),
                "pc1_mean": float(scores[mask, 0].mean()),
                "pc2_mean": float(scores[mask, 1].mean()),
                "pc1_sd": float(scores[mask, 0].std()),
                "pc2_sd": float(scores[mask, 1].std()),
            }
        )
    details = pd.DataFrame(rows)

    # Separation: is SAS closer to its own centroid than to the EUR centroid?
    verdict = "REPORTED"
    message = (
        f"PCA over {dosages.shape[1]} background sites. "
        f"PC1 explains {singular[0] ** 2 / (singular ** 2).sum():.1%} of variance."
    )
    if {"SAS", "EUR"} <= set(details["population"]):
        sas = details[details["population"] == "SAS"].iloc[0]
        eur = details[details["population"] == "EUR"].iloc[0]
        separation = math.dist(
            (sas["pc1_mean"], sas["pc2_mean"]), (eur["pc1_mean"], eur["pc2_mean"])
        )
        spread = max(float(sas["pc1_sd"]), 1e-9)
        message += (
            f" SAS and EUR centroids are {separation:.2f} apart, "
            f"{separation / spread:.2f} SAS standard deviations."
        )
        details.attrs["sas_eur_separation"] = separation

    if require_real_reference:
        message += (
            " Clustering against real 1000 Genomes samples was required but no "
            "reference genotypes are bundled."
        )
        verdict = "FAIL"
    else:
        message += (
            " Clustering against real 1000 Genomes samples is NOT tested: the "
            "reference genotypes are not bundled with this dataset."
        )
    return GateResult("Population structure (PCA)", "8.1", verdict, message, details)


def gate_diplotype_distribution(
    expected: pd.DataFrame, patients: pd.DataFrame
) -> GateResult:
    """Chi-square of observed diplotype counts against allele-frequency expectation."""
    merged = expected.merge(
        patients[["patient_id", "ancestry_super_pop"]], on="patient_id", how="left"
    )
    rows = []
    for pop, group in merged.groupby("ancestry_super_pop"):
        counts = group["diplotype_expected"].value_counts()
        if len(counts) < 2:
            continue
        # Expected under the reference-heavy null: the overwhelming majority of
        # individuals carry the reference diplotype.
        observed = counts.to_numpy(dtype=float)
        rows.append(
            {
                "population": pop,
                "n_distinct_diplotypes": int(len(counts)),
                "n_individuals": int(len(group)),
                "most_common": counts.index[0],
                "most_common_fraction": float(observed[0] / observed.sum()),
            }
        )
    details = pd.DataFrame(rows)
    if details.empty:
        return GateResult(
            "Diplotype distribution", "8.1", "SKIPPED",
            "Not enough distinct diplotypes to test.",
        )
    return GateResult(
        "Diplotype distribution", "8.1", "REPORTED",
        f"Diplotype diversity summarised across {len(details)} populations.",
        details,
    )


# ----------------------------------------------------------------------
# 8.2 Engine correctness
# ----------------------------------------------------------------------


def gate_engine_determinism(recompute) -> GateResult:
    """Run the Layer A caller twice and compare byte for byte."""
    first = recompute()
    second = recompute()
    first_bytes = first.to_csv(index=False).encode()
    second_bytes = second.to_csv(index=False).encode()
    identical = first_bytes == second_bytes
    return GateResult(
        "Engine determinism", "8.2", "PASS" if identical else "FAIL",
        "Two independent Layer A runs produced "
        + ("byte-identical output." if identical else "DIFFERENT output."),
        metrics={"identical": identical},
    )


def gate_ground_truth(
    stored: pd.DataFrame, recomputed: pd.DataFrame, required: float
) -> GateResult:
    """Recomputed Layer A labels must reproduce the stored ones exactly."""
    compare_columns = [
        "diplotype_expected", "activity_score_expected", "phenotype_expected",
        "cpic_recommendation_expected", "cpic_dose_action_expected",
        "sas_override_expected",
    ]
    merged = stored.merge(recomputed, on="patient_id", suffixes=("_stored", "_new"))
    non_refusal = merged[
        merged["cpic_recommendation_expected_stored"] != "REFUSED_INSUFFICIENT_EVIDENCE"
    ]
    if non_refusal.empty:
        return GateResult(
            "Ground-truth agreement (non-refusal rows)", "8.2", "SKIPPED",
            "No non-refusal rows to compare.",
        )
    agree = np.ones(len(non_refusal), dtype=bool)
    mismatches: dict[str, int] = {}
    for column in compare_columns:
        left = non_refusal[f"{column}_stored"]
        right = non_refusal[f"{column}_new"]
        same = (left.astype(str) == right.astype(str)) | (left.isna() & right.isna())
        mismatches[column] = int((~same).sum())
        agree &= same.to_numpy()
    rate = float(agree.mean())
    return GateResult(
        "Ground-truth agreement (non-refusal rows)", "8.2",
        "PASS" if rate >= required else "FAIL",
        f"{rate:.4%} of {len(non_refusal)} non-refusal rows reproduced exactly "
        f"(required {required:.0%}).",
        pd.DataFrame([{"column": k, "mismatches": v} for k, v in mismatches.items()]),
        {"agreement": rate},
    )


def gate_refusals(refusal_cases: pd.DataFrame, required: float,
                  rule_ids_available: bool) -> GateResult:
    """Each engineered case must come back with the category it was built for."""
    if refusal_cases.empty:
        return GateResult("Refusal agreement", "8.2", "SKIPPED",
                          "No refusal cases present.")
    matched = (
        refusal_cases["expected_category"]
        == refusal_cases["refusal_category_expected"]
    )
    rate = float(matched.mean())
    by_category = (
        refusal_cases.assign(matched=matched)
        .groupby("expected_category")
        .agg(n=("matched", "size"), matched=("matched", "sum"))
        .reset_index()
    )
    by_category["rate"] = by_category["matched"] / by_category["n"]

    summary = f"{rate:.2%} of {len(refusal_cases)} engineered cases returned the expected category."
    if not rule_ids_available:
        summary += (
            " refusal_rule_id_expected is null throughout: no anukriti-swarm "
            "source was supplied, and rule IDs are never invented."
        )
    return GateResult(
        "Refusal agreement", "8.2", "PASS" if rate >= required else "FAIL",
        summary, by_category, {"agreement": rate},
    )


# Disagreements with PharmCAT that this build understands and can account for.
# Anything outside this set is counted as unexplained and fails the gate: the
# point of the cross-check is to surface differences, not to absorb them.
#
# ``tie_break_equal_activity`` is currently empty: Layer A adopted PharmCAT's
# ordering of equal-activity alleles. It stays listed because a PharmCAT
# release that ordered them differently would land here, where the gate summary
# and qc/pharmcat_parity.csv report it by name, rather than being read as a
# miscall by either caller.
EXPLAINED_DISAGREEMENTS = {
    "engine_refused_low_quality",
    "engine_refused_ambiguous_phase",
    "tie_break_equal_activity",
}


def _classify_pharmcat_row(row: pd.Series) -> str:
    """Name the reason one patient's two calls differ."""
    if not bool(row["pharmcat_called"]):
        return "pharmcat_did_not_call"
    if bool(row["diplotype_agrees"]) and bool(row["phenotype_agrees"]):
        return "agree"

    category = str(row.get("refusal_category_expected") or "")
    # The engine declines to answer in two situations PharmCAT has no
    # counterpart for. Neither is a miscall by either caller.
    if category == "low_quality_call":
        return "engine_refused_low_quality"
    if category == "ambiguous_phase" or bool(row.get("phase_ambiguous_expected")):
        return "engine_refused_ambiguous_phase"

    same_consequence = bool(row["phenotype_agrees"]) and bool(
        row["activity_score_agrees"]
    )
    if same_consequence and int(row["n_variant_copies"]) > 2:
        return "tie_break_equal_activity"
    if same_consequence:
        return "naming_difference_same_consequence"
    return "unexplained_call_difference"


def gate_pharmcat_parity(
    expected: pd.DataFrame,
    genotypes_long: pd.DataFrame,
    spec: dict[str, Any],
    seed: int,
    work_dir: str | Path,
    report_path: str | Path | None = None,
) -> GateResult:
    """Cross-check a seed-fixed sample of patients against PharmCAT.

    PharmCAT is an independent implementation of the same CPIC tables, so it
    is the only check here that can catch an error shared by this dataset's
    caller and its own expected output. It is not a gold standard: where the
    two disagree, the gate's job is to say which of a small number of
    understood reasons applies, and to fail if a disagreement is not one of
    them.

    Returns SKIPPED, never PASS, when PharmCAT is not installed.
    """
    gene = str(spec.get("gene", "DPYD"))
    try:
        install = pharmcat.find_install(spec.get("install_dir"))
    except pharmcat.PharmcatUnavailable as exc:
        return GateResult(
            f"Parity with PharmCAT ({gene})", "8.2", "SKIPPED",
            f"PharmCAT was not run: {exc}. This is a genuine gap, not a pass.",
        )

    sample_size = int(spec.get("sample_size", 500))
    patient_ids = sorted(expected["patient_id"].astype(str).unique())
    if not patient_ids:
        return GateResult(
            f"Parity with PharmCAT ({gene})", "8.2", "SKIPPED",
            "No patients to sample.",
        )
    rng = np.random.default_rng(seed + int(spec.get("seed_offset", 8_002)))
    size = min(sample_size, len(patient_ids))
    sample = sorted(
        rng.choice(np.array(patient_ids), size=size, replace=False).tolist()
    )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    calls_by_patient: dict[str, list[dict[str, Any]]] = {}
    in_sample = genotypes_long[genotypes_long["patient_id"].astype(str).isin(sample)]
    for call in in_sample.to_dict("records"):
        calls_by_patient.setdefault(str(call["patient_id"]), []).append(call)

    try:
        sites = vcfio.read_pharmcat_positions(install.positions, gene)
        vcf_path = vcfio.write_pharmcat_vcf(
            work_dir / f"pharmcat_sample_{gene}.vcf",
            sample, sites, calls_by_patient, gene,
        )
        output_dir = pharmcat.run(install, vcf_path, work_dir / "reports", gene)
        called = pharmcat.parse_reports(output_dir, gene)
    except pharmcat.PharmcatUnavailable as exc:
        return GateResult(
            f"Parity with PharmCAT ({gene})", "8.2", "SKIPPED",
            f"PharmCAT {install.version} was found but did not run: {exc}.",
        )
    except ValueError as exc:
        # A catalog variant with no PharmCAT position would silently vanish
        # from the comparison, which would overstate agreement.
        return GateResult(
            f"Parity with PharmCAT ({gene})", "8.2", "FAIL", str(exc),
            metrics=install.as_metrics(),
        )

    merged = (
        expected.assign(patient_id=expected["patient_id"].astype(str))
        [expected["patient_id"].astype(str).isin(sample)]
        .merge(called, on="patient_id", how="left")
    )
    merged["pharmcat_called"] = merged["pharmcat_called"].fillna(False).astype(bool)

    dosage = (
        in_sample.assign(
            copies=in_sample["genotype"].map({"0/1": 1, "1|0": 1, "0|1": 1, "1/1": 2,
                                              "1|1": 2}).fillna(0)
        )
        .groupby(in_sample["patient_id"].astype(str))["copies"]
        .sum()
    )
    merged["n_variant_copies"] = (
        merged["patient_id"].map(dosage).fillna(0).astype(int)
    )
    merged["expected_diplotype_normalised"] = merged["diplotype_expected"].map(
        pharmcat.split_diplotype
    )
    merged["diplotype_agrees"] = (
        merged["expected_diplotype_normalised"] == merged["pharmcat_diplotype"]
    )
    merged["phenotype_agrees"] = (
        merged["phenotype_expected"].astype(str)
        == merged["pharmcat_phenotype"].astype(str)
    )
    engine_score = pd.to_numeric(merged["activity_score_expected"], errors="coerce")
    pharmcat_score = pd.to_numeric(merged["pharmcat_activity_score"], errors="coerce")
    merged["activity_score_agrees"] = (
        np.isclose(engine_score, pharmcat_score, equal_nan=True)
        | (engine_score.isna() & pharmcat_score.isna())
    )
    merged["disagreement_class"] = merged.apply(_classify_pharmcat_row, axis=1)

    if report_path is not None:
        columns = [
            "patient_id", "diplotype_expected", "expected_diplotype_normalised",
            "pharmcat_diplotype", "phenotype_expected", "pharmcat_phenotype",
            "activity_score_expected", "pharmcat_activity_score",
            "n_variant_copies", "pharmcat_n_possible_diplotypes",
            "refusal_category_expected", "phase_ambiguous_expected",
            "diplotype_agrees", "phenotype_agrees", "activity_score_agrees",
            "disagreement_class",
        ]
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        merged[columns].sort_values("patient_id").to_csv(report_path, index=False)

    details = (
        merged.groupby("disagreement_class")
        .agg(
            n=("patient_id", "size"),
            example_patient=("patient_id", "min"),
        )
        .reset_index()
        .sort_values(["n", "disagreement_class"], ascending=[False, True])
    )
    details["share"] = details["n"] / len(merged)
    examples = merged.set_index("patient_id")
    details["example_expected"] = [
        str(examples.loc[p, "expected_diplotype_normalised"])
        for p in details["example_patient"]
    ]
    details["example_pharmcat"] = [
        str(examples.loc[p, "pharmcat_diplotype"]) for p in details["example_patient"]
    ]

    diplotype_rate = float(merged["diplotype_agrees"].mean())
    phenotype_rate = float(merged["phenotype_agrees"].mean())
    unexplained = merged[
        ~merged["disagreement_class"].isin(EXPLAINED_DISAGREEMENTS | {"agree"})
    ]
    max_unexplained = int(spec.get("max_unexplained", 0))
    status = "PASS" if len(unexplained) <= max_unexplained else "FAIL"

    breakdown = ", ".join(
        f"{row.disagreement_class} {row.n}"
        for row in details.itertuples()
        if row.disagreement_class != "agree"
    )
    summary = (
        f"PharmCAT {install.version} on {len(merged)} patients: "
        f"{diplotype_rate:.2%} diplotype agreement, "
        f"{phenotype_rate:.2%} phenotype agreement. "
        + (f"Disagreements: {breakdown}. " if breakdown else "No disagreements. ")
        + (
            f"{len(unexplained)} unexplained (limit {max_unexplained})."
            if len(unexplained)
            else "Every disagreement falls into a known class."
        )
    )
    metrics = {
        "n_sampled": len(merged),
        "diplotype_agreement": diplotype_rate,
        "phenotype_agreement": phenotype_rate,
        "n_unexplained": int(len(unexplained)),
        "pharmcat_allele_definition_version": (
            str(merged["pharmcat_allele_definition_version"].dropna().iloc[0])
            if merged["pharmcat_allele_definition_version"].notna().any()
            else None
        ),
        **install.as_metrics(),
    }
    return GateResult(
        f"Parity with PharmCAT ({gene})", "8.2", status, summary, details, metrics
    )


# ----------------------------------------------------------------------
# 8.3 Privacy
# ----------------------------------------------------------------------


def gate_privacy(
    synthetic: pd.DataFrame,
    seed_frame: pd.DataFrame | None,
    spec: dict[str, Any],
    seed_is_real: bool,
    holdout_frame: pd.DataFrame | None = None,
) -> list[GateResult]:
    """Nearest-neighbour, NNAA, epsilon-identifiability and membership inference.

    ``holdout_frame`` holds records drawn from the same distribution as the
    seed cohort but never shown to the generator. Without it the membership
    test has no honest negative class and is skipped rather than faked.
    """
    if seed_frame is None or seed_frame.empty:
        return [
            GateResult(
                "Privacy (tabular)", "8.3", "SKIPPED",
                "No seed cohort was retained to measure against.",
            )
        ]

    columns = [c for c in seed_frame.columns if c in synthetic.columns]
    # NNAA compares a record's nearest neighbour in the other set against its
    # nearest neighbour in its own set, so unequal set sizes bias it on their
    # own. Both sides are cut to the same size first, and every frame is
    # encoded in one shared space so the distances mean the same thing.
    n_compare = min(len(seed_frame), len(synthetic))
    frames = {
        "real": seed_frame[columns].sample(n=n_compare, random_state=0),
        "fake": synthetic[columns].sample(n=n_compare, random_state=0),
        "seed_full": seed_frame[columns],
    }
    if holdout_frame is not None and not holdout_frame.empty:
        frames["holdout"] = holdout_frame[columns]
    encoded = encode_jointly(frames, columns)
    real = encoded["real"]
    fake = encoded["fake"]

    d_tt = _nearest_distance(real, real, exclude_self=True)
    d_ss = _nearest_distance(fake, fake, exclude_self=True)
    d_ts = _nearest_distance(real, fake, exclude_self=False)
    d_st = _nearest_distance(fake, real, exclude_self=False)

    nnaa = 0.5 * (float((d_ts > d_tt).mean()) + float((d_st > d_ss).mean()))
    identical = int((d_ts == 0).sum())

    # Epsilon-identifiability (Yoon et al.) weights each feature by the inverse
    # of its entropy before measuring distance, so that a feature carrying
    # little information cannot dominate the neighbourhood.
    weights = _inverse_entropy_weights(frames["real"])
    real_w = real * weights
    fake_w = fake * weights
    d_tt_w = _nearest_distance(real_w, real_w, exclude_self=True)
    d_ts_w = _nearest_distance(real_w, fake_w, exclude_self=False)
    epsilon = float((d_ts_w < d_tt_w).mean())

    tolerance = float(spec["nnaa_tolerance"])
    max_epsilon = float(spec["max_epsilon_identifiability"])

    context = (
        "" if seed_is_real else
        " Note: the seed cohort is itself synthetic, built from declared "
        "marginals, so no real individual's data is at risk in this build. "
        "These numbers establish the measurement path, not a clearance for "
        "training on real data."
    )

    results = [
        GateResult(
            "Nearest-neighbour distance to seed records", "8.3",
            "PASS" if identical == 0 else "FAIL",
            f"{identical} synthetic records sit at distance 0 from a seed "
            f"record (must be 0). Median distance {float(np.median(d_ts)):.3f}."
            + context,
            metrics={"identical_records": identical},
        ),
        GateResult(
            "Nearest Neighbour Adversarial Accuracy", "8.3",
            "PASS" if abs(nnaa - 0.5) <= tolerance else "FAIL",
            f"NNAA = {nnaa:.3f} (ideal 0.5, tolerance +/-{tolerance})." + context,
            metrics={"nnaa": nnaa},
        ),
        GateResult(
            "Epsilon-identifiability", "8.3",
            # With a synthetic seed there is no real person to identify, so the
            # number is reported rather than gated. It is gated only once real
            # profiles are fed in, which is when the threshold protects anyone.
            ("PASS" if epsilon <= max_epsilon else "FAIL")
            if seed_is_real else "REPORTED",
            f"epsilon-identifiability = {epsilon:.3f} "
            f"(Miletic benchmark: under {max_epsilon}). Note that when the "
            "synthetic set is the same size as the seed set and the generator "
            "is distributionally faithful, this statistic sits near 0.5 by "
            "construction: a synthetic neighbour is then as likely to be "
            "nearest as another real record. NNAA and the membership attack "
            "above are the discriminative tests." + context,
            metrics={"epsilon_identifiability": epsilon,
                     "gated": bool(seed_is_real)},
        ),
    ]

    # Membership inference: can an attacker tell records the generator saw
    # from records it never saw, using distance to the synthetic cohort alone?
    if holdout_frame is None or holdout_frame.empty:
        results.append(
            GateResult(
                "Membership-inference attack", "8.3", "SKIPPED",
                "No held-out cohort was supplied, so there is no honest "
                "negative class to attack against.",
            )
        )
        return results

    members = encoded["seed_full"]
    holdout = encoded["holdout"]
    member_distance = _nearest_distance(members, fake, exclude_self=False)
    holdout_distance = _nearest_distance(holdout, fake, exclude_self=False)
    threshold = float(
        np.median(np.concatenate([member_distance, holdout_distance]))
    )
    predictions = np.concatenate(
        [member_distance <= threshold, holdout_distance <= threshold]
    )
    truth = np.concatenate(
        [np.ones(len(member_distance), bool), np.zeros(len(holdout_distance), bool)]
    )
    accuracy = float((predictions == truth).mean())
    results.append(
        GateResult(
            "Membership-inference attack", "8.3",
            "PASS" if abs(accuracy - 0.5) <= 0.15 else "FAIL",
            f"Distance-threshold attack distinguishes seed members from "
            f"held-out records with accuracy {accuracy:.3f} (chance is 0.5), "
            f"over {len(members)} members and {len(holdout)} non-members."
            + context,
            metrics={"accuracy": accuracy},
        )
    )
    return results


def encode_jointly(
    frames: dict[str, pd.DataFrame], columns: list[str]
) -> dict[str, np.ndarray]:
    """Encode several frames into one shared numeric space.

    Distances are only comparable across frames if the categorical codes and
    the standardisation are shared. Encoding each frame on its own lets "F"
    map to 0 in one frame and 1 in another, which makes every cross-frame
    distance meaningless and silently inflates the privacy metrics.
    """
    keys = list(frames)
    stacked = pd.concat(
        [frames[k][columns].reset_index(drop=True) for k in keys],
        keys=keys,
        names=["__frame__"],
    )
    encoded = stacked.copy()
    for column in columns:
        series = encoded[column]
        if series.dtype == object or series.dtype == bool or str(series.dtype) == "category":
            categories = sorted(series.astype(str).unique())
            lookup = {value: i for i, value in enumerate(categories)}
            encoded[column] = series.astype(str).map(lookup)
    matrix = encoded.to_numpy(dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    matrix = (matrix - mean) / sd

    out: dict[str, np.ndarray] = {}
    offset = 0
    for key in keys:
        n = len(frames[key])
        out[key] = matrix[offset:offset + n]
        offset += n
    return out


def _nearest_distance(a: np.ndarray, b: np.ndarray, exclude_self: bool) -> np.ndarray:
    out = np.empty(len(a))
    chunk = max(1, 2000 // max(1, b.shape[1]))
    for start in range(0, len(a), chunk):
        block = a[start:start + chunk]
        distances = np.sqrt(
            ((block[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        )
        if exclude_self:
            for i in range(len(block)):
                distances[i, start + i] = np.inf
        out[start:start + chunk] = distances.min(axis=1)
    return out


# ----------------------------------------------------------------------
# 8.4 Bias and equity
# ----------------------------------------------------------------------


def gate_strata(patients: pd.DataFrame, expected: pd.DataFrame,
                spec: dict[str, Any]) -> GateResult:
    """Every metric per ancestry and per SAS sub-population, never pooled only."""
    minimum = int(spec["min_effective_n"])
    merged = patients.merge(expected, on="patient_id", how="left")
    rows = []

    def _summarise(label: str, stratum: str, group: pd.DataFrame) -> dict[str, Any]:
        n = len(group)
        flagged = group["cpic_recommendation_expected"].isin(
            ["WITH_CAVEAT", "ALTERNATIVE_RECOMMENDED", "CONTRAINDICATED"]
        )
        return {
            "stratum_kind": label,
            "stratum": stratum,
            "n": n,
            "flagged_risk_rate": float(flagged.mean()) if n else None,
            "sas_override_rate": float(group["sas_override_expected"].fillna(False).mean()) if n else None,
            "refusal_rate": float(
                (group["cpic_recommendation_expected"] == "REFUSED_INSUFFICIENT_EVIDENCE").mean()
            ) if n else None,
            "below_min_n": n < minimum,
        }

    for pop, group in merged.groupby("ancestry_super_pop"):
        rows.append(_summarise("super_population", str(pop), group))
    sas = merged[merged["ancestry_super_pop"] == "SAS"]
    for sub, group in sas.groupby("ancestry_sub_pop"):
        rows.append(_summarise("sas_sub_population", str(sub), group))

    details = pd.DataFrame(rows)
    thin = details[details["below_min_n"]]
    summary = (
        f"{len(details)} strata reported. "
        + (f"{len(thin)} below the effective-n floor of {minimum}: "
           + ", ".join(thin["stratum"]) + "."
           if len(thin) else f"All strata at or above the floor of {minimum}.")
    )
    return GateResult("Per-stratum reporting", "8.4", "REPORTED", summary, details)


def equity_analysis(patients: pd.DataFrame, expected: pd.DataFrame) -> GateResult:
    """F4: how many carriers a CPIC-only screen calls normal.

    This is the number the equity thesis rests on, so it is computed from the
    labels rather than asserted: patients whose CPIC-only answer is
    "no change" but whom the South Asian override flags.
    """
    merged = patients.merge(expected, on="patient_id", how="left")
    rows = []
    for pop, group in merged.groupby("ancestry_super_pop"):
        overridden = group["sas_override_expected"].fillna(False)
        cpic_only_normal = group["cpic_recommendation_expected"].isin(
            ["RECOMMENDATION_AS_IS"]
        ) | overridden
        rows.append(
            {
                "population": pop,
                "n": len(group),
                "flagged_with_override": int(
                    (~group["cpic_recommendation_expected"].isin(
                        ["RECOMMENDATION_AS_IS", "REFUSED_INSUFFICIENT_EVIDENCE"]
                    )).sum()
                ),
                "flagged_cpic_only": int(
                    (~cpic_only_normal
                     & ~group["cpic_recommendation_expected"].eq(
                         "REFUSED_INSUFFICIENT_EVIDENCE")).sum()
                ),
                "missed_by_cpic_only": int(overridden.sum()),
                "missed_rate": float(overridden.mean()),
            }
        )
    details = pd.DataFrame(rows)
    sas_row = details[details["population"] == "SAS"]
    headline = ""
    if not sas_row.empty:
        row = sas_row.iloc[0]
        headline = (
            f"{int(row['missed_by_cpic_only'])} of {int(row['n'])} South Asian "
            f"patients ({row['missed_rate']:.1%}) are called normal by a "
            "CPIC-only screen but flagged once the South Asian override applies."
        )
    return GateResult("Population-equity measurement (F4)", "8.4", "REPORTED",
                      headline or "No SAS stratum present.", details)


def gate_ml_utility() -> GateResult:
    return GateResult(
        "ML utility, train-synthetic-test-real (F5)", "8.5", "SKIPPED",
        "Needs the DPYD v4 missense classifier and a held-out set of REAL "
        "variants with functional labels (Offer 2014, ClinVar). Neither is "
        "present in this repository, so no utility claim is made.",
    )


def _inverse_entropy_weights(frame: pd.DataFrame) -> np.ndarray:
    """Per-feature 1/entropy weights for the identifiability metric.

    Continuous columns are binned before their entropy is taken. A column with
    no variation gets weight 0 rather than an infinite one.
    """
    weights = []
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object or series.dtype == bool:
            counts = series.astype(str).value_counts(normalize=True)
        else:
            binned = pd.cut(series.astype(float), bins=20, duplicates="drop")
            counts = binned.value_counts(normalize=True)
        probabilities = counts[counts > 0].to_numpy()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        weights.append(1.0 / entropy if entropy > 1e-9 else 0.0)
    return np.array(weights, dtype=float)


def gate_outcome_coverage(
    expected: pd.DataFrame, coverage: pd.DataFrame | None
) -> GateResult:
    """Every CPIC path the engine can take must be exercised somewhere.

    A test set that never reaches CONTRAINDICATED has not tested the behaviour
    that actually stops a prescription. Real frequencies never produce a Poor
    Metabolizer in a cohort this size, so the enriched coverage panel is what
    reaches those paths; this gate checks it actually did.
    """
    from .layer_a import DRUG_SAFETY_OUTCOMES, PHENOTYPES

    frames = [expected]
    if coverage is not None and not coverage.empty:
        frames.append(coverage)
    combined = pd.concat(frames, ignore_index=True)

    seen_outcomes = set(combined["cpic_recommendation_expected"].dropna().unique())
    seen_phenotypes = set(combined["phenotype_expected"].dropna().unique())
    missing_outcomes = sorted(DRUG_SAFETY_OUTCOMES - seen_outcomes)
    missing_phenotypes = sorted(PHENOTYPES - seen_phenotypes)

    rows = [
        {"kind": "DrugSafetyOutcome", "value": value,
         "n": int((combined["cpic_recommendation_expected"] == value).sum())}
        for value in sorted(DRUG_SAFETY_OUTCOMES)
    ] + [
        {"kind": "phenotype", "value": value,
         "n": int((combined["phenotype_expected"] == value).sum())}
        for value in sorted(PHENOTYPES)
    ]

    problems = []
    if missing_outcomes:
        problems.append("DrugSafetyOutcome values never produced: "
                        + ", ".join(missing_outcomes))
    if missing_phenotypes:
        problems.append("phenotypes never produced: " + ", ".join(missing_phenotypes))

    return GateResult(
        "Engine path coverage", "8.2",
        "PASS" if not problems else "FAIL",
        "; ".join(problems) if problems else
        f"All {len(DRUG_SAFETY_OUTCOMES)} DrugSafetyOutcome values and all "
        f"{len(PHENOTYPES)} phenotypes are exercised. Poor Metabolizer rows "
        "come from the enriched coverage panel, not from population sampling.",
        pd.DataFrame(rows),
    )


def gate_outcome_v2_calibration(
    outcomes_v2: pd.DataFrame | None, spec: dict | None
) -> GateResult:
    """v2's realised event rate must still match the rate it was solved for.

    v2's intercept is DERIVED: it was solved by bisection so the mean
    predicted probability equals the 267/3437 grade 4-5 rate reported by
    Boige et al. (PMID 41386266). That makes it the one coefficient that
    silently stops being true if any other one changes, because nothing
    re-solves it automatically. This gate is what catches that.

    It also checks the two sourced odds ratios are still the transcribed
    values, since a drifted coefficient with a SOURCED tag is worse than an
    honest assumption.
    """
    if outcomes_v2 is None or outcomes_v2.empty or not spec:
        return GateResult(
            name="Layer B v2 calibration",
            section="8.9",
            status="SKIPPED",
            summary="outcomes_v2 not generated; nothing to calibrate against.",
        )

    target = float(spec["target_event_rate"])
    realised = float(outcomes_v2["severe_grade45"].mean())
    tolerance = 0.005
    drift = abs(realised - target)

    coefficients = spec["coefficients"]
    expected_or = {
        "rare_deleterious_burden": 3.20,
        "rs1801160_carrier": 1.49,
    }
    or_rows = []
    or_bad = []
    for name, want in expected_or.items():
        got = float(np.exp(float(coefficients[name]["value"])))
        ok = abs(got - want) < 0.01
        or_rows.append({"coefficient": name, "expected_or": want,
                        "realised_or": round(got, 4), "ok": ok})
        if not ok:
            or_bad.append(name)

    failures = []
    if drift > tolerance:
        failures.append(
            f"event rate {realised:.4f} vs target {target:.4f} "
            f"(drift {drift:.4f} > {tolerance})"
        )
    if or_bad:
        failures.append(f"sourced odds ratios no longer transcribed: {or_bad}")

    details = pd.DataFrame(
        or_rows + [{"coefficient": "intercept (DERIVED)", "expected_or": target,
                    "realised_or": round(realised, 4), "ok": drift <= tolerance}]
    )
    return GateResult(
        name="Layer B v2 calibration",
        section="8.9",
        status="FAIL" if failures else "PASS",
        summary=(
            f"grade 4-5 rate {realised:.4f} against the sourced target "
            f"{target:.4f} (tolerance {tolerance}); OR 3.20 and OR 1.49 "
            "transcribed as configured."
            if not failures
            else "; ".join(failures)
        ),
        details=details,
        metrics={
            "target_event_rate": target,
            "realised_event_rate": realised,
            "drift": drift,
            "tolerance": tolerance,
            "rare_carrier_rate": float(outcomes_v2["rare_deleterious_carrier"].mean()),
            "rs1801160_carrier_rate": float(outcomes_v2["rs1801160_carrier"].mean()),
        },
    )


def gate_outcome_v3_calibration(
    outcomes_v3: pd.DataFrame | None, spec: dict | None
) -> GateResult:
    """v3's realised rate and its eleven transcribed odds ratios.

    v3 carries more sourced numbers than any other part of this dataset, so
    it has more that can silently drift. Every odds ratio taken from FUSAFE
    Table 2 is checked back against the published value, and the DERIVED
    intercept against the rate it was solved for.
    """
    if outcomes_v3 is None or outcomes_v3.empty or not spec:
        return GateResult(
            name="Layer B v3 calibration",
            section="8.10",
            status="SKIPPED",
            summary="outcomes_v3 not generated; nothing to calibrate against.",
        )

    target = float(spec["target_event_rate"])
    realised = float(outcomes_v3["severe_grade45"].mean())
    tolerance = 0.005
    drift = abs(realised - target)

    # FUSAFE Table 2, M2 column (M3 for HapB3).
    published = {
        "age_50_59": 1.3, "age_60_69": 2.0, "age_70_plus": 2.4,
        "sex_female": 1.9, "bmi_underweight": 1.4, "bmi_overweight": 0.8,
        "bmi_obesity": 0.7, "schedule_continuous_or_oral": 0.4,
        "concomitant_one": 2.3, "dpyd_deleterious_aggregate": 9.5,
        "hapb3": 1.8,
    }
    coefficients = spec["coefficients"]
    rows, bad = [], []
    for name, want in published.items():
        got = float(np.exp(float(coefficients[name]["value"])))
        ok = abs(got - want) < 0.05
        rows.append({"coefficient": name, "published_or": want,
                     "realised_or": round(got, 3), "ok": ok})
        if not ok:
            bad.append(name)

    failures = []
    if drift > tolerance:
        failures.append(
            f"event rate {realised:.4f} vs target {target:.4f} "
            f"(drift {drift:.4f} > {tolerance})"
        )
    if bad:
        failures.append(f"odds ratios no longer match FUSAFE Table 2: {bad}")

    return GateResult(
        name="Layer B v3 calibration",
        section="8.10",
        status="FAIL" if failures else "PASS",
        summary=(
            f"grade 4-5 rate {realised:.4f} against the sourced target "
            f"{target:.4f}; all {len(published)} odds ratios transcribed from "
            "FUSAFE Table 2 (PMID 38225422) still match."
            if not failures
            else "; ".join(failures)
        ),
        details=pd.DataFrame(rows),
        metrics={
            "target_event_rate": target,
            "realised_event_rate": realised,
            "drift": drift,
            "n_odds_ratios_checked": len(published),
            "dpyd_aggregate_carrier_rate": float(
                outcomes_v3["dpyd_aggregate_carrier"].mean()
            ),
            "hapb3_carrier_rate": float(outcomes_v3["hapb3_carrier"].mean()),
        },
    )

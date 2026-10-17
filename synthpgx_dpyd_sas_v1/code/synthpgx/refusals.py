"""The engineered edge-case panel, where the correct answer is a refusal.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Every row here is built so the engine *should* decline to give a confident
answer. Honest refusal is a product behaviour worth testing as hard as any
positive call, so each case states the category it exercises and the engine has
to name that same category back.

Two of the categories need a variant the rulebook has never seen. Those are
constructed at DPYD positions absent from the catalog and carry ``rsid = None``
and ``is_synthetic_construct = True``. They are deliberately not given a real
rsID, because a fabricated identifier would be indistinguishable from a sourced
one later.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CATEGORIES = [
    "unlisted_variant_no_evidence",
    "unsupported_ancestry",
    "frameshift_not_in_tables",
    "ambiguous_phase",
    "low_quality_call",
    "non_cpic_population_override",
]

#: Ancestry labels outside the closed enum, used by the unsupported-ancestry
#: case. These are strings the engine must reject, not populations to model.
OUT_OF_ENUM_ANCESTRIES = ["OCE", "MID", "UNKNOWN", "NOT_REPORTED"]


def _synthetic_positions(region, catalog_positions: set[int], n: int, rng) -> list[int]:
    """Pick DPYD positions that no catalog variant occupies."""
    out: list[int] = []
    low = region.start + 1000
    high = region.end - 1000
    while len(out) < n:
        candidate = int(rng.integers(low, high))
        if candidate in catalog_positions or candidate in out:
            continue
        out.append(candidate)
    return sorted(out)


def build_refusal_panel(
    catalog,
    region,
    n_per_category: int,
    seed: int,
    min_dp: int,
    min_gq: int,
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the edge-case cohort.

    Returns ``(patients, calls)``: one patient record per engineered case and a
    flat list of that patient's variant calls.
    """
    rng = np.random.default_rng(seed)
    catalog_positions = {
        int(p) for p in catalog["grch38_pos"].dropna().astype(int).tolist()
    }

    snv_catalog = catalog[catalog["is_snv"].fillna(False).astype(bool)].copy()
    known = snv_catalog.to_dict("records")
    decreased = [
        v for v in known
        if str(v.get("cpic_function")) in ("No function", "Decreased function")
    ]
    override_eligible = [v for v in known if bool(v.get("sas_override_eligible"))]

    patients: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    index = start_index

    def _new_patient(category: str, ancestry: str, sub_pop: str | None) -> dict[str, Any]:
        nonlocal index
        patient_id = f"SYN-R{index:06d}"
        index += 1
        return {
            "patient_id": patient_id,
            "case_category": category,
            "ancestry_super_pop": ancestry,
            "ancestry_sub_pop": sub_pop,
        }

    def _call(patient_id: str, variant: dict[str, Any], genotype: str, dp: int, gq: int):
        return {
            "patient_id": patient_id,
            "variant_id": variant["variant_id"],
            "hgvs_c": variant.get("hgvs_c"),
            "rsid": variant.get("rsid"),
            "chrom": "chr1",
            "pos_grch38": int(variant["grch38_pos"]),
            "ref": variant["ref"],
            "alt": variant["alt"],
            "genotype": genotype,
            "phased": None,
            "dp": int(dp),
            "gq": int(gq),
            "is_synthetic_construct": bool(variant.get("is_synthetic_construct", False)),
        }

    def _good_quality() -> tuple[int, int]:
        return int(rng.integers(min_dp + 20, min_dp + 120)), int(
            rng.integers(min_gq + 15, 99)
        )

    # --- 1. unlisted variant, no evidence ---------------------------
    positions = _synthetic_positions(region, catalog_positions, n_per_category * 2, rng)
    for i in range(n_per_category):
        patient = _new_patient("unlisted_variant_no_evidence", "SAS", "GIH")
        ref, alt = _random_snv(rng)
        variant = {
            "variant_id": f"SYNTH-UNLISTED-{positions[i]}",
            "hgvs_c": None,
            "rsid": None,
            "grch38_pos": positions[i],
            "ref": ref,
            "alt": alt,
            "is_synthetic_construct": True,
        }
        dp, gq = _good_quality()
        calls.append(_call(patient["patient_id"], variant, "0/1", dp, gq))
        patient["expected_category"] = "unlisted_variant_no_evidence"
        patient["construction"] = (
            "Heterozygous variant at a DPYD position absent from the CPIC allele "
            "table; no rsID is asserted."
        )
        patients.append(patient)

    # --- 2. unsupported / low-data ancestry -------------------------
    for i in range(n_per_category):
        ancestry = OUT_OF_ENUM_ANCESTRIES[i % len(OUT_OF_ENUM_ANCESTRIES)]
        patient = _new_patient("unsupported_ancestry", ancestry, None)
        variant = decreased[i % len(decreased)] if decreased else known[0]
        dp, gq = _good_quality()
        calls.append(_call(patient["patient_id"], variant, "0/1", dp, gq))
        patient["expected_category"] = "unsupported_ancestry"
        patient["construction"] = (
            f"Ancestry label {ancestry!r} lies outside the closed enum "
            "AFR/AMR/EAS/EUR/SAS, so no population layer applies."
        )
        patients.append(patient)

    # --- 3. frameshift / stop-gain not in the tables -----------------
    for i in range(n_per_category):
        patient = _new_patient("frameshift_not_in_tables", "SAS", "ITU")
        position = positions[n_per_category + i]
        ref = "".join(rng.choice(list("ACGT"), size=17))
        variant = {
            "variant_id": f"SYNTH-FS16-{position}",
            "hgvs_c": None,
            "rsid": None,
            "grch38_pos": position,
            "ref": ref,
            "alt": ref[0],
            "is_synthetic_construct": True,
        }
        dp, gq = _good_quality()
        calls.append(_call(patient["patient_id"], variant, "0/1", dp, gq))
        patient["expected_category"] = "frameshift_not_in_tables"
        patient["construction"] = (
            "16 bp frameshift deletion at a DPYD position absent from the CPIC "
            "allele table. Loss of function is obvious to a human reader but the "
            "rulebook has no entry, which is exactly the unequivocal-exception "
            "path. No rsID is asserted."
        )
        patients.append(patient)

    # --- 4. ambiguous phase ------------------------------------------
    pool = decreased if len(decreased) >= 2 else known
    for i in range(n_per_category):
        patient = _new_patient("ambiguous_phase", "SAS", "PJL")
        first = pool[(2 * i) % len(pool)]
        second = pool[(2 * i + 1) % len(pool)]
        if first["variant_id"] == second["variant_id"]:
            second = pool[(2 * i + 2) % len(pool)]
        for variant in (first, second):
            dp, gq = _good_quality()
            calls.append(_call(patient["patient_id"], variant, "0/1", dp, gq))
        # A third variant copy is what makes the phase genuinely unresolvable.
        third = pool[(2 * i + 3) % len(pool)]
        if third["variant_id"] not in (first["variant_id"], second["variant_id"]):
            dp, gq = _good_quality()
            calls.append(_call(patient["patient_id"], third, "0/1", dp, gq))
        patient["expected_category"] = "ambiguous_phase"
        patient["construction"] = (
            "Three heterozygous reduced-function variants with no phase "
            "information. The diplotype, and therefore the activity score, "
            "changes with phase."
        )
        patients.append(patient)

    # --- 5. low-quality call ------------------------------------------
    for i in range(n_per_category):
        patient = _new_patient("low_quality_call", "SAS", "BEB")
        variant = decreased[i % len(decreased)] if decreased else known[0]
        dp = int(rng.integers(2, max(min_dp, 3)))
        gq = int(rng.integers(1, max(min_gq, 2)))
        calls.append(_call(patient["patient_id"], variant, "0/1", dp, gq))
        patient["expected_category"] = "low_quality_call"
        patient["construction"] = (
            f"Reduced-function variant called at DP={dp}, GQ={gq}, below the "
            f"DP>={min_dp} / GQ>={min_gq} thresholds."
        )
        patients.append(patient)

    # --- 6. conflicting evidence / population override ----------------
    if override_eligible:
        for i in range(n_per_category):
            patient = _new_patient("non_cpic_population_override", "SAS", "STU")
            variant = override_eligible[i % len(override_eligible)]
            dp, gq = _good_quality()
            calls.append(_call(patient["patient_id"], variant, "0/1", dp, gq))
            patient["expected_category"] = "non_cpic_population_override"
            patient["construction"] = (
                f"South Asian patient carrying {variant['variant_id']} "
                f"({variant.get('hgvs_p')}), which CPIC assigns "
                f"{variant.get('cpic_function')!r} but South Asian population "
                "evidence associates with fluoropyrimidine toxicity. The engine "
                "should flag, not silently pass."
            )
            patients.append(patient)

    return patients, calls


def _random_snv(rng) -> tuple[str, str]:
    bases = list("ACGT")
    ref = str(rng.choice(bases))
    alt = str(rng.choice([b for b in bases if b != ref]))
    return ref, alt


# ----------------------------------------------------------------------
# Enriched coverage panel
# ----------------------------------------------------------------------
#
# At real population frequencies a Poor Metabolizer is rarer than one in
# 20,500, so a naturally sampled cohort never exercises the two CPIC paths
# that actually stop a prescription: "avoid" and "avoid unless no alternative".
# An engine test suite that never tests those paths is not a test suite.
#
# These rows are ENRICHED, not sampled. They say so in `is_enriched`, and the
# frequency gates must exclude them, because treating them as a population
# sample would misstate how common Poor Metabolizers are by orders of
# magnitude.

COVERAGE_TARGETS = [
    ("0.0", "two no-function alleles"),
    ("0.5", "one no-function and one decreased-function allele, in trans"),
    ("1.0", "two decreased-function alleles"),
    ("1.5", "one decreased-function allele"),
]


def build_coverage_panel(
    catalog,
    per_target: int,
    seed: int,
    min_dp: int,
    min_gq: int,
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Patients constructed to hit each CPIC activity score at least once.

    Compound heterozygotes are emitted **phased**, because an unphased pair of
    reduced-function variants is genuinely ambiguous and the engine is right to
    refuse it. Phasing is what lets these rows test the recommendation path
    rather than the refusal path.
    """
    rng = np.random.default_rng(seed)
    snv = catalog[catalog["is_snv"].fillna(False).astype(bool)]
    no_function = [
        v for v in snv.to_dict("records")
        if str(v.get("cpic_function")) == "No function"
    ]
    decreased = [
        v for v in snv.to_dict("records")
        if str(v.get("cpic_function")) == "Decreased function"
    ]
    if not no_function or not decreased:
        return [], []

    patients: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    index = start_index

    def _emit(patient_id, variant, genotype, phased):
        calls.append(
            {
                "patient_id": patient_id,
                "variant_id": variant["variant_id"],
                "hgvs_c": variant.get("hgvs_c"),
                "rsid": variant.get("rsid"),
                "chrom": "chr1",
                "pos_grch38": int(variant["grch38_pos"]),
                "ref": variant["ref"],
                "alt": variant["alt"],
                "genotype": genotype,
                "phased": phased,
                "dp": int(rng.integers(min_dp + 20, min_dp + 120)),
                "gq": int(rng.integers(min_gq + 15, 99)),
                "is_synthetic_construct": False,
            }
        )

    for target, description in COVERAGE_TARGETS:
        for i in range(per_target):
            patient_id = f"SYN-C{index:06d}"
            index += 1
            if target == "0.0":
                variant = no_function[i % len(no_function)]
                _emit(patient_id, variant, "1/1", "1|1")
            elif target == "0.5":
                first = no_function[i % len(no_function)]
                second = decreased[i % len(decreased)]
                _emit(patient_id, first, "0/1", "1|0")
                _emit(patient_id, second, "0/1", "0|1")
            elif target == "1.0":
                variant = decreased[i % len(decreased)]
                _emit(patient_id, variant, "1/1", "1|1")
            else:  # 1.5
                variant = decreased[i % len(decreased)]
                _emit(patient_id, variant, "0/1", "1|0")
            patients.append(
                {
                    "patient_id": patient_id,
                    "case_category": f"coverage_activity_score_{target}",
                    "ancestry_super_pop": "SAS",
                    "ancestry_sub_pop": "GIH",
                    "target_activity_score": float(target),
                    "construction": description
                    + ". Phased, and ENRICHED: these genotypes are far rarer "
                    "than this panel implies and must be excluded from any "
                    "frequency statistic.",
                    "is_enriched": True,
                }
            )
    return patients, calls

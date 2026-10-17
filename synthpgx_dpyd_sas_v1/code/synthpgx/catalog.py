"""Build the DPYD variant catalog and the frozen CPIC reference tables.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Nothing in this module types a frequency, an rsID, a PMID or an activity score
from memory.  Every field is either

  * copied from a fetched source record (CPIC, gnomAD, Ensembl, PubMed), or
  * computed by an explicit rule from such fields, or
  * ``None`` -- in which case it is reported to ``qc/open_questions.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import sources
from .sources import Fetcher

SNV_RE = re.compile(r"^g\.(\d+)([ACGT])>([ACGT])$")

SUPER_POPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SAS_SUB_POPS = ["GIH", "ITU", "PJL", "BEB", "STU"]


# ----------------------------------------------------------------------
# Curated South Asian evidence, transcribed from verified PubMed records.
# ----------------------------------------------------------------------
#
# Each entry records a claim *and* where it came from.  ``maf_indian`` values
# are quoted verbatim from the Naushad 2021 abstract (PMID 33105068, n = 2000
# Indian subjects, Infinium Global Screening Array).  ``effect_direction`` is
# that paper's own reported association, not an effect size: the paper is
# paywalled, so no odds ratio is transcribed here.  Layer B effect sizes live
# in ``config.yaml`` and are explicitly flagged as assumptions.

SAS_EVIDENCE: list[dict[str, Any]] = [
    {
        "protein": "V732I",
        "pmid": "33105068",
        "cohort_n": 2000,
        "cohort_description": "Indian subjects screened on Infinium Global Screening Array",
        "maf_indian": 0.0844,
        "effect_direction": "toxicity_associated",
        "effect_note": "Pooled Indian data: V732I associated with 5-FU/capecitabine toxicity",
        "effect_size_reported": None,
        "in_silico": "damaging (paper), thermolabile (CUPSAT)",
    },
    {
        "protein": "S534N",
        "pmid": "33105068",
        "cohort_n": 2000,
        "cohort_description": "Indian subjects screened on Infinium Global Screening Array",
        "maf_indian": None,  # not quoted in the abstract
        "effect_direction": "toxicity_associated",
        "effect_note": "Pooled Indian data: S534N associated with 5-FU/capecitabine toxicity",
        "effect_size_reported": None,
        "in_silico": "damaging (paper), thermolabile (CUPSAT)",
    },
    {
        "protein": "C29R",
        "pmid": "33105068",
        "cohort_n": 2000,
        "cohort_description": "Indian subjects screened on Infinium Global Screening Array",
        "maf_indian": 0.2491,
        "effect_direction": "null_association",
        "effect_note": "Pooled Indian data: C29R exhibited null association (negative control)",
        "effect_size_reported": None,
        "in_silico": "not reported as damaging",
    },
    {
        "protein": "I543V",
        "pmid": "33105068",
        "cohort_n": 2000,
        "cohort_description": "Indian subjects screened on Infinium Global Screening Array",
        "maf_indian": 0.09047,
        "effect_direction": "null_association",
        "effect_note": "Pooled Indian data: I543V exhibited null association (negative control)",
        "effect_size_reported": None,
        "in_silico": "thermolabile (CUPSAT), not damaging",
    },
    {
        "protein": "M166V",
        "pmid": "33105068",
        "cohort_n": 2000,
        "cohort_description": "Indian subjects screened on Infinium Global Screening Array",
        "maf_indian": 0.08993,
        "effect_direction": "null_association",
        "effect_note": (
            "Pooled Indian data: M166V exhibited null association (negative control). "
            "Same variant as rs2297595 / c.496A>G cited from Hariprakash 2018."
        ),
        "effect_size_reported": None,
        "in_silico": "damaging (paper)",
    },
]

#: Papers whose PMIDs must resolve for the build to be considered sourced.
REQUIRED_PMIDS = ["33105068", "29239269", "29152729", "38886557"]


# ----------------------------------------------------------------------


@dataclass
class CatalogBuild:
    variants: list[dict[str, Any]] = field(default_factory=list)
    alleles: list[dict[str, Any]] = field(default_factory=list)
    allele_variants: list[dict[str, Any]] = field(default_factory=list)
    diplotypes: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    gene: dict[str, Any] = field(default_factory=dict)
    pmids: dict[str, dict[str, str]] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    frequency_sources: dict[str, Any] = field(default_factory=dict)


def _parse_snv(chromosome_location: str | None) -> tuple[int, str, str] | None:
    if not chromosome_location:
        return None
    match = SNV_RE.match(chromosome_location.strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


def _star_allele(name: str) -> str | None:
    """Extract a ``*N`` designation from a CPIC allele name, if present."""
    match = re.search(r"\(\s*(\*\d+[A-Z]?)\s*\)", name)
    if match:
        return match.group(1)
    match = re.search(r"\((HapB3)\)", name)
    return match.group(1) if match else None


def _allele_base_name(name: str) -> str:
    """Strip a CPIC allele's trailing star designation.

    CPIC names an allele ``c.1905+1G>A (*2A)`` while the variant it is built
    from is ``c.1905+1G>A``. Without this the two never join, and the most
    consequential DPYD alleles silently lose their function assignment.
    """
    return re.sub(r"\s*\((\*\d+[A-Z]?|HapB3)\)\s*$", "", name).strip()


def _is_compound(name: str) -> bool:
    """True when a CPIC allele is defined by more than one variant."""
    return "," in _allele_base_name(name)


def build_catalog(
    fetcher: Fetcher,
    gnomad_primary: str = "gnomad_r3",
    gnomad_secondary: str = "gnomad_r4",
) -> CatalogBuild:
    """Assemble the catalog from live sources."""
    build = CatalogBuild()
    cpic = sources.fetch_cpic(fetcher)

    build.gene = cpic["gene"][0] if cpic["gene"] else {}
    build.diplotypes = cpic["diplotype"]
    build.recommendations = cpic["recommendation"]
    drug_names = {d["drugid"]: d["name"] for d in cpic["drug"]}
    for rec in build.recommendations:
        rec["drug_name"] = drug_names.get(rec.get("drugid"), "")

    # --- allele function table -------------------------------------
    allele_by_name: dict[str, dict[str, Any]] = {}
    allele_by_base: dict[str, dict[str, Any]] = {}
    for allele in cpic["allele"]:
        row = {
            "allele_name": allele["name"],
            "allele_base_name": _allele_base_name(allele["name"]),
            "star_allele": _star_allele(allele["name"]),
            "cpic_function": allele.get("clinicalfunctionalstatus"),
            "cpic_activity_score": allele.get("activityvalue"),
            "evidence_level": allele.get("strength"),
            "cpic_citations": ";".join(allele.get("citations") or []),
            "cpic_findings": allele.get("findings") or "",
            "cpic_frequency": allele.get("frequency") or {},
        }
        allele_by_name[allele["name"]] = row
        if not _is_compound(allele["name"]):
            allele_by_base[row["allele_base_name"]] = row
        build.alleles.append(row)

    # --- allele definition -> which variants define which allele ----
    def_by_id = {d["id"]: d for d in cpic["allele_definition"]}
    loc_by_id = {s["id"]: s for s in cpic["sequence_location"]}
    for value in cpic["allele_location_value"]:
        definition = def_by_id.get(value["alleledefinitionid"])
        location = loc_by_id.get(value["locationid"])
        if not definition or not location:
            continue
        build.allele_variants.append(
            {
                "allele_name": definition["name"],
                "variant_id": location["name"],
                "rsid": location.get("dbsnpid"),
                "position_grch38": location.get("position"),
                "variant_allele": value.get("variantallele"),
            }
        )

    # --- per-variant rows -------------------------------------------
    sas_by_protein = {e["protein"]: e for e in SAS_EVIDENCE}
    defining_alleles: dict[str, list[str]] = {}
    for link in build.allele_variants:
        defining_alleles.setdefault(link["variant_id"], []).append(link["allele_name"])

    for location in cpic["sequence_location"]:
        name = location["name"]
        protein = (location.get("proteinlocation") or "").replace("p.", "") or None
        snv = _parse_snv(location.get("chromosomelocation"))
        allele_names = sorted(
            a for a in set(defining_alleles.get(name, [])) if a != "Reference"
        )

        # Function comes from the CPIC allele this variant defines on its own,
        # matched on the allele's base name so that "c.1905+1G>A (*2A)" joins
        # to the variant "c.1905+1G>A".
        own_allele = allele_by_base.get(name)
        function_source = "cpic_allele" if own_allele else None

        if own_allele is None:
            # A tag or partner variant of a compound allele carries no
            # standalone function. When it belongs to exactly one non-reference
            # allele, it inherits that allele's function, and the inheritance is
            # recorded rather than hidden.
            compounds = [
                allele_by_name[a]
                for a in allele_names
                if a in allele_by_name and _is_compound(a)
            ]
            if len(compounds) == 1:
                own_allele = compounds[0]
                function_source = f"inherited_from_allele:{own_allele['allele_name']}"
            elif allele_names:
                build.open_questions.append(
                    f"Variant {name} has no standalone CPIC function and belongs "
                    f"to {len(allele_names)} alleles ({', '.join(allele_names)}); "
                    "function left null rather than guessed."
                )

        cpic_function = own_allele["cpic_function"] if own_allele else None
        cpic_activity = own_allele["cpic_activity_score"] if own_allele else None
        evidence_level = own_allele["evidence_level"] if own_allele else None
        citations = own_allele["cpic_citations"] if own_allele else ""

        star = None
        for allele_name in allele_names:
            star = star or _star_allele(allele_name)

        row: dict[str, Any] = {
            "variant_id": name,
            "gene": "DPYD",
            "hgvs_c": name,
            "hgvs_p": f"p.{protein}" if protein else None,
            "rsid": location.get("dbsnpid"),
            "grch38_pos": location.get("position"),
            "ref": snv[1] if snv else None,
            "alt": snv[2] if snv else None,
            "is_snv": bool(snv),
            "star_allele": star,
            "defines_alleles": ";".join(allele_names),
            "cpic_function": cpic_function,
            "cpic_activity_score": cpic_activity,
            "evidence_level": evidence_level,
            "cpic_citation_pmids": citations,
            "in_cpic_table": True,
            "cpic_function_source": function_source,
        }

        if not snv:
            build.open_questions.append(
                f"Variant {name} ({location.get('chromosomelocation')}) is not a simple "
                "SNV; genomic ref/alt left null and the variant is excluded from "
                "frequency-calibrated injection."
            )

        # --- gnomAD frequencies -------------------------------------
        for super_pop in SUPER_POPS:
            row[f"freq_gnomad_{super_pop.lower()}"] = None
            row[f"freq_gnomad_{super_pop.lower()}_an"] = None
        row["gnomad_variant_id"] = None
        if snv:
            variant_id = f"1-{snv[0]}-{snv[1]}-{snv[2]}"
            row["gnomad_variant_id"] = variant_id
            record = sources.gnomad_variant(fetcher, variant_id, gnomad_primary)
            if record is None:
                record = sources.gnomad_variant(fetcher, variant_id, gnomad_secondary)
                row["gnomad_dataset"] = gnomad_secondary if record else None
            else:
                row["gnomad_dataset"] = gnomad_primary
            if record:
                af = sources.gnomad_ancestry_af(record)
                for super_pop in SUPER_POPS:
                    row[f"freq_gnomad_{super_pop.lower()}"] = af[super_pop]["af"]
                    row[f"freq_gnomad_{super_pop.lower()}_an"] = af[super_pop]["an"]
                rsids = record.get("rsids") or []
                row["gnomad_rsids"] = ";".join(rsids)
                if row["rsid"] and rsids and row["rsid"] not in rsids:
                    build.open_questions.append(
                        f"rsID mismatch for {name}: CPIC says {row['rsid']}, "
                        f"gnomAD {row['gnomad_dataset']} says {','.join(rsids)}."
                    )
            else:
                build.open_questions.append(
                    f"No gnomAD record for {name} ({variant_id}); all gnomAD "
                    "frequencies left null."
                )

        # --- 1000 Genomes SAS sub-populations (supplemental) ---------
        for sub in SAS_SUB_POPS:
            row[f"freq_1kg_{sub.lower()}"] = None
        if row["rsid"]:
            kg = sources.ensembl_1kg_frequencies(fetcher, row["rsid"])
            for sub, value in kg.items():
                row[f"freq_1kg_{sub.lower()}"] = value

        # --- GenomeIndia ---------------------------------------------
        # No machine-readable public endpoint was found for GenomeIndia
        # aggregate frequencies; the IBDC portal serves them interactively.
        row["freq_genomeindia"] = None

        # --- South Asian population-paper evidence --------------------
        evidence = sas_by_protein.get(protein or "")
        row["population_paper_pmid"] = evidence["pmid"] if evidence else None
        row["paper_effect_note"] = evidence["effect_note"] if evidence else None
        row["paper_effect_direction"] = (
            evidence["effect_direction"] if evidence else None
        )
        row["paper_cohort_n"] = evidence["cohort_n"] if evidence else None
        row["paper_maf_indian"] = evidence["maf_indian"] if evidence else None
        row["sas_panel"] = bool(evidence)
        build.variants.append(row)

    # --- GenomeIndia gap ---------------------------------------------
    build.open_questions.append(
        "GenomeIndia (9,768 genomes, IBDC portal ibdc.dbt.gov.in) exposes no "
        "machine-readable aggregate-frequency endpoint that this build could "
        "reach; every freq_genomeindia value is null. Supply a downloaded "
        "frequency file to populate it."
    )
    build.open_questions.append(
        "PharmVar's allele API (pharmvar.org/api-service) returned HTTP 401 "
        "without credentials, so pharmvar_version is recorded as null and "
        "star-allele names are taken from CPIC allele names instead."
    )

    # --- PMID verification ------------------------------------------
    pmids = set(REQUIRED_PMIDS)
    for row in build.variants:
        for pmid in (row.get("cpic_citation_pmids") or "").split(";"):
            if pmid.strip().isdigit():
                pmids.add(pmid.strip())
    build.pmids = sources.verify_pmids(fetcher, sorted(pmids))
    for pmid in REQUIRED_PMIDS:
        if pmid not in build.pmids:
            build.open_questions.append(
                f"PMID {pmid} did not resolve through NCBI esummary; provenance "
                "for the claims resting on it is unverified."
            )

    # --- SAS override eligibility (computed, never hand-set) ---------
    for row in build.variants:
        row["sas_override_eligible"] = _sas_override_eligible(row)
        row["sas_override_reason"] = _sas_override_reason(row)

    build.frequency_sources = {
        "primary": {
            "name": "gnomAD",
            "dataset": gnomad_primary,
            "note": (
                "Spec 6.4.3: gnomAD v3.1.2 is the primary frequency source. "
                "1000 Genomes is supplemental only, because 1000 Genomes "
                "samples are also inside gnomAD and must not be double-counted."
            ),
            "endpoint": sources.GNOMAD_API,
            "populations_used": sources.GNOMAD_POP_MAP,
        },
        "secondary": {
            "name": "gnomAD",
            "dataset": gnomad_secondary,
            "note": "Used only when the variant is absent from the primary dataset.",
        },
        "supplemental": {
            "name": "1000 Genomes phase 3 via Ensembl REST",
            "endpoint": sources.ENSEMBL_API,
            "populations_used": SAS_SUB_POPS,
            "note": "Never used for injection targets; reported for comparison only.",
        },
        "layer_a_truth": {
            "name": "CPIC",
            "endpoint": sources.CPIC_API,
            "note": "Allele function, activity value, diplotype->phenotype, recommendations.",
        },
        "unavailable": {
            "GenomeIndia": "No machine-readable endpoint reached; all values null.",
            "PharmVar": "API returned HTTP 401 without credentials.",
        },
    }
    return build


#: Minimum South Asian minor allele frequency for override eligibility.
#:
#: Ratified at 5 percent as originally specified. The decision was taken
#: knowingly: S534N (c.1601G>A, rs1801158) carries a reported toxicity
#: association in Naushad et al. 2021 but its gnomAD South Asian MAF is 0.70
#: percent, so it falls below this line and is excluded. Only V732I qualifies.
#: Lowering this threshold would readmit S534N and would also need a fresh look
#: at every other variant between the old and new line.
SAS_OVERRIDE_MIN_MAF = 0.05

#: Minimum cohort size for the published toxicity evidence to count.
SAS_OVERRIDE_MIN_COHORT_N = 50


def _sas_override_eligible(row: dict[str, Any]) -> bool:
    """Spec 4.7 rule, computed from sourced fields only.

    A variant is SAS-override eligible when all three hold:

    1. MAF above :data:`SAS_OVERRIDE_MIN_MAF` in SAS (gnomAD, primary source);
    2. published toxicity evidence with n above
       :data:`SAS_OVERRIDE_MIN_COHORT_N` in that population;
    3. CPIC has no recommendation for it, or only a European-based default
       (operationalised as: CPIC assigns it normal function, or does not list
       it as an allele at all).

    A missing input makes the rule False, never True.
    """
    maf = row.get("freq_gnomad_sas")
    if maf is None or maf <= SAS_OVERRIDE_MIN_MAF:
        return False
    if row.get("paper_effect_direction") != "toxicity_associated":
        return False
    cohort_n = row.get("paper_cohort_n")
    if not cohort_n or cohort_n <= SAS_OVERRIDE_MIN_COHORT_N:
        return False
    function = row.get("cpic_function")
    return function is None or function == "Normal function"


def _sas_override_reason(row: dict[str, Any]) -> str | None:
    if not row.get("sas_override_eligible"):
        return None
    protein = (row.get("hgvs_p") or "").replace("p.", "") or row["variant_id"]
    maf = row.get("freq_gnomad_sas") or 0.0
    return (
        f"{protein} SAS MAF {maf * 100:.1f}% (>5%) + "
        f"PMID{row.get('population_paper_pmid')} n={row.get('paper_cohort_n')} + "
        f"CPIC function '{row.get('cpic_function')}'"
    )


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

CATALOG_COLUMNS = [
    "variant_id", "gene", "hgvs_c", "hgvs_p", "rsid", "grch38_pos", "ref", "alt",
    "star_allele", "cpic_function", "cpic_activity_score", "evidence_level",
    "freq_gnomad_sas", "freq_gnomad_eur", "freq_gnomad_afr", "freq_gnomad_eas",
    "freq_gnomad_amr", "freq_genomeindia",
    "freq_1kg_gih", "freq_1kg_itu", "freq_1kg_pjl", "freq_1kg_beb", "freq_1kg_stu",
    "population_paper_pmid", "paper_effect_note", "sas_override_eligible",
    # provenance / build fields beyond the spec's minimum
    "defines_alleles", "is_snv", "in_cpic_table", "sas_panel",
    "paper_effect_direction", "paper_cohort_n", "paper_maf_indian",
    "cpic_citation_pmids", "cpic_function_source", "gnomad_variant_id", "gnomad_dataset",
    "freq_gnomad_sas_an", "sas_override_reason",
]


def write_reference(build: CatalogBuild, reference_dir: Path) -> None:
    import pandas as pd

    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)

    variants = pd.DataFrame(build.variants)
    for column in CATALOG_COLUMNS:
        if column not in variants.columns:
            variants[column] = None
    variants = variants[CATALOG_COLUMNS].sort_values("grch38_pos", na_position="last")
    variants.to_csv(reference_dir / "variant_catalog.csv", index=False)

    pd.DataFrame(build.alleles).to_csv(
        reference_dir / "cpic_allele_functions.csv", index=False
    )
    pd.DataFrame(build.allele_variants).to_csv(
        reference_dir / "cpic_allele_definitions.csv", index=False
    )

    diplo = pd.DataFrame(
        [
            {
                "diplotype": d["diplotype"],
                "activity_score": d.get("totalactivityscore"),
                "phenotype": d.get("generesult"),
                "function1": d.get("function1"),
                "function2": d.get("function2"),
                "lookupkey": json.dumps(d.get("lookupkey") or {}, sort_keys=True),
            }
            for d in build.diplotypes
        ]
    )
    diplo.to_csv(reference_dir / "cpic_diplotype_phenotype.csv", index=False)

    recs = pd.DataFrame(
        [
            {
                "drug": r.get("drug_name"),
                "drugid": r.get("drugid"),
                "phenotype": (r.get("phenotypes") or {}).get("DPYD"),
                "activity_score": (r.get("activityscore") or {}).get("DPYD"),
                "lookupkey": (r.get("lookupkey") or {}).get("DPYD"),
                "recommendation": r.get("drugrecommendation"),
                "implications": (r.get("implications") or {}).get("DPYD"),
                "classification": r.get("classification"),
                "comments": r.get("comments"),
            }
            for r in build.recommendations
        ]
    )
    recs.to_csv(reference_dir / "cpic_recommendations.csv", index=False)

    pd.DataFrame(SAS_EVIDENCE).to_csv(
        reference_dir / "sas_population_evidence.csv", index=False
    )

    (reference_dir / "frequency_sources.json").write_text(
        json.dumps(
            {
                "sources": build.frequency_sources,
                "gene": build.gene,
                "pmid_verification": build.pmids,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

"""VCF emission.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Writes one small VCF per synthetic patient plus a gzipped cohort VCF. The files
carry simulated DP and GQ so that quality filters, and the refusal path that
depends on them, can actually be exercised downstream.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Iterable

VCF_VERSION = "VCFv4.2"
ASSEMBLY = "GRCh38"
CONTIG_LINE = "##contig=<ID=chr1,length=248956422,assembly=GRCh38>"

HEADER_META = [
    f"##fileformat={VCF_VERSION}",
    f"##reference={ASSEMBLY}",
    CONTIG_LINE,
    '##FILTER=<ID=PASS,Description="All filters passed">',
    '##FILTER=<ID=LowQual,Description="DP or GQ below the configured threshold">',
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth">',
    '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
    '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">',
    '##INFO=<ID=HGVSC,Number=1,Type=String,Description="HGVS coding change on NM_000110.4">',
    '##INFO=<ID=SYNTHETIC,Number=0,Type=Flag,Description="Simulated call. In silico cohort validation. Not clinical evidence.">',
    "##source=synthpgx",
    "##synthpgx_warning=SYNTHETIC. In silico cohort validation. Not clinical evidence.",
]

COLUMNS = [
    "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT",
]


def _record(call: dict[str, Any], min_dp: int, min_gq: int) -> str:
    dp = call.get("dp")
    gq = call.get("gq")
    failed = (dp is not None and dp < min_dp) or (gq is not None and gq < min_gq)
    filt = "LowQual" if failed else "PASS"
    info = f"GENE=DPYD;HGVSC={call['hgvs_c']};SYNTHETIC"
    genotype = call.get("phased") or call.get("genotype", "./.")
    sample = f"{genotype}:{dp if dp is not None else '.'}:{gq if gq is not None else '.'}"
    return "\t".join(
        [
            call.get("chrom", "chr1"),
            str(call["pos_grch38"]),
            call.get("rsid") or ".",
            call["ref"],
            call["alt"],
            ".",
            filt,
            info,
            "GT:DP:GQ",
            sample,
        ]
    )


def write_patient_vcf(
    path: str | Path,
    sample_id: str,
    calls: Iterable[dict[str, Any]],
    min_dp: int,
    min_gq: int,
    meta: dict[str, str] | None = None,
) -> Path:
    """Write one patient's DPYD calls."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = list(HEADER_META)
    for key, value in (meta or {}).items():
        lines.append(f"##synthpgx_{key}={value}")
    lines.append("\t".join(COLUMNS + [sample_id]))
    ordered = sorted(calls, key=lambda c: int(c["pos_grch38"]))
    lines.extend(_record(call, min_dp, min_gq) for call in ordered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def write_cohort_vcf(
    path: str | Path,
    sample_ids: list[str],
    sites: list[dict[str, Any]],
    genotypes: dict[str, list[dict[str, Any]]],
    min_dp: int,
    min_gq: int,
) -> Path:
    """Write every patient into one gzipped multi-sample VCF.

    Plain gzip, not bgzip: the file is readable by any VCF parser, but it is not
    tabix-indexable. Run ``bgzip``/``tabix`` over it if an index is needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for line in HEADER_META:
            handle.write(line + "\n")
        handle.write("\t".join(COLUMNS + sample_ids) + "\n")
        for site in sorted(sites, key=lambda s: int(s["pos_grch38"])):
            fields = [
                site.get("chrom", "chr1"),
                str(site["pos_grch38"]),
                site.get("rsid") or ".",
                site["ref"],
                site["alt"],
                ".",
                "PASS",
                f"GENE=DPYD;HGVSC={site['hgvs_c']};SYNTHETIC",
                "GT:DP:GQ",
            ]
            for sample_id in sample_ids:
                call = genotypes[sample_id].get(site["variant_id"])
                if call is None:
                    fields.append("./.:.:.")
                    continue
                genotype = call.get("phased") or call.get("genotype", "./.")
                fields.append(
                    f"{genotype}:{call.get('dp', '.')}:{call.get('gq', '.')}"
                )
            handle.write("\t".join(fields) + "\n")
    return path


# ----------------------------------------------------------------------
# PharmCAT-shaped emission (build brief Part 8.2)
# ----------------------------------------------------------------------
#
# The per-patient VCFs above carry only non-reference calls, because that is
# what a variant caller emits. PharmCAT's named-allele matcher needs the
# opposite: every position in its own allele definition must be present, so
# that a missing row means "not sequenced" rather than "reference". The two
# emitters are kept separate rather than one being made to serve both, because
# widening the clinical VCFs with reference rows would misrepresent what the
# simulated caller produced.

PHARMCAT_FORMAT = "GT"


def read_pharmcat_positions(
    path: str | Path, gene: str = "DPYD"
) -> list[dict[str, Any]]:
    """Read one gene's definition positions out of ``pharmcat_positions.vcf``.

    PharmCAT ships this file with each release. Taking REF and ALT from it
    rather than from the local catalog is deliberate: the two agree on every
    SNV, but PharmCAT anchors its indels one base to the left, and a REF that
    disagrees with PharmCAT's own definition would be rejected or, worse,
    silently matched against the wrong allele.
    """
    sites: list[dict[str, Any]] = []
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8 or f"PX={gene}" not in fields[7]:
                continue
            sites.append(
                {
                    "chrom": fields[0],
                    "pos": int(fields[1]),
                    "rsid": fields[2],
                    "ref": fields[3],
                    "alt": fields[4],
                }
            )
    sites.sort(key=lambda s: (s["chrom"], s["pos"]))
    return sites


def write_pharmcat_vcf(
    path: str | Path,
    sample_ids: list[str],
    sites: list[dict[str, Any]],
    calls_by_patient: dict[str, list[dict[str, Any]]],
    gene: str = "DPYD",
) -> Path:
    """Write a multi-sample VCF shaped for PharmCAT's named-allele matcher.

    Every site in ``sites`` is written for every sample. A sample with no call
    at a site is written ``0/0``: the synthetic generator only ever injects
    catalog variants, so absence here really is homozygous reference, not a
    gap in coverage.

    The FORMAT column is ``GT`` alone. DP and GQ are deliberately omitted
    rather than invented for the reference rows, which means PharmCAT takes
    every genotype at face value and has no counterpart to the engine's
    quality-refusal path. That difference is a reported parity finding, not a
    defect in this file.

    Raises ``ValueError`` if a patient carries a variant that has no site in
    ``sites``; such a call would otherwise vanish from the comparison.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_rsid = {str(site["rsid"]): site for site in sites}
    unmapped: dict[str, int] = {}
    # sample -> rsid -> GT, resolved once so the row loop stays a lookup.
    genotype_by_sample: dict[str, dict[str, str]] = {}
    for sample_id in sample_ids:
        resolved: dict[str, str] = {}
        for call in calls_by_patient.get(sample_id, []):
            rsid = str(call.get("rsid") or "")
            if rsid not in by_rsid:
                key = str(call.get("variant_id") or rsid or "?")
                unmapped[key] = unmapped.get(key, 0) + 1
                continue
            resolved[rsid] = str(
                call.get("phased") or call.get("genotype") or "0/0"
            )
        genotype_by_sample[sample_id] = resolved

    if unmapped:
        listed = ", ".join(f"{k} (x{v})" for k, v in sorted(unmapped.items()))
        raise ValueError(
            f"{len(unmapped)} variant(s) have no position in the PharmCAT "
            f"{gene} definition and would be dropped silently: {listed}"
        )

    lines = [
        f"##fileformat={VCF_VERSION}",
        f"##reference={ASSEMBLY}",
        CONTIG_LINE,
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##source=synthpgx",
        f"##synthpgx_shape=pharmcat_{gene}_definition_positions",
        "##synthpgx_warning=SYNTHETIC. In silico cohort validation. Not clinical evidence.",
        "\t".join(COLUMNS + list(sample_ids)),
    ]
    for site in sites:
        fields = [
            site["chrom"],
            str(site["pos"]),
            site.get("rsid") or ".",
            site["ref"],
            site["alt"],
            ".",
            "PASS",
            f"PX={gene}",
            PHARMCAT_FORMAT,
        ]
        rsid = str(site["rsid"])
        fields.extend(
            genotype_by_sample[sample_id].get(rsid, "0/0")
            for sample_id in sample_ids
        )
        lines.append("\t".join(fields))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path

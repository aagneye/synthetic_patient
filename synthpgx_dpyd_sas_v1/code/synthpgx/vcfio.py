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

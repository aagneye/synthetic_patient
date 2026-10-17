"""Locating, running and parsing PharmCAT (build brief Part 8.2).

SYNTHETIC. In silico cohort validation. Not clinical evidence.

PharmCAT is an external, independently maintained caller. It is deliberately
not vendored into this repository: the parity check is only worth anything if
it runs against a PharmCAT release the reader can download themselves, at a
version this module records rather than pins.

Nothing here decides whether the dataset is good. It produces PharmCAT's call
for each sample; :func:`synthpgx.qc.gate_pharmcat_parity` does the comparing.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

JAVA_HOME_ENV = "JAVA_HOME"
PHARMCAT_HOME_ENV = "PHARMCAT_HOME"
MIN_JAVA_MAJOR = 17

# PharmCAT spells DPYD allele names with the star designation appended, as in
# "c.85T>C (*9A)"; this dataset's caller uses the bare CPIC variant name. Only
# a "(*...)" suffix is stripped, so the compound allele
# "c.1129-5923C>G, c.1236G>A (HapB3)" keeps its name intact on both sides.
_STAR_SUFFIX = re.compile(r"\s*\(\*[^)]*\)$")


@dataclass(frozen=True)
class PharmcatInstall:
    """A usable PharmCAT: the jar, its positions file, and a Java to run it."""

    jar: Path
    positions: Path
    java: Path
    version: str
    java_version: str

    def as_metrics(self) -> dict[str, str]:
        # The jar's name, not its path: the path is wherever this machine
        # happens to have unpacked PharmCAT, and qc_metrics.json is committed.
        return {
            "pharmcat_version": self.version,
            "pharmcat_jar": self.jar.name,
            "java_version": self.java_version,
        }


class PharmcatUnavailable(RuntimeError):
    """PharmCAT, or a Java new enough to run it, could not be found."""


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


def _java_major(java: Path) -> int | None:
    try:
        proc = subprocess.run(
            [str(java), "-version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'version "(\d+)(?:\.(\d+))?', proc.stderr + proc.stdout)
    if not match:
        return None
    major = int(match.group(1))
    # Java 8 and earlier report as 1.8.x; the leading 1 is not the major.
    if major == 1 and match.group(2):
        major = int(match.group(2))
    return major


def find_java() -> tuple[Path, str] | None:
    """A ``java`` executable at version 17 or above, or None."""
    candidates: list[Path] = []
    java_home = os.environ.get(JAVA_HOME_ENV)
    if java_home:
        candidates.append(Path(java_home) / "bin" / "java.exe")
        candidates.append(Path(java_home) / "bin" / "java")
    on_path = shutil.which("java")
    if on_path:
        candidates.append(Path(on_path))
    # Standard JDK install roots, so the gate still runs on a machine where
    # Java was installed but never put on PATH.
    for root in (
        r"C:\Program Files\Java",
        r"C:\Program Files\Eclipse Adoptium",
        r"C:\Program Files\Microsoft\jdk",
        r"C:\Program Files\Amazon Corretto",
        "/usr/lib/jvm",
        "/Library/Java/JavaVirtualMachines",
    ):
        for entry in sorted(glob.glob(str(Path(root) / "*"))):
            candidates.append(Path(entry) / "bin" / "java.exe")
            candidates.append(Path(entry) / "bin" / "java")
            candidates.append(Path(entry) / "Contents" / "Home" / "bin" / "java")

    for candidate in candidates:
        if not candidate.exists():
            continue
        major = _java_major(candidate)
        if major is not None and major >= MIN_JAVA_MAJOR:
            return candidate, str(major)
    return None


def find_install(install_dir: str | Path | None = None) -> PharmcatInstall:
    """Locate a PharmCAT release.

    Looks in ``install_dir`` when given, then ``$PHARMCAT_HOME``. The directory
    must hold a ``pharmcat-*.jar`` and the matching ``pharmcat_positions*.vcf``
    that ships with the same release, because the positions file is what the
    PharmCAT-shaped VCF is built from.

    Raises :class:`PharmcatUnavailable` with a reason the gate can report.
    """
    roots = [Path(p) for p in (install_dir, os.environ.get(PHARMCAT_HOME_ENV)) if p]
    if not roots:
        raise PharmcatUnavailable(
            "no PharmCAT location configured (set $" + PHARMCAT_HOME_ENV
            + " or validation.pharmcat.install_dir in config.yaml)"
        )

    for root in roots:
        if not root.is_dir():
            continue
        jars = sorted(glob.glob(str(root / "pharmcat-*.jar")))
        jars = [j for j in jars if "preprocessor" not in Path(j).name]
        positions = sorted(glob.glob(str(root / "pharmcat_positions*.vcf")))
        if not jars or not positions:
            continue
        jar = Path(jars[-1])
        match = re.search(r"pharmcat-([0-9][^-]*)-", jar.name)
        version = match.group(1) if match else "unknown"
        java = find_java()
        if java is None:
            raise PharmcatUnavailable(
                f"PharmCAT was found at {root} but no Java "
                f"{MIN_JAVA_MAJOR}+ is available to run it"
            )
        return PharmcatInstall(
            jar=jar,
            positions=Path(positions[-1]),
            java=java[0],
            version=version,
            java_version=java[1],
        )

    listed = ", ".join(str(r) for r in roots)
    raise PharmcatUnavailable(
        f"no pharmcat-*.jar and pharmcat_positions*.vcf pair found in {listed}"
    )


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------


def run(
    install: PharmcatInstall,
    vcf: Path,
    output_dir: Path,
    gene: str = "DPYD",
    timeout: int = 3600,
) -> Path:
    """Run PharmCAT over a multi-sample VCF, one JSON report per sample."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(install.java), "-jar", str(install.jar),
        "-vcf", str(vcf),
        "-o", str(output_dir),
        "-g", gene,
        "-reporterJson",
        "-del",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise PharmcatUnavailable(f"PharmCAT exited {proc.returncode}: {tail.strip()}")
    return output_dir


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def normalise_allele(name: Any) -> str:
    """PharmCAT's allele name in this dataset's spelling."""
    if name is None:
        return "Reference"
    return _STAR_SUFFIX.sub("", str(name)).strip()


def normalise_diplotype(allele_1: Any, allele_2: Any) -> str:
    """An order-independent diplotype string, for comparing two callers.

    Neither caller promises an order for the two alleles, so both sides are
    sorted. Any difference that survives this is a difference in which alleles
    were named, not in how they were written down.
    """
    return " / ".join(sorted([normalise_allele(allele_1), normalise_allele(allele_2)]))


def split_diplotype(diplotype: Any) -> str:
    """This dataset's diplotype string, normalised the same way."""
    text = str(diplotype).strip()
    parts = text.split("/")
    if len(parts) != 2:
        # "Indeterminate", and anything else that is not a pair, compares as
        # itself rather than being forced into one.
        return text
    return " / ".join(sorted(part.strip() for part in parts))


def parse_reports(output_dir: Path, gene: str = "DPYD") -> pd.DataFrame:
    """Read the gene's recommendation diplotype out of each sample's report.

    PharmCAT reports every diplotype consistent with an unphased genotype in
    ``sourceDiplotypes``, but exactly one ``recommendationDiplotypes`` entry:
    the pair CPIC's lookup actually uses, which for DPYD is the two lowest
    activity values. That is the call this dataset's Layer A also makes, so it
    is the one compared.
    """
    rows = []
    for path in sorted(Path(output_dir).glob("*.report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        node = (payload.get("genes") or {}).get(gene)
        # "<base>.<sample>.report.json"; sample ids here never contain dots.
        sample_id = path.name.split(".")[-3]
        recommended = None
        if node:
            recommended = (node.get("recommendationDiplotypes") or [None])[0]
        if recommended is None:
            rows.append({"patient_id": sample_id, "pharmcat_called": False})
            continue
        rows.append(
            {
                "patient_id": sample_id,
                "pharmcat_called": True,
                "pharmcat_diplotype": normalise_diplotype(
                    (recommended.get("allele1") or {}).get("name"),
                    (recommended.get("allele2") or {}).get("name"),
                ),
                "pharmcat_phenotype": (recommended.get("phenotypes") or [None])[0],
                "pharmcat_activity_score": recommended.get("activityScore"),
                "pharmcat_n_possible_diplotypes": len(
                    node.get("sourceDiplotypes") or []
                ),
                "pharmcat_allele_definition_version": node.get(
                    "alleleDefinitionVersion"
                ),
            }
        )
    return pd.DataFrame(rows)

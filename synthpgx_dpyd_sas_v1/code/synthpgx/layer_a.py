"""Layer A: the deterministic DPYD call.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

This module is the dataset's ground truth. It is a lookup over frozen CPIC
tables plus the population layer, and it contains no model, no sampling and no
randomness. Given the same genotype it returns the same answer every time.

Order of operations (build brief 1.4)::

    variants -> diplotype -> activity score -> metabolizer status
             -> CPIC recommendation -> population override / refusal

If ``anukriti-pgx-core`` is importable, :func:`make_caller` returns an adapter
around its ``DPYDCaller`` instead, so the dataset is labelled by the real engine
wherever that engine is installed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# The five DrugSafetyOutcome values (build brief 4.4).
RECOMMENDATION_AS_IS = "RECOMMENDATION_AS_IS"
WITH_CAVEAT = "WITH_CAVEAT"
ALTERNATIVE_RECOMMENDED = "ALTERNATIVE_RECOMMENDED"
CONTRAINDICATED = "CONTRAINDICATED"
REFUSED_INSUFFICIENT_EVIDENCE = "REFUSED_INSUFFICIENT_EVIDENCE"

DRUG_SAFETY_OUTCOMES = {
    RECOMMENDATION_AS_IS,
    WITH_CAVEAT,
    ALTERNATIVE_RECOMMENDED,
    CONTRAINDICATED,
    REFUSED_INSUFFICIENT_EVIDENCE,
}

DOSE_ACTIONS = {
    "standard",
    "reduce_50pct",
    "avoid",
    "avoid_or_lt25pct_with_TDM",
}

PHENOTYPES = {
    "Normal Metabolizer",
    "Intermediate Metabolizer",
    "Poor Metabolizer",
    "Indeterminate",
}

ANCESTRY_ENUM = ("AFR", "AMR", "EAS", "EUR", "SAS")

REFERENCE_ALLELE = "Reference"
REFERENCE_ACTIVITY = 1.0


# ----------------------------------------------------------------------


@dataclass
class Call:
    """One Layer A answer, with the provenance that produced it."""

    diplotype: str
    activity_score: float | None
    phenotype: str
    recommendation: str
    dose_action: str | None
    cpic_recommendation_text: str | None = None
    sas_override: bool = False
    override_reason: str | None = None
    refusal_rule_id: str | None = None
    refusal_category: str | None = None
    phase_ambiguous: bool = False
    pmids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "diplotype_expected": self.diplotype,
            "activity_score_expected": self.activity_score,
            "phenotype_expected": self.phenotype,
            "cpic_recommendation_expected": self.recommendation,
            "cpic_dose_action_expected": self.dose_action,
            "sas_override_expected": self.sas_override,
            "override_reason_expected": self.override_reason,
            "refusal_rule_id_expected": self.refusal_rule_id,
            "refusal_category_expected": self.refusal_category,
            "phase_ambiguous_expected": self.phase_ambiguous,
            "cpic_recommendation_text": self.cpic_recommendation_text,
            "notes": "; ".join(self.notes) if self.notes else None,
        }


class RuleRegistry:
    """Valid refusal rule IDs, read from an ``anukriti-swarm`` checkout.

    The build brief (4.6) forbids inventing rule IDs and requires the build to
    fail on an unknown one. When no swarm source is supplied there is no list to
    validate against, so every ``refusal_rule_id`` is emitted as ``None`` and the
    semantic ``refusal_category`` carries the meaning instead. That gap is
    reported in ``qc/open_questions.md``.
    """

    def __init__(self, ids: set[str] | None, source: str | None):
        self.ids = ids
        self.source = source

    @property
    def available(self) -> bool:
        return bool(self.ids)

    @classmethod
    def from_source(cls, path: str | Path | None) -> "RuleRegistry":
        if not path:
            return cls(None, None)
        root = Path(path)
        if not root.exists():
            return cls(None, None)
        pattern = re.compile(r"\b([RVU](?:1[0-2]|[1-9]))\b")
        found: set[str] = set()
        for file in root.rglob("*.py"):
            try:
                found.update(pattern.findall(file.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
        return cls(found or None, str(root))

    def validate(self, rule_id: str | None) -> str | None:
        if rule_id is None:
            return None
        if not self.available:
            return None
        if rule_id not in self.ids:
            raise ValueError(
                f"refusal rule id {rule_id!r} is not defined in {self.source}"
            )
        return rule_id


# ----------------------------------------------------------------------


class CpicTables:
    """The frozen CPIC slice this engine looks up."""

    def __init__(self, reference_dir: str | Path):
        reference_dir = Path(reference_dir)
        self.reference_dir = reference_dir
        self.catalog = pd.read_csv(reference_dir / "variant_catalog.csv")
        self.alleles = pd.read_csv(reference_dir / "cpic_allele_functions.csv")
        self.diplotypes = pd.read_csv(reference_dir / "cpic_diplotype_phenotype.csv")
        self.recommendations = pd.read_csv(reference_dir / "cpic_recommendations.csv")
        meta = json.loads((reference_dir / "frequency_sources.json").read_text("utf-8"))
        self.gene = meta.get("gene", {})
        self.pmid_verification = meta.get("pmid_verification", {})

        self.activity_by_variant: dict[str, float] = {}
        self.function_by_variant: dict[str, str] = {}
        self.pmids_by_variant: dict[str, list[str]] = {}
        for _, row in self.catalog.iterrows():
            score = row.get("cpic_activity_score")
            if pd.notna(score):
                try:
                    self.activity_by_variant[row["variant_id"]] = float(score)
                except (TypeError, ValueError):
                    pass
            if pd.notna(row.get("cpic_function")):
                self.function_by_variant[row["variant_id"]] = row["cpic_function"]
            cites = row.get("cpic_citation_pmids")
            if isinstance(cites, str) and cites:
                self.pmids_by_variant[row["variant_id"]] = [
                    p for p in cites.split(";") if p
                ]

        # activity score -> phenotype, taken from the CPIC diplotype table.
        self.phenotype_by_score: dict[str, str] = {}
        for _, row in self.diplotypes.iterrows():
            score = row.get("activity_score")
            result = row.get("phenotype")
            if pd.notna(score) and pd.notna(result):
                self.phenotype_by_score.setdefault(_fmt_score(score), result)

        # (drug, activity score) -> CPIC recommendation row.
        self.recommendation_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for _, row in self.recommendations.iterrows():
            key = row.get("lookupkey")
            if pd.isna(key):
                key = row.get("activity_score")
            if pd.isna(key) or pd.isna(row.get("drug")):
                continue
            self.recommendation_by_key[(str(row["drug"]), _fmt_score(key))] = row.to_dict()

        self.override_variants = {
            row["variant_id"]: row
            for _, row in self.catalog.iterrows()
            if bool(row.get("sas_override_eligible"))
        }

        # Compound alleles: a haplotype defined by more than one variant, such
        # as HapB3 (c.1129-5923C>G together with c.1236G>A). A carrier has both
        # variants in cis on one chromosome, so treating them as two
        # independent reduced-function alleles would both halve the activity
        # score and make the phase look unresolvable.
        self.compound_alleles: dict[str, dict[str, Any]] = {}
        definitions = pd.read_csv(reference_dir / "cpic_allele_definitions.csv")
        function_by_allele = {
            row["allele_name"]: row for _, row in self.alleles.iterrows()
        }
        for allele_name, group in definitions.groupby("allele_name"):
            if allele_name == "Reference":
                continue
            members = sorted(set(group["variant_id"].astype(str)))
            if len(members) < 2:
                continue
            allele_row = function_by_allele.get(allele_name)
            if allele_row is None or pd.isna(allele_row.get("cpic_activity_score")):
                continue
            self.compound_alleles[str(allele_name)] = {
                "members": members,
                "activity": float(allele_row["cpic_activity_score"]),
                "function": allele_row.get("cpic_function"),
            }

    @property
    def cpic_version(self) -> str | None:
        return self.gene.get("version") and str(self.gene.get("version")) or None

    def is_known_variant(self, variant_id: str) -> bool:
        return variant_id in self.activity_by_variant


_DIGITS = re.compile(r"(\d+)")


def _allele_sort_key(name: str) -> tuple[object, ...]:
    """Order allele names the way a reader reads them: c.85 before c.1627.

    CPIC's two-lowest rule fixes *which activity values* the diplotype is
    built from, but when three or more alleles share the lowest value it does
    not say which of them to name. Some order has to be picked, and a plain
    string sort picks a misleading one: ``"c.1627A>G" < "c.85T>C"`` because
    ``"1" < "8"``, so the allele further along the transcript wins. Comparing
    each run of digits as a number instead puts the alleles in transcript
    order, which is also the order PharmCAT's DPYD caller uses, so the two
    callers name the same pair (qc/open_questions.md, "PharmCAT parity").

    The key alternates string and integer parts, so two keys always compare
    like with like.
    """
    return tuple(
        int(part) if part.isdigit() else part for part in _DIGITS.split(name)
    )


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value).strip()


# ----------------------------------------------------------------------


class DpydTableCaller:
    """Deterministic DPYD caller over the frozen CPIC tables."""

    name = "synthpgx.layer_a.DpydTableCaller"

    def __init__(
        self,
        tables: CpicTables,
        rules: RuleRegistry | None = None,
        outcome_map: dict[str, dict[str, str]] | None = None,
        min_dp: int = 20,
        min_gq: int = 30,
        supported_ancestries: Iterable[str] = ANCESTRY_ENUM,
    ):
        self.tables = tables
        self.rules = rules or RuleRegistry(None, None)
        self.outcome_map = outcome_map or {}
        self.min_dp = min_dp
        self.min_gq = min_gq
        self.supported_ancestries = set(supported_ancestries)

    # -- Layer 1 ----------------------------------------------------
    @staticmethod
    def _phase_changes_score(
        homozygous: list[float], heterozygous: list[float]
    ) -> tuple[bool, float]:
        """Does the activity score depend on how unphased variants are phased?

        Each haplotype's activity is the lowest activity among the variants on
        it, so a normal-function variant can never lower a haplotype below 1.0
        and can never change the total. Only reduced-function copies are passed
        in here, which is why carrying three common normal-function South Asian
        variants does not produce a spurious refusal.

        Returns ``(ambiguous, lowest_possible_score)``.
        """
        if not homozygous and not heterozygous:
            return False, 2.0 * REFERENCE_ACTIVITY
        if len(heterozygous) > 12:  # too many to enumerate; treat as ambiguous
            base = min(homozygous + heterozygous)
            return True, 2.0 * base

        totals: set[float] = set()
        for assignment in range(1 << len(heterozygous)):
            side_a = list(homozygous)
            side_b = list(homozygous)
            for i, activity in enumerate(heterozygous):
                if assignment >> i & 1:
                    side_a.append(activity)
                else:
                    side_b.append(activity)
            total = (min(side_a) if side_a else REFERENCE_ACTIVITY) + (
                min(side_b) if side_b else REFERENCE_ACTIVITY
            )
            totals.add(round(total, 1))
        return len(totals) > 1, min(totals)

    def _alleles(
        self, calls: list[dict[str, Any]]
    ) -> tuple[list[tuple[str, float]], bool, float]:
        """Resolve observed variant calls into two alleles and a score.

        CPIC's DPYD approach treats each variant as its own allele and bases
        the activity score on the two lowest-activity alleles, so that is the
        diplotype reported. Separately, the phasing is enumerated to find out
        whether the score would actually differ under another phasing; only
        then is the call ambiguous.
        """
        dosage: dict[str, int] = {}
        phase: dict[str, str] = {}
        for call in calls:
            genotype = call.get("genotype", "0/0")
            if genotype in ("0/0", "./."):
                continue
            dosage[call["variant_id"]] = 2 if genotype == "1/1" else 1
            phased = call.get("phased")
            if isinstance(phased, str) and "|" in phased:
                phase[call["variant_id"]] = phased

        copies: list[tuple[str, float]] = []
        homozygous_reduced: list[float] = []
        heterozygous_reduced: list[float] = []

        # Collapse compound alleles first. The number of copies of the
        # haplotype is the lowest dosage across its defining variants, and
        # those copies are then removed from the individual variants so they
        # are not counted twice.
        for allele_name, allele in self.tables.compound_alleles.items():
            members = allele["members"]
            if not all(dosage.get(m) for m in members):
                continue
            n_copies = min(dosage[m] for m in members)
            for member in members:
                dosage[member] -= n_copies
                if dosage[member] <= 0:
                    dosage.pop(member, None)
            activity = allele["activity"]
            copies.extend([(allele_name, activity)] * n_copies)
            if activity < REFERENCE_ACTIVITY:
                if n_copies == 2:
                    homozygous_reduced.append(activity)
                else:
                    heterozygous_reduced.extend([activity] * n_copies)

        for variant_id, n in sorted(dosage.items()):
            activity = self.tables.activity_by_variant.get(variant_id)
            if activity is None:
                continue
            copies.extend([(variant_id, activity)] * n)
            if activity < REFERENCE_ACTIVITY:
                if n == 2:
                    homozygous_reduced.append(activity)
                else:
                    heterozygous_reduced.append(activity)

        # A fully phased call has no ambiguity to resolve: each variant sits on
        # a known chromosome, so the haplotype activities are read off rather
        # than enumerated.
        reduced_ids = [
            variant_id
            for variant_id, activity in copies
            if activity < REFERENCE_ACTIVITY
        ]
        fully_phased = bool(reduced_ids) and all(
            variant_id in phase for variant_id in set(reduced_ids)
        )
        if fully_phased:
            side_a: list[float] = []
            side_b: list[float] = []
            for variant_id, activity in copies:
                if activity >= REFERENCE_ACTIVITY:
                    continue
                code = phase.get(variant_id, "0|1")
                left, _, right = code.partition("|")
                if left == "1":
                    side_a.append(activity)
                if right == "1":
                    side_b.append(activity)
            lowest = round(
                (min(side_a) if side_a else REFERENCE_ACTIVITY)
                + (min(side_b) if side_b else REFERENCE_ACTIVITY),
                1,
            )
            ambiguous = False
        else:
            ambiguous, lowest = self._phase_changes_score(
                homozygous_reduced, heterozygous_reduced
            )

        # Lowest activity first, then transcript order. Sorting the copy
        # multiset keeps the two copies of a homozygous allele adjacent, so
        # when that allele sorts first the pair named is the phase-consistent
        # one; with three heterozygous alleles no two-allele label can be,
        # and the pair named is a lookup key rather than a genotype.
        copies.sort(key=lambda item: (item[1], _allele_sort_key(item[0])))
        chosen = copies[:2]
        while len(chosen) < 2:
            chosen.append((REFERENCE_ALLELE, REFERENCE_ACTIVITY))
        chosen.sort(key=lambda item: (item[0] != REFERENCE_ALLELE, item[0]))
        # ``lowest`` is authoritative in every branch: when the phasing is
        # known it is the phased total, when enumeration finds one possible
        # total it is that total, and when the call is ambiguous it is the
        # most conservative of the possibilities.
        return chosen, ambiguous, lowest

    @staticmethod
    def _diplotype_string(alleles: list[tuple[str, float]]) -> str:
        return "/".join(a[0] for a in alleles)

    # -- full call ---------------------------------------------------
    def call(
        self,
        calls: list[dict[str, Any]],
        ancestry_super_pop: str,
        drug: str,
        ancestry_sub_pop: str | None = None,
    ) -> Call:
        # --- quality gate (refusal path) ---------------------------
        low_quality = [
            c for c in calls
            if c.get("genotype") not in ("0/0", None)
            and (
                (c.get("dp") is not None and c["dp"] < self.min_dp)
                or (c.get("gq") is not None and c["gq"] < self.min_gq)
            )
        ]
        if low_quality:
            worst = low_quality[0]
            return Call(
                diplotype="Indeterminate",
                activity_score=None,
                phenotype="Indeterminate",
                recommendation=REFUSED_INSUFFICIENT_EVIDENCE,
                dose_action=None,
                refusal_rule_id=self.rules.validate(None),
                refusal_category="low_quality_call",
                notes=[
                    f"{worst['variant_id']} DP={worst.get('dp')} GQ={worst.get('gq')} "
                    f"below thresholds DP>={self.min_dp} GQ>={self.min_gq}"
                ],
            )

        # --- unknown variant (refusal path) ------------------------
        unknown = [
            c
            for c in calls
            if c.get("genotype") not in ("0/0", "./.", None)
            and not self.tables.is_known_variant(c["variant_id"])
        ]
        if unknown:
            # An indel that shifts the reading frame is unequivocally damaging
            # to a human reader even though the rulebook has no entry for it.
            # That is a different situation from an uncharacterised missense
            # change, and the engine says so rather than collapsing both into
            # one refusal.
            frameshift = [
                c for c in unknown if _is_frameshift(c.get("ref"), c.get("alt"))
            ]
            if frameshift:
                category = "frameshift_not_in_tables"
                note = (
                    "carries a frameshift or other length-changing variant "
                    "absent from the CPIC allele table: "
                    + ", ".join(sorted(c["variant_id"] for c in frameshift))
                    + ". Loss of function is unequivocal but no table entry "
                    "exists, so no activity score is asserted."
                )
            else:
                category = "unlisted_variant_no_evidence"
                note = "carries variant(s) absent from the CPIC allele table: " + (
                    ", ".join(sorted(c["variant_id"] for c in unknown))
                )
            return Call(
                diplotype="Indeterminate",
                activity_score=None,
                phenotype="Indeterminate",
                recommendation=REFUSED_INSUFFICIENT_EVIDENCE,
                dose_action=None,
                refusal_rule_id=self.rules.validate(None),
                refusal_category=category,
                notes=[note],
            )

        # --- unsupported ancestry ----------------------------------
        if ancestry_super_pop not in self.supported_ancestries:
            return Call(
                diplotype="Indeterminate",
                activity_score=None,
                phenotype="Indeterminate",
                recommendation=REFUSED_INSUFFICIENT_EVIDENCE,
                dose_action=None,
                refusal_rule_id=self.rules.validate(None),
                refusal_category="unsupported_ancestry",
                notes=[f"ancestry {ancestry_super_pop!r} is outside the closed enum"],
            )

        # --- Layers 1 and 2 ----------------------------------------
        alleles, ambiguous, resolved_score = self._alleles(calls)
        diplotype = self._diplotype_string(alleles)
        activity_score = round(resolved_score, 1)
        score_key = _fmt_score(activity_score)
        phenotype = self.tables.phenotype_by_score.get(score_key)

        pmids: list[str] = []
        for variant_id, _ in alleles:
            pmids.extend(self.tables.pmids_by_variant.get(variant_id, []))

        if ambiguous:
            # Reduced-function variants whose phasing actually changes the
            # activity score. The honest answer is Indeterminate plus a caveat,
            # with the most conservative score reported.
            return Call(
                diplotype=diplotype,
                activity_score=activity_score,
                phenotype="Indeterminate",
                recommendation=REFUSED_INSUFFICIENT_EVIDENCE,
                dose_action=None,
                refusal_rule_id=self.rules.validate(None),
                refusal_category="ambiguous_phase",
                phase_ambiguous=True,
                pmids=sorted(set(pmids)),
                notes=[
                    "unphased reduced-function variants whose activity score "
                    "differs between cis and trans; the lowest possible score "
                    "is reported"
                ],
            )

        if phenotype is None:
            return Call(
                diplotype=diplotype,
                activity_score=activity_score,
                phenotype="Indeterminate",
                recommendation=REFUSED_INSUFFICIENT_EVIDENCE,
                dose_action=None,
                refusal_rule_id=self.rules.validate(None),
                refusal_category="no_phenotype_for_activity_score",
                pmids=sorted(set(pmids)),
                notes=[f"activity score {score_key} absent from CPIC diplotype table"],
            )

        # --- Layer 3 -----------------------------------------------
        rec_row = self.tables.recommendation_by_key.get((drug, score_key))
        mapping = self.outcome_map.get(score_key)
        if rec_row is None or mapping is None:
            return Call(
                diplotype=diplotype,
                activity_score=activity_score,
                phenotype=phenotype,
                recommendation=REFUSED_INSUFFICIENT_EVIDENCE,
                dose_action=None,
                refusal_rule_id=self.rules.validate(None),
                refusal_category="no_cpic_recommendation",
                pmids=sorted(set(pmids)),
                notes=[f"no CPIC recommendation for {drug} at activity score {score_key}"],
            )

        recommendation = mapping["outcome"]
        dose_action = mapping["dose_action"]
        call = Call(
            diplotype=diplotype,
            activity_score=activity_score,
            phenotype=phenotype,
            recommendation=recommendation,
            dose_action=dose_action,
            cpic_recommendation_text=rec_row.get("recommendation"),
            pmids=sorted(set(pmids)),
        )

        # --- population layer --------------------------------------
        self._apply_sas_override(call, calls, ancestry_super_pop)
        return call

    def _apply_sas_override(
        self, call: Call, calls: list[dict[str, Any]], ancestry_super_pop: str
    ) -> None:
        """Flag SAS patients whose risk CPIC's European-based default misses.

        The override never relaxes a CPIC call; it can only escalate a
        CPIC-normal result to a caveat. It is applied only to SAS patients and
        only for variants the catalog marked override-eligible by the spec 4.7
        rule.
        """
        if ancestry_super_pop != "SAS":
            return
        if call.recommendation != RECOMMENDATION_AS_IS:
            return

        carried = []
        for item in calls:
            if item.get("genotype") in ("0/0", "./.", None):
                continue
            row = self.tables.override_variants.get(item["variant_id"])
            if row is not None:
                carried.append(row)
        if not carried:
            return

        reasons = [str(r["sas_override_reason"]) for r in carried]
        pmids = [str(r["population_paper_pmid"]) for r in carried if r.get("population_paper_pmid")]
        call.sas_override = True
        call.override_reason = " | ".join(reasons)
        call.recommendation = WITH_CAVEAT
        call.dose_action = "standard"
        call.refusal_category = "non_cpic_population_override"
        call.pmids = sorted(set(call.pmids + pmids))
        call.notes.append(
            "Non-CPIC population override: CPIC assigns these variants normal "
            "function, but South Asian population evidence associates them with "
            "fluoropyrimidine toxicity. Flagged, not dose-changing."
        )


# ----------------------------------------------------------------------


def make_caller(tables: CpicTables, **kwargs) -> Any:
    """Return the real engine's caller when installed, else the table caller.

    The build brief requires Layer A to come from ``anukriti-pgx-core`` where it
    is available. The adapter is intentionally thin: if the import succeeds the
    dataset is labelled by the product engine, and the QC gate that compares the
    two is what catches any divergence.
    """
    try:  # pragma: no cover - depends on the environment
        from anukriti_pgx_core import DPYDCaller  # type: ignore

        return _CoreCallerAdapter(DPYDCaller(), tables, **kwargs)
    except Exception:
        return DpydTableCaller(tables, **kwargs)


class _CoreCallerAdapter:  # pragma: no cover - requires anukriti-pgx-core
    """Adapter around ``anukriti_pgx_core.DPYDCaller``."""

    name = "anukriti_pgx_core.DPYDCaller"

    def __init__(self, caller: Any, tables: CpicTables, **kwargs):
        self.caller = caller
        self.tables = tables
        self.fallback = DpydTableCaller(tables, **kwargs)

    def call(self, calls, ancestry_super_pop, drug, ancestry_sub_pop=None) -> Call:
        result = self.caller.call(
            variants=calls,
            population=ancestry_super_pop,
            drug=drug,
            sub_population=ancestry_sub_pop,
        )
        return Call(
            diplotype=result.diplotype,
            activity_score=result.activity_score,
            phenotype=result.phenotype,
            recommendation=result.recommendation,
            dose_action=getattr(result, "dose_action", None),
            cpic_recommendation_text=getattr(result, "recommendation_text", None),
            sas_override=bool(getattr(result, "population_override", False)),
            override_reason=getattr(result, "override_reason", None),
            refusal_rule_id=getattr(result, "refusal_rule_id", None),
            refusal_category=getattr(result, "refusal_category", None),
            pmids=list(getattr(result, "pmids", []) or []),
        )


def _is_frameshift(ref: Any, alt: Any) -> bool:
    """True when an indel changes the reading frame.

    A length change that is not a multiple of three shifts the frame. A
    multiple-of-three indel is in-frame and is not claimed to be a frameshift
    here, even though it may still be damaging.
    """
    if not isinstance(ref, str) or not isinstance(alt, str):
        return False
    if not ref or not alt:
        return False
    delta = len(ref) - len(alt)
    return delta != 0 and delta % 3 != 0

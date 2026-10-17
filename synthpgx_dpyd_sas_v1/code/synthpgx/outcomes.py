"""Layer B: simulated fluoropyrimidine toxicity outcomes.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

**These labels are sampled, not observed.** They come from a logistic model
whose coefficients are assumptions held in ``config.yaml``, and they exist to
stress the pipeline and to train the ML components that sit beside the engine.
They do not demonstrate that any variant causes toxicity, and they must never
feed back into Layer A.

Every coefficient carries a ``provenance`` tag. ``SOURCED`` means the number was
taken from a cited record; ``ASSUMPTION`` means it was chosen to be plausible
and is unverified. Both are written into the output so a reader can tell them
apart without opening the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

OUTCOME_COLUMNS = [
    "patient_id",
    "drug",
    "cycle_count",
    "ctcae_grade_max",
    "tox_neutropenia_g3plus",
    "tox_diarrhea_g3plus",
    "tox_mucositis_g3plus",
    "tox_handfoot_g3plus",
    "tox_neurotox_g3plus",
    "hospitalised",
    "treatment_related_death",
    "dose_reduced_by_clinician",
    "outcome_model_version",
    "effect_size_source_pmids",
    "effect_size_provenance",
    "label_type",
]

OXALIPLATIN_REGIMENS = {"FOLFOX", "XELOX"}
ORAL_REGIMENS = {"capecitabine", "XELOX"}


@dataclass
class EffectSizes:
    """One realisation of the Layer B coefficients."""

    values: dict[str, float]
    provenance: dict[str, str]
    pmids: list[str]
    version: str

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.values.get(key, default))


def draw_effect_sizes(
    spec: dict[str, Any], draw_index: int, seed: int
) -> EffectSizes:
    """Realise the coefficient set for one sensitivity draw.

    Draw 0 always uses the central values exactly, so the primary build is
    reproducible and readable. Later draws perturb each coefficient on the log
    odds scale by its declared uncertainty, which is how the brief asks for the
    sensitivity analysis to be produced.
    """
    rng = np.random.default_rng(seed + 9_000 + draw_index)
    values: dict[str, float] = {}
    provenance: dict[str, str] = {}
    pmids: set[str] = set()

    for name, entry in spec["coefficients"].items():
        central = float(entry["value"])
        if draw_index == 0:
            values[name] = central
        else:
            sd = float(entry.get("uncertainty_sd", 0.0))
            values[name] = float(rng.normal(central, sd)) if sd > 0 else central
        provenance[name] = str(entry.get("provenance", "ASSUMPTION"))
        for pmid in entry.get("source_pmids", []) or []:
            pmids.add(str(pmid))

    version = f"{spec['version']}-draw{draw_index}"
    return EffectSizes(values, provenance, sorted(pmids), version)


# ----------------------------------------------------------------------


def _logit_to_probability(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logit))


def simulate_outcomes(
    patients: pd.DataFrame,
    layer_a: pd.DataFrame,
    spec: dict[str, Any],
    seed: int,
    draw_index: int = 0,
) -> tuple[pd.DataFrame, EffectSizes]:
    """Sample one drug course per patient.

    The genotype term uses the Layer A activity score, plus a separate term for
    carriers the South Asian override flags. That second term is the part of the
    model that makes a CPIC-only screen look falsely reassuring for those
    patients, which is the behaviour the equity analysis is meant to measure.
    """
    effects = draw_effect_sizes(spec, draw_index, seed)
    rng = np.random.default_rng(seed + 50_000 + draw_index)

    frame = patients.merge(
        layer_a[
            [
                "patient_id",
                "activity_score_expected",
                "phenotype_expected",
                "sas_override_expected",
            ]
        ],
        on="patient_id",
        how="left",
    )
    n = len(frame)

    activity = frame["activity_score_expected"].to_numpy(dtype=float)
    # An indeterminate call carries no activity score; treat those patients as
    # metabolically normal for the purposes of the simulated outcome only.
    activity = np.where(np.isfinite(activity), activity, 2.0)
    deficit = np.clip(2.0 - activity, 0.0, 2.0)

    age = frame["age_years"].to_numpy(dtype=float)
    egfr = frame["renal_function_egfr"].to_numpy(dtype=float)
    comed = frame["co_medication_interacting"].to_numpy(dtype=bool)
    regimen = frame["regimen"].astype(str).to_numpy()
    override = frame["sas_override_expected"].fillna(False).to_numpy(dtype=bool)

    age_spec = spec["centring"]["age_years"]
    egfr_spec = spec["centring"]["renal_function_egfr"]

    logit = np.full(n, effects.get("intercept"))
    logit += effects.get("activity_deficit_per_unit") * deficit
    logit += effects.get("sas_override_carrier") * override
    logit += effects.get("age_per_decade") * (age - float(age_spec)) / 10.0
    logit += effects.get("egfr_per_10_below_ref") * np.clip(
        (float(egfr_spec) - egfr) / 10.0, 0.0, None
    )
    logit += effects.get("co_medication") * comed
    for name, value in spec["regimen_effects"].items():
        logit += float(value) * (regimen == name)

    probability = _logit_to_probability(logit)
    grade3plus = rng.random(n) < probability

    # --- CTCAE grade ------------------------------------------------
    grade = np.zeros(n, dtype=int)
    low_grades = np.array(spec["grade_distribution"]["no_g3plus"]["grades"])
    low_probs = np.array(spec["grade_distribution"]["no_g3plus"]["weights"], dtype=float)
    low_probs = low_probs / low_probs.sum()
    high_grades = np.array(spec["grade_distribution"]["g3plus"]["grades"])
    high_probs = np.array(spec["grade_distribution"]["g3plus"]["weights"], dtype=float)
    high_probs = high_probs / high_probs.sum()
    grade[~grade3plus] = rng.choice(
        low_grades, size=int((~grade3plus).sum()), p=low_probs
    )
    grade[grade3plus] = rng.choice(
        high_grades, size=int(grade3plus.sum()), p=high_probs
    )

    # --- toxicity types, conditioned on regimen ----------------------
    oral = np.isin(regimen, list(ORAL_REGIMENS))
    oxali = np.isin(regimen, list(OXALIPLATIN_REGIMENS))
    type_spec = spec["toxicity_types"]

    def _draw(kind: str, modifier: np.ndarray | None = None) -> np.ndarray:
        base = float(type_spec[kind]["p_given_g3plus"])
        probs = np.full(n, base)
        if modifier is not None:
            probs = probs * modifier
        return grade3plus & (rng.random(n) < np.clip(probs, 0.0, 1.0))

    handfoot_mod = np.where(oral, float(type_spec["handfoot"]["oral_multiplier"]), 1.0)
    neurotox_mod = np.where(
        oxali, float(type_spec["neurotox"]["oxaliplatin_multiplier"]), 0.05
    )

    neutropenia = _draw("neutropenia")
    diarrhea = _draw("diarrhea")
    mucositis = _draw("mucositis")
    handfoot = _draw("handfoot", handfoot_mod)
    neurotox = _draw("neurotox", neurotox_mod)

    # --- downstream clinical events ----------------------------------
    hospitalised = grade3plus & (
        rng.random(n) < float(spec["p_hospitalised_given_g3plus"])
    )
    death_logit = (
        np.log(float(spec["mortality_baseline"]) / (1 - float(spec["mortality_baseline"])))
        + effects.get("mortality_activity_deficit_per_unit") * deficit
    )
    death = (rng.random(n) < _logit_to_probability(death_logit)) & grade3plus
    grade[death] = 5

    dose_reduced = (grade3plus & (rng.random(n) < float(spec["p_dose_reduced_given_g3plus"]))) | (
        ~grade3plus & (rng.random(n) < float(spec["p_dose_reduced_given_no_g3plus"]))
    )

    cycles = rng.integers(
        int(spec["cycles"]["min"]), int(spec["cycles"]["max"]) + 1, size=n
    )
    cycles = np.where(death, np.minimum(cycles, int(spec["cycles"]["max_if_death"])), cycles)

    drug = np.where(oral, "capecitabine", "fluorouracil")

    out = pd.DataFrame(
        {
            "patient_id": frame["patient_id"].to_numpy(),
            "drug": drug,
            "cycle_count": cycles.astype(int),
            "ctcae_grade_max": grade.astype(int),
            "tox_neutropenia_g3plus": neutropenia,
            "tox_diarrhea_g3plus": diarrhea,
            "tox_mucositis_g3plus": mucositis,
            "tox_handfoot_g3plus": handfoot,
            "tox_neurotox_g3plus": neurotox,
            "hospitalised": hospitalised,
            "treatment_related_death": death,
            "dose_reduced_by_clinician": dose_reduced,
            "outcome_model_version": effects.version,
            "effect_size_source_pmids": ";".join(effects.pmids),
            "effect_size_provenance": _provenance_summary(effects),
            "label_type": "SIMULATED",
        }
    )
    return out[OUTCOME_COLUMNS], effects


def _provenance_summary(effects: EffectSizes) -> str:
    assumed = sorted(k for k, v in effects.provenance.items() if v.startswith("ASSUMPTION"))
    sourced = sorted(k for k, v in effects.provenance.items() if v.startswith("SOURCED"))
    return f"SOURCED=[{','.join(sourced)}];ASSUMPTION=[{','.join(assumed)}]"

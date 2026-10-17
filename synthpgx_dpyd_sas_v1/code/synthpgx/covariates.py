"""Method B: clinical covariates from a Gaussian copula.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

The build brief picks a copula for the tabular side on the strength of Miletic
et al. 2025 (PMID 41049602), which compared seven generators on 142
pharmacogenetics patients and found copula and synthpop gave the best privacy
and utility balance, with deep generators carrying higher re-identification
risk.

There are no real patient profiles to fit yet. So the copula is fitted to a
**seed cohort built from marginals declared in ``config.yaml``**, every one of
which is tagged as an assumption. The moment real profiles exist, pass them with
``--clinical-seed`` and the same copula expands them instead; that is the use
the brief actually wants this component for.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REGIMENS = [
    "5FU_bolus",
    "5FU_infusional",
    "capecitabine",
    "FOLFOX",
    "FOLFIRI",
    "XELOX",
]


def build_seed_frame(spec: dict[str, Any], n: int, seed: int) -> pd.DataFrame:
    """Construct the seed cohort the copula is fitted to.

    The correlations here are clinical assumptions, declared in config:
    renal function falls with age, and older patients are likelier to receive
    an oral or non-oxaliplatin regimen. They shape covariates only; Layer A
    never reads them.
    """
    rng = np.random.default_rng(seed)
    age_spec = spec["age_years"]
    egfr_spec = spec["renal_function_egfr"]

    # Correlated age / eGFR draw via a bivariate normal, then transformed.
    rho = float(spec["age_egfr_correlation"])
    normals = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)

    age = np.clip(
        normals[:, 0] * float(age_spec["sd"]) + float(age_spec["mean"]),
        float(age_spec["min"]),
        float(age_spec["max"]),
    ).round()
    egfr = np.clip(
        normals[:, 1] * float(egfr_spec["sd"]) + float(egfr_spec["mean"]),
        float(egfr_spec["min"]),
        float(egfr_spec["max"]),
    )

    sex = rng.choice(["F", "M"], size=n, p=_probabilities(spec["sex"], ["F", "M"]))

    base_weights = np.array(_probabilities(spec["regimen"], REGIMENS))
    age_shift = float(spec["regimen_age_shift"])
    oral = np.array([r in ("capecitabine", "XELOX") for r in REGIMENS], dtype=float)
    regimen = np.empty(n, dtype=object)
    centred = (age - float(age_spec["mean"])) / max(float(age_spec["sd"]), 1e-9)
    for i in range(n):
        weights = base_weights * np.exp(age_shift * centred[i] * oral)
        weights = weights / weights.sum()
        regimen[i] = REGIMENS[rng.choice(len(REGIMENS), p=weights)]

    dose_by_regimen = spec["dose_mg_m2"]
    dose = np.array(
        [
            float(dose_by_regimen[r]["mean"])
            * float(rng.normal(1.0, float(dose_by_regimen[r]["cv"])))
            for r in regimen
        ]
    )
    dose = np.clip(dose, 0.5 * dose.mean(), 1.5 * dose.mean()).round(1)

    comed = rng.random(n) < float(spec["co_medication_probability"])

    return pd.DataFrame(
        {
            "age_years": age.astype(int),
            "sex": sex,
            "regimen": regimen,
            "dose_mg_m2": dose,
            "renal_function_egfr": egfr.round(1),
            "co_medication_interacting": comed,
        }
    )


def _probabilities(mapping: dict[str, float], order: list[str]) -> list[float]:
    values = np.array([float(mapping[k]) for k in order], dtype=float)
    return list(values / values.sum())


# ----------------------------------------------------------------------


def sample_covariates(
    spec: dict[str, Any],
    n: int,
    seed: int,
    clinical_seed: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a Gaussian copula and draw ``n`` covariate rows.

    Returns the sampled frame and a provenance record naming the generator that
    actually ran, so the dataset can never claim SDV produced rows that the
    fallback produced.
    """
    if clinical_seed:
        seed_frame = pd.read_csv(clinical_seed)
        seed_provenance = {
            "seed_source": str(clinical_seed),
            "seed_is_real_data": True,
            "seed_rows": int(len(seed_frame)),
        }
    else:
        seed_rows = int(spec.get("seed_rows", 500))
        seed_frame = build_seed_frame(spec, seed_rows, seed)
        seed_provenance = {
            "seed_source": "config.yaml clinical_covariates marginals",
            "seed_is_real_data": False,
            "seed_rows": seed_rows,
            "warning": (
                "No real patient profiles were available. The copula was fitted "
                "to a seed cohort built from declared assumptions, so these "
                "covariates carry no empirical support."
            ),
        }

    sampled, generator = _fit_and_sample(seed_frame, n, seed)
    provenance = {"generator": generator, **seed_provenance}
    return sampled, provenance


def _fit_and_sample(
    seed_frame: pd.DataFrame, n: int, seed: int
) -> tuple[pd.DataFrame, str]:
    try:
        return _sdv_sample(seed_frame, n, seed), "sdv.GaussianCopulaSynthesizer"
    except Exception as exc:  # pragma: no cover - depends on the environment
        warnings.warn(
            f"SDV GaussianCopula unavailable or failed ({exc!r}); falling back to "
            "the copulas GaussianMultivariate engine.",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        return (
            _copulas_sample(seed_frame, n, seed),
            "copulas.multivariate.GaussianMultivariate",
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Gaussian copula unavailable ({exc!r}); resampling the seed frame "
            "with replacement instead. Covariate correlations are preserved only "
            "as far as bootstrap resampling preserves them.",
            RuntimeWarning,
            stacklevel=2,
        )
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(seed_frame), size=n)
    return seed_frame.iloc[index].reset_index(drop=True), "bootstrap_resample"


def _sdv_sample(seed_frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    import numpy as _np
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    _np.random.seed(seed)
    table = {"patients": seed_frame}
    metadata = Metadata.detect_from_dataframes(table)
    synthesizer = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synthesizer.fit(table)
        sampled = synthesizer.sample(num_rows=n)
    # SDV 2.x returns a {table_name: frame} mapping for multi-table metadata.
    if isinstance(sampled, dict):
        sampled = sampled["patients"]
    return sampled.reset_index(drop=True)


def _copulas_sample(seed_frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    import numpy as _np
    from copulas.multivariate import GaussianMultivariate

    _np.random.seed(seed)
    numeric = seed_frame.select_dtypes(include=[np.number])
    model = GaussianMultivariate()
    model.fit(numeric)
    sampled = model.sample(n).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    for column in seed_frame.columns:
        if column in sampled.columns:
            continue
        values = seed_frame[column].to_numpy()
        sampled[column] = values[rng.integers(0, len(values), size=n)]
    return sampled[seed_frame.columns]


def reconcile_dose(
    frame: pd.DataFrame, spec: dict[str, Any], seed: int
) -> pd.DataFrame:
    """Make the starting dose consistent with the regimen.

    A copula fitted over mixed types will happily pair capecitabine with a
    2400 mg/m2 infusional dose, because it models the dose column's marginal
    rather than its dependence on the regimen. Clinically the starting dose is
    a function of the regimen, so it is derived from the regimen here instead
    of being sampled alongside it.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    table = spec["dose_mg_m2"]
    means = np.array([float(table[r]["mean"]) for r in out["regimen"]])
    cvs = np.array([float(table[r]["cv"]) for r in out["regimen"]])
    noise = rng.normal(1.0, cvs)
    out["dose_mg_m2"] = np.clip(means * noise, means * 0.6, means * 1.4).round(1)
    return out


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Force the sampled frame back onto the schema's types and enums."""
    out = frame.copy()
    out["age_years"] = out["age_years"].astype(float).round().astype(int)
    out["dose_mg_m2"] = out["dose_mg_m2"].astype(float).round(1)
    out["renal_function_egfr"] = out["renal_function_egfr"].astype(float).round(1)
    out["sex"] = out["sex"].astype(str).str.upper().str[0].map(
        lambda s: s if s in ("F", "M") else "F"
    )
    out["regimen"] = out["regimen"].astype(str).where(
        out["regimen"].isin(REGIMENS), REGIMENS[0]
    )
    out["co_medication_interacting"] = out["co_medication_interacting"].astype(bool)
    return out

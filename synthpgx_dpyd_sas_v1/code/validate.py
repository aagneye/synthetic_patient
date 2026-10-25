#!/usr/bin/env python
"""Run every Part 8 quality gate and write the QC report.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Usage::

    python validate.py --config config.yaml

Exits non-zero if any gate fails. A gate whose reference data is missing is
reported SKIPPED and does not pass silently.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthpgx import covariates, layer_a, qc  # noqa: E402
from synthpgx.sources import Fetcher, ensembl_1kg_frequencies  # noqa: E402

LABEL = "SYNTHETIC. In silico cohort validation. Not clinical evidence."


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------


SAS_1KG_PANELS = {
    "1000GENOMES:phase_3:GIH",
    "1000GENOMES:phase_3:ITU",
    "1000GENOMES:phase_3:PJL",
    "1000GENOMES:phase_3:BEB",
    "1000GENOMES:phase_3:STU",
}


def fetch_reference_ld(
    fetcher: Fetcher, catalog: pd.DataFrame, max_variants: int = 12
) -> pd.DataFrame:
    """Pairwise r-squared from 1000 Genomes via Ensembl, for common variants.

    Ensembl rejects the ``population_name`` filter for some panels but will
    return every population for the pair, so the filter is applied here
    instead. The synthetic SAS cohort mixes all five South Asian panels, so
    the reference is their mean r-squared.
    """
    common = (
        catalog[catalog["freq_gnomad_sas"].fillna(0) >= 0.02]
        .dropna(subset=["rsid"])
        .sort_values("freq_gnomad_sas", ascending=False)
    )
    rsids = list(dict.fromkeys(common["rsid"].astype(str)))[:max_variants]
    rows = []
    for i in range(len(rsids)):
        for j in range(i + 1, len(rsids)):
            a, b = rsids[i], rsids[j]
            url = (
                f"https://rest.ensembl.org/ld/human/pairwise/{a}/{b}"
                "?content-type=application/json"
            )
            payload = fetcher.try_json(f"ensembl_ld_all_{a}_{b}", url)
            if not payload:
                continue
            values = [
                float(e["r2"])
                for e in payload
                if e.get("population_name") in SAS_1KG_PANELS and e.get("r2")
            ]
            if values:
                # Same canonical ordering as the synthetic side, so the two
                # tables join on the pair rather than on its spelling.
                key_a, key_b = sorted([a, b])
                rows.append(
                    {
                        "rsid_a": key_a,
                        "rsid_b": key_b,
                        "r2_reference": float(np.mean(values)),
                        "n_sas_panels": len(values),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument("--root", default=str(here.parent))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-ld-reference", action="store_true")
    parser.add_argument(
        "--pharmcat-work-dir",
        default=None,
        help="Keep the PharmCAT-shaped VCF and PharmCAT's own reports here. "
             "By default they are staged in a temporary directory and deleted "
             "afterwards: the 500 reports run to a few hundred megabytes, and "
             "the per-patient comparison is kept in qc/pharmcat_parity.csv "
             "either way.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.root)
    data_dir = root / "data"
    reference_dir = root / "reference"
    qc_dir = root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    log("loading dataset ...")
    patients = pd.read_parquet(data_dir / "patients.parquet")
    genotypes_long = pd.read_parquet(data_dir / "genotypes_long.parquet")
    expected = pd.read_parquet(data_dir / "expected_engine_output.parquet")
    refusal_cases = pd.read_parquet(data_dir / "refusal_cases.parquet")
    refusal_calls = pd.read_parquet(data_dir / "refusal_genotypes_long.parquet")
    outcomes_frame = pd.read_parquet(data_dir / "outcomes.parquet")
    catalog = pd.read_csv(reference_dir / "variant_catalog.csv")
    manifest = json.loads((qc_dir / "generation_manifest.json").read_text("utf-8"))
    log(f"  {len(patients)} patients, {len(genotypes_long)} calls, "
        f"{len(refusal_cases)} refusal cases")

    validation = config["validation"]
    results: list[qc.GateResult] = []

    # --- 8.1 -------------------------------------------------------
    log("8.1 genetic realism ...")
    results.append(
        qc.gate_allele_frequency(
            genotypes_long, patients, catalog, validation["allele_frequency"]
        )
    )
    results.append(
        qc.gate_hardy_weinberg(genotypes_long, patients, validation["hardy_weinberg"])
    )

    reference_ld = None
    if not args.skip_ld_reference:
        fetcher = Fetcher(reference_dir / "raw", offline=args.offline)
        try:
            reference_ld = fetch_reference_ld(fetcher, catalog)
            log(f"  1000 Genomes LD reference: {len(reference_ld)} pairs")
        except Exception as exc:  # pragma: no cover - network
            log(f"  LD reference unavailable: {exc!r}")
    results.append(
        qc.gate_linkage_disequilibrium(
            genotypes_long, patients, catalog, reference_ld
        )
    )
    results.append(
        qc.gate_population_structure(
            qc_dir / "background_genotypes.npz",
            bool(validation["population_structure"]["require_real_reference"]),
        )
    )
    results.append(qc.gate_diplotype_distribution(expected, patients))

    # --- 8.2 -------------------------------------------------------
    log("8.2 engine correctness ...")
    tables = layer_a.CpicTables(reference_dir)
    rules = layer_a.RuleRegistry.from_source(config["engine"].get("swarm_source"))
    calls_by_patient = _calls_by_patient(genotypes_long, refusal_calls)

    def recompute(frame: pd.DataFrame, default_drug: str | None = None) -> pd.DataFrame:
        caller = layer_a.make_caller(
            tables,
            rules=rules,
            outcome_map=config["engine"]["outcome_map"],
            min_dp=int(config["engine"]["min_dp"]),
            min_gq=int(config["engine"]["min_gq"]),
            supported_ancestries=config["engine"]["supported_ancestries"],
        )
        rows = []
        for _, row in frame.iterrows():
            calls = calls_by_patient.get(row["patient_id"], [])
            drug = default_drug or row.get("drug") or "fluorouracil"
            call = caller.call(
                calls, row["ancestry_super_pop"], drug, row.get("ancestry_sub_pop")
            )
            out = {"patient_id": row["patient_id"]}
            out.update(call.as_row())
            rows.append(out)
        return pd.DataFrame(rows)

    patient_drug = expected[["patient_id", "drug"]]
    engine_input = patients.merge(patient_drug, on="patient_id", how="left")

    results.append(qc.gate_engine_determinism(lambda: recompute(engine_input)))
    recomputed = recompute(engine_input)
    results.append(
        qc.gate_ground_truth(
            expected, recomputed, float(validation["engine"]["require_ground_truth_match"])
        )
    )
    results.append(
        qc.gate_refusals(
            refusal_cases,
            float(validation["engine"]["require_refusal_match"]),
            bool(manifest.get("refusal_rule_ids_available")),
        )
    )
    coverage_path = data_dir / "coverage_cases.parquet"
    coverage_cases = (
        pd.read_parquet(coverage_path) if coverage_path.exists() else None
    )
    results.append(qc.gate_outcome_coverage(expected, coverage_cases))

    # v2's intercept is DERIVED to match a sourced event rate, so nothing
    # re-solves it when another coefficient moves. This gate is what notices.
    v2_path = data_dir / "outcomes_v2.parquet"
    outcomes_v2 = pd.read_parquet(v2_path) if v2_path.exists() else None
    results.append(
        qc.gate_outcome_v2_calibration(outcomes_v2, config.get("outcomes_v2"))
    )
    log("8.2 PharmCAT parity (skipped quickly if PharmCAT is not installed) ...")
    with tempfile.TemporaryDirectory(prefix="synthpgx_pharmcat_") as scratch:
        results.append(
            qc.gate_pharmcat_parity(
                expected,
                genotypes_long,
                validation.get("pharmcat", {}),
                int(manifest["seed"]),
                work_dir=Path(args.pharmcat_work_dir or scratch),
                report_path=qc_dir / "pharmcat_parity.csv",
            )
        )

    # --- 8.3 -------------------------------------------------------
    log("8.3 privacy ...")
    provenance = manifest.get("covariate_provenance", {})
    seed_rows = int(
        provenance.get("seed_rows", config["clinical_covariates"]["seed_rows"])
    )
    seed_frame = covariates.build_seed_frame(
        config["clinical_covariates"], seed_rows, int(manifest["seed"]) + 23
    )
    # Records from the same distribution that the generator never saw. They are
    # the negative class the membership attack needs; drawing them with a
    # different seed is what makes them genuine non-members.
    holdout_frame = covariates.build_seed_frame(
        config["clinical_covariates"], seed_rows, int(manifest["seed"]) + 9_973
    )
    results.extend(
        qc.gate_privacy(
            patients, seed_frame, validation["privacy"],
            bool(provenance.get("seed_is_real_data")),
            holdout_frame=holdout_frame,
        )
    )

    # --- 8.4 / 8.5 --------------------------------------------------
    log("8.4 bias and equity ...")
    results.append(qc.gate_strata(patients, expected, validation["strata"]))
    results.append(qc.equity_analysis(patients, expected))
    results.append(qc.gate_ml_utility())

    # --- report -----------------------------------------------------
    log("writing QC report ...")
    summary = write_report(
        results, qc_dir / "qc_report.html", manifest, patients, expected,
        outcomes_frame, catalog,
    )
    (qc_dir / "qc_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    failed = [r for r in results if r.failed]
    for result in results:
        marker = {"PASS": "PASS   ", "FAIL": "FAIL   ",
                  "SKIPPED": "SKIP   ", "REPORTED": "REPORT "}[result.status]
        log(f"  {marker} [{result.section}] {result.name}: {result.summary}")

    if failed:
        log(f"BUILD FAILED: {len(failed)} gate(s) failed.")
        return 1
    skipped = [r for r in results if r.status == "SKIPPED"]
    log(f"All gates passed. {len(skipped)} gate(s) skipped for missing reference data.")
    return 0


def _calls_by_patient(
    genotypes_long: pd.DataFrame, refusal_calls: pd.DataFrame
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for frame in (genotypes_long, refusal_calls):
        if frame.empty:
            continue
        for row in frame.to_dict("records"):
            out.setdefault(row["patient_id"], []).append(row)
    return out


# ----------------------------------------------------------------------


def write_report(results, path: Path, manifest, patients, expected,
                 outcomes_frame, catalog) -> dict:
    counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0, "REPORTED": 0}
    for result in results:
        counts[result.status] += 1

    sections: list[str] = []
    for result in results:
        colour = {"PASS": "#1a7f37", "FAIL": "#b42318",
                  "SKIPPED": "#9a6700", "REPORTED": "#0969da"}[result.status]
        table = ""
        if result.details is not None and not result.details.empty:
            frame = result.details
            if len(frame) > 60:
                frame = frame.head(60)
                note = f"<p class='note'>Showing the first 60 of {len(result.details)} rows.</p>"
            else:
                note = ""
            table = frame.to_html(index=False, border=0, classes="data") + note
        sections.append(
            f"""
            <section>
              <h3><span class="badge" style="background:{colour}">{result.status}</span>
                  {html.escape(result.name)}
                  <span class="sec">Part {html.escape(result.section)}</span></h3>
              <p>{html.escape(result.summary)}</p>
              {table}
            </section>"""
        )

    overview = pd.DataFrame(
        [
            {"metric": "Patients", "value": len(patients)},
            {"metric": "Catalog variants", "value": len(catalog)},
            {"metric": "Injected variants", "value": manifest.get("n_injected_variants")},
            {"metric": "Refusal cases", "value": manifest.get("n_refusal_cases")},
            {"metric": "Layer A caller", "value": manifest.get("layer_a_caller")},
            {"metric": "CPIC version", "value": manifest.get("cpic_version")},
            {"metric": "PharmVar version", "value": manifest.get("pharmvar_version") or "unavailable (API needs credentials)"},
            {"metric": "Seed", "value": manifest.get("seed")},
            {"metric": "Generated (UTC)", "value": manifest.get("generated_utc")},
            {"metric": "Covariate generator",
             "value": (manifest.get("covariate_provenance") or {}).get("generator")},
            {"metric": "Simulated grade 3+ rate",
             "value": f"{outcomes_frame['ctcae_grade_max'].ge(3).mean():.3f}"},
        ]
    )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SynthPGx-DPYD-SAS QC report</title>
<style>
 :root {{ color-scheme: light dark; --fg:#1f2328; --bg:#ffffff; --muted:#656d76;
          --line:#d0d7de; --panel:#f6f8fa; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#e6edf3; --bg:#0d1117; --muted:#9198a1; --line:#30363d; --panel:#161b22; }}
 }}
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin:0;
         padding:2rem clamp(1rem,4vw,3rem); color:var(--fg); background:var(--bg);
         line-height:1.5; }}
 h1 {{ margin:0 0 .25rem; font-size:1.6rem; }}
 h3 {{ font-size:1rem; margin:0 0 .4rem; display:flex; align-items:center; gap:.6rem;
       flex-wrap:wrap; }}
 .warn {{ background:#fff4e5; border:1px solid #d4a72c; color:#7a4f01;
          padding:.75rem 1rem; border-radius:8px; font-weight:600; margin:1rem 0 1.5rem; }}
 @media (prefers-color-scheme: dark) {{ .warn {{ background:#2d2206; color:#f0c674; }} }}
 .badge {{ color:#fff; border-radius:999px; padding:.1rem .6rem; font-size:.7rem;
           letter-spacing:.04em; }}
 .sec {{ color:var(--muted); font-weight:400; font-size:.8rem; }}
 section {{ border:1px solid var(--line); border-radius:10px; padding:1rem 1.2rem;
            margin-bottom:1rem; background:var(--panel); }}
 table.data {{ border-collapse:collapse; font-size:.8rem; margin-top:.5rem;
               display:block; overflow-x:auto; max-width:100%; }}
 table.data th, table.data td {{ border:1px solid var(--line); padding:.25rem .5rem;
                                 text-align:left; white-space:nowrap; }}
 table.data th {{ background:var(--bg); }}
 .tally {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }}
 .tally div {{ border:1px solid var(--line); border-radius:8px; padding:.5rem 1rem;
               background:var(--panel); }}
 .note {{ color:var(--muted); font-size:.8rem; }}
</style></head><body>
<h1>SynthPGx-DPYD-SAS &mdash; QC report</h1>
<div class="warn">{LABEL}</div>
<div class="tally">
  <div><strong>{counts['PASS']}</strong> passed</div>
  <div><strong>{counts['FAIL']}</strong> failed</div>
  <div><strong>{counts['SKIPPED']}</strong> skipped</div>
  <div><strong>{counts['REPORTED']}</strong> reported</div>
</div>
<section><h3>Dataset</h3>{overview.to_html(index=False, border=0, classes="data")}</section>
{''.join(sections)}
<p class="note">Skipped gates are gaps, not passes. Every value in this report
is derived from the generated dataset and the frozen reference tables.</p>
</body></html>"""
    path.write_text(document, encoding="utf-8")

    return {
        "label": LABEL,
        "counts": counts,
        "gates": [
            {"name": r.name, "section": r.section, "status": r.status,
             "summary": r.summary, "metrics": r.metrics}
            for r in results
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())

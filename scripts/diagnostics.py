from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from meridian.analysis import analyzer, visualizer
from meridian.schema.serde import meridian_serde

from spendlens.config import settings
from spendlens.meridian_adapter import model_frame

MODELS = {
    "base": {
        "path": Path("models/spendlens.binpb"),
        "manifest": Path("models/run_manifest.json"),
        "suffix": "",
    },
    "calibrated": {
        "path": Path("models/spendlens_calibrated.binpb"),
        "manifest": None,  # calibrated fit does not write a manifest
        "suffix": "_calibrated",
    },
}
MAX_RHAT = 1.05
MIN_ESS = 400
MAX_DIVERGENCE_RATE = 0.001  # documented tolerance: <=0.1% of post-warmup draws
MAX_HOLDOUT_MAPE = 0.20


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default="base")
    args = p.parse_args()
    spec = MODELS[args.model]
    model_path, sfx = spec["path"], spec["suffix"]

    if not model_path.exists():
        raise SystemExit(f"MODEL NOT FOUND: {model_path}")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    if spec["manifest"] is not None:
        run = json.loads(spec["manifest"].read_text())
        if run.get("mode") != "full":
            raise SystemExit("FULL MODEL MANIFEST REQUIRED")
        if run.get("model_sha256") != model_hash:
            raise SystemExit("MODEL HASH MISMATCH")

    mmm = meridian_serde.load_meridian(str(model_path))
    Path("reports/html").mkdir(parents=True, exist_ok=True)
    Path("data/exports").mkdir(parents=True, exist_ok=True)

    # 1. Convergence
    idata = mmm.inference_data
    summary = az.summary(idata, round_to=4)
    summary.to_csv(f"reports/diagnostics_summary{sfx}.csv")
    max_rhat = float(summary["r_hat"].dropna().max())
    min_ess = float(summary["ess_bulk"].dropna().min())
    if "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum())
        total_draws = int(idata.sample_stats["diverging"].size)
    else:
        divergences, total_draws = 0, 0
    divergence_rate = divergences / total_draws if total_draws else 0.0

    # 2. Model fit chart
    fit = visualizer.ModelFit(
        mmm, confidence_level=settings()["constraints"]["confidence_level"]
    )
    fit.plot_model_fit(show_geo_level=False, include_ci=True).save(
        f"reports/html/model_fit{sfx}.html"
    )

    # 3. Holdout check. The model was built with holdout_id=True for the final
    # 13 weeks: those KPI values did not enter fitting, but their media/control
    # tensors are present, so expected_outcome gives a true future-period check.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        an = analyzer.Analyzer(mmm)
    draws = np.asarray(an.expected_outcome(
        use_posterior=True, aggregate_geos=False, aggregate_times=False,
        inverse_transform_outcome=True, use_kpi=True,
    ))
    draws = draws.reshape((-1,) + draws.shape[-2:])  # sample x geo x time
    mean = np.mean(draws, axis=0)
    lower = np.quantile(draws, 0.05, axis=0)
    upper = np.quantile(draws, 0.95, axis=0)

    raw = model_frame()
    geos = sorted(raw.geo.unique())
    times = sorted(raw.time.unique())
    actual = raw.pivot(index="geo", columns="time", values="kpi").loc[geos, times].to_numpy()
    rows = [
        (geo, time, actual[gi, ti], mean[gi, ti], lower[gi, ti], upper[gi, ti])
        for gi, geo in enumerate(geos)
        for ti, time in enumerate(times)
    ]
    pred = pd.DataFrame(rows, columns=["geo", "time", "actual", "predicted", "lower", "upper"])
    split = json.loads(Path("data/gold/split_manifest.json").read_text())
    pred["split"] = np.where(
        pd.to_datetime(pred.time) >= pd.Timestamp(split["holdout_start"]), "holdout", "train"
    )
    pred.to_csv(f"data/exports/holdout_predictions{sfx}.csv", index=False)
    h = pred.query("split == 'holdout'")
    holdout_mape = float(np.mean(np.abs(h.actual - h.predicted) / np.maximum(np.abs(h.actual), 1)))

    # 4. Gate
    gate = {
        "model": args.model,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": divergences,
        "divergence_rate": round(divergence_rate, 5),
        "max_divergence_rate": MAX_DIVERGENCE_RATE,
        "holdout_mape": holdout_mape,
        "pass_convergence": max_rhat <= MAX_RHAT and min_ess >= MIN_ESS
        and divergence_rate <= MAX_DIVERGENCE_RATE,
        "pass_holdout": holdout_mape <= MAX_HOLDOUT_MAPE,
    }
    gate["approved_for_insight"] = gate["pass_convergence"] and gate["pass_holdout"]
    out = Path(f"reports/model_health_gate{sfx}.json")
    out.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    if not gate["approved_for_insight"]:
        raise SystemExit(f"{args.model.upper()} MODEL NOT APPROVED FOR INSIGHT")


if __name__ == "__main__":
    main()
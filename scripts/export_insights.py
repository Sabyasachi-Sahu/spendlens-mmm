from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings

os.environ.setdefault("MERIDIAN_ENABLE_JAX_X64", "False")

from pathlib import Path

import pandas as pd
from meridian.analysis import analyzer, summarizer, visualizer
from meridian.schema.serde import meridian_serde

from spendlens.config import settings

CONFIDENCE = 0.90
MODELS = {
    "calibrated": {
        "path": Path("models/spendlens_calibrated.binpb"),
        "gate": Path("reports/model_health_gate_calibrated.json"),
        "suffix": "",
    },
    "base": {
        "path": Path("models/spendlens.binpb"),
        "gate": Path("reports/model_health_gate.json"),
        "suffix": "_uncalibrated",
    },
}
METRICS = [
    "spend",
    "pct_of_spend",
    "impressions",
    "pct_of_impressions",
    "cpm",
    "incremental_outcome",
    "pct_of_contribution",
    "roi",
    "effectiveness",
    "mroi",
    "cpik",
]
STATS = {"mean": "mean", "ci_lo": "lo90", "ci_hi": "hi90"}


def numeric_channel_table(mmm) -> pd.DataFrame:
    """Posterior metrics per channel as real numbers (mean and 90% interval)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ds = analyzer.Analyzer(mmm).summary_metrics(confidence_level=CONFIDENCE)
    ds = ds.sel(distribution="posterior")

    out = pd.DataFrame({"channel": ds.coords["channel"].values.astype(str)})
    for name in METRICS:
        if name not in ds.data_vars:
            continue
        da = ds[name]
        if "metric" in da.dims:  # modelled metric: mean / 90% interval
            for stat, suffix in STATS.items():
                out[f"{name}_{suffix}"] = da.sel(metric=stat).values.astype(float)
        else:  # fixed input such as spend: single value
            out[f"{name}_mean"] = da.values.astype(float)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default="calibrated")
    args = p.parse_args()
    spec = MODELS[args.model]
    model_path, gate_path, sfx = spec["path"], spec["gate"], spec["suffix"]

    # 1. Health gate and hash check
    if not gate_path.exists():
        raise SystemExit(f"Run first: python -m scripts.diagnostics --model {args.model}")
    gate = json.loads(gate_path.read_text())
    if not gate.get("approved_for_insight"):
        raise SystemExit(f"{args.model.upper()} model is not approved for insight")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if gate.get("model_sha256") != model_hash:
        raise SystemExit("MODEL HASH MISMATCH: model changed since its health gate. Rerun diagnostics.")

    cfg = settings()
    currency = cfg["project"].get("currency", "INR")
    mmm = meridian_serde.load_meridian(str(model_path))

    out = Path("data/exports")
    out.mkdir(parents=True, exist_ok=True)
    html = Path("reports/html")
    html.mkdir(parents=True, exist_ok=True)

    # 2. Numeric channel summary (for Power BI / Tableau)
    table = numeric_channel_table(mmm)
    table.insert(0, "model", args.model)
    table.insert(2, "currency", currency)
    table.to_csv(out / f"channel_summary{sfx}.csv", index=False)
    table.to_parquet(out / f"channel_summary{sfx}.parquet", index=False)

    # 3. Meridian HTML charts and report (Meridian formats these itself)
    ms = visualizer.MediaSummary(mmm, confidence_level=CONFIDENCE)
    charts = {
        "spend_vs_contribution": ms.plot_spend_vs_contribution(),
        "roi_vs_effectiveness": ms.plot_roi_vs_effectiveness(),
        "roi_vs_mroi": ms.plot_roi_vs_mroi(),
        "contribution_waterfall": ms.plot_contribution_waterfall_chart(),
    }
    for name, chart in charts.items():
        chart.save(str(html / f"{name}{sfx}.html"))
    summarizer.Summarizer(mmm).output_model_results_summary(
        filename=f"spendlens_model_summary{sfx}.html", filepath=str(html.resolve())
    )

    # 4. Export manifest for traceability
    manifest = {
        "model": args.model,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "health_gate": str(gate_path),
        "currency": currency,
        "confidence_level": CONFIDENCE,
        "rows": int(len(table)),
        "channel_summary": f"channel_summary{sfx}.csv",
    }
    (out / f"export_manifest{sfx}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    view = table[
        ["channel", "spend_mean", "roi_mean", "roi_lo90", "roi_hi90", "mroi_mean", "pct_of_contribution_mean"]
    ].round(2)
    print(f"model={args.model} | currency={currency} | rows={len(table)}")
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings

os.environ.setdefault("MERIDIAN_ENABLE_JAX_X64", "False")

from pathlib import Path

import numpy as np
import pandas as pd
from meridian.analysis import optimizer
from meridian.schema.serde import meridian_serde

from spendlens.config import settings

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
TOL = 1e-3


def mean_metric(ds, name):
    """Posterior mean of a metric as a plain array (None if not present)."""
    if name not in ds.data_vars:
        return None
    da = ds[name]
    for dim in list(da.dims):
        labels = [str(x) for x in da.coords[dim].values] if dim in da.coords else []
        if dim == "distribution" and "posterior" in labels:
            da = da.sel({dim: "posterior"})
        if dim == "metric" and "mean" in labels:
            da = da.sel({dim: "mean"})
    return np.asarray(da).squeeze()


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
    channels = cfg["channels"]
    lo = abs(cfg["constraints"]["min_spend_change"])
    hi = cfg["constraints"]["max_spend_change"]

    mmm = meridian_serde.load_meridian(str(model_path))
    args_builder = mmm.input_data.get_paid_channels_argument_builder()
    constraint_low = args_builder(**{c: lo for c in channels})
    constraint_high = args_builder(**{c: hi for c in channels})

    # 2. Optimise with a fixed total budget
    results = optimizer.BudgetOptimizer(mmm).optimize(
        fixed_budget=True,
        spend_constraint_lower=constraint_low,
        spend_constraint_upper=constraint_high,
        use_posterior=True,
        confidence_level=cfg["constraints"]["confidence_level"],
    )

    out = Path("data/exports")
    html = Path("reports/html")
    out.mkdir(parents=True, exist_ok=True)
    html.mkdir(parents=True, exist_ok=True)

    base, opt = results.nonoptimized_data, results.optimized_data
    base.to_dataframe().reset_index().to_csv(out / f"budget_metrics_baseline_long{sfx}.csv", index=False)
    opt.to_dataframe().reset_index().to_csv(out / f"budget_metrics_optimized_long{sfx}.csv", index=False)

    # 3. Channel-level allocation table
    channel_coord = "channel" if "channel" in base.coords else next(c for c in base.coords if "channel" in c)
    channel = [str(x) for x in base.coords[channel_coord].values]
    cols = {"channel": channel}
    for name in ("spend", "incremental_outcome", "roi", "mroi"):
        b, o = mean_metric(base, name), mean_metric(opt, name)
        if b is not None and np.ndim(b) == 1 and len(b) == len(channel):
            cols[f"baseline_{name}"] = b
            cols[f"optimized_{name}"] = o
    allocation = pd.DataFrame(cols)
    allocation.insert(1, "currency", currency)
    allocation["change_pct"] = allocation.optimized_spend / allocation.baseline_spend - 1
    allocation.insert(0, "model", args.model)
    allocation.to_csv(out / f"budget_allocation{sfx}.csv", index=False)

    results.get_response_curves().to_dataframe().reset_index().to_csv(
        out / f"response_curves{sfx}.csv", index=False
    )
    results.output_optimization_summary(
        filename=f"spendlens_optimization{sfx}.html", filepath=str(html.resolve())
    )
    results.plot_spend_delta().save(str(html / f"optimization_spend_delta{sfx}.html"))

    # 4. Sanity checks
    paid = allocation[~allocation.channel.str.contains("All", case=False)]
    total_base, total_opt = paid.baseline_spend.sum(), paid.optimized_spend.sum()
    # Meridian reports spend rounded to the nearest 100,000, so allow half a rounding step.
    ROUND_STEP = 100_000
    low_ok = paid.baseline_spend * (1 - lo) - ROUND_STEP / 2
    high_ok = paid.baseline_spend * (1 + hi) + ROUND_STEP / 2
    within = bool(((paid.optimized_spend >= low_ok) & (paid.optimized_spend <= high_ok)).all())
    same_budget = abs(total_opt / total_base - 1) <= TOL

    view = paid[["channel", "baseline_spend", "optimized_spend", "change_pct", "baseline_mroi", "optimized_mroi"]].copy()
    view["change_pct"] = (view.change_pct * 100).round(1)
    print(f"model={args.model} | currency={currency} | budget constraint: -{lo:.0%} to +{hi:.0%}")
    print(view.round(2).to_string(index=False))
    print(f"\nCHECK total budget unchanged: {'PASS' if same_budget else 'FAIL'} ({total_base:,.0f} -> {total_opt:,.0f})")
    print(f"CHECK every channel within +/-{lo:.0%} limits (allowing 50,000 rounding): {'PASS' if within else 'FAIL'}")
    if "baseline_incremental_outcome" in paid and "optimized_incremental_outcome" in paid:
        b, o = paid.baseline_incremental_outcome.sum(), paid.optimized_incremental_outcome.sum()
        print(f"Incremental outcome (sum of channels): {b:,.0f} -> {o:,.0f} ({o / b - 1:+.1%})")


if __name__ == "__main__":
    main()
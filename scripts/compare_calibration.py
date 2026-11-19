from __future__ import annotations

import os

os.environ.setdefault("MERIDIAN_ENABLE_JAX_X64", "False")

from pathlib import Path

import pandas as pd
from meridian.analysis import analyzer
from meridian.schema.serde import meridian_serde

from spendlens.config import settings

CONFIDENCE = 0.90
MODELS = {
    "uncalibrated": "models/spendlens.binpb",
    "calibrated": "models/spendlens_calibrated.binpb",
}
METRICS = [
    "spend",
    "pct_of_spend",
    "incremental_outcome",
    "pct_of_contribution",
    "roi",
    "mroi",
    "effectiveness",
    "cpik",
]
STATS = {"mean": "mean", "ci_lo": "lo90", "ci_hi": "hi90"}
OUT = Path("data/exports")


def make_analyzer(mmm):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return analyzer.Analyzer(mmm)


def posterior_metrics(model_path: str, label: str) -> pd.DataFrame:
    """Numeric posterior metrics per channel (no prior rows, no text formatting)."""
    mmm = meridian_serde.load_meridian(model_path)
    ds = make_analyzer(mmm).summary_metrics(confidence_level=CONFIDENCE)
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

    out.insert(0, "model", label)
    return out


def main() -> None:
    cfg = settings()
    currency = cfg["project"].get("currency", "INR")

    long = pd.concat(
        [posterior_metrics(path, label) for label, path in MODELS.items()],
        ignore_index=True,
    )
    long.insert(1, "currency", currency)

    ids = (
        pd.read_csv(OUT / "calibration_inputs.csv")
        .set_index("channel")["experiment_id"]
        .to_dict()
    )
    long["experiment_id"] = long["channel"].map(ids).fillna("none")

    # Wide: one row per channel, uncalibrated vs calibrated side by side
    key_cols = ["channel", "currency", "experiment_id"]
    u = long[long.model == "uncalibrated"].drop(columns="model")
    c = long[long.model == "calibrated"].drop(columns="model")
    value_cols = [col for col in u.columns if col not in key_cols]
    wide = u.merge(
        c, on=key_cols, how="outer", suffixes=("_uncal", "_cal"), validate="one_to_one"
    )
    for metric in ("roi", "mroi", "pct_of_contribution"):
        if f"{metric}_mean_uncal" in wide and f"{metric}_mean_cal" in wide:
            wide[f"{metric}_delta"] = wide[f"{metric}_mean_cal"] - wide[f"{metric}_mean_uncal"]

    OUT.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT / "calibration_comparison_long.csv", index=False)
    wide.to_csv(OUT / "calibration_comparison.csv", index=False)

    view = wide[
        ["channel", "experiment_id", "roi_mean_uncal", "roi_mean_cal", "roi_delta",
         "roi_lo90_cal", "roi_hi90_cal", "mroi_mean_cal"]
    ].round(2)
    print(f"currency={currency} | confidence={CONFIDENCE} | posterior only")
    print(view.to_string(index=False))
    print(f"\nrows: long={len(long)} wide={len(wide)} (expected wide = channels + totals)")


if __name__ == "__main__":
    main()
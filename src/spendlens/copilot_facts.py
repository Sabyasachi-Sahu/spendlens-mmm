"""Build numbered facts from the exported CSV files for the SpendLens copilot."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPORTS = Path("data/exports")
ALL = "All Channels"


def _read(name: str) -> pd.DataFrame:
    path = EXPORTS / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the export scripts first.")
    return pd.read_csv(path)


def _ranked(df: pd.DataFrame, col: str, fmt: str) -> str:
    """'name value, name value, ...' sorted from highest to lowest."""
    d = df.sort_values(col, ascending=False)
    return ", ".join(f"{r['channel']} {format(r[col], fmt)}" for _, r in d.iterrows())


def build_facts() -> list[dict]:
    facts: list[dict] = []

    def add(kind: str, channel: str, text: str, source: str) -> None:
        facts.append(
            {
                "id": f"F{len(facts) + 1:03d}",
                "kind": kind,
                "channel": channel,
                "text": text,
                "source": source,
            }
        )

    cs = _read("channel_summary.csv")
    ba = _read("budget_allocation.csv")
    cc = _read("calibration_comparison.csv")
    ci = _read("calibration_inputs.csv")

    # Ranks are computed here so the model copies them instead of comparing numbers.
    paid = cs[cs["channel"] != ALL].copy()
    n = len(paid)
    paid["mroi_rank"] = paid["mroi_mean"].rank(ascending=False, method="min").astype(int)
    paid["roi_rank"] = paid["roi_mean"].rank(ascending=False, method="min").astype(int)
    ranks = paid.set_index("channel")[["mroi_rank", "roi_rank"]].to_dict("index")

    ba = ba.copy()
    nb = len(ba)
    ba["change_rank"] = ba["change_pct"].rank(ascending=False, method="min").astype(int)
    change_rank = ba.set_index("channel")["change_rank"].to_dict()
    ba["change_amount"] = ba["optimized_spend"] - ba["baseline_spend"]
    ba["amount_rank"] = ba["change_amount"].rank(ascending=False, method="min").astype(int)
    amount_rank = ba.set_index("channel")["amount_rank"].to_dict()
    # 0. Ranking facts first, so a long fact list never drops them
    add(
        "overview", ALL,
        "Marginal ROI at current spend, highest first (rank 1 = best): "
        + _ranked(paid, "mroi_mean", ".2f") + ".",
        "channel_summary.csv, all channels",
    )
    add(
        "overview", ALL,
        "ROI, highest first (rank 1 = best): " + _ranked(paid, "roi_mean", ".2f") + ".",
        "channel_summary.csv, all channels",
    )
    ba["change_m"] = ba["change_amount"] / 1e6
    add(
        "budget_overview", ALL,
        "Budget change in million INR, largest increase first: "
        + _ranked(ba, "change_m", "+.1f") + ".",
        "budget_allocation.csv, all rows",
    )
    add(
        "budget_overview", ALL,
        "Budget change in percent, largest increase first (budgets are rounded to 100,000, "
        "so channels at the 20% limit can differ by a tenth of a point): "
        + _ranked(ba, "change_pct", "+.1%") + ".",
        "budget_allocation.csv, all rows",
    )

    # 1. Channel summary (calibrated model)
    for _, r in cs.iterrows():
        ch, cur = r["channel"], r["currency"]
        src = f"channel_summary.csv, channel={ch}"
        if ch == ALL:
            add(
                "overview", ch,
                f"All paid channels together: ROI {r['roi_mean']:.2f} "
                f"(90% range {r['roi_lo90']:.2f} to {r['roi_hi90']:.2f}); paid media drives "
                f"{r['pct_of_contribution_mean']:.1f}% of outcomes; spend {r['spend_mean']:,.0f} {cur}.",
                src,
            )
        else:
            rk = ranks[ch]
            add(
                "channel", ch,
                f"{ch}: spend {r['spend_mean']:,.0f} {cur}; ROI {r['roi_mean']:.2f} "
                f"(90% range {r['roi_lo90']:.2f} to {r['roi_hi90']:.2f}), ROI rank {rk['roi_rank']} of {n}; "
                f"marginal ROI at current spend {r['mroi_mean']:.2f}, marginal ROI rank {rk['mroi_rank']} of {n} "
                f"(1 = highest).",
                src,
            )

    # 2. Budget optimiser
    cur = ba["currency"].iloc[0]
    for _, r in ba.iterrows():
        ch = r["channel"]
        add(
            "budget", ch,
            f"{ch}: budget {r['baseline_spend']:,.0f} -> {r['optimized_spend']:,.0f} {cur}, "
            f"a change of {r['change_amount'] / 1e6:+.1f} million {cur} ({r['change_pct'] * 100:+.1f}%); "
            f"rank by amount {amount_rank[ch]} of {nb}, by percent {change_rank[ch]} of {nb} "
            f"(1 = largest increase); marginal ROI {r['baseline_mroi']:.2f} at the optimiser's "
            f"baseline -> {r['optimized_mroi']:.2f} after the move.",
            f"budget_allocation.csv, channel={ch}",
        )
    base_total, opt_total = ba["baseline_spend"].sum(), ba["optimized_spend"].sum()
    base_out = ba["baseline_incremental_outcome"].sum()
    opt_out = ba["optimized_incremental_outcome"].sum()
    same = "unchanged" if abs(opt_total / base_total - 1) < 1e-3 else "changed"
    add(
        "budget_overview", ALL,
        f"Total paid budget {base_total:,.0f} -> {opt_total:,.0f} {cur} ({same}); "
        f"incremental outcome {base_out:,.0f} -> {opt_out:,.0f} ({opt_out / base_out - 1:+.1%}). "
        "Each channel may move by at most 20% either way. "
        "The optimiser reports spend rounded to the nearest 100,000.",
        "budget_allocation.csv, all rows",
    )

    # 3. Calibration: uncalibrated vs calibrated model, with the change stated
    cal = cc[cc["channel"] != ALL]
    add(
        "calibration_overview", ALL,
        "ROI change from calibration, largest increase first: " + _ranked(cal, "roi_delta", "+.2f") + ".",
        "calibration_comparison.csv, all channels",
    )    
    for _, r in cc.iterrows():
        ch = r["channel"]
        add(
            "calibration_overview" if ch == ALL else "calibration", ch,
            f"{ch}: ROI {r['roi_mean_uncal']:.2f} before calibration -> {r['roi_mean_cal']:.2f} after, "
            f"a change of {r['roi_delta']:+.2f} (90% range after: {r['roi_lo90_cal']:.2f} to {r['roi_hi90_cal']:.2f}).",
            f"calibration_comparison.csv, channel={ch}",
        )

    # 4. Lift-test results, with the KPI range converted to ROI units
    for _, r in ci.iterrows():
        scale = r["roi_point"] / r["incremental_kpi"]
        lo, hi = r["lift_lower"] * scale, r["lift_upper"] * scale
        add(
            "experiment", r["channel"],
            f"{r['experiment_id']} ({r['design']}): the lift test on {r['channel']} measured ROI "
            f"{r['roi_point']:.2f} (90% range {lo:.2f} to {hi:.2f}, converted from the KPI range).",
            f"calibration_inputs.csv, experiment_id={r['experiment_id']}",
        )

    # 5. Fixed caveats from the README
    for text in [
        "All data in this project is synthetic. No real client or patient data is used.",
        "Each KPI unit is valued at an assumed 5,200 INR (a setting in the synthetic data, not a measured price).",
        "The outcome lift from the budget move is a point estimate with no uncertainty range, "
        "and six of eight channels sit at their +/-20% limit.",
        "Only two channels (rep_visits and hcp_email) have lift tests. The other six are calibrated indirectly.",
    ]:
        add("caveat", ALL, text, "README.md")

    return facts


if __name__ == "__main__":
    df = pd.DataFrame(build_facts())
    out = EXPORTS / "copilot_facts.csv"
    df.to_csv(out, index=False)
    print(df.groupby("kind").size().to_string())
    print(f"\ntotal facts: {len(df)} -> {out}\n")
    for _, r in df.head(3).iterrows():
        print(r["id"], r["text"])
    print()
    for _, r in df[df["channel"] == "rep_visits"].iterrows():
        print(r["id"], r["text"])
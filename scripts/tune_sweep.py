from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KNOTS = [4, 6, 8]
ETA_SCALES = [0.25, 0.3, 0.4]
OUT_DIR = Path("reports/tuning")
MAX_MAPE = 0.05  # reject settings that predict badly on the holdout


def run(cmd: list[str], env: dict, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    grid = list(itertools.product(KNOTS, ETA_SCALES))
    print(f"[sweep] {len(grid)} combinations: {grid}")
    if args.dry_run:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (knots, eta) in enumerate(grid, start=1):
        tag = f"k{knots}_eta{eta}"
        run_dir = OUT_DIR / tag
        gate_file = run_dir / "model_health_gate.json"

        if gate_file.exists():
            print(f"[{i}/{len(grid)}] {tag}: already done, skipping")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": "src",
                "MERIDIAN_ENABLE_JAX_X64": "False",
                "TF_ENABLE_ONEDNN_OPTS": "0",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "SPENDLENS_KNOTS": str(knots),
                "SPENDLENS_ETA_M_SCALE": str(eta),
            })
            Path("reports/model_health_gate.json").unlink(missing_ok=True)
            t0 = time.perf_counter()
            rc = run([sys.executable, "-m", "scripts.fit_meridian", "--mode", "full"],
                     env, run_dir / "fit.log")
            if rc != 0:
                print(f"    fit FAILED (see {run_dir / 'fit.log'})")
                continue
            rc = run([sys.executable, "-m", "scripts.diagnostics"], env, run_dir / "diagnostics.log")
            gate_src = Path("reports/model_health_gate.json")
            if not gate_src.exists():
                print(f"    diagnostics FAILED (see {run_dir / 'diagnostics.log'})")
                continue
            shutil.copy(gate_src, gate_file)
            shutil.copy("models/run_manifest.json", run_dir / "run_manifest.json")
            print(f"    done in {time.perf_counter() - t0:.0f}s", flush=True)

        gate = json.loads(gate_file.read_text())
        rows.append({
            "knots": knots,
            "eta_m_scale": eta,
            "divergences": gate["divergences"],
            "max_rhat": round(gate["max_rhat"], 4),
            "min_ess_bulk": round(gate["min_ess_bulk"], 1),
            "holdout_mape": round(gate["holdout_mape"], 4),
            "approved": gate["approved_for_insight"],
        })

    # Rank: reject poor holdout, then fewest divergences, lowest R-hat, highest ESS
    ok = [r for r in rows if r["holdout_mape"] <= MAX_MAPE]
    ok.sort(key=lambda r: (r["divergences"], r["max_rhat"], -r["min_ess_bulk"]))
    rejected = [r for r in rows if r["holdout_mape"] > MAX_MAPE]
    ranked = ok + rejected

    summary = OUT_DIR / "sweep_summary.csv"
    if ranked:
        with summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()) + ["rank"])
            w.writeheader()
            for n, r in enumerate(ranked, start=1):
                w.writerow({**r, "rank": n if r in ok else "rejected"})
        print(f"\n[sweep] summary written to {summary}")
        for n, r in enumerate(ranked, start=1):
            print(n if r in ok else "X", r)


if __name__ == "__main__":
    main()
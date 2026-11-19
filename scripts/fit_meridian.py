from __future__ import annotations
import os
# These must be set before Meridian/JAX is imported.
os.environ.setdefault("MERIDIAN_ENABLE_JAX_X64", "False")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
from meridian.schema.serde import meridian_serde
from spendlens.meridian_adapter import build_model

class FrozenKwargs(dict):
    """Hashable dict so JAX can treat sampler settings as static arguments."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))
    
PROFILES = {
    "smoke": {"chains": 2, "adapt": 50, "burnin": 50, "keep": 100, "prior": 50},
    # CPU-balanced candidate. Step 9.8 still decides whether it is usable.
    "full": {"chains": 4, "adapt": 1000, "burnin": 500, "keep": 1000, "prior": 300},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "uncommitted"


def heartbeat(stop: threading.Event, started: float, seconds: int) -> None:
    while not stop.wait(seconds):
        elapsed = time.perf_counter() - started
        print(f"[heartbeat] posterior still running; elapsed={elapsed:.1f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--chains", type=int)
    p.add_argument("--adapt", type=int)
    p.add_argument("--burnin", type=int)
    p.add_argument("--keep", type=int)
    p.add_argument("--heartbeat-seconds", type=int, default=60)
    a = p.parse_args()

    cfg = dict(PROFILES[a.mode])
    for name in ("chains", "adapt", "burnin", "keep"):
        value = getattr(a, name)
        if value is not None:
            cfg[name] = value
    if min(cfg.values()) <= 0:
        raise SystemExit("All sampling values must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    output = model_dir / ("spendlens_smoke.binpb" if a.mode == "smoke" else "spendlens.binpb")
    temp_output = model_dir / f".{output.stem}.{stamp}.tmp{output.suffix}"
    manifest_path = model_dir / ("run_manifest_smoke.json" if a.mode == "smoke" else "run_manifest.json")
    data_path = Path("data/gold/spendlens_gold.parquet")

    devices = [str(x) for x in jax.devices()]
    print(f"[start] mode={a.mode} config={cfg}", flush=True)
    print(f"[environment] devices={devices} x64={os.environ['MERIDIAN_ENABLE_JAX_X64']}", flush=True)
    total_start = time.perf_counter()

    print("[1/4] building Meridian model", flush=True)
    mmm = build_model()
    print(f"[2/4] sampling prior: {cfg['prior']} draws", flush=True)
    prior_start = time.perf_counter()
    mmm.sample_prior(cfg["prior"])
    prior_seconds = time.perf_counter() - prior_start

    print(
        f"[3/4] sampling posterior: chains={cfg['chains']} adapt={cfg['adapt']} "
        f"burnin={cfg['burnin']} keep={cfg['keep']}",
        flush=True,
    )
    posterior_start = time.perf_counter()
    stop = threading.Event()
    monitor = threading.Thread(
        target=heartbeat,
        args=(stop, posterior_start, a.heartbeat_seconds),
        daemon=True,
    )
    monitor.start()
    try:
        mmm.sample_posterior(
            n_chains=cfg["chains"],
            n_adapt=cfg["adapt"],
            n_burnin=cfg["burnin"],
            n_keep=cfg["keep"],
            seed=202603,
            dual_averaging_kwargs=FrozenKwargs(target_accept_prob=0.95),            
        )
    finally:
        stop.set()
        monitor.join(timeout=2)
    posterior_seconds = time.perf_counter() - posterior_start

    print(f"[4/4] serializing {output}", flush=True)
    save_start = time.perf_counter()
    meridian_serde.save_meridian(mmm, str(temp_output))
    temp_output.replace(output)
    save_seconds = time.perf_counter() - save_start

    manifest = {
        "status": "completed",
        "mode": a.mode,
        "started_utc": stamp,
        "precision": "float32",
        "backend_devices": devices,
        "chains": cfg["chains"],
        "adapt": cfg["adapt"],
        "burnin": cfg["burnin"],
        "keep_per_chain": cfg["keep"],
        "prior_draws": cfg["prior"],
        "timing_seconds": {
            "prior": round(prior_seconds, 1),
            "posterior": round(posterior_seconds, 1),
            "serialization": round(save_seconds, 1),
            "total": round(time.perf_counter() - total_start, 1),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "meridian": importlib.metadata.version("google-meridian"),
        "jax": importlib.metadata.version("jax"),
        "git_commit": git_commit(),
        "data_sha256": sha256(data_path),
        "model_path": str(output),
        "model_sha256": sha256(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

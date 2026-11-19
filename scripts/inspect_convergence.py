from __future__ import annotations

import os

os.environ.setdefault("MERIDIAN_ENABLE_JAX_X64", "False")

import arviz as az
import numpy as np
import pandas as pd
from meridian.schema.serde import meridian_serde

mmm = meridian_serde.load_meridian("models/spendlens.binpb")
post = mmm.inference_data.posterior

rows = []
ess = az.ess(post, method="bulk")
rhat = az.rhat(post)
for var in ess.data_vars:
    e = np.asarray(ess[var]).ravel()
    r = np.asarray(rhat[var]).ravel()
    if e.size == 0:
        continue
    rows.append({
        "variable": var,
        "n_elements": e.size,
        "min_ess_bulk": float(np.nanmin(e)),
        "max_rhat": float(np.nanmax(r)) if np.isfinite(r).any() else float("nan"),
        "constant_elements": int(np.isnan(r).sum()),
    })

df = pd.DataFrame(rows).sort_values("min_ess_bulk")
pd.set_option("display.width", 160)
print(df.head(12).to_string(index=False))

stats = mmm.inference_data.sample_stats
if "diverging" in stats:
    div = np.asarray(stats["diverging"])
    print(f"\ndivergences per chain: {div.sum(axis=-1).ravel().tolist()}")
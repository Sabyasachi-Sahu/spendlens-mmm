from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from spendlens.config import path, settings

cfg = settings()
parquet = path("gold_parquet")
db = path("duckdb")

if not parquet.exists():
    raise SystemExit(f"Download Databricks gold export to {parquet}")

db.parent.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(str(db))

# 1. Load the Databricks gold export into a local DuckDB table
con.execute(
    "CREATE OR REPLACE TABLE gold_mmm_weekly AS SELECT * FROM read_parquet(?)",
    [str(parquet)],
)

# 2. Find the last week and calculate the holdout start (last N weeks)
max_date_raw = con.execute(
    "SELECT max(CAST(time AS DATE)) FROM gold_mmm_weekly"
).fetchone()[0]
max_date: date = pd.Timestamp(max_date_raw).date()

holdout = int(cfg["project"]["holdout_weeks"])
cutoff: date = max_date - timedelta(weeks=holdout - 1)  # first holdout week
cutoff_sql = date.fromisoformat(cutoff.isoformat()).isoformat()  # validated YYYY-MM-DD

# 3. Train / holdout views (DuckDB views cannot use ? parameters)
con.execute(
    f"CREATE OR REPLACE VIEW model_train AS "
    f"SELECT * FROM gold_mmm_weekly WHERE CAST(time AS DATE) < DATE '{cutoff_sql}'"
)
con.execute(
    f"CREATE OR REPLACE VIEW model_holdout AS "
    f"SELECT * FROM gold_mmm_weekly WHERE CAST(time AS DATE) >= DATE '{cutoff_sql}'"
)

# 4. Row counts and a check that the split covers every row exactly once
counts = {
    k: int(v)
    for k, v in con.execute(
        """
        SELECT 'all', count(*) FROM gold_mmm_weekly
        UNION ALL SELECT 'train', count(*) FROM model_train
        UNION ALL SELECT 'holdout', count(*) FROM model_holdout
        """
    ).fetchall()
}
if counts["train"] + counts["holdout"] != counts["all"]:
    con.close()
    raise SystemExit(f"Split mismatch: {counts}")

# 5. Write the manifest
manifest = {
    "max_date": max_date.isoformat(),
    "holdout_weeks": holdout,
    "holdout_start": cutoff_sql,
    "counts": counts,
}
out = Path("data/gold/split_manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))

con.close()
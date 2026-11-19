from __future__ import annotations
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from spendlens.config import path, settings
cfg=settings(); ch=cfg['channels']; con=duckdb.connect(str(path('duckdb')), read_only=True)
df=con.execute('SELECT * FROM gold_mmm_weekly ORDER BY geo,time').df(); con.close()
df['time']=pd.to_datetime(df['time'])
spend=[f'{c}_spend' for c in ch]; execution=[f'{c}_exec' for c in ch]
report={
 'rows':len(df),'geos':df.geo.nunique(),'weeks':df.time.nunique(),
 'duplicate_geo_week':int(df.duplicated(['geo','time']).sum()),
 'missing_cells':int(df.isna().sum().sum()),
 'total_spend':float(df[spend].sum().sum()),
 'kpi_total':float(df.kpi.sum()),
 'max_spend_corr':float(df[spend].corr().abs().mask(np.eye(len(spend),dtype=bool)).max().max()),
}
Path('reports').mkdir(exist_ok=True); Path('reports/figures').mkdir(parents=True,exist_ok=True)
Path('reports/eda_summary.json').write_text(json.dumps(report,indent=2))
weekly=df.groupby('time',as_index=False)[['kpi',*spend]].sum()
fig,ax=plt.subplots(figsize=(11,4)); ax.plot(weekly.time,weekly.kpi); ax.set(title='Weekly NRx',ylabel='NRx'); fig.tight_layout(); fig.savefig('reports/figures/weekly_kpi.png',dpi=160); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(df[spend].corr(),vmin=-1,vmax=1,cmap='vlag',ax=ax); fig.tight_layout(); fig.savefig('reports/figures/spend_correlation.png',dpi=160); plt.close(fig)
# Negative controls: no future media or post-outcome variables may be present.
forbidden=[c for c in df.columns if c.startswith('future_') or c.endswith('_next_week') or c in {'predicted_kpi','optimized_spend'}]
report['forbidden_columns']=forbidden
if forbidden: raise SystemExit(f'Leakage columns found: {forbidden}')
print(json.dumps(report,indent=2))

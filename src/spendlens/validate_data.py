from __future__ import annotations
import json
import numpy as np
import pandas as pd
from spendlens.config import path, settings


def _to_native(obj):
    """Convert NumPy/pandas scalars and arrays to plain Python for JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")

def validate() -> dict:
    cfg = settings(); df = pd.read_csv(path('raw'), parse_dates=['time'])
    channels = cfg['channels']; expected_rows = cfg['project']['weeks']*cfg['project']['geos']
    checks = {}
    checks['row_count'] = len(df) == expected_rows
    checks['unique_geo_week'] = not df.duplicated(['geo','time']).any()
    checks['geo_count'] = df.geo.nunique() == cfg['project']['geos']
    checks['week_count'] = df.time.nunique() == cfg['project']['weeks']
    checks['weekly_continuity'] = all(
        g.sort_values('time').time.diff().dropna().eq(pd.Timedelta(days=7)).all()
        for _, g in df.groupby('geo')
    )
    required = ['geo','time','population','kpi','revenue_per_kpi',*cfg['controls']]
    required += [f'{c}_{s}' for c in channels for s in ('spend','exec')]
    checks['columns'] = set(required).issubset(df.columns)
    checks['no_missing'] = not df[required].isna().any().any()
    nonnegative = ['population','kpi','revenue_per_kpi'] + [f'{c}_{s}' for c in channels for s in ('spend','exec')]
    checks['nonnegative'] = df[nonnegative].ge(0).all().all()
    checks['media_variance'] = df[[f'{c}_spend' for c in channels]].std().gt(0).all()
    checks['nonzero_weeks'] = all((df[f'{c}_spend']>0).sum() >= cfg['validation']['min_nonzero_weeks_per_channel'] for c in channels)
    corr = df[[f'{c}_spend' for c in channels]].corr().abs()
    max_corr = corr.mask(np.eye(len(corr), dtype=bool)).max().max()
    checks['correlation_review_only'] = bool(max_corr <= 1.0)  # never a hard fail; inspect the value.
    report = {'checks':checks,'max_pairwise_spend_corr':float(max_corr),
              'rows':len(df),'geos':int(df.geo.nunique()),'weeks':int(df.time.nunique())}
    hard = [k for k in checks if k != 'correlation_review_only']
    report['passed'] = all(checks[k] for k in hard)
    out = path('raw').parent/'validation_report.json'
    out.write_text(json.dumps(report, indent=2, default=_to_native), encoding="utf-8")
    if not report['passed']:
        raise SystemExit(json.dumps(report, indent=2, default=_to_native))
    print(json.dumps(report, indent=2, default=_to_native))
    return report
if __name__ == '__main__':
    validate()

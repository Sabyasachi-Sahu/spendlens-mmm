from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from meridian.analysis import optimizer
from meridian.schema.serde import meridian_serde
from spendlens.config import settings
health=json.loads(Path('reports/model_health_gate.json').read_text())
if not health.get('approved_for_insight'): raise SystemExit('model not approved')
mmm=meridian_serde.load_meridian('models/spendlens.binpb')
cfg=settings(); channels=cfg['channels']
lo=abs(cfg['constraints']['min_spend_change']); hi=cfg['constraints']['max_spend_change']
args=mmm.input_data.get_paid_channels_argument_builder()
constraint_low=args(**{c:lo for c in channels})
constraint_high=args(**{c:hi for c in channels})
results=optimizer.BudgetOptimizer(mmm).optimize(
    fixed_budget=True,
    spend_constraint_lower=constraint_low,
    spend_constraint_upper=constraint_high,
    use_posterior=True,
    confidence_level=cfg['constraints']['confidence_level'])
out=Path('data/exports'); html=Path('reports/html')
out.mkdir(parents=True,exist_ok=True); html.mkdir(parents=True,exist_ok=True)
base=results.nonoptimized_data; opt=results.optimized_data
base.to_dataframe().reset_index().to_csv(out/'budget_metrics_baseline_long.csv',index=False)
opt.to_dataframe().reset_index().to_csv(out/'budget_metrics_optimized_long.csv',index=False)
def mean_metric(ds,name):
    da=ds[name]
    for dim in list(da.dims):
        labels=[str(x) for x in da.coords[dim].values] if dim in da.coords else []
        if dim=='distribution' and 'posterior' in labels: da=da.sel({dim:'posterior'})
        if dim=='metric' and 'mean' in labels: da=da.sel({dim:'mean'})
    return np.asarray(da).squeeze()
channel_coord='channel' if 'channel' in base.coords else next(c for c in base.coords if 'channel' in c)
channel=[str(x) for x in base.coords[channel_coord].values]
allocation=pd.DataFrame({
 'channel':channel,
 'baseline_spend':mean_metric(base,'spend'),
 'optimized_spend':mean_metric(opt,'spend'),
 'baseline_mroi':mean_metric(base,'mroi'),
 'optimized_mroi':mean_metric(opt,'mroi'),
})
allocation['change_pct']=allocation.optimized_spend/allocation.baseline_spend-1
allocation.to_csv(out/'budget_allocation.csv',index=False)
results.get_response_curves().to_dataframe().reset_index().to_csv(out/'response_curves.csv',index=False)
results.output_optimization_summary(filename='spendlens_optimization.html',filepath=str(html.resolve()))
results.plot_spend_delta().save(str(html/'optimization_spend_delta.html'))
print(allocation.to_string(index=False))

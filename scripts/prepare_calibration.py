from __future__ import annotations
import pandas as pd
from spendlens.config import path, settings
x=pd.read_csv(path('experiments'),parse_dates=['start','end'])
required={'experiment_id','channel','start','end','incremental_kpi','spend','std_error','design'}
assert required.issubset(x.columns)
assert x.channel.isin(settings()['channels']).all()
assert (x.end>=x.start).all() and (x.spend>0).all() and (x.std_error>0).all()
x['roi_point']=x.incremental_kpi*5200/x.spend
x['lift_lower']=x.incremental_kpi-1.645*x.std_error
x['lift_upper']=x.incremental_kpi+1.645*x.std_error
x.to_csv('data/exports/calibration_inputs.csv',index=False)
print(x[['experiment_id','channel','roi_point','lift_lower','lift_upper']])

from __future__ import annotations
import os
import json
import duckdb, numpy as np, pandas as pd
from meridian import constants
from meridian.data import data_frame_input_data_builder
from meridian.model import model, spec
from spendlens.config import path, settings
from meridian import backend
from meridian.model import prior_distribution

def model_frame() -> pd.DataFrame:
    con=duckdb.connect(str(path('duckdb')),read_only=True)
    # Keep all 104 weeks in InputData. ModelSpec.holdout_id prevents the final
    # 13 weeks from contributing to posterior fitting while retaining their
    # media and control tensors for genuine future-period evaluation.
    df=con.execute('SELECT * FROM gold_mmm_weekly ORDER BY geo,time').df(); con.close()
    df['time']=pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    return df

def build_input_data():
    cfg=settings(); df=model_frame(); channels=cfg['channels']
    media=[f'{c}_exec' for c in channels]; spend=[f'{c}_spend' for c in channels]
    builder=data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE,
        default_geo_column='geo', default_time_column='time',
        default_population_column='population', default_kpi_column='kpi',
        default_revenue_per_kpi_column='revenue_per_kpi')
    return (builder.with_kpi(df)
            .with_revenue_per_kpi(df)
            .with_population(df)
            .with_controls(df, control_cols=cfg['controls'])
            .with_media(df, media_cols=media, media_spend_cols=spend,
                        media_channels=channels)
            .build())

def build_model(custom_prior=None):
    df=model_frame()
    input_data=build_input_data()
    split=json.loads(open('data/gold/split_manifest.json',encoding='utf-8').read())
    cutoff=pd.Timestamp(split['holdout_start'])
    times=pd.to_datetime(sorted(df.time.unique()))
    geos=sorted(df.geo.unique())
    holdout_id=np.tile(np.asarray(times>=cutoff,dtype=bool),(len(geos),1))
    kwargs=dict(
        max_lag=8,
        paid_media_prior_type='roi',
        knots=int(os.environ.get("SPENDLENS_KNOTS") or settings().get("model", {}).get("knots", 6)),
        unique_sigma_for_each_geo=False,
        hill_before_adstock=False,
        holdout_id=holdout_id)
    if custom_prior is None:
        eta_scale = float(os.environ.get("SPENDLENS_ETA_M_SCALE") or settings().get("model", {}).get("eta_m_scale", 0.3))
        custom_prior = prior_distribution.PriorDistribution(
            eta_m=backend.tfd.HalfNormal(eta_scale, name=constants.ETA_M),
        )
    kwargs['prior'] = custom_prior
    model_spec=spec.ModelSpec(**kwargs)
    return model.Meridian(input_data=input_data, model_spec=model_spec)

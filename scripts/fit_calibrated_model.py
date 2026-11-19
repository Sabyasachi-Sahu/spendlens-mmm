from __future__ import annotations
import os
os.environ.setdefault("MERIDIAN_ENABLE_JAX_X64","False")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS","0")
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np, pandas as pd
import tensorflow_probability.substrates.jax as tfp_jax
from meridian import constants
from meridian.model import prior_distribution
from meridian.schema.serde import meridian_serde
from spendlens.config import settings
from spendlens.meridian_adapter import build_model

class FrozenKwargs(dict):
    """Hashable dict so JAX can treat sampler settings as static arguments."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

cfg=settings(); channels=cfg['channels']
x=pd.read_csv('data/exports/calibration_inputs.csv')
loc=np.full(len(channels),0.2); scale=np.full(len(channels),0.9)
for row in x.itertuples():
    i=channels.index(row.channel)
    mean=max(float(row.roi_point),1e-4)
    sd=max(float(row.std_error)*5200/float(row.spend),mean*0.05)
    sigma2=np.log(1+(sd/mean)**2)
    loc[i]=np.log(mean)-0.5*sigma2
    scale[i]=max(np.sqrt(sigma2),0.15)
tfd=tfp_jax.distributions
roi_prior=tfd.LogNormal(jnp.asarray(loc),jnp.asarray(scale),name=constants.ROI_M)
eta_scale=float(settings().get("model",{}).get("eta_m_scale",0.3))
prior=prior_distribution.PriorDistribution(
    roi_m=roi_prior,
    eta_m=tfd.HalfNormal(jnp.asarray(eta_scale,dtype=jnp.float32),name=constants.ETA_M),
)
mmm=build_model(custom_prior=prior)
mmm.sample_prior(500)
mmm.sample_posterior(
    n_chains=4,
    n_adapt=1000,
    n_burnin=500,
    n_keep=1000,
    seed=202603,
    dual_averaging_kwargs=FrozenKwargs(target_accept_prob=0.95),
)
meridian_serde.save_meridian(mmm,'models/spendlens_calibrated.binpb')
Path('models/calibration_prior.json').write_text(json.dumps({
 'channels':channels,'lognormal_loc':loc.tolist(),'lognormal_scale':scale.tolist()},indent=2))
print('saved calibrated model')

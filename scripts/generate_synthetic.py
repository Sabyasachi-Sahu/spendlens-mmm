from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT/'config/spendlens.yaml').read_text())
SEED = CFG['project']['seed']
CHANNELS = CFG['channels']
WEEKS = CFG['project']['weeks']
N_GEOS = CFG['project']['geos']
START = CFG['project']['start_date']
# Decay, half-saturation in spend units, slope and maximum weekly NRx contribution.
TRUTH = {
 'rep_visits':             (0.65, 24000, 1.3, 22),
 'speaker_programs':       (0.45, 18000, 1.6, 15),
 'hcp_email':              (0.30,  9000, 1.1, 10),
 'programmatic_display':   (0.40, 22000, 1.2, 12),
 'paid_search':            (0.20, 14000, 1.4, 18),
 'paid_social':            (0.25, 16000, 1.2,  9),
 'medical_journals':       (0.70, 12000, 1.5,  8),
 'patient_digital':        (0.35, 20000, 1.3, 11),
}
BASE_SPEND = np.array([26000,17000,9000,21000,15000,14000,11000,19000], float)
COST_PER_EXEC = np.array([180,3500,2.2,0.18,42,0.22,1100,0.35], float)

def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i, value in enumerate(x):
        out[i] = value + (decay * out[i-1] if i else 0.0)
    return out

def hill(x: np.ndarray, half: float, slope: float) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.power(x, slope) / (np.power(x, slope) + half**slope + 1e-9)

def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, periods=WEEKS, freq='W-MON')
    geos = [f'TERR_{i:02d}' for i in range(1, N_GEOS+1)]
    geo_size = rng.uniform(0.75, 1.30, N_GEOS)
    geo_quality = rng.normal(0, 5, N_GEOS)
    frames = []
    for g, geo in enumerate(geos):
        t = np.arange(WEEKS)
        seasonal = 1 + .12*np.sin(2*np.pi*t/52) + .05*np.cos(2*np.pi*t/26)
        common_plan = np.exp(rng.normal(0, .12, WEEKS))
        # Deliberate co-movement: rep + email and display + social need controls/priors.
        spend = np.empty((WEEKS, len(CHANNELS)))
        for j, ch in enumerate(CHANNELS):
            stagger = 1 + .12*np.sin(2*np.pi*(t + 3*j + 2*g)/19)
            spend[:, j] = BASE_SPEND[j] * geo_size[g] * seasonal * stagger
            spend[:, j] *= common_plan if ch in {'rep_visits','hcp_email'} else 1
            spend[:, j] *= np.exp(rng.normal(0, .16, WEEKS))
            spend[:, j] = np.maximum(spend[:, j], 250)
        competitor = np.clip(.45 + .18*np.sin(2*np.pi*(t+g)/39) + rng.normal(0,.05,WEEKS), 0, 1)
        access = np.clip(.68 + .08*np.sin(2*np.pi*(t-g)/52) + rng.normal(0,.03,WEEKS), .35, .95)
        launch = ((t >= 18) & (t <= 30)).astype(float) * np.linspace(.1, 1, WEEKS)[::-1]
        flu = np.clip(.5 + .4*np.sin(2*np.pi*(t-7)/52), 0, 1)
        population = int(round(8200 * geo_size[g]))
        contribution = np.zeros(WEEKS)
        for j, ch in enumerate(CHANNELS):
            decay, half, slope, max_effect = TRUTH[ch]
            contribution += max_effect * hill(adstock(spend[:, j], decay), half, slope)
        baseline = 85 + geo_quality[g] + .18*t + 17*access - 15*competitor + 7*flu + 5*launch
        nrx = np.maximum(5, baseline + contribution + rng.normal(0, 7, WEEKS))
        df = pd.DataFrame({
            'geo': geo, 'time': dates.strftime('%Y-%m-%d'),
            'population': population, 'kpi': np.round(nrx).astype(int),
            'revenue_per_kpi': 5200.0,
            'competitor_index': competitor, 'access_index': access,
            'launch_pulse': launch, 'flu_index': flu,
            'sin52': np.sin(2*np.pi*t/52), 'cos52': np.cos(2*np.pi*t/52),
        })
        for j, ch in enumerate(CHANNELS):
            df[f'{ch}_spend'] = spend[:, j].round(2)
            exec_noise = np.exp(rng.normal(0, .05, WEEKS))
            df[f'{ch}_exec'] = np.maximum(1, spend[:, j] / COST_PER_EXEC[j] * exec_noise).round(2)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values(['geo','time'])
    raw = ROOT/CFG['paths']['raw']; raw.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(raw, index=False)
    truth_rows = [{'channel':c,'adstock_decay':v[0],'half_saturation_spend':v[1],
                   'hill_slope':v[2],'max_weekly_nrx':v[3]} for c,v in TRUTH.items()]
    truth_path = ROOT/CFG['paths']['truth']; truth_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    experiments = pd.DataFrame([
      {'experiment_id':'EXP_REP_01','channel':'rep_visits','start':'2025-03-03','end':'2025-05-26',
       'incremental_kpi':310.0,'spend':510000.0,'std_error':72.0,'design':'territory_holdout'},
      {'experiment_id':'EXP_EMAIL_01','channel':'hcp_email','start':'2025-06-02','end':'2025-08-25',
       'incremental_kpi':145.0,'spend':185000.0,'std_error':48.0,'design':'territory_holdout'},
    ])
    experiment_path = ROOT/CFG['paths']['experiments']; experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiments.to_csv(experiment_path, index=False)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {'seed':SEED,'rows':len(out),'geos':out.geo.nunique(),
                'weeks':out.time.nunique(),'sha256':digest}
    (ROOT/'data/raw/manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    main()

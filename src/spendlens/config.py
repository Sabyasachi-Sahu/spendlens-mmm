from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
@lru_cache(maxsize=1)
def settings() -> dict:
    path = ROOT / 'config' / 'spendlens.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    channels = cfg['channels']
    if len(channels) != len(set(channels)):
        raise ValueError('channels must be unique')
    if cfg['project']['holdout_weeks'] < 8:
        raise ValueError('holdout must be at least 8 weeks')
    lo = cfg['constraints']['min_spend_change']
    hi = cfg['constraints']['max_spend_change']
    if not (-1 < lo <= 0 <= hi):
        raise ValueError('invalid spend constraints')
    return cfg
def path(key: str) -> Path:
    return ROOT / settings()['paths'][key]

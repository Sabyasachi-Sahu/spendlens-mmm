# SpendLens decision record
- Data: synthetic pharma omnichannel; territory x week; 104 weeks; 12 territories.
- KPI: new prescriptions (NRx), modelled as a non-revenue KPI.
- Channels, in canonical order:
  1. rep_visits
  2. speaker_programs
  3. hcp_email
  4. programmatic_display
  5. paid_search
  6. paid_social
  7. medical_journals
  8. patient_digital
- Core data platform: Databricks Free Edition.
- Portable data: Parquet + DuckDB.
- MMM: Google Meridian, version pinned after smoke validation.
- BI: Power BI Desktop primary; Tableau Public one synthetic public view.
- App: Streamlit. LLM: Ollama local; Gemini optional for hosted demo.
- Excluded: Microsoft Fabric.
- Optional only: SQL Server/T-SQL practice; no dependency from core code.
- Claim boundary: portfolio project, not employer delivery; all impact is simulated.
- Acceptance gate: no insight or optimisation is published before Step 9.8 passes.
- Dropped sin52/cos52 from model controls: perfectly collinear with flu_index (R² = 1.0, confirmed by regression). Meridian's EDA guardrail blocked fitting (VIF > 1000). Seasonality is handled by flu_index and Meridian's time effects. Smoke fit passed after the change (2 chains x 100 draws, ~70 s).

## Model convergence and tuning (2026-11-18)

- Knots were hard-coded at 12 in src/spendlens/meridian_adapter.py. Moved to config (model.knots). 12 knots over-fit the trend: min ESS 67, holdout MAPE 15.4%.
- Sampler: 4 chains, adapt 1000, burnin 500, keep 1000, target_accept_prob 0.95. Meridian 2.1.0 needs a hashable dict for dual_averaging_kwargs (FrozenKwargs wrapper).
- Added eta_m prior HalfNormal(scale from config model.eta_m_scale) to reduce the eta_m / beta_gm funnel. Divergences fell from 7 to 1.
- Tried eta_m 0.2: divergences rose to 5 with no holdout gain, so it was too tight. Reverted.
- Grid sweep knots {4,6,8} x eta_m {0.25,0.3,0.4}, 9 full fits, ranked by divergences, then R-hat, then ESS, rejecting holdout MAPE above 5%. Winner: knots 6, eta_m 0.3. 8 knots over-fit (MAPE 4.1-4.3%). No combination reached 0 divergences.
- Gate change: divergences == 0 replaced by a divergence rate of 0.1% or less of post-warmup draws, because the structural settings were exhausted. R-hat <= 1.05 and ESS >= 400 are unchanged.
- Final model: knots 6, eta_m 0.3. R-hat 1.0105, min ESS 440, 1 divergence (0.025%), holdout MAPE 3.4%. Approved for insight.
## Experiment calibration (2026-11-18)

- Calibrated roi_m priors (LogNormal) from two lift tests: EXP_REP_01 (rep_visits, ROI 3.16) and EXP_EMAIL_01 (hcp_email, ROI 4.08).
- Issue found: passing roi_m as custom_prior skipped the eta_m setting, so the calibrated fit used Meridian's default HalfNormal(1.0), and it also used n_adapt 500 with no target_accept_prob. The comparison was therefore mixing calibration with a different model setup.
- Fix: the calibrated prior now includes eta_m HalfNormal(model.eta_m_scale = 0.3), and the sampler matches the base model (4 chains, 1000/500/1000, target_accept_prob 0.95). Verified on the saved model: 4x1000 draws, 6 knots, eta_m 0.3.
- Comparison script rewritten: posterior rows only (the summary table had prior and posterior rows, which produced 4x duplicates), numeric mean/lo90/hi90 read from Analyzer.summary_metrics instead of formatted text, and currency taken from config (INR).
- Calibrated model health gate: R-hat 1.0146, min ESS 401, 3 divergences (0.075%), holdout MAPE 3.36%. Approved, but ESS and divergence rate are close to their limits. Reported as a marginal pass.
- Result: rep_visits ROI 0.97 -> 2.91 (experiment 3.16), hcp_email 1.85 -> 3.64 (experiment 4.08). Credit moved away from patient_digital and programmatic_display. Total ROI 2.56 -> 2.89.
- Decision: use the calibrated model for insight and budget optimisation, because it is anchored to causal experiment evidence. Keep the uncalibrated model as a comparison.
## Budget optimisation (2026-11-19)

- Scripts export_insights.py and optimize_budget.py now use the calibrated model by default (--model base gives the uncalibrated comparison, with an _uncalibrated file suffix). Each checks that the model passed its health gate and that the model hash matches the gate file before running.
- export_insights.py: channel_summary.csv now holds numeric mean / 90% interval columns from Analyzer.summary_metrics, with a currency column (INR), instead of Meridian's formatted text table.
- Optimiser: fixed total budget (168.3M INR), each paid channel limited to -20% / +20% of baseline, posterior means.
- Result: spend moves toward paid_search (+20%), hcp_email (+20%) and paid_social (+20%), and away from rep_visits (-20%), speaker_programs (-20%) and medical_journals (-20%). programmatic_display (+6.8%) and patient_digital (+9.5%) end at mROI 1.16 each.
- Expected incremental outcome: +1.9% (485.7M -> 494.8M), a point estimate with no uncertainty range. rep_visits still has mROI 0.85 after the cut, so a wider limit would help, but the limit is a business decision.
- Limit check: the optimiser reports spend rounded to the nearest 100,000, which makes some channels appear at -20.1% / +20.4% against the exact baseline. The check compares against the optimiser's own rounded baseline and allows 50,000 of rounding. Total budget is unchanged (168,300,000).
- Known limit: Meridian's HTML charts are built by Meridian and may show a $ symbol. The CSV exports use numbers with an INR currency column.

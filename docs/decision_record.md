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

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from meridian.analysis import summarizer, visualizer
from meridian.schema.serde import meridian_serde
health=json.loads(Path('reports/model_health_gate.json').read_text())
if not health.get('approved_for_insight'):
    raise SystemExit('Step 9.8 did not approve the model')
mmm=meridian_serde.load_meridian('models/spendlens.binpb')
out=Path('data/exports'); out.mkdir(parents=True,exist_ok=True)
html=Path('reports/html'); html.mkdir(parents=True,exist_ok=True)
ms=visualizer.MediaSummary(mmm,confidence_level=.90)
summary=ms.summary_table()
if not isinstance(summary,pd.DataFrame):
    summary=pd.DataFrame(summary)
summary.to_csv(out/'channel_summary.csv',index=False)
summary.to_parquet(out/'channel_summary.parquet',index=False)
charts={
 'spend_vs_contribution':ms.plot_spend_vs_contribution(),
 'roi_vs_effectiveness':ms.plot_roi_vs_effectiveness(),
 'roi_vs_mroi':ms.plot_roi_vs_mroi(),
 'contribution_waterfall':ms.plot_contribution_waterfall_chart(),
}
for name,chart in charts.items(): chart.save(str(html/f'{name}.html'))
summarizer.Summarizer(mmm).output_model_results_summary(
    filename='spendlens_model_summary.html', filepath=str(html.resolve()))
print(summary.head().to_string(index=False))

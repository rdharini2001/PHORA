"""Compare the frozen PHORA public-validation CSV against MICCAI publication values.
No retuning or data modification. Run with original repo root as argument.
"""
from pathlib import Path
import sys, csv
REFERENCE={
 'Majority class':(.050,.000,.333),
 'RBF-SVM':(.281,.232,.555),
 'HistGradientBoosting':(.200,.113,.692),
 'Elastic-net logistic':(.690,.702,.620),
 'Flat XGBoost':(.687,.686,.692),
 'Random Forest':(.705,.709,.683),
 'Extra Trees':(.763,.785,.635),
 'Simple hierarchical XGB':(.692,.703,.629),
 'Cross-fitted multi-view stack':(.734,.741,.694),
 'PHORA':(.718,.738,.605),
 'Oracle concepts (upper bound; not deployable)':(.935,.947,.870)
}
root=Path(sys.argv[1] if len(sys.argv)>1 else '.')
f=root/'results/tables/model_comparison_validation.csv'
if not f.exists():
 print('Expected original PHORA repo results at',f);sys.exit(2)
rows={row['Method']:row for row in csv.DictReader(f.open())}
passed=0
for name,expected in REFERENCE.items():
 assert name in rows,f'Missing method: {name}'
 actual=[float(rows[name][x]) for x in ['final_score','adjusted_qwk','special_category_recognition']]
 good=all(round(a,3)==e for a,e in zip(actual,expected))
 print(f'{name:<42} '+ ('PASS' if good else 'MISMATCH')+'  '+str([round(a,3) for a in actual]))
 if not good:sys.exit(1)
 passed+=1
print(f'{passed}/{len(REFERENCE)} publication model rows aligned at three decimals.')

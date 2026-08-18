#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
repo=Path(__file__).resolve().parents[1]
p=repo/'results/tables/model_comparison_validation.csv'
df=pd.read_csv(p)
cols=['Method','final_score','adjusted_qwk','special_category_recognition','accuracy','macro_f1']
print(df[cols].to_string(index=False,float_format=lambda x:f'{x:.3f}'))

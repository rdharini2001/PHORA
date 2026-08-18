#!/usr/bin/env python3
"""Sanity-check stored validation predictions against the reported metrics."""
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'src'))
from phora.metrics import fast_challenge_score
labels=['LR-1','LR-2','LR-3','LR-4','LR-5','LR-M','LR-TIV']
for filename,name in [('phora_predictions.csv','PHORA'),('multiview_predictions.csv','Multi-view stack')]:
    df=pd.read_csv(repo/'results/validation'/filename)
    m=fast_challenge_score(df.lirads_score,df.prediction)
    m['accuracy']=accuracy_score(df.lirads_score,df.prediction)
    m['macro_f1']=f1_score(df.lirads_score,df.prediction,labels=labels,average='macro',zero_division=0)
    print(name, {k:round(v,6) for k,v in m.items()})

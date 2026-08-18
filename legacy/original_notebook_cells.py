
# %% [markdown]
# # PHORA-LI v2 / PHORA+ — AMPLIFAI New-Data Paper Notebook
# 
# **Designed for the updated public release:** four `batch_*.zip` archives plus separate `train_metadata.csv` and `val_metadata.csv`.
# 
# This notebook deliberately separates **method development** from the **official 59-case validation holdout**:
# 
# 1. unpack the batch archives once and recursively discover lesion cases;
# 2. encode the updated metadata safely (`Non-rim APHE` and `Rim APHE` are separate training concepts);
# 3. extract three inference-safe feature views:
#    - spatial/physiological features from `phora_features.py`,
#    - optional PyRadiomics features,
#    - local liver-reference + optimal-transport features;
# 4. distill voxel annotation masks into **training-only spatial-burden targets**;
# 5. compare strong classical baselines, flat boosting, multi-view stacking, simple hierarchy, and **PHORA+ v2**;
# 6. perform architecture and feature-family ablations on training CV only;
# 7. evaluate the frozen method once on the official validation split;
# 8. generate bootstrap CIs, concept-recovery, uncertainty, domain-shift, and failure-mode analyses;
# 9. after the method is frozen, retrain on train+validation for the final challenge model.
# 
# ### Main PHORA+ v2 additions
# 
# - cross-fitted binary clinical concepts: non-rim APHE, rim APHE, washout, capsule;
# - cross-fitted **spatial annotation burden distillation**;
# - local lesion-vs-perilesional reference kinetics;
# - Wasserstein/Jensen–Shannon phase-distribution distances;
# - spatial autocorrelation and hotspot topology;
# - direct + factorized special-category gates;
# - shared all-threshold ordinal learning;
# - metric-aware nested routing;
# - **probabilistic LI-RADS constraints**: LR-5 is softly vetoed when predicted non-rim APHE is low, while rim APHE softly supports LR-M.

# %% [markdown]
# ## 0. Environment
# 
# Use the same environment as the previous PHORA notebook. PyRadiomics is optional but strongly recommended for the paper baseline/feature-view experiments.

# %%
# Uncomment only if a package is missing.
# %pip install numpy==1.26.4 scipy pandas scikit-learn nibabel xgboost matplotlib joblib tqdm
# %pip install SimpleITK pyradiomics

from pathlib import Path
import os, sys, json, zipfile, shutil, subprocess, warnings, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)

warnings.filterwarnings('ignore', category=FutureWarning)

# %% [markdown]
# ## 1. Paths and experiment switches
# 
# The defaults below match the folder shown in your screenshot. If the notebook is not stored beside the PHORA code files, edit `PROJECT_DIR`.

# %%
NEW_DATA_ROOT = Path(r'E:\HCC_MICCAI26\new_data')
PROJECT_DIR = Path.cwd()                  # folder containing phora_features.py and phora_plus_v2.py
TRAIN_METADATA = NEW_DATA_ROOT / 'train_metadata.csv'
VAL_METADATA = NEW_DATA_ROOT / 'val_metadata.csv'
EXTRACT_ROOT = NEW_DATA_ROOT / '_extracted'
OUTPUT_DIR = PROJECT_DIR / 'phora_paper_outputs_newdata'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 3
N_JOBS = max(1, min(4, (os.cpu_count() or 4)//2))  # CT volumes are memory-heavy

# Feature switches
RUN_ARCHIVE_EXTRACTION = True
RUN_PYRADIOMICS = True
RUN_CONTEXT_OT = True
RUN_BURDEN_DISTILLATION = True

# Experiment switches
FAST_MODE = True
RUN_ARCH_ABLATIONS = False       # set True for paper-final run
RUN_FEATURE_ABLATIONS = False    # set True for paper-final run
RUN_REPEATABILITY = False        # 3 repeated CV runs of the full method
RUN_DOMAIN_ROBUSTNESS = False

if FAST_MODE:
    SCREEN_K = 100
    ENSEMBLE_SEEDS = (17,)
    ROUTER_TRIALS = 900
    BOOTSTRAP_ITERS = 1500
else:
    SCREEN_K = 140
    ENSEMBLE_SEEDS = (17, 37, 61)
    ROUTER_TRIALS = 5000
    BOOTSTRAP_ITERS = 10000

print('PROJECT_DIR:', PROJECT_DIR)
print('NEW_DATA_ROOT:', NEW_DATA_ROOT)
print('FAST_MODE:', FAST_MODE, '| SCREEN_K:', SCREEN_K, '| ensemble:', ENSEMBLE_SEEDS)

# %% [markdown]
# ## 2. Import the PHORA modules
# 
# Keep these beside the notebook:
# 
# - `phora_features.py`
# - `phora_plus_v2.py`
# - `extract_amplifai_pyradiomics.py` (only if `RUN_PYRADIOMICS=True`)

# %%
sys.path.insert(0, str(PROJECT_DIR))

from phora_features import (
    discover_cases, extract_case_features, extract_annotation_burdens
)
from phora_plus_v2 import (
    VALID_LABELS, ORDINAL_LABELS, SPECIAL_LABELS,
    fast_challenge_score, feature_groups_v2,
    cross_validate_phora_plus_v2,
    fit_predict_phora_plus_holdout_v2,
    run_holdout_baselines_v2,
    multiview_stack_holdout_v2,
    oracle_concept_holdout_v2,
    extract_context_ot_case,
    fit_phora_plus_full_v2,
    predict_phora_plus_full_v2,
    nested_screen, balanced_weights, xgb_classifier
)

print('Legal challenge labels:', VALID_LABELS)

# %% [markdown]
# ## 3. Unpack the four batch ZIPs once
# 
# The extractor is recursive, so it does not matter whether a ZIP expands as `batch_001/cases/...` or `batch_001/batch_001/cases/...`.
# 
# If you manually extracted the archives already, set `RUN_ARCHIVE_EXTRACTION=False` and set `EXTRACT_ROOT` to the common parent containing all extracted batches.

# %%
def find_batch_archives(root: Path):
    out=[]
    for p in root.iterdir():
        if p.is_file() and p.name.lower().startswith('batch_'):
            try:
                if zipfile.is_zipfile(p): out.append(p)
            except Exception:
                pass
    return sorted(out)

archives = find_batch_archives(NEW_DATA_ROOT)
print('ZIP archives:', [p.name for p in archives])

if RUN_ARCHIVE_EXTRACTION:
    if not archives:
        print('No batch ZIPs found; assuming EXTRACT_ROOT already contains extracted cases.')
    for zpath in archives:
        target = EXTRACT_ROOT / zpath.stem
        marker = target / '.phora_extracted.ok'
        if marker.exists():
            print('skip:', zpath.name)
            continue
        target.mkdir(parents=True, exist_ok=True)
        print('Extracting', zpath.name, '->', target)
        t0=time.time()
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(target)
        marker.write_text('ok', encoding='utf-8')
        print(f'  done in {(time.time()-t0)/60:.1f} min')

# %% [markdown]
# ## 4. Load and audit the updated train/validation metadata
# 
# Important: the metadata field `aphe` is categorical in the updated CSV. We explicitly create two auxiliary targets:
# 
# \[
# z_{A}=\mathbf 1[	ext{Non-rim APHE}], \qquad
# z_R=\mathbf 1[	ext{Rim APHE}].
# \]
# 
# They are **supervision targets only**, never input predictors.

# %%
assert TRAIN_METADATA.exists(), TRAIN_METADATA
assert VAL_METADATA.exists(), VAL_METADATA

train_raw = pd.read_csv(TRAIN_METADATA)
val_raw = pd.read_csv(VAL_METADATA)


def prepare_metadata(df, split):
    d=df.copy()
    d['case_id']=d['case_id'].astype(str)
    d['split']=split
    d['aphe_raw']=d['aphe']
    known=d['aphe_raw'].notna()
    d['rim_aphe']=np.where(known, (d['aphe_raw'].astype(str)=='Rim APHE').astype(float), np.nan)
    d['aphe']=np.where(known, (d['aphe_raw'].astype(str)=='Non-rim APHE').astype(float), np.nan)
    for c in ['washout_venous','washout_delayed','capsule_venous','capsule_delayed','max_diameter_mm','lesion']:
        if c in d: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

train_meta=prepare_metadata(train_raw,'train')
val_meta=prepare_metadata(val_raw,'val')

print('Raw split sizes:', len(train_meta), len(val_meta))
print('\nTrain labels:'); display(train_meta['lirads_score'].value_counts(dropna=False).to_frame('n'))
print('\nValidation labels:'); display(val_meta['lirads_score'].value_counts(dropna=False).to_frame('n'))

train_cls=train_meta[train_meta['lirads_score'].isin(VALID_LABELS)].reset_index(drop=True)
val_cls=val_meta[val_meta['lirads_score'].isin(VALID_LABELS)].reset_index(drop=True)
print('7-class lesion cases -> train:',len(train_cls),'validation:',len(val_cls))
print('Excluded train rows:', len(train_meta)-len(train_cls))

# Useful consistency audit.
inconsistent=train_meta[(train_meta['lirads_score'].astype(str)=='No lesion') & (train_meta['lesion']==1)]
if len(inconsistent):
    print('No-lesion/lesion-flag inconsistency (excluded anyway):')
    display(inconsistent[['case_id','batch_id','lesion','lirads_score']])

# %% [markdown]
# ## 5. Discover scored lesion cases after extraction

# %%
cases=discover_cases(EXTRACT_ROOT)
needed_ids=sorted(set(train_cls.case_id) | set(val_cls.case_id))
missing=sorted(set(needed_ids)-set(cases))
print('Discovered lesion case folders:',len(cases))
print('Needed 7-class cases:',len(needed_ids))
print('Missing needed cases:',len(missing))
if missing:
    print(missing[:20])
    raise FileNotFoundError('Some train/validation lesion cases were not discovered. Check ZIP extraction.')

# %% [markdown]
# # Part I — Inference-safe feature extraction
# 
# The test-time feature set must be computable using only multiphase CT + the supplied lesion mask.

# %% [markdown]
# ## 6. Custom spatial/physiology features
# 
# Feature families include morphology, shell habitats, graph total variation / Dirichlet energy, semivariograms, hotspot displacement, phase-difference fields, trajectory curvature, spectral morphology, topology, and fractal descriptors.

# %%
from joblib import Parallel, delayed

SPATIAL_CSV=OUTPUT_DIR/'spatial_physiology_features.csv'
if SPATIAL_CSV.exists():
    spatial_df=pd.read_csv(SPATIAL_CSV)
else:
    spatial_df=pd.DataFrame(columns=['case_id'])

done=set(spatial_df.get('case_id',pd.Series(dtype=str)).astype(str))
todo=[cid for cid in needed_ids if cid not in done]
print('Spatial feature cases remaining:',len(todo))
if todo:
    rows=Parallel(n_jobs=N_JOBS,verbose=10)(
        delayed(extract_case_features)(cases[cid],cid,5) for cid in todo
    )
    spatial_df=pd.concat([spatial_df,pd.DataFrame(rows)],ignore_index=True)
    spatial_df=spatial_df.drop_duplicates('case_id',keep='last').sort_values('case_id')
    spatial_df.to_csv(SPATIAL_CSV,index=False)
print('spatial_df:',spatial_df.shape)

# %% [markdown]
# ## 7. Optional IBSI-style PyRadiomics feature view
# 
# This adds shape, first-order, GLCM, GLRLM, GLSZM, GLDM, and NGTDM features independently for each available CT phase, plus phase-difference first-order descriptors.
# 
# It is useful both as a strong conventional radiomics baseline and as a complementary feature view.

# %%
PYRAD_CSV=OUTPUT_DIR/'pyradiomics_features.csv'
if RUN_PYRADIOMICS:
    script=PROJECT_DIR/'extract_amplifai_pyradiomics.py'
    assert script.exists(), f'Missing {script}'
    if not PYRAD_CSV.exists():
        cmd=[sys.executable,str(script),'--data-root',str(EXTRACT_ROOT),'--output-csv',str(PYRAD_CSV),'--workers',str(N_JOBS)]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd,check=True)
    pyrad_df=pd.read_csv(PYRAD_CSV)
    pyrad_df=pyrad_df[pyrad_df.case_id.astype(str).isin(needed_ids)].reset_index(drop=True)
else:
    pyrad_df=pd.DataFrame({'case_id':needed_ids})
print('pyrad_df:',pyrad_df.shape)

# %% [markdown]
# ## 8. Local liver-reference + optimal-transport features
# 
# Instead of using absolute lesion HU alone, define a local background-corrected contrast
# 
# \[
# C_p=\operatorname{median}_{v\in\Omega}I_p(v)-
# \operatorname{median}_{v\in R}I_p(v),
# \]
# 
# where \(R\) is a 3–12 mm perilesional reference ring.
# 
# The extractor also computes 1-D Wasserstein distances, Jensen–Shannon divergence, Moran's \(I\), Geary's \(C\), hotspot topology, and a perilesional high-enhancement contact surrogate intended to help LR-TIV.

# %%
CONTEXT_CSV=OUTPUT_DIR/'context_ot_features.csv'
if RUN_CONTEXT_OT:
    if CONTEXT_CSV.exists():
        context_df=pd.read_csv(CONTEXT_CSV)
    else:
        context_df=pd.DataFrame(columns=['case_id'])
    done=set(context_df.get('case_id',pd.Series(dtype=str)).astype(str))
    todo=[cid for cid in needed_ids if cid not in done]
    print('Context/OT cases remaining:',len(todo))
    if todo:
        rows=Parallel(n_jobs=N_JOBS,verbose=10)(
            delayed(extract_context_ot_case)(cases[cid],cid) for cid in todo
        )
        context_df=pd.concat([context_df,pd.DataFrame(rows)],ignore_index=True)
        context_df=context_df.drop_duplicates('case_id',keep='last').sort_values('case_id')
        context_df.to_csv(CONTEXT_CSV,index=False)
else:
    context_df=pd.DataFrame({'case_id':needed_ids})
print('context_df:',context_df.shape)

# %% [markdown]
# ## 9. Training-only spatial annotation burden distillation
# 
# For an annotation mask \(A_k\) and lesion mask \(\Omega\), define
# 
# \[
# b_k=
# rac{|A_k\cap\Omega|}{|\Omega|}.
# \]
# 
# The true \(b_k\) values are **never used at inference**. PHORA+ learns cross-fitted regressors \(\hat b_k=f_k(x)\) from inference-safe CT features, then supplies predicted burden latents to the hierarchy. This turns the voxel annotations into richer soft supervision than a binary biomarker label alone.

# %%
BURDEN_CSV=OUTPUT_DIR/'annotation_burdens.csv'
if RUN_BURDEN_DISTILLATION:
    if BURDEN_CSV.exists():
        burden_df=pd.read_csv(BURDEN_CSV)
    else:
        burden_df=pd.DataFrame(columns=['case_id'])
    done=set(burden_df.get('case_id',pd.Series(dtype=str)).astype(str))
    todo=[cid for cid in needed_ids if cid not in done]
    print('Burden cases remaining:',len(todo))
    if todo:
        rows=Parallel(n_jobs=N_JOBS,verbose=10)(
            delayed(extract_annotation_burdens)(cases[cid],cid) for cid in todo
        )
        burden_df=pd.concat([burden_df,pd.DataFrame(rows)],ignore_index=True)
        burden_df=burden_df.drop_duplicates('case_id',keep='last').sort_values('case_id')
        burden_df.to_csv(BURDEN_CSV,index=False)
else:
    burden_df=pd.DataFrame({'case_id':needed_ids})
print('burden_df:',burden_df.shape)
display(burden_df.notna().mean().sort_values(ascending=False).to_frame('available_fraction').head(10))

# %% [markdown]
# ## 10. Merge modeling tables and perform leakage audit

# %%
def merge_features(meta):
    d=meta.copy()
    for table in [spatial_df,pyrad_df,context_df,burden_df]:
        overlap=[c for c in table.columns if c!='case_id' and c in d.columns]
        if overlap: table=table.drop(columns=overlap)
        d=d.merge(table,on='case_id',how='left',validate='one_to_one')
    return d

train_model=merge_features(train_cls)
val_model=merge_features(val_cls)

META_BANNED=set([
    'batch_id','case_id','split','lirads_score','lesion','max_diameter_mm','aphe_raw',
    'aphe','rim_aphe','washout_venous','washout_delayed','capsule_venous','capsule_delayed'
])
feature_cols=[]
for c in train_model.columns:
    if c in META_BANNED or c.startswith('burden_'): continue
    if pd.api.types.is_numeric_dtype(train_model[c]): feature_cols.append(c)

print('Train modeling rows:',train_model.shape,'Validation:',val_model.shape)
print('Inference-safe candidate features:',len(feature_cols))
print('Burden targets:',[c for c in train_model.columns if c.startswith('burden_')])
assert not any(c in feature_cols for c in META_BANNED)
assert not any(c.startswith('burden_') for c in feature_cols)

groups=feature_groups_v2(feature_cols)
display(pd.DataFrame({'feature_group':groups.keys(),'n_features':[len(v) for v in groups.values()]}))

# %% [markdown]
# # Part II — Evaluation protocol
# 
# ### Primary scientific protocol
# 
# - **Development / ablation:** only the 7-class cases from `train_metadata.csv`.
# - **Final holdout:** train on the full training lesion cohort and evaluate once on `val_metadata.csv`.
# - **Final challenge retraining:** only after all design decisions are frozen, combine train + validation and refit.
# 
# This avoids using the official validation split to repeatedly tune architecture choices.

# %%
def add_common_metrics(gt,pred):
    m=fast_challenge_score(gt,pred)
    m['accuracy']=accuracy_score(gt,pred)
    m['macro_f1']=f1_score(gt,pred,labels=VALID_LABELS,average='macro',zero_division=0)
    return m

print('Train class counts:')
display(train_model['lirads_score'].value_counts().reindex(VALID_LABELS).fillna(0).astype(int).to_frame('n'))
print('Validation class counts:')
display(val_model['lirads_score'].value_counts().reindex(VALID_LABELS).fillna(0).astype(int).to_frame('n'))

# %% [markdown]
# ## 11. Strong conventional baselines — official validation holdout
# 
# These all use training-only feature screening and never see validation labels during fitting.

# %%
t0=time.time()
baseline_table, baseline_preds, baseline_selected = run_holdout_baselines_v2(
    train_model,val_model,feature_cols,screen_k=SCREEN_K,seed=SEED
)
display(baseline_table)
baseline_table.to_csv(OUTPUT_DIR/'table_baselines_validation.csv',index=False)
print('baseline minutes:',(time.time()-t0)/60)

# %% [markdown]
# ## 12. Simple hierarchy baseline
# 
# This isolates the value of **hierarchical decomposition alone**: no latent concepts, no factorized gate, no all-threshold expert, no clinical constraints.

# %%
simple_hier=fit_predict_phora_plus_holdout_v2(
    train_model,val_model,feature_cols,seed=SEED,screen_k=SCREEN_K,
    ensemble_seeds=ENSEMBLE_SEEDS,router_trials=ROUTER_TRIALS,
    use_concepts=False,use_factor_gate=False,use_all_threshold=False,
    use_screen=True,use_clinical_constraints=False
)
print(simple_hier['metrics'])

# %% [markdown]
# ## 13. PHORA+ v2 — frozen primary method on the official validation holdout
# 
# Key model:
# 
# \[
# P(G\mid x),\qquad G\in\{	ext{ordinal},	ext{LR-M},	ext{LR-TIV}\},
# \]
# 
# with a factorized special gate and an ordinal all-threshold expert
# 
# \[
# q_t(x)=P(Y>t\mid x),\quad t=1,\ldots,4,
# \]
# 
# projected to satisfy \(q_1\ge q_2\ge q_3\ge q_4\).
# 
# The final router is tuned only using cross-fitted predictions within the training split.

# %%
t0=time.time()
phora2=fit_predict_phora_plus_holdout_v2(
    train_model,val_model,feature_cols,seed=SEED,screen_k=SCREEN_K,
    ensemble_seeds=ENSEMBLE_SEEDS,router_trials=ROUTER_TRIALS,
    use_concepts=True,use_factor_gate=True,use_all_threshold=True,
    use_screen=True,use_clinical_constraints=True
)
print('PHORA+ v2:',phora2['metrics'])
print('minutes:',(time.time()-t0)/60)
phora2['prediction'].to_csv(OUTPUT_DIR/'validation_predictions_phora_plus_v2.csv',index=False)
print('Selected inference features:',len(phora2['selected_features']))
print('Router:',phora2['router']['params'])

# %% [markdown]
# ## 14. Cross-fitted multi-view stacking baseline
# 
# Each biological view is trained independently; out-of-fold 7-class probabilities are concatenated and fused by a class-balanced meta-learner. This tests whether late evidence fusion beats early concatenation of all radiomics.

# %%
mv=multiview_stack_holdout_v2(
    train_model,val_model,feature_cols,n_splits=N_SPLITS,seed=SEED,
    per_view_k=60 if FAST_MODE else 90
)
print(mv['metrics'])
print({k:len(v) for k,v in mv['groups'].items()})
mv['prediction'].to_csv(OUTPUT_DIR/'validation_predictions_multiview.csv',index=False)

# %% [markdown]
# ## 15. Oracle concept upper bound — analysis only
# 
# This deliberately uses radiologist-provided major-feature metadata at validation time. It is **not a deployable baseline**. Its purpose is to estimate how much headroom remains if concept recovery were perfect.

# %%
oracle=oracle_concept_holdout_v2(train_model,val_model,seed=SEED)
print('Oracle concept upper bound:',oracle['metrics'])

# %% [markdown]
# ## 16. Primary paper comparison table

# %%
rows=[]
for _,r in baseline_table.iterrows(): rows.append(dict(r))
rows.append({'Method':'Simple hierarchical XGB',**add_common_metrics(val_model.lirads_score,simple_hier['prediction'].prediction)})
rows.append({'Method':'Cross-fitted multi-view stack',**mv['metrics']})
rows.append({'Method':'PHORA+ v2 (proposed)',**add_common_metrics(val_model.lirads_score,phora2['prediction'].prediction)})
rows.append({'Method':'Oracle concepts (upper bound; not deployable)',**oracle['metrics']})
primary=pd.DataFrame(rows).sort_values('final_score',ascending=False)
display(primary)
primary.to_csv(OUTPUT_DIR/'table_primary_validation.csv',index=False)

# %% [markdown]
# # Part III — Ablations (training split only)
# 
# These are the experiments to use for mechanism claims. Run them with `FAST_MODE=False` and the ablation switches enabled for the final manuscript.

# %% [markdown]
# ## 17. Architecture ablations
# 
# - no concept/burden distillation;
# - no factorized gate;
# - no all-threshold ordinal expert;
# - no clinical constraint router;
# - no feature screening;
# - no spatial burden distillation;
# - full PHORA+ v2.

# %%
arch_rows=[]
if RUN_ARCH_ABLATIONS:
    specs=[
        ('PHORA+ v2 full',dict()),
        ('w/o concept + burden distillation',dict(use_concepts=False)),
        ('w/o factorized special gate',dict(use_factor_gate=False)),
        ('w/o all-threshold ordinal expert',dict(use_all_threshold=False)),
        ('w/o clinical constraints',dict(use_clinical_constraints=False)),
        ('w/o nested feature screening',dict(use_screen=False)),
    ]
    for i,(name,kw) in enumerate(specs):
        print('\n',name)
        rr=cross_validate_phora_plus_v2(
            train_model,feature_cols,n_splits=N_SPLITS,seed=SEED+i,
            screen_k=SCREEN_K,ensemble_seeds=ENSEMBLE_SEEDS,router_trials=ROUTER_TRIALS,
            use_concepts=kw.get('use_concepts',True),
            use_factor_gate=kw.get('use_factor_gate',True),
            use_all_threshold=kw.get('use_all_threshold',True),
            use_screen=kw.get('use_screen',True),
            use_clinical_constraints=kw.get('use_clinical_constraints',True)
        )
        arch_rows.append({'Method':name,**rr['metrics']})

    # Burden distillation ablation: retain binary concepts, remove only burden targets.
    no_burden=train_model.drop(columns=[c for c in train_model if c.startswith('burden_')],errors='ignore')
    rr=cross_validate_phora_plus_v2(
        no_burden,feature_cols,n_splits=N_SPLITS,seed=SEED+99,
        screen_k=SCREEN_K,ensemble_seeds=ENSEMBLE_SEEDS,router_trials=ROUTER_TRIALS
    )
    arch_rows.append({'Method':'w/o spatial burden distillation',**rr['metrics']})

arch_ablation=pd.DataFrame(arch_rows)
if len(arch_ablation):
    display(arch_ablation.sort_values('final_score',ascending=False))
    arch_ablation.to_csv(OUTPUT_DIR/'table_architecture_ablation_traincv.csv',index=False)
else:
    print('Set RUN_ARCH_ABLATIONS=True for the final ablation table.')

# %% [markdown]
# ## 18. Feature-family ablations
# 
# This answers whether the new spatial mathematics adds information beyond ordinary radiomics.

# %%
feature_rows=[]
if RUN_FEATURE_ABLATIONS:
    G=feature_groups_v2(feature_cols)
    specs={
        'Morphology only': G.get('Morphology',[]),
        'PyRadiomics only': G.get('PyRadiomics',[]),
        'Custom kinetics + spatial': list(dict.fromkeys(G.get('Custom kinetics',[])+G.get('Spatial fields',[]))),
        'Context + optimal transport only': G.get('Context + OT',[]),
        'All except PyRadiomics': [c for c in feature_cols if c not in set(G.get('PyRadiomics',[]))],
        'All inference-safe features': feature_cols,
    }
    for i,(name,cols) in enumerate(specs.items()):
        if len(cols)<3: continue
        print('\n',name,'n=',len(cols))
        rr=cross_validate_phora_plus_v2(
            train_model,cols,n_splits=N_SPLITS,seed=SEED+200+i,
            screen_k=min(SCREEN_K,len(cols)),ensemble_seeds=ENSEMBLE_SEEDS,
            router_trials=ROUTER_TRIALS
        )
        feature_rows.append({'Feature set':name,'n_features':len(cols),**rr['metrics']})
feature_ablation=pd.DataFrame(feature_rows)
if len(feature_ablation):
    display(feature_ablation.sort_values('final_score',ascending=False))
    feature_ablation.to_csv(OUTPUT_DIR/'table_feature_ablation_traincv.csv',index=False)
else:
    print('Set RUN_FEATURE_ABLATIONS=True for the final feature-family ablation table.')

# %% [markdown]
# ## 19. Repeatability / seed sensitivity of the full method
# 
# Run three repeated 3-fold CV experiments on the training split. Report mean ± SD in supplementary material.

# %%
repeat_rows=[]
if RUN_REPEATABILITY:
    for s in [42,142,242]:
        rr=cross_validate_phora_plus_v2(
            train_model,feature_cols,n_splits=N_SPLITS,seed=s,screen_k=SCREEN_K,
            ensemble_seeds=ENSEMBLE_SEEDS,router_trials=ROUTER_TRIALS
        )
        repeat_rows.append({'seed':s,**rr['metrics']})
repeatability=pd.DataFrame(repeat_rows)
if len(repeatability):
    display(repeatability)
    display(repeatability[['final_score','adjusted_qwk','special_category_recognition']].agg(['mean','std']))
    repeatability.to_csv(OUTPUT_DIR/'repeatability_traincv.csv',index=False)

# %% [markdown]
# # Part IV — Validation analyses for the paper

# %% [markdown]
# ## 20. Confusion matrix and per-class recall

# %%
vp=phora2['prediction']
gt=vp['lirads_score'].to_numpy(); pred=vp['prediction'].to_numpy()
cm=confusion_matrix(gt,pred,labels=VALID_LABELS)
cm_df=pd.DataFrame(cm,index=VALID_LABELS,columns=VALID_LABELS)
display(cm_df)
cm_df.to_csv(OUTPUT_DIR/'confusion_matrix_validation.csv')

rec=[]
for i,lab in enumerate(VALID_LABELS):
    denom=cm[i].sum(); rec.append({'class':lab,'n':int(denom),'recall':float(cm[i,i]/denom) if denom else np.nan})
per_class=pd.DataFrame(rec)
display(per_class)
per_class.to_csv(OUTPUT_DIR/'per_class_recall_validation.csv',index=False)

# %% [markdown]
# ## 21. Concept recovery on the untouched validation split
# 
# This measures whether the learned representation recovers the radiologists' major imaging concepts without receiving them as predictors.

# %%
concept_pairs={
    'Non-rim APHE':('aphe','latent_prob_aphe'),
    'Rim APHE':('rim_aphe','latent_prob_rim_aphe'),
    'Venous washout':('washout_venous','latent_prob_washout_venous'),
    'Delayed washout':('washout_delayed','latent_prob_washout_delayed'),
    'Venous capsule':('capsule_venous','latent_prob_capsule_venous'),
    'Delayed capsule':('capsule_delayed','latent_prob_capsule_delayed'),
}
concept_rows=[]
for name,(target,pcol) in concept_pairs.items():
    if pcol not in vp.columns: continue
    y=pd.to_numeric(val_model[target],errors='coerce').to_numpy(float)
    p=pd.to_numeric(vp[pcol],errors='coerce').to_numpy(float)
    ok=np.isfinite(y)&np.isfinite(p)
    if ok.sum()>=5 and len(np.unique(y[ok]))>1:
        concept_rows.append({'Concept':name,'n':int(ok.sum()),'AUROC':roc_auc_score(y[ok],p[ok]),'AUPRC':average_precision_score(y[ok],p[ok])})
concept_table=pd.DataFrame(concept_rows)
display(concept_table)
concept_table.to_csv(OUTPUT_DIR/'table_concept_recovery_validation.csv',index=False)

# %% [markdown]
# ## 22. Spatial-burden distillation fidelity
# 
# For feature masks that exist in validation, correlate the predicted burden with the true annotated burden. This is an interpretability result, not a test-time input.

# %%
from scipy.stats import spearmanr, pearsonr
burden_rows=[]
for target in [c for c in val_model.columns if c.startswith('burden_')]:
    pcol='latent_'+target
    if pcol not in vp.columns: continue
    y=pd.to_numeric(val_model[target],errors='coerce').to_numpy(float)
    p=pd.to_numeric(vp[pcol],errors='coerce').to_numpy(float)
    ok=np.isfinite(y)&np.isfinite(p)
    if ok.sum()>=8:
        burden_rows.append({
            'target':target,'n':int(ok.sum()),
            'spearman_r':spearmanr(y[ok],p[ok]).statistic,
            'pearson_r':pearsonr(y[ok],p[ok]).statistic,
            'MAE':float(np.mean(np.abs(y[ok]-p[ok])))
        })
burden_fidelity=pd.DataFrame(burden_rows)
display(burden_fidelity)
burden_fidelity.to_csv(OUTPUT_DIR/'burden_distillation_validation.csv',index=False)

# %% [markdown]
# ## 23. Uncertainty via hierarchy disagreement
# 
# Two independent ordinal experts are available: regression and all-threshold. Their disagreement
# 
# \[
# U_{ord}=|s_{reg}-s_{AT}|
# \]
# 
# is combined with gate entropy to identify unstable cases. The challenge still receives one label; this is a failure-analysis / calibration result for the paper.

# %%
eps=1e-9
beta=float(phora2['router']['params'][0])
gd=vp[['p_direct_ord','p_direct_M','p_direct_TIV']].to_numpy(float)
gf=vp[['p_factor_ord','p_factor_M','p_factor_TIV']].to_numpy(float)
g=beta*gd+(1-beta)*gf; g=np.clip(g,eps,None); g/=g.sum(axis=1,keepdims=True)
vp_analysis=vp[['case_id','lirads_score','prediction','ord_reg_score','ord_AT_score']].copy()
vp_analysis['gate_entropy']=-(g*np.log(g)).sum(axis=1)/np.log(3)
vp_analysis['ordinal_disagreement']=np.abs(vp_analysis.ord_reg_score-vp_analysis.ord_AT_score)
vp_analysis['uncertainty']=vp_analysis['gate_entropy'] + .25*vp_analysis['ordinal_disagreement']
vp_analysis['correct']=(vp_analysis.lirads_score==vp_analysis.prediction).astype(int)
vp_analysis['uncertainty_quartile']=pd.qcut(vp_analysis.uncertainty,4,duplicates='drop')
uncertainty_table=vp_analysis.groupby('uncertainty_quartile',observed=True).agg(n=('case_id','size'),accuracy=('correct','mean'),mean_uncertainty=('uncertainty','mean')).reset_index()
display(uncertainty_table)
uncertainty_table.to_csv(OUTPUT_DIR/'uncertainty_validation.csv',index=False)

# %% [markdown]
# ## 24. Stratified bootstrap CIs + paired comparison against the best non-oracle baseline

# %%
def stratified_bootstrap_metrics(gt,pred,n_iter=5000,seed=42):
    gt=np.asarray(gt,object); pred=np.asarray(pred,object); rng=np.random.default_rng(seed)
    by={c:np.where(gt==c)[0] for c in np.unique(gt)}; rows=[]
    for _ in range(n_iter):
        idx=np.concatenate([rng.choice(ii,size=len(ii),replace=True) for ii in by.values()])
        rows.append(fast_challenge_score(gt[idx],pred[idx]))
    b=pd.DataFrame(rows)
    return pd.DataFrame([{'metric':c,'mean':b[c].mean(),'ci_low':b[c].quantile(.025),'ci_high':b[c].quantile(.975)} for c in b])

ci_phora=stratified_bootstrap_metrics(gt,pred,n_iter=BOOTSTRAP_ITERS,seed=SEED)
display(ci_phora)
ci_phora.to_csv(OUTPUT_DIR/'bootstrap_ci_phora_validation.csv',index=False)

# Choose the best deployable baseline by its already-computed holdout score.
best_name=baseline_table.iloc[0]['Method']
best_df=baseline_preds[best_name]
pair=vp[['case_id','lirads_score','prediction']].merge(best_df,on=['case_id','lirads_score'],suffixes=('_phora','_base'),validate='one_to_one')

def paired_bootstrap(pair,n_iter=5000,seed=42):
    y=pair.lirads_score.to_numpy(object); a=pair.prediction_base.to_numpy(object); b=pair.prediction_phora.to_numpy(object)
    rng=np.random.default_rng(seed); by={c:np.where(y==c)[0] for c in np.unique(y)}; ds=[]
    for _ in range(n_iter):
        idx=np.concatenate([rng.choice(ii,size=len(ii),replace=True) for ii in by.values()])
        ma=fast_challenge_score(y[idx],a[idx]); mb=fast_challenge_score(y[idx],b[idx])
        ds.append([mb[k]-ma[k] for k in ['final_score','adjusted_qwk','special_category_recognition']])
    z=np.asarray(ds)
    return pd.DataFrame({'metric':['final_score','adjusted_qwk','special_category_recognition'],
                         'mean_delta':z.mean(0),'ci_low':np.quantile(z,.025,axis=0),'ci_high':np.quantile(z,.975,axis=0),'P_delta_gt_0':(z>0).mean(0)})
paired=paired_bootstrap(pair,n_iter=BOOTSTRAP_ITERS,seed=SEED+1)
print('Comparator:',best_name)
display(paired)
paired.to_csv(OUTPUT_DIR/'paired_bootstrap_validation.csv',index=False)

# %% [markdown]
# ## 25. Train→validation covariate shift: MMD + two-sample classifier
# 
# This is a quantitative domain-shift check. MMD is computed after training-only feature screening, median imputation, and standardization.

# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression

sel=phora2['selected_features']
Xtr=train_model[sel].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(float)
Xva=val_model[sel].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(float)
imp=SimpleImputer(strategy='median').fit(Xtr); Xtr=imp.transform(Xtr); Xva=imp.transform(Xva)
sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xva=sc.transform(Xva)

rng=np.random.default_rng(SEED); xs=Xtr[rng.choice(len(Xtr),min(250,len(Xtr)),replace=False)]; ys=Xva
D=pairwise_distances(np.vstack([xs,ys]),metric='euclidean'); nz=D[D>0]; sigma=np.median(nz) if len(nz) else 1.0; gamma=1/(2*sigma*sigma+1e-12)
def K(a,b): return np.exp(-gamma*pairwise_distances(a,b,metric='sqeuclidean'))
mmd2=float(K(xs,xs).mean()+K(ys,ys).mean()-2*K(xs,ys).mean())

Xd=np.vstack([Xtr,Xva]); yd=np.r_[np.zeros(len(Xtr)),np.ones(len(Xva))]
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
dp=cross_val_predict(LogisticRegression(max_iter=3000,class_weight='balanced'),Xd,yd,cv=cv,method='predict_proba')[:,1]
domain_auc=roc_auc_score(yd,dp)
shift=pd.DataFrame([{'MMD2_rbf':mmd2,'two_sample_domain_AUC':domain_auc,'n_features':len(sel)}])
display(shift)
shift.to_csv(OUTPUT_DIR/'train_validation_shift.csv',index=False)

# %% [markdown]
# ## 26. Optional source/batch robustness
# 
# Because one batch is very small, treat leave-one-batch-out results as exploratory. The cell skips held-out batches with fewer than 10 lesion cases.

# %%
robust_rows=[]
if RUN_DOMAIN_ROBUSTNESS:
    pooled=pd.concat([train_model,val_model],ignore_index=True)
    for i,b in enumerate(sorted(pooled.batch_id.dropna().unique())):
        te=pooled[pooled.batch_id==b].reset_index(drop=True); tr=pooled[pooled.batch_id!=b].reset_index(drop=True)
        if len(te)<10: continue
        sel=nested_screen(tr,feature_cols,k=SCREEN_K,seed=SEED+i)
        Xtr=tr[sel].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
        Xte=te[sel].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
        enc={c:j for j,c in enumerate(VALID_LABELS)}; yi=np.array([enc[z] for z in tr.lirads_score],int)
        m=xgb_classifier(7,SEED+i,n_estimators=420,depth=3,lr=.025)
        m.fit(Xtr,yi,sample_weight=balanced_weights(yi,power=.65))
        pp=m.predict(Xte).astype(int); pr=np.array([VALID_LABELS[j] for j in pp],object)
        robust_rows.append({'held_out_batch':b,'n':len(te),**fast_challenge_score(te.lirads_score,pr)})
robust=pd.DataFrame(robust_rows)
if len(robust):
    display(robust)
    robust.to_csv(OUTPUT_DIR/'leave_one_batch_out_exploratory.csv',index=False)

# %% [markdown]
# # Part V — Final freeze and challenge retraining
# 
# Only run this section **after** the method, feature set, and routing configuration are frozen from training CV + the single official validation analysis.
# 
# The saved object is a local research model (`joblib`) containing XGBoost estimators. It is useful for reproducing the frozen model and for building the final Codabench runtime. It is not by itself a Codabench ZIP.

# %%
RUN_FINAL_REFIT = False  # change to True only after method freeze

if RUN_FINAL_REFIT:
    import joblib
    all_public=pd.concat([train_model,val_model],ignore_index=True)
    final_model=fit_phora_plus_full_v2(
        all_public,feature_cols,seed=SEED,screen_k=SCREEN_K,
        ensemble_seeds=ENSEMBLE_SEEDS,router_trials=max(ROUTER_TRIALS,5000)
    )
    model_path=OUTPUT_DIR/'phora_plus_v2_all_public.joblib'
    joblib.dump(final_model,model_path,compress=3)
    manifest={
        'method':'PHORA+ v2','n_cases':int(len(all_public)),
        'class_counts':all_public.lirads_score.value_counts().to_dict(),
        'n_selected_features':len(final_model['selected']),
        'screen_k':SCREEN_K,'ensemble_seeds':list(ENSEMBLE_SEEDS),
        'router_trials':max(ROUTER_TRIALS,5000),
        'train_metadata':str(TRAIN_METADATA),'val_metadata':str(VAL_METADATA),
    }
    (OUTPUT_DIR/'final_training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Saved',model_path)
else:
    print('Final refit is OFF. Turn it on only after the method is frozen.')

# %% [markdown]
# ## 28. Export manuscript-ready tables

# %%
# Save compact LaTeX tables where available.
def save_tex(df,name,floatfmt='%.3f'):
    if df is None or len(df)==0: return
    (OUTPUT_DIR/name).write_text(df.to_latex(index=False,float_format=lambda x: floatfmt % x),encoding='utf-8')

save_tex(primary,'table_primary_validation.tex')
if 'arch_ablation' in globals(): save_tex(arch_ablation,'table_architecture_ablation.tex')
if 'feature_ablation' in globals(): save_tex(feature_ablation,'table_feature_ablation.tex')
save_tex(concept_table,'table_concept_recovery.tex')
save_tex(per_class,'table_per_class_recall.tex')

print('Paper outputs:')
for p in sorted(OUTPUT_DIR.glob('*')):
    print(' ',p.name)

# %% [markdown]
# # Recommended final experiment matrix for the paper
# 
# ### Main table — official validation split
# 1. Majority class
# 2. Elastic-net logistic regression
# 3. RBF-SVM
# 4. Random Forest
# 5. Extra Trees
# 6. Histogram Gradient Boosting
# 7. Flat XGBoost
# 8. Simple hierarchical XGBoost
# 9. Cross-fitted multi-view stacking
# 10. **PHORA+ v2**
# 11. Oracle clinical concepts *(upper bound only; clearly marked non-deployable)*
# 
# ### Architecture ablation — training CV
# 1. PHORA+ v2 full
# 2. − concept + burden distillation
# 3. − factorized special gate
# 4. − shared all-threshold ordinal expert
# 5. − clinically constrained router
# 6. − nested feature screening
# 7. − spatial burden distillation only
# 
# ### Feature ablation — training CV
# 1. morphology only
# 2. conventional PyRadiomics only
# 3. custom kinetics + spatial fields
# 4. local context + optimal transport only
# 5. all except PyRadiomics
# 6. full multimodal radiomics
# 
# ### Analyses
# - concept AUROC/AUPRC;
# - spatial-burden correlation;
# - confusion matrix and class recall;
# - stratified bootstrap 95% CI;
# - paired bootstrap vs best baseline;
# - ordinal-expert disagreement / uncertainty;
# - MMD and train→validation two-sample AUC;
# - optional source/batch robustness;
# - repeated-seed CV.
# 
# ## Final-run settings
# Before producing numbers for the manuscript, set:
# 
# ```python
# FAST_MODE = False
# RUN_ARCH_ABLATIONS = True
# RUN_FEATURE_ABLATIONS = True
# RUN_REPEATABILITY = True
# ```
# 
# Then restart the kernel and run top-to-bottom. Feature extraction is cached, so the expensive CT processing should not repeat.

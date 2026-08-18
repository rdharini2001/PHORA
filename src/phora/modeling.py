"""PHORA+ v2 modeling, hierarchical routing, baselines, and analysis utilities."""
from __future__ import annotations
import math, warnings, json, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

ORDINAL_LABELS = ["LR-1", "LR-2", "LR-3", "LR-4", "LR-5"]
SPECIAL_LABELS = ["LR-M", "LR-TIV"]
VALID_LABELS = ORDINAL_LABELS + SPECIAL_LABELS
ORD_TO_INT = {x:i+1 for i,x in enumerate(ORDINAL_LABELS)}
LABEL_TO_INT = {x:i for i,x in enumerate(VALID_LABELS)}
EPS = 1e-8


def fast_challenge_score(y_true, y_pred):
    y_true=np.asarray(y_true,dtype=object); y_pred=np.asarray(y_pred,dtype=object)
    # official adjusted QWK: only ordinal GT + ordinal predictions
    ord_mask=np.isin(y_true, ORDINAL_LABELS) & np.isin(y_pred, ORDINAL_LABELS)
    if not ord_mask.any():
        qwk=0.0
    else:
        gi=np.array([ORDINAL_LABELS.index(v) for v in y_true[ord_mask]],dtype=int)
        pi=np.array([ORDINAL_LABELS.index(v) for v in y_pred[ord_mask]],dtype=int)
        O=np.zeros((5,5),dtype=float)
        np.add.at(O,(gi,pi),1)
        n=O.sum()
        if n<=0:
            qwk=0.0
        elif len(np.unique(np.r_[gi,pi]))<2:
            qwk=1.0 if np.array_equal(gi,pi) else 0.0
        else:
            a=O.sum(axis=1); b=O.sum(axis=0)
            E=np.outer(a,b)/n
            ii,jj=np.indices((5,5)); W=((ii-jj)**2)/16.0
            den=(W*E).sum(); num=(W*O).sum()
            qwk=max(0.0, 1.0-num/(den+EPS))
    def collapse(a):
        return np.array(["ordinal" if v in ORDINAL_LABELS else v for v in a],dtype=object)
    g=collapse(y_true); p=collapse(y_pred)
    recalls=[]
    for lab in ["ordinal","LR-M","LR-TIV"]:
        m=(g==lab)
        if m.any(): recalls.append(float(np.mean(p[m]==lab)))
    scr=float(np.mean(recalls)) if recalls else 0.0
    return {"final_score":0.85*qwk+0.15*scr,"adjusted_qwk":qwk,"special_category_recognition":scr}


def balanced_weights(y, power=.65, lo=.2, hi=10.0):
    y=np.asarray(y)
    vals, cnt=np.unique(y,return_counts=True)
    mp=dict(zip(vals,cnt)); n=len(y); k=len(vals)
    w=np.array([(n/(k*mp[v]))**power for v in y],float)
    w/=w.mean()+EPS
    return np.clip(w,lo,hi)


def xgb_classifier(n_classes, seed, n_estimators=320, depth=3, lr=.025):
    params=dict(n_estimators=n_estimators,max_depth=depth,learning_rate=lr,
                min_child_weight=2.0,subsample=.88,colsample_bytree=.72,
                reg_alpha=.2,reg_lambda=4.0,gamma=0.0,tree_method='hist',
                random_state=seed,n_jobs=-1)
    if n_classes==2:
        return xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',**params)
    return xgb.XGBClassifier(objective='multi:softprob',num_class=n_classes,eval_metric='mlogloss',**params)


def xgb_regressor(seed, n_estimators=380, depth=3, lr=.02):
    return xgb.XGBRegressor(objective='reg:squarederror',n_estimators=n_estimators,max_depth=depth,
                            learning_rate=lr,min_child_weight=2.0,subsample=.88,colsample_bytree=.72,
                            reg_alpha=.2,reg_lambda=4.0,tree_method='hist',random_state=seed,n_jobs=-1)


def _numeric_matrix(df, cols):
    return df.loc[:,cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)


def nested_screen(train_df, feature_cols, k=120, corr_threshold=.985, seed=42):
    """Training-fold-only multi-objective screening for gate + ordinal + concepts."""
    cols=[c for c in feature_cols if c in train_df.columns]
    X=train_df[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    miss=X.isna().mean().to_numpy()
    keep=miss<.40
    cols=[c for c,z in zip(cols,keep) if z]
    X=X.loc[:,cols]
    imp=SimpleImputer(strategy='median')
    Xi=imp.fit_transform(X)
    var=np.nanvar(Xi,axis=0)
    keep=var>1e-10
    Xi=Xi[:,keep]; cols=[c for c,z in zip(cols,keep) if z]
    if len(cols)<=k:
        return cols
    gate_y=np.array([0 if z in ORDINAL_LABELS else (1 if z=='LR-M' else 2) for z in train_df.lirads_score],int)
    try:
        fg=np.nan_to_num(f_classif(Xi,gate_y)[0],nan=0,posinf=0,neginf=0)
    except Exception:
        fg=np.zeros(len(cols))
    om=np.array([z in ORDINAL_LABELS for z in train_df.lirads_score])
    oy=np.array([ORD_TO_INT[z] for z in train_df.loc[om,'lirads_score']],float)
    try:
        fo=np.nan_to_num(f_regression(Xi[om],oy)[0],nan=0,posinf=0,neginf=0)
    except Exception:
        fo=np.zeros(len(cols))
    concept_scores=[]
    for t in ['aphe','washout_venous','washout_delayed','capsule_venous','capsule_delayed']:
        if t not in train_df: continue
        yy=pd.to_numeric(train_df[t],errors='coerce').to_numpy()
        ok=np.isfinite(yy)
        if ok.sum()>20 and len(np.unique(yy[ok]))>1:
            try: concept_scores.append(np.nan_to_num(f_classif(Xi[ok],yy[ok].astype(int))[0],nan=0,posinf=0,neginf=0))
            except Exception: pass
    fc=np.mean(concept_scores,axis=0) if concept_scores else np.zeros(len(cols))
    def rank01(v):
        order=np.argsort(np.argsort(v))
        return order/(max(1,len(v)-1))
    combined=np.maximum.reduce([rank01(fg),rank01(fo),rank01(fc)])
    idx=np.argsort(combined)[::-1][:min(k*2,len(cols))]
    # Greedy correlation pruning among high-scoring candidates.
    cand=[cols[i] for i in idx]
    Xc=pd.DataFrame(Xi[:,idx],columns=cand)
    corr=Xc.corr().abs().fillna(0)
    selected=[]
    for c in cand:
        if len(selected)>=k: break
        if all(corr.loc[c,s] < corr_threshold for s in selected): selected.append(c)
    return selected


def fit_binary_concept_heads(X, df, targets, seed=42, n_estimators=220):
    models={}
    for j,t in enumerate(targets):
        if t not in df: continue
        yy=pd.to_numeric(df[t],errors='coerce').to_numpy(float)
        ok=np.isfinite(yy)
        if ok.sum()<20 or len(np.unique(yy[ok]))<2: continue
        m=xgb_classifier(2,seed+17*j,n_estimators=n_estimators,depth=3,lr=.03)
        m.fit(X[ok],yy[ok].astype(int),sample_weight=balanced_weights(yy[ok].astype(int),power=.55))
        models[t]=m
    return models


def predict_concepts(models, X, targets):
    arr=[]; names=[]
    for t in targets:
        if t in models:
            arr.append(models[t].predict_proba(X)[:,1]); names.append('latent_prob_'+t)
    return (np.column_stack(arr).astype(np.float32) if arr else np.zeros((len(X),0),np.float32)), names


def inner_oof_concepts(X, df, targets, seed=42, n_splits=3):
    full=fit_binary_concept_heads(X,df,targets,seed)
    order=[t for t in targets if t in full]
    Z=np.full((len(X),len(order)),np.nan,np.float32)
    kf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    yy=df.lirads_score.astype(str).to_numpy()
    strata=np.array(['ordinal' if z in ORDINAL_LABELS else z for z in yy],dtype=object)
    for f,(tr,va) in enumerate(kf.split(X,strata)):
        mm=fit_binary_concept_heads(X[tr],df.iloc[tr],order,seed+100+f,n_estimators=180)
        z,_=predict_concepts(mm,X[va],order)
        if z.shape[1]==len(order): Z[va]=z
        else:
            z,_=predict_concepts(full,X[va],order); Z[va]=z
    return Z,['latent_prob_'+t for t in order],full,order


def sigmoid(z):
    z=np.clip(np.asarray(z,float),-30,30)
    return 1/(1+np.exp(-z))


def structured_lirads_basis(Z, names, size_mm):
    """Differentiable clinical interaction basis; uses predicted concepts only."""
    mp={n:Z[:,i] for i,n in enumerate(names)}
    def g(n,default=.5):
        v=mp.get(n)
        return np.full(len(size_mm),default,float) if v is None else np.nan_to_num(v,nan=default)
    pA=g('latent_prob_aphe')
    pWV=g('latent_prob_washout_venous'); pWD=g('latent_prob_washout_delayed')
    pCV=g('latent_prob_capsule_venous'); pCD=g('latent_prob_capsule_delayed')
    pW=1-(1-pWV)*(1-pWD); pC=1-(1-pCV)*(1-pCD)
    d=np.nan_to_num(np.asarray(size_mm,float),nan=np.nanmedian(size_mm))
    s10=sigmoid((d-10)/2.5); s20=sigmoid((d-20)/3.5); mid=s10*(1-s20)
    major_count=pW+pC
    lr5_plaus=np.clip(pA*(s20*(1-(1-pW)*(1-pC)) + mid*pW),0,1)
    lr4_plaus=np.clip(pA*(1-lr5_plaus) + (1-pA)*0.5*(pW+pC),0,1)
    basis=np.column_stack([pA,pW,pC,major_count,s10,s20,pA*pW,pA*pC,pW*pC,
                           pA*s10,pA*s20,lr4_plaus,lr5_plaus]).astype(np.float32)
    bnames=['rule_pAPHE','rule_pWashout','rule_pCapsule','rule_expected_major_count','rule_size10_soft','rule_size20_soft',
            'rule_APHE_x_W','rule_APHE_x_C','rule_W_x_C','rule_APHE_x_size10','rule_APHE_x_size20','rule_LR4_plaus','rule_LR5_plaus']
    return basis,bnames


def fit_direct_gate_ensemble(X,y,seeds=(11,29,47)):
    gy=np.array([0 if z in ORDINAL_LABELS else (1 if z=='LR-M' else 2) for z in y],int)
    models=[]
    for s in seeds:
        m=xgb_classifier(3,s,n_estimators=360,depth=3,lr=.022)
        m.fit(X,gy,sample_weight=balanced_weights(gy,power=.7))
        models.append(m)
    return models


def predict_direct_gate(models,X):
    return np.mean([m.predict_proba(X) for m in models],axis=0)


def fit_factor_gate_ensemble(X,y,seeds=(13,31,53)):
    y=np.asarray(y,dtype=object)
    special=np.array([z in SPECIAL_LABELS for z in y],int)
    sp_models=[]; tiv_models=[]
    sm=(special==1); tiv=np.array([1 if z=='LR-TIV' else 0 for z in y[sm]],int)
    for s in seeds:
        a=xgb_classifier(2,s,n_estimators=330,depth=3,lr=.025)
        a.fit(X,special,sample_weight=balanced_weights(special,power=.7))
        b=xgb_classifier(2,s+100,n_estimators=260,depth=3,lr=.025)
        b.fit(X[sm],tiv,sample_weight=balanced_weights(tiv,power=.8))
        sp_models.append(a); tiv_models.append(b)
    return sp_models,tiv_models


def predict_factor_gate(models,X):
    sp,tiv=models
    ps=np.mean([m.predict_proba(X)[:,1] for m in sp],axis=0)
    pt=np.mean([m.predict_proba(X)[:,1] for m in tiv],axis=0)
    out=np.column_stack([1-ps,ps*(1-pt),ps*pt])
    return out/(out.sum(axis=1,keepdims=True)+EPS)


def fit_shared_all_threshold(X,y,seed=42,n_estimators=360):
    y=np.asarray(y,dtype=object); ok=np.array([z in ORDINAL_LABELS for z in y])
    Xo=X[ok]; yo=np.array([ORD_TO_INT[z] for z in y[ok]],int)
    n=len(yo); thresholds=np.tile(np.arange(1,5),n); yrep=np.repeat(yo,4)
    Xrep=np.repeat(Xo,4,axis=0)
    t=(thresholds/4.0).reshape(-1,1).astype(np.float32)
    Xexp=np.column_stack([Xrep,t,t*t]).astype(np.float32)
    target=(yrep>thresholds).astype(int)
    base_w=np.repeat(balanced_weights(yo,power=.65),4)
    # Additional balance within each threshold task.
    tw=np.ones_like(base_w)
    for k in range(1,5):
        m=thresholds==k
        tw[m]=balanced_weights(target[m],power=.5)
    model=xgb_classifier(2,seed,n_estimators=n_estimators,depth=3,lr=.022)
    model.fit(Xexp,target,sample_weight=base_w*tw)
    return model


def predict_all_threshold(model,X):
    n=len(X); thresholds=np.tile(np.arange(1,5),n)
    Xrep=np.repeat(X,4,axis=0); t=(thresholds/4.0).reshape(-1,1).astype(np.float32)
    Xexp=np.column_stack([Xrep,t,t*t]).astype(np.float32)
    q=model.predict_proba(Xexp)[:,1].reshape(n,4)
    # q_k=P(Y>k) must be non-increasing in k. Euclidean isotonic projection surrogate.
    q=np.minimum.accumulate(q,axis=1)
    q=np.clip(q,0,1)
    p=np.column_stack([1-q[:,0],q[:,0]-q[:,1],q[:,1]-q[:,2],q[:,2]-q[:,3],q[:,3]])
    p=np.clip(p,0,None); p/=p.sum(axis=1,keepdims=True)+EPS
    score=(p*np.arange(1,6)[None,:]).sum(axis=1)
    return score,p


def fit_ordinal_ensemble(X,y,seeds=(17,37,61)):
    y=np.asarray(y,dtype=object); ok=np.array([z in ORDINAL_LABELS for z in y]); yi=np.array([ORD_TO_INT[z] for z in y[ok]],int)
    regs=[]; ats=[]
    for s in seeds:
        r=xgb_regressor(s,n_estimators=420,depth=3,lr=.018)
        r.fit(X[ok],yi.astype(float),sample_weight=balanced_weights(yi,power=.7))
        regs.append(r)
        ats.append(fit_shared_all_threshold(X,y,seed=s+200,n_estimators=330))
    return regs,ats


def predict_ordinal_ensemble(models,X):
    regs,ats=models
    reg=np.mean([np.clip(m.predict(X),.5,5.5) for m in regs],axis=0)
    ats_scores=[]; probs=[]
    for m in ats:
        s,p=predict_all_threshold(m,X); ats_scores.append(s); probs.append(p)
    return reg,np.mean(ats_scores,axis=0),np.mean(probs,axis=0)


def labels_from_router(gd,gf,reg,ats,params):
    beta,alpha,c1,c2,c3,c4,tM,tT,margin=params
    g=beta*gd+(1-beta)*gf; g/=g.sum(axis=1,keepdims=True)+EPS
    score=alpha*reg+(1-alpha)*ats
    cps=np.array([c1,c2,c3,c4],float)
    k=1+(score[:,None]>cps[None,:]).sum(axis=1)
    out=np.array([f'LR-{int(v)}' for v in np.clip(k,1,5)],dtype=object)
    pord,pm,pt=g[:,0],g[:,1],g[:,2]
    chooseM=(pm>=pt)&(pm>=tM)&(pm>=pord+margin)
    chooseT=(pt>pm)&(pt>=tT)&(pt>=pord+margin)
    out[chooseM]='LR-M'; out[chooseT]='LR-TIV'
    return out


def tune_router(y,gd,gf,reg,ats,n_trials=3000,seed=42):
    rng=np.random.default_rng(seed)
    candidates=[(0.5,0.5,1.5,2.5,3.5,4.5,.40,.35,0.0)]
    betas=np.array([0,.25,.5,.75,1.0]); alphas=np.array([0,.25,.5,.75,1.0])
    for _ in range(n_trials):
        cp=np.array([rng.uniform(1.1,2.0),rng.uniform(1.8,3.0),rng.uniform(2.8,4.1),rng.uniform(3.8,5.1)])
        cp=np.sort(cp)
        if np.min(np.diff(cp))<.18: continue
        candidates.append((float(rng.choice(betas)),float(rng.choice(alphas)),*map(float,cp),
                           float(rng.uniform(.20,.70)),float(rng.uniform(.15,.70)),float(rng.uniform(-.08,.18))))
    best=None
    for p in candidates:
        pred=labels_from_router(gd,gf,reg,ats,p); m=fast_challenge_score(y,pred)
        key=(m['final_score'],m['adjusted_qwk'],m['special_category_recognition'])
        if best is None or key>best[0]: best=(key,p,pred,m)
    return {'params':best[1],'prediction':best[2],'metrics':best[3]}


def inner_router_oof(X,y,n_splits=3,seeds=(17,37),seed=42,use_factor_gate=True,use_all_threshold=True):
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    gd=np.zeros((len(y),3)); gf=np.zeros((len(y),3)); reg=np.zeros(len(y)); ats=np.zeros(len(y))
    strata=np.array(['ordinal' if z in ORDINAL_LABELS else z for z in y],dtype=object)
    for f,(tr,va) in enumerate(skf.split(X,strata)):
        ss=tuple(s+1000*f for s in seeds)
        dm=fit_direct_gate_ensemble(X[tr],y[tr],seeds=ss)
        fm=fit_factor_gate_ensemble(X[tr],y[tr],seeds=tuple(s+10 for s in ss)) if use_factor_gate else None
        om=fit_ordinal_ensemble(X[tr],y[tr],seeds=tuple(s+20 for s in ss))
        gd[va]=predict_direct_gate(dm,X[va])
        gf[va]=predict_factor_gate(fm,X[va]) if use_factor_gate else gd[va]
        reg[va],ats0,_=predict_ordinal_ensemble(om,X[va])
        ats[va]=ats0 if use_all_threshold else reg[va]
    return gd,gf,reg,ats


def cross_validate_phora_plus(data, feature_cols, n_splits=3, seed=42, screen_k=120, ensemble_seeds=(17,37), router_trials=2500,
                              use_concepts=True,use_factor_gate=True,use_all_threshold=True,use_screen=True):
    data=data[data.lirads_score.isin(VALID_LABELS)].reset_index(drop=True).copy()
    y=data.lirads_score.astype(str).to_numpy(); n=len(data)
    oof=np.empty(n,dtype=object); oof_gd=np.zeros((n,3)); oof_gf=np.zeros((n,3)); oof_reg=np.zeros(n); oof_ats=np.zeros(n)
    selected_by_fold=[]; router_by_fold=[]
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    concept_targets=['aphe','washout_venous','washout_delayed','capsule_venous','capsule_delayed']
    for f,(tr,va) in enumerate(skf.split(np.zeros(n),y),start=1):
        train=data.iloc[tr].reset_index(drop=True); val=data.iloc[va].reset_index(drop=True)
        selected=nested_screen(train,feature_cols,k=screen_k,seed=seed+f) if use_screen else list(feature_cols)
        selected_by_fold.append(selected)
        Xtr=_numeric_matrix(train,selected); Xva=_numeric_matrix(val,selected)
        if use_concepts:
            Ztr,znames,concept_full,order=inner_oof_concepts(Xtr,train,concept_targets,seed=seed+100*f,n_splits=3)
            Zva,_=predict_concepts(concept_full,Xva,order)
            size_col='geom_max_bbox_diameter_mm' if 'geom_max_bbox_diameter_mm' in train else 'max_diameter_mm'
            Btr,bnames=structured_lirads_basis(Ztr,znames,pd.to_numeric(train[size_col],errors='coerce').to_numpy())
            Bva,_=structured_lirads_basis(Zva,znames,pd.to_numeric(val[size_col],errors='coerce').to_numpy())
            Atr=np.column_stack([Xtr,Ztr,Btr]).astype(np.float32); Ava=np.column_stack([Xva,Zva,Bva]).astype(np.float32)
        else:
            Atr=Xtr.astype(np.float32); Ava=Xva.astype(np.float32)
        # Router/postprocessing tuned strictly inside outer training fold.
        igd,igf,ireg,iats=inner_router_oof(Atr,y[tr],n_splits=3,seeds=ensemble_seeds,seed=seed+200*f,use_factor_gate=use_factor_gate,use_all_threshold=use_all_threshold)
        tuned=tune_router(y[tr],igd,igf,ireg,iats,n_trials=router_trials,seed=seed+300*f)
        router_by_fold.append(tuned['params'])
        ss=tuple(s+5000*f for s in ensemble_seeds)
        dm=fit_direct_gate_ensemble(Atr,y[tr],seeds=ss)
        fm=fit_factor_gate_ensemble(Atr,y[tr],seeds=tuple(s+10 for s in ss)) if use_factor_gate else None
        om=fit_ordinal_ensemble(Atr,y[tr],seeds=tuple(s+20 for s in ss))
        gd=predict_direct_gate(dm,Ava); gf=predict_factor_gate(fm,Ava) if use_factor_gate else gd.copy(); reg,ats0,_=predict_ordinal_ensemble(om,Ava); ats=ats0 if use_all_threshold else reg.copy()
        pred=labels_from_router(gd,gf,reg,ats,tuned['params'])
        oof[va]=pred; oof_gd[va]=gd; oof_gf[va]=gf; oof_reg[va]=reg; oof_ats[va]=ats
        print(f'fold {f}: selected={len(selected)} inner={tuned["metrics"]["final_score"]:.4f} outer_partial={fast_challenge_score(y[va],pred)["final_score"]:.4f}')
    metrics=fast_challenge_score(y,oof)
    out=pd.DataFrame({'case_id':data.case_id,'lirads_score':y,'prediction':oof,'ord_reg_score':oof_reg,'ord_AT_score':oof_ats})
    out[['p_direct_ord','p_direct_M','p_direct_TIV']]=oof_gd
    out[['p_factor_ord','p_factor_M','p_factor_TIV']]=oof_gf
    return {'metrics':metrics,'oof':out,'selected_by_fold':selected_by_fold,'router_by_fold':router_by_fold}

# ---------- additional CT features ----------
def robust_stats(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)<8: return (np.nan,)*4
    lo,hi=np.percentile(v,[2.5,97.5]); v=np.clip(v,lo,hi)
    q10,q25,q50,q75,q90=np.percentile(v,[10,25,50,75,90])
    return float(q50),float(q75-q25),float(q10),float(q90)


def moran_geary(arr,mask):
    x=np.asarray(arr,float); m=mask & np.isfinite(x); vals=x[m]
    if len(vals)<20 or np.var(vals)<1e-8: return np.nan,np.nan
    mu=vals.mean(); denom=np.sum((vals-mu)**2); cross=0.; sq=0.; W=0
    for ax in range(3):
        a=[slice(None)]*3; b=[slice(None)]*3; a[ax]=slice(1,None); b[ax]=slice(None,-1)
        mm=m[tuple(a)]&m[tuple(b)]; xa=x[tuple(a)][mm]; xb=x[tuple(b)][mm]
        if len(xa):
            cross+=2*np.sum((xa-mu)*(xb-mu)); sq+=2*np.sum((xa-xb)**2); W+=2*len(xa)
    N=len(vals)
    return float(N*cross/(W*denom+EPS)), float((N-1)*sq/(2*W*denom+EPS))


def hot_topology(arr,mask):
    vals=arr[mask & np.isfinite(arr)]
    if len(vals)<30: return (np.nan,)*4
    comps=[]; lfrac=[]
    for q in [60,70,80,90]:
        th=np.percentile(vals,q); hot=mask & np.isfinite(arr) & (arr>=th)
        lab,n=ndimage.label(hot,structure=ndimage.generate_binary_structure(3,1)); comps.append(n)
        if n:
            cnt=np.bincount(lab.ravel())[1:]; lfrac.append(float(cnt.max()/(hot.sum()+EPS)))
        else: lfrac.append(0.)
    return float(np.mean(comps)),float(np.std(comps)),float(np.mean(lfrac)),float(lfrac[-1])


def perivascular_contact(arr,mask,spacing,dout):
    ring=(~mask)&(dout>0)&(dout<=12)&np.isfinite(arr)
    vals=arr[ring]
    if len(vals)<50: return (np.nan,)*3
    th=np.percentile(vals,90); hot=ring&(arr>=th); contact=hot&(dout<=3.0)
    contact_fraction=float(contact.sum()/(((~mask)&(dout<=3.0)).sum()+EPS))
    lab,n=ndimage.label(hot,structure=ndimage.generate_binary_structure(3,1))
    ids=np.unique(lab[contact]); ids=ids[ids>0]
    if len(ids)==0: return contact_fraction,0.0,np.nan
    counts=[np.sum(lab==i) for i in ids]; best=int(ids[int(np.argmax(counts))]); pts=np.argwhere(lab==best)*spacing[None,:]
    frac=float(len(pts)/(ring.sum()+EPS))
    if len(pts)>=5:
        ev=np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
        elong=float(np.sqrt((ev[0]+EPS)/(ev[1]+EPS)))
    else: elong=np.nan
    return contact_fraction,frac,elong


def extract_context_ot_case(case_dir,case_id):
    from phora_features import _load_case_on_art_grid
    d=_load_case_on_art_grid(Path(case_dir),case_id,margin_vox=14)
    if d is None: return {'case_id':case_id}
    mask=d['mask']; sp=np.asarray(d['spacing'],float); imgs=d['images']
    din=ndimage.distance_transform_edt(mask,sampling=sp); dout=ndimage.distance_transform_edt(~mask,sampling=sp)
    outer03=(~mask)&(dout>0)&(dout<=3); bg=(~mask)&(dout>=3)&(dout<=12)
    inner03=mask&(din<=3); maxd=float(np.max(din[mask])) if mask.any() else 0
    core=mask&(din>=max(3.0,.55*maxd))
    out={'case_id':case_id}; rel={}; lesvals={}
    for ph,arr in imgs.items():
        if arr is None: continue
        lv=arr[mask & np.isfinite(arr)]; bv=arr[bg & np.isfinite(arr)]
        lm,li,lp10,lp90=robust_stats(lv); bm,bi,bp10,bp90=robust_stats(bv)
        im,*_=robust_stats(arr[inner03 & np.isfinite(arr)]); om,*_=robust_stats(arr[outer03 & np.isfinite(arr)]); cm,*_=robust_stats(arr[core & np.isfinite(arr)])
        out[f'ctx_{ph}_lesion_median']=lm; out[f'ctx_{ph}_background_median']=bm
        out[f'ctx_{ph}_lesion_minus_bg']=lm-bm; out[f'ctx_{ph}_inner_minus_outer']=im-om; out[f'ctx_{ph}_rim_minus_core']=im-cm
        out[f'ctx_{ph}_lesion_iqr']=li; out[f'ctx_{ph}_bg_iqr']=bi
        mor,gea=moran_geary(arr,mask); out[f'ctx_{ph}_moranI']=mor; out[f'ctx_{ph}_gearyC']=gea
        tc,ts,lf,l90=hot_topology(arr,mask); out[f'ctx_{ph}_topo_components_mean']=tc; out[f'ctx_{ph}_topo_components_sd']=ts; out[f'ctx_{ph}_hot_largest_frac_mean']=lf; out[f'ctx_{ph}_hot_largest_frac_q90']=l90
        cf,vf,ve=perivascular_contact(arr,mask,sp,dout); out[f'ctx_{ph}_vascular_contact']=cf; out[f'ctx_{ph}_vascular_component_frac']=vf; out[f'ctx_{ph}_vascular_elongation']=ve
        rel[ph]=lm-bm; lesvals[ph]=lv
    # Background-corrected kinetic features.
    pairs=[('ART','DRY'),('ART','VEN'),('ART','DEL'),('VEN','DEL')]
    for a,b in pairs:
        if a in rel and b in rel:
            out[f'ctx_rel_{a}_minus_{b}']=rel[a]-rel[b]
        if a in lesvals and b in lesvals:
            va,vb=lesvals[a],lesvals[b]
            if len(va)>20 and len(vb)>20:
                # subsample deterministic quantiles if huge
                if len(va)>5000: va=np.quantile(va,np.linspace(0,1,5000))
                if len(vb)>5000: vb=np.quantile(vb,np.linspace(0,1,5000))
                out[f'ot_W1_{a}_{b}']=float(wasserstein_distance(va,vb))
                lo=min(np.percentile(va,1),np.percentile(vb,1)); hi=max(np.percentile(va,99),np.percentile(vb,99))
                if hi>lo:
                    bins=np.linspace(lo,hi,49); ha,_=np.histogram(va,bins=bins,density=True); hb,_=np.histogram(vb,bins=bins,density=True)
                    ha=ha+EPS; hb=hb+EPS; ha/=ha.sum(); hb/=hb.sum()
                    out[f'ot_JSD_{a}_{b}']=float(jensenshannon(ha,hb,base=2.0)**2)
    if 'ART' in rel and 'VEN' in rel: out['phys_APHE_washout_axis']=rel['ART']-rel['VEN']
    if 'ART' in rel and 'DEL' in rel: out['phys_APHE_delayed_axis']=rel['ART']-rel['DEL']
    return out

# ============================================================================
# PHORA+ v2: new-data / paper protocol extensions
# ============================================================================
CONCEPT_TARGETS_V2 = [
    'aphe', 'rim_aphe', 'washout_venous', 'washout_delayed',
    'capsule_venous', 'capsule_delayed'
]


def safe_strata(labels, n_splits=3):
    """Stable CV strata when LR-1/LR-2 are too rare for ordinary 7-class stratification."""
    y = np.asarray(labels, dtype=object)
    s = y.copy()
    vals, cnt = np.unique(s, return_counts=True)
    if len(cnt) and cnt.min() < n_splits:
        s = np.array(['LR-1-3' if z in ('LR-1','LR-2','LR-3') else z for z in y], dtype=object)
    vals, cnt = np.unique(s, return_counts=True)
    if len(cnt) and cnt.min() < n_splits:
        s = np.array(['ordinal' if z in ORDINAL_LABELS else z for z in y], dtype=object)
    return s


def structured_lirads_basis_v2(Z, names, size_mm):
    """Soft LI-RADS concept basis including non-rim and rim APHE."""
    mp = {n: Z[:, i] for i, n in enumerate(names)}
    n = len(size_mm)
    def g(nm, default=.5):
        v = mp.get(nm)
        return np.full(n, default, float) if v is None else np.nan_to_num(v, nan=default)

    pA = g('latent_prob_aphe')
    pR = g('latent_prob_rim_aphe', default=0.15)
    pWV = g('latent_prob_washout_venous')
    pWD = g('latent_prob_washout_delayed')
    pCV = g('latent_prob_capsule_venous')
    pCD = g('latent_prob_capsule_delayed')
    pW = 1 - (1-pWV)*(1-pWD)
    pC = 1 - (1-pCV)*(1-pCD)
    d = np.asarray(size_mm, float)
    med = np.nanmedian(d) if np.isfinite(d).any() else 30.0
    d = np.nan_to_num(d, nan=med)
    s10 = sigmoid((d-10)/2.5)
    s20 = sigmoid((d-20)/3.5)
    mid = s10*(1-s20)
    major_count = pW + pC
    lr5_plaus = np.clip(pA * (s20*(1-(1-pW)*(1-pC)) + mid*pW), 0, 1)
    lr4_plaus = np.clip(pA*(1-lr5_plaus) + (1-pA)*0.60*(pW+pC), 0, 1)
    lrm_plaus = np.clip(pR * (0.55 + 0.45*(1-pA)), 0, 1)
    basis = np.column_stack([
        pA,pR,pW,pC,major_count,s10,s20,
        pA*pW,pA*pC,pW*pC,pA*s10,pA*s20,
        pR*(1-pA),pR*pW,lr4_plaus,lr5_plaus,lrm_plaus
    ]).astype(np.float32)
    names_out = [
        'rule_pNonRimAPHE','rule_pRimAPHE','rule_pWashout','rule_pCapsule',
        'rule_expected_major_count','rule_size10_soft','rule_size20_soft',
        'rule_APHE_x_W','rule_APHE_x_C','rule_W_x_C','rule_APHE_x_size10',
        'rule_APHE_x_size20','rule_Rim_x_noAPHE','rule_Rim_x_W',
        'rule_LR4_plaus','rule_LR5_plaus','rule_LRM_plaus'
    ]
    return basis, names_out


def _latent_col(Z, names, target, default=.5):
    nm = f'latent_prob_{target}'
    if nm not in names:
        return np.full(len(Z), default, float)
    return np.nan_to_num(Z[:, names.index(nm)].astype(float), nan=default)


def labels_from_router_v2(gd, gf, reg, ats, p_aphe, p_rim, size_mm, params):
    """Metric-aware soft hierarchy with probabilistic clinical constraints."""
    beta, alpha, c1, c2, c3, c4, tM, tT, margin, tA, gammaR = params
    g = beta*gd + (1-beta)*gf
    g = np.clip(g, EPS, None)
    if p_rim is not None:
        # Rim APHE is highly informative for LR-M but is kept probabilistic, not a hard label.
        g[:,1] *= np.exp(gammaR*(np.asarray(p_rim,float)-0.5))
    g /= g.sum(axis=1, keepdims=True) + EPS

    score = alpha*reg + (1-alpha)*ats
    cps = np.array([c1,c2,c3,c4], float)
    k = 1 + (score[:,None] > cps[None,:]).sum(axis=1)
    out = np.array([f'LR-{int(v)}' for v in np.clip(k,1,5)], dtype=object)

    pord, pm, pt = g[:,0], g[:,1], g[:,2]
    chooseM = (pm >= pt) & (pm >= tM) & (pm >= pord + margin)
    chooseT = (pt > pm) & (pt >= tT) & (pt >= pord + margin)
    out[chooseM] = 'LR-M'
    out[chooseT] = 'LR-TIV'

    # Clinical constraints apply only on the ordinal branch.
    ord_mask = np.isin(out, ORDINAL_LABELS)
    size = np.asarray(size_mm, float)
    too_small = ord_mask & (out == 'LR-5') & np.isfinite(size) & (size < 10.0)
    out[too_small] = 'LR-4'
    if p_aphe is not None:
        veto = ord_mask & (out == 'LR-5') & (np.asarray(p_aphe,float) < tA)
        out[veto] = 'LR-4'
    return out


def tune_router_v2(y, gd, gf, reg, ats, p_aphe, p_rim, size_mm, n_trials=3500, seed=42):
    rng = np.random.default_rng(seed)
    candidates = [(.5,.5,1.5,2.5,3.5,4.5,.40,.35,0.0,.45,1.0)]
    betas = np.array([0,.25,.5,.75,1.0])
    alphas = np.array([0,.25,.5,.75,1.0])
    for _ in range(int(n_trials)):
        cp = np.array([
            rng.uniform(1.05,2.05), rng.uniform(1.75,3.05),
            rng.uniform(2.75,4.15), rng.uniform(3.75,5.10)
        ])
        cp = np.sort(cp)
        if np.min(np.diff(cp)) < .15:
            continue
        candidates.append((
            float(rng.choice(betas)), float(rng.choice(alphas)), *map(float,cp),
            float(rng.uniform(.18,.70)), float(rng.uniform(.15,.70)),
            float(rng.uniform(-.10,.20)), float(rng.uniform(.20,.80)),
            float(rng.uniform(0.0,2.5))
        ))
    best = None
    for p in candidates:
        pred = labels_from_router_v2(gd,gf,reg,ats,p_aphe,p_rim,size_mm,p)
        m = fast_challenge_score(y,pred)
        key = (m['final_score'],m['adjusted_qwk'],m['special_category_recognition'])
        if best is None or key > best[0]:
            best = (key,p,pred,m)
    return {'params':best[1], 'prediction':best[2], 'metrics':best[3]}


def _augment_with_concepts_v2(Xtr, Xte, train_df, test_df, concept_targets, seed):
    Ztr, znames, concept_full, order = inner_oof_concepts(
        Xtr, train_df, concept_targets, seed=seed, n_splits=3
    )
    Zte, _ = predict_concepts(concept_full, Xte, order)
    size_col = 'geom_max_bbox_diameter_mm' if 'geom_max_bbox_diameter_mm' in train_df.columns else 'max_diameter_mm'
    strn = pd.to_numeric(train_df[size_col], errors='coerce').to_numpy()
    stst = pd.to_numeric(test_df[size_col], errors='coerce').to_numpy()
    Btr, bnames = structured_lirads_basis_v2(Ztr, znames, strn)
    Bte, _ = structured_lirads_basis_v2(Zte, znames, stst)
    Atr = np.column_stack([Xtr,Ztr,Btr]).astype(np.float32)
    Ate = np.column_stack([Xte,Zte,Bte]).astype(np.float32)
    return Atr,Ate,Ztr,Zte,znames,bnames,concept_full,order,strn,stst


def cross_validate_phora_plus_v2(data, feature_cols, n_splits=3, seed=42, screen_k=140,
                                 ensemble_seeds=(17,37,61), router_trials=4000,
                                 use_concepts=True, use_factor_gate=True,
                                 use_all_threshold=True, use_screen=True,
                                 use_clinical_constraints=True):
    data = data[data.lirads_score.isin(VALID_LABELS)].reset_index(drop=True).copy()
    y = data.lirads_score.astype(str).to_numpy(); n = len(data)
    oof = np.empty(n,dtype=object)
    oof_gd=np.zeros((n,3)); oof_gf=np.zeros((n,3)); oof_reg=np.zeros(n); oof_ats=np.zeros(n)
    oof_pA=np.full(n,np.nan); oof_pR=np.full(n,np.nan)
    selected_by_fold=[]; router_by_fold=[]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strata = safe_strata(y,n_splits)
    for f,(tr,va) in enumerate(skf.split(np.zeros(n),strata),start=1):
        train=data.iloc[tr].reset_index(drop=True); val=data.iloc[va].reset_index(drop=True)
        selected=nested_screen(train,feature_cols,k=screen_k,seed=seed+f) if use_screen else list(feature_cols)
        selected_by_fold.append(selected)
        Xtr=_numeric_matrix(train,selected); Xva=_numeric_matrix(val,selected)
        size_col='geom_max_bbox_diameter_mm' if 'geom_max_bbox_diameter_mm' in train.columns else 'max_diameter_mm'
        size_tr=pd.to_numeric(train[size_col],errors='coerce').to_numpy()
        size_va=pd.to_numeric(val[size_col],errors='coerce').to_numpy()
        if use_concepts:
            Atr,Ava,Ztr,Zva,znames,_,_,_,size_tr,size_va = _augment_with_concepts_v2(
                Xtr,Xva,train,val,CONCEPT_TARGETS_V2,seed+100*f
            )
            pA_tr=_latent_col(Ztr,znames,'aphe',.5); pR_tr=_latent_col(Ztr,znames,'rim_aphe',.1)
            pA_va=_latent_col(Zva,znames,'aphe',.5); pR_va=_latent_col(Zva,znames,'rim_aphe',.1)
        else:
            Atr=Xtr.astype(np.float32); Ava=Xva.astype(np.float32)
            pA_tr=np.full(len(train),1.0); pR_tr=np.zeros(len(train))
            pA_va=np.full(len(val),1.0); pR_va=np.zeros(len(val))

        igd,igf,ireg,iats=inner_router_oof(
            Atr,y[tr],n_splits=3,seeds=ensemble_seeds,seed=seed+200*f,
            use_factor_gate=use_factor_gate,use_all_threshold=use_all_threshold
        )
        if use_clinical_constraints:
            tuned=tune_router_v2(y[tr],igd,igf,ireg,iats,pA_tr,pR_tr,size_tr,
                                 n_trials=router_trials,seed=seed+300*f)
        else:
            tuned=tune_router(y[tr],igd,igf,ireg,iats,n_trials=router_trials,seed=seed+300*f)
        router_by_fold.append(tuned['params'])

        ss=tuple(s+5000*f for s in ensemble_seeds)
        dm=fit_direct_gate_ensemble(Atr,y[tr],seeds=ss)
        fm=fit_factor_gate_ensemble(Atr,y[tr],seeds=tuple(s+10 for s in ss)) if use_factor_gate else None
        om=fit_ordinal_ensemble(Atr,y[tr],seeds=tuple(s+20 for s in ss))
        gd=predict_direct_gate(dm,Ava)
        gf=predict_factor_gate(fm,Ava) if use_factor_gate else gd.copy()
        reg,ats0,_=predict_ordinal_ensemble(om,Ava); ats=ats0 if use_all_threshold else reg.copy()
        if use_clinical_constraints:
            pred=labels_from_router_v2(gd,gf,reg,ats,pA_va,pR_va,size_va,tuned['params'])
        else:
            pred=labels_from_router(gd,gf,reg,ats,tuned['params'])
        oof[va]=pred; oof_gd[va]=gd; oof_gf[va]=gf; oof_reg[va]=reg; oof_ats[va]=ats
        oof_pA[va]=pA_va; oof_pR[va]=pR_va
        print(f'fold {f}: selected={len(selected)} inner={tuned["metrics"]["final_score"]:.4f} outer={fast_challenge_score(y[va],pred)["final_score"]:.4f}')
    metrics=fast_challenge_score(y,oof)
    out=pd.DataFrame({'case_id':data.case_id,'lirads_score':y,'prediction':oof,
                      'ord_reg_score':oof_reg,'ord_AT_score':oof_ats,
                      'p_concept_nonrim_aphe':oof_pA,'p_concept_rim_aphe':oof_pR})
    out[['p_direct_ord','p_direct_M','p_direct_TIV']]=oof_gd
    out[['p_factor_ord','p_factor_M','p_factor_TIV']]=oof_gf
    return {'metrics':metrics,'oof':out,'selected_by_fold':selected_by_fold,'router_by_fold':router_by_fold}


def fit_predict_phora_plus_holdout_v2(train_df, test_df, feature_cols, seed=42, screen_k=140,
                                      ensemble_seeds=(17,37,61), router_trials=5000,
                                      use_concepts=True, use_factor_gate=True,
                                      use_all_threshold=True, use_screen=True,
                                      use_clinical_constraints=True):
    train=train_df[train_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True).copy()
    test=test_df[test_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True).copy()
    ytr=train.lirads_score.astype(str).to_numpy(); yte=test.lirads_score.astype(str).to_numpy()
    selected=nested_screen(train,feature_cols,k=screen_k,seed=seed) if use_screen else list(feature_cols)
    Xtr=_numeric_matrix(train,selected); Xte=_numeric_matrix(test,selected)
    size_col='geom_max_bbox_diameter_mm' if 'geom_max_bbox_diameter_mm' in train.columns else 'max_diameter_mm'
    size_tr=pd.to_numeric(train[size_col],errors='coerce').to_numpy()
    size_te=pd.to_numeric(test[size_col],errors='coerce').to_numpy()
    concept_meta={}
    if use_concepts:
        Atr,Ate,Ztr,Zte,znames,bnames,concept_full,order,size_tr,size_te = _augment_with_concepts_v2(
            Xtr,Xte,train,test,CONCEPT_TARGETS_V2,seed+100
        )
        pA_tr=_latent_col(Ztr,znames,'aphe',.5); pR_tr=_latent_col(Ztr,znames,'rim_aphe',.1)
        pA_te=_latent_col(Zte,znames,'aphe',.5); pR_te=_latent_col(Zte,znames,'rim_aphe',.1)
        concept_meta={'models':concept_full,'order':order,'names':znames,'basis_names':bnames}
    else:
        Atr=Xtr.astype(np.float32); Ate=Xte.astype(np.float32)
        pA_tr=np.ones(len(train)); pR_tr=np.zeros(len(train)); pA_te=np.ones(len(test)); pR_te=np.zeros(len(test))

    igd,igf,ireg,iats=inner_router_oof(Atr,ytr,n_splits=3,seeds=ensemble_seeds,seed=seed+200,
                                       use_factor_gate=use_factor_gate,use_all_threshold=use_all_threshold)
    if use_clinical_constraints:
        tuned=tune_router_v2(ytr,igd,igf,ireg,iats,pA_tr,pR_tr,size_tr,n_trials=router_trials,seed=seed+300)
    else:
        tuned=tune_router(ytr,igd,igf,ireg,iats,n_trials=router_trials,seed=seed+300)

    dm=fit_direct_gate_ensemble(Atr,ytr,seeds=ensemble_seeds)
    fm=fit_factor_gate_ensemble(Atr,ytr,seeds=tuple(s+10 for s in ensemble_seeds)) if use_factor_gate else None
    om=fit_ordinal_ensemble(Atr,ytr,seeds=tuple(s+20 for s in ensemble_seeds))
    gd=predict_direct_gate(dm,Ate); gf=predict_factor_gate(fm,Ate) if use_factor_gate else gd.copy()
    reg,ats,pord=predict_ordinal_ensemble(om,Ate)
    if not use_all_threshold: ats=reg.copy()
    if use_clinical_constraints:
        pred=labels_from_router_v2(gd,gf,reg,ats,pA_te,pR_te,size_te,tuned['params'])
    else:
        pred=labels_from_router(gd,gf,reg,ats,tuned['params'])

    # Useful 7-class soft distribution for analysis (ATS ordinal distribution + soft gate).
    beta=float(tuned['params'][0]); g=beta*gd+(1-beta)*gf; g=np.clip(g,EPS,None)
    if use_clinical_constraints and pR_te is not None:
        gammaR=float(tuned['params'][-1]); g[:,1]*=np.exp(gammaR*(pR_te-.5))
    g/=g.sum(axis=1,keepdims=True)+EPS
    p7=np.column_stack([g[:,0,None]*pord, g[:,1], g[:,2]])
    p7/=p7.sum(axis=1,keepdims=True)+EPS

    out=pd.DataFrame({'case_id':test.case_id,'lirads_score':yte,'prediction':pred,
                      'ord_reg_score':reg,'ord_AT_score':ats,
                      'p_concept_nonrim_aphe':pA_te,'p_concept_rim_aphe':pR_te})
    out[['p_direct_ord','p_direct_M','p_direct_TIV']]=gd
    out[['p_factor_ord','p_factor_M','p_factor_TIV']]=gf
    for j,lab in enumerate(VALID_LABELS): out[f'p7_{lab}']=p7[:,j]
    if use_concepts:
        for j,nm in enumerate(znames): out[nm]=Zte[:,j]
    models={'selected':selected,'concept':concept_meta,'direct_gate':dm,'factor_gate':fm,'ordinal':om,
            'router':tuned['params'],'use_concepts':use_concepts,'use_factor_gate':use_factor_gate,
            'use_all_threshold':use_all_threshold,'use_clinical_constraints':use_clinical_constraints}
    return {'metrics':fast_challenge_score(yte,pred),'prediction':out,'selected_features':selected,
            'router':tuned,'models':models}


def run_holdout_baselines_v2(train_df, test_df, feature_cols, screen_k=140, seed=42):
    """Strong non-hierarchical baselines trained only on train_df."""
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    train=train_df[train_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True)
    test=test_df[test_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True)
    selected=nested_screen(train,feature_cols,k=screen_k,seed=seed)
    Xtr=_numeric_matrix(train,selected); Xte=_numeric_matrix(test,selected)
    ytr=train.lirads_score.astype(str).to_numpy(); yte=test.lirads_score.astype(str).to_numpy()
    enc={c:i for i,c in enumerate(VALID_LABELS)}; yi=np.array([enc[z] for z in ytr],int)
    models={
        'Elastic-net logistic': Pipeline([
            ('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),
            ('clf',LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.2,C=.25,
                                      class_weight='balanced',max_iter=6000,random_state=seed))]),
        'RBF-SVM': Pipeline([
            ('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),
            ('clf',SVC(C=1.5,gamma='scale',class_weight='balanced',probability=True,random_state=seed))]),
        'Random Forest': Pipeline([
            ('imp',SimpleImputer(strategy='median',add_indicator=True)),
            ('clf',RandomForestClassifier(n_estimators=900,max_features=.45,min_samples_leaf=2,
                                          class_weight='balanced_subsample',n_jobs=-1,random_state=seed))]),
        'Extra Trees': Pipeline([
            ('imp',SimpleImputer(strategy='median',add_indicator=True)),
            ('clf',ExtraTreesClassifier(n_estimators=1000,max_features=.5,min_samples_leaf=2,
                                        class_weight='balanced',n_jobs=-1,random_state=seed))]),
        'HistGradientBoosting': Pipeline([
            ('imp',SimpleImputer(strategy='median',add_indicator=True)),
            ('clf',HistGradientBoostingClassifier(learning_rate=.045,max_iter=350,max_leaf_nodes=15,
                                                   l2_regularization=3.0,random_state=seed))]),
    }
    rows=[]; preds={}
    # Majority baseline is intentionally trivial.
    maj=pd.Series(ytr).value_counts().idxmax(); pmaj=np.array([maj]*len(test),object)
    m=fast_challenge_score(yte,pmaj); rows.append({'Method':'Majority class',**m,'accuracy':accuracy_score(yte,pmaj),'macro_f1':f1_score(yte,pmaj,labels=VALID_LABELS,average='macro',zero_division=0)})
    preds['Majority class']=pd.DataFrame({'case_id':test.case_id,'lirads_score':yte,'prediction':pmaj})
    for name,model in models.items():
        try:
            model.fit(Xtr,yi,clf__sample_weight=balanced_weights(yi,power=.65))
        except TypeError:
            model.fit(Xtr,yi)
        pp=model.predict(Xte).astype(int); pred=np.array([VALID_LABELS[i] for i in pp],object)
        m=fast_challenge_score(yte,pred); rows.append({'Method':name,**m,'accuracy':accuracy_score(yte,pred),'macro_f1':f1_score(yte,pred,labels=VALID_LABELS,average='macro',zero_division=0)})
        preds[name]=pd.DataFrame({'case_id':test.case_id,'lirads_score':yte,'prediction':pred})
    xm=xgb_classifier(7,seed+77,n_estimators=520,depth=3,lr=.022)
    xm.fit(Xtr,yi,sample_weight=balanced_weights(yi,power=.68))
    pp=xm.predict(Xte).astype(int); pred=np.array([VALID_LABELS[i] for i in pp],object)
    m=fast_challenge_score(yte,pred); rows.append({'Method':'Flat XGBoost',**m,'accuracy':accuracy_score(yte,pred),'macro_f1':f1_score(yte,pred,labels=VALID_LABELS,average='macro',zero_division=0)})
    preds['Flat XGBoost']=pd.DataFrame({'case_id':test.case_id,'lirads_score':yte,'prediction':pred})
    return pd.DataFrame(rows).sort_values('final_score',ascending=False),preds,selected


def feature_groups_v2(feature_cols):
    cols=list(feature_cols)
    groups={
        'Morphology': [c for c in cols if c.startswith(('geom_','spectral_','topology_','fractal_','shape_'))],
        'Custom kinetics': [c for c in cols if c.startswith(('phase_','shell_','delta_','traj_'))],
        'Spatial fields': [c for c in cols if c.startswith('spatial_') or '_spatial_' in c],
        'Context + OT': [c for c in cols if c.startswith(('ctx_','ot_','phys_'))],
        'Handcrafted radiomics': [c for c in cols if any(tag in c for tag in ('_original_firstorder_','_original_glcm_','_original_glrlm_','_original_glszm_','_original_gldm_','_original_ngtdm_'))],
    }
    return {k:list(dict.fromkeys(v)) for k,v in groups.items() if len(v)>0}


def multiview_stack_holdout_v2(train_df, test_df, feature_cols, n_splits=3, seed=42, per_view_k=60):
    """Cross-fitted late-fusion baseline used for the final paper experiments.

    One XGBoost expert is fit per biological feature view. Out-of-fold class
    probabilities are concatenated and augmented by cross-view mean/std
    probabilities, then fused with a class-balanced logistic meta-learner.
    """
    from sklearn.metrics import accuracy_score, f1_score

    train = train_df[train_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True)
    test = test_df[test_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True)
    y = train["lirads_score"].astype(str).to_numpy()
    yte = test["lirads_score"].astype(str).to_numpy()
    label_to_int = {c: i for i, c in enumerate(VALID_LABELS)}
    yi = np.array([label_to_int[z] for z in y], dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = feature_groups_v2(feature_cols)
    oof_blocks, test_blocks, used_groups = [], [], {}

    for view_name, cols in groups.items():
        cols = [c for c in cols if c in train.columns]
        if not cols:
            continue
        selected = nested_screen(train, cols, k=min(per_view_k, len(cols)), seed=seed)
        if not selected:
            continue
        used_groups[view_name] = selected
        X = _numeric_matrix(train, selected)
        Xt = _numeric_matrix(test, selected)
        oof_prob = np.zeros((len(train), len(VALID_LABELS)), dtype=np.float32)
        test_prob = np.zeros((len(test), len(VALID_LABELS)), dtype=np.float32)

        for fold, (tr, va) in enumerate(skf.split(X, yi)):
            model = xgb_classifier(n_classes=len(VALID_LABELS), seed=seed + fold + 100)
            model.fit(X[tr], yi[tr], sample_weight=balanced_weights(yi[tr], power=.60))
            oof_prob[va] = model.predict_proba(X[va])
            test_prob += model.predict_proba(Xt) / n_splits

        oof_blocks.append(oof_prob)
        test_blocks.append(test_prob)

    if not oof_blocks:
        raise ValueError('No non-empty feature views were found.')

    Ztr = np.column_stack(oof_blocks)
    Zte = np.column_stack(test_blocks)
    if len(oof_blocks) > 1:
        mean_tr = np.mean(np.stack(oof_blocks, axis=0), axis=0)
        mean_te = np.mean(np.stack(test_blocks, axis=0), axis=0)
        std_tr = np.std(np.stack(oof_blocks, axis=0), axis=0)
        std_te = np.std(np.stack(test_blocks, axis=0), axis=0)
        Ztr = np.column_stack([Ztr, mean_tr, std_tr])
        Zte = np.column_stack([Zte, mean_te, std_te])

    meta = LogisticRegression(C=.35, class_weight='balanced', max_iter=5000,
                              solver='lbfgs', random_state=seed)
    meta.fit(Ztr, yi, sample_weight=balanced_weights(yi, power=.55))
    prob = meta.predict_proba(Zte)
    pp = np.argmax(prob, axis=1)
    pred = np.array([VALID_LABELS[i] for i in pp], dtype=object)
    out = pd.DataFrame({'case_id': test.case_id.astype(str), 'lirads_score': yte, 'prediction': pred})
    for j, label in enumerate(VALID_LABELS):
        out[f'p_{label}'] = prob[:, j]

    metrics = fast_challenge_score(yte, pred)
    metrics['accuracy'] = accuracy_score(yte, pred)
    metrics['macro_f1'] = f1_score(yte, pred, labels=VALID_LABELS, average='macro', zero_division=0)
    return {'metrics': metrics, 'prediction': out, 'groups': used_groups,
            'meta_model': meta, 'train_stack': Ztr, 'test_stack': Zte}

def oracle_concept_holdout_v2(train_df,test_df,seed=42):
    """Upper-bound analysis using radiologist metadata concepts. NOT inference-valid."""
    from sklearn.metrics import accuracy_score, f1_score
    cols=['max_diameter_mm','aphe','rim_aphe','washout_venous','washout_delayed','capsule_venous','capsule_delayed']
    tr=train_df[train_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True); te=test_df[test_df.lirads_score.isin(VALID_LABELS)].reset_index(drop=True)
    Xtr=_numeric_matrix(tr,cols); Xte=_numeric_matrix(te,cols)
    enc={c:i for i,c in enumerate(VALID_LABELS)}; yi=np.array([enc[z] for z in tr.lirads_score],int)
    m=xgb_classifier(7,seed,n_estimators=260,depth=3,lr=.035)
    m.fit(Xtr,yi,sample_weight=balanced_weights(yi,power=.65))
    pp=m.predict(Xte).astype(int); pred=np.array([VALID_LABELS[i] for i in pp],object)
    gt=te.lirads_score.to_numpy(); met=fast_challenge_score(gt,pred); met['accuracy']=accuracy_score(gt,pred); met['macro_f1']=f1_score(gt,pred,labels=VALID_LABELS,average='macro',zero_division=0)
    return {'metrics':met,'prediction':pd.DataFrame({'case_id':te.case_id,'lirads_score':gt,'prediction':pred})}


def fit_phora_plus_full_v2(data, feature_cols, seed=42, screen_k=140, ensemble_seeds=(17,37,61), router_trials=6000):
    """Fit the frozen PHORA+ v2 method on all labeled lesion cases. Returns Python model objects."""
    d=data[data.lirads_score.isin(VALID_LABELS)].reset_index(drop=True).copy(); y=d.lirads_score.astype(str).to_numpy()
    selected=nested_screen(d,feature_cols,k=screen_k,seed=seed); X=_numeric_matrix(d,selected)
    Z,znames,concept_full,order=inner_oof_concepts(X,d,CONCEPT_TARGETS_V2,seed=seed+100,n_splits=3)
    size_col='geom_max_bbox_diameter_mm' if 'geom_max_bbox_diameter_mm' in d.columns else 'max_diameter_mm'; size=pd.to_numeric(d[size_col],errors='coerce').to_numpy()
    B,bnames=structured_lirads_basis_v2(Z,znames,size); A=np.column_stack([X,Z,B]).astype(np.float32)
    pA=_latent_col(Z,znames,'aphe',.5); pR=_latent_col(Z,znames,'rim_aphe',.1)
    gd,gf,reg,ats=inner_router_oof(A,y,n_splits=3,seeds=ensemble_seeds,seed=seed+200,use_factor_gate=True,use_all_threshold=True)
    tuned=tune_router_v2(y,gd,gf,reg,ats,pA,pR,size,n_trials=router_trials,seed=seed+300)
    dm=fit_direct_gate_ensemble(A,y,seeds=ensemble_seeds); fm=fit_factor_gate_ensemble(A,y,seeds=tuple(s+10 for s in ensemble_seeds)); om=fit_ordinal_ensemble(A,y,seeds=tuple(s+20 for s in ensemble_seeds))
    return {'selected':selected,'concept_models':concept_full,'concept_order':order,'latent_names':znames,'basis_names':bnames,
            'direct_gate':dm,'factor_gate':fm,'ordinal':om,'router':tuned['params'],'size_col':size_col,'training_metrics_inner':tuned['metrics']}


def predict_phora_plus_full_v2(model, df):
    d=df.reset_index(drop=True).copy(); X=_numeric_matrix(d,model['selected'])
    Z,znames=predict_concepts(model['concept_models'],X,model['concept_order'])
    size=pd.to_numeric(d[model['size_col']],errors='coerce').to_numpy(); B,_=structured_lirads_basis_v2(Z,znames,size); A=np.column_stack([X,Z,B]).astype(np.float32)
    gd=predict_direct_gate(model['direct_gate'],A); gf=predict_factor_gate(model['factor_gate'],A); reg,ats,_=predict_ordinal_ensemble(model['ordinal'],A)
    pA=_latent_col(Z,znames,'aphe',.5); pR=_latent_col(Z,znames,'rim_aphe',.1)
    return labels_from_router_v2(gd,gf,reg,ats,pA,pR,size,model['router'])

# ---- burden-aware overrides (training-only spatial annotation distillation) ----
def fit_burden_heads_v2(X, df, burden_cols, seed=42, n_estimators=220):
    models={}
    for j,t in enumerate(burden_cols):
        if t not in df.columns: continue
        yy=pd.to_numeric(df[t],errors='coerce').to_numpy(float)
        ok=np.isfinite(yy)
        if ok.sum()<25 or np.nanstd(yy[ok])<1e-5: continue
        m=xgb_regressor(seed+37*j,n_estimators=n_estimators,depth=3,lr=.025)
        m.fit(X[ok],yy[ok])
        models[t]=m
    return models


def predict_burdens_v2(models,X,order):
    arr=[]; names=[]
    for t in order:
        if t in models:
            arr.append(np.clip(models[t].predict(X),0,1)); names.append('latent_'+t)
    return (np.column_stack(arr).astype(np.float32) if arr else np.zeros((len(X),0),np.float32)), names


def inner_oof_burdens_v2(X,df,burden_cols,seed=42,n_splits=3):
    full=fit_burden_heads_v2(X,df,burden_cols,seed)
    order=[t for t in burden_cols if t in full]
    Z=np.full((len(X),len(order)),np.nan,np.float32)
    if not order:
        return Z,[],full,order
    strata=safe_strata(df.lirads_score.astype(str).to_numpy(),n_splits)
    kf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for f,(tr,va) in enumerate(kf.split(X,strata)):
        mm=fit_burden_heads_v2(X[tr],df.iloc[tr],order,seed+100+f,n_estimators=180)
        z,_=predict_burdens_v2(mm,X[va],order)
        if z.shape[1]!=len(order): z,_=predict_burdens_v2(full,X[va],order)
        Z[va]=z
    return Z,['latent_'+t for t in order],full,order


def structured_lirads_basis_v2(Z, names, size_mm):
    """Soft LI-RADS basis using predicted binary concepts and predicted spatial burdens."""
    mp={n:Z[:,i] for i,n in enumerate(names)}; n=len(size_mm)
    def g(nm,default=.5):
        v=mp.get(nm)
        return np.full(n,default,float) if v is None else np.nan_to_num(v,nan=default)
    pA=g('latent_prob_aphe'); pR=g('latent_prob_rim_aphe',.15)
    pWV=g('latent_prob_washout_venous'); pWD=g('latent_prob_washout_delayed')
    pCV=g('latent_prob_capsule_venous'); pCD=g('latent_prob_capsule_delayed')
    bA=g('latent_burden_aphe',.25); bWV=g('latent_burden_washout_venous',.20); bWD=g('latent_burden_washout_delayed',.20)
    bCV=g('latent_burden_capsule_venous',.10); bCD=g('latent_burden_capsule_delayed',.10)
    pW=1-(1-pWV)*(1-pWD); pC=1-(1-pCV)*(1-pCD)
    bW=np.maximum(bWV,bWD); bC=np.maximum(bCV,bCD)
    d=np.asarray(size_mm,float); med=np.nanmedian(d) if np.isfinite(d).any() else 30.0; d=np.nan_to_num(d,nan=med)
    s10=sigmoid((d-10)/2.5); s20=sigmoid((d-20)/3.5); mid=s10*(1-s20)
    major_count=pW+pC
    lr5_plaus=np.clip(pA*(s20*(1-(1-pW)*(1-pC)) + mid*pW),0,1)
    lr4_plaus=np.clip(pA*(1-lr5_plaus)+(1-pA)*.60*(pW+pC),0,1)
    lrm_plaus=np.clip(pR*(.55+.45*(1-pA)),0,1)
    burden_coherence=np.clip(.5*bA+.3*bW+.2*bC,0,1)
    basis=np.column_stack([
        pA,pR,pW,pC,major_count,s10,s20,pA*pW,pA*pC,pW*pC,
        pA*s10,pA*s20,pR*(1-pA),pR*pW,lr4_plaus,lr5_plaus,lrm_plaus,
        bA,bW,bC,burden_coherence,pA*bA,pW*bW,pC*bC
    ]).astype(np.float32)
    bnames=['rule_pNonRimAPHE','rule_pRimAPHE','rule_pWashout','rule_pCapsule','rule_expected_major_count',
            'rule_size10_soft','rule_size20_soft','rule_APHE_x_W','rule_APHE_x_C','rule_W_x_C',
            'rule_APHE_x_size10','rule_APHE_x_size20','rule_Rim_x_noAPHE','rule_Rim_x_W','rule_LR4_plaus',
            'rule_LR5_plaus','rule_LRM_plaus','rule_burden_APHE','rule_burden_Washout','rule_burden_Capsule',
            'rule_burden_coherence','rule_pA_x_bA','rule_pW_x_bW','rule_pC_x_bC']
    return basis,bnames


def _augment_with_concepts_v2(Xtr, Xte, train_df, test_df, concept_targets, seed):
    Zc_tr,cnames,concept_full,cord=inner_oof_concepts(Xtr,train_df,concept_targets,seed=seed,n_splits=3)
    Zc_te,_=predict_concepts(concept_full,Xte,cord)
    burden_cols=[c for c in train_df.columns if c.startswith('burden_')]
    Zb_tr,bnames_lat,burden_full,bord=inner_oof_burdens_v2(Xtr,train_df,burden_cols,seed=seed+51,n_splits=3)
    Zb_te,_=predict_burdens_v2(burden_full,Xte,bord)
    Ztr=np.column_stack([Zc_tr,Zb_tr]).astype(np.float32); Zte=np.column_stack([Zc_te,Zb_te]).astype(np.float32)
    znames=list(cnames)+list(bnames_lat)
    size_col='geom_max_bbox_diameter_mm' if 'geom_max_bbox_diameter_mm' in train_df.columns else 'max_diameter_mm'
    strn=pd.to_numeric(train_df[size_col],errors='coerce').to_numpy(); stst=pd.to_numeric(test_df[size_col],errors='coerce').to_numpy()
    Btr,bnames=structured_lirads_basis_v2(Ztr,znames,strn); Bte,_=structured_lirads_basis_v2(Zte,znames,stst)
    Atr=np.column_stack([Xtr,Ztr,Btr]).astype(np.float32); Ate=np.column_stack([Xte,Zte,Bte]).astype(np.float32)
    meta={'concept_models':concept_full,'concept_order':cord,'burden_models':burden_full,'burden_order':bord}
    return Atr,Ate,Ztr,Zte,znames,bnames,meta,cord,strn,stst

# Re-override full fit/predict to retain burden heads.
def fit_phora_plus_full_v2(data, feature_cols, seed=42, screen_k=140, ensemble_seeds=(17,37,61), router_trials=6000):
    d=data[data.lirads_score.isin(VALID_LABELS)].reset_index(drop=True).copy(); y=d.lirads_score.astype(str).to_numpy()
    selected=nested_screen(d,feature_cols,k=screen_k,seed=seed); X=_numeric_matrix(d,selected)
    # use a self-prediction target container solely to trigger cross-fitted latent creation
    A,_,Z,_,znames,bnames,meta,_,size,_=_augment_with_concepts_v2(X,X,d,d,CONCEPT_TARGETS_V2,seed+100)
    pA=_latent_col(Z,znames,'aphe',.5); pR=_latent_col(Z,znames,'rim_aphe',.1)
    gd,gf,reg,ats=inner_router_oof(A,y,n_splits=3,seeds=ensemble_seeds,seed=seed+200,use_factor_gate=True,use_all_threshold=True)
    tuned=tune_router_v2(y,gd,gf,reg,ats,pA,pR,size,n_trials=router_trials,seed=seed+300)
    dm=fit_direct_gate_ensemble(A,y,seeds=ensemble_seeds); fm=fit_factor_gate_ensemble(A,y,seeds=tuple(s+10 for s in ensemble_seeds)); om=fit_ordinal_ensemble(A,y,seeds=tuple(s+20 for s in ensemble_seeds))
    size_col='geom_max_bbox_diameter_mm' if 'geom_max_bbox_diameter_mm' in d.columns else 'max_diameter_mm'
    return {'selected':selected,'concept_models':meta['concept_models'],'concept_order':meta['concept_order'],
            'burden_models':meta['burden_models'],'burden_order':meta['burden_order'],'latent_names':znames,'basis_names':bnames,
            'direct_gate':dm,'factor_gate':fm,'ordinal':om,'router':tuned['params'],'size_col':size_col,'training_metrics_inner':tuned['metrics']}


def predict_phora_plus_full_v2(model, df):
    d=df.reset_index(drop=True).copy(); X=_numeric_matrix(d,model['selected'])
    Zc,cnames=predict_concepts(model['concept_models'],X,model['concept_order']); Zb,bnames_lat=predict_burdens_v2(model.get('burden_models',{}),X,model.get('burden_order',[]))
    Z=np.column_stack([Zc,Zb]).astype(np.float32); znames=list(cnames)+list(bnames_lat)
    size=pd.to_numeric(d[model['size_col']],errors='coerce').to_numpy(); B,_=structured_lirads_basis_v2(Z,znames,size); A=np.column_stack([X,Z,B]).astype(np.float32)
    gd=predict_direct_gate(model['direct_gate'],A); gf=predict_factor_gate(model['factor_gate'],A); reg,ats,_=predict_ordinal_ensemble(model['ordinal'],A)
    pA=_latent_col(Z,znames,'aphe',.5); pR=_latent_col(Z,znames,'rim_aphe',.1)
    return labels_from_router_v2(gd,gf,reg,ats,pA,pR,size,model['router'])

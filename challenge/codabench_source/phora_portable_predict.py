from __future__ import annotations
import gzip,pickle
import numpy as np
from portable_xgb import predict as xgb_predict

ORDINAL_LABELS=["LR-1","LR-2","LR-3","LR-4","LR-5"]
EPS=1e-8

def sigmoid(z):
    z=np.clip(np.asarray(z,float),-30,30); return 1/(1+np.exp(-z))

def predict_concepts(models,X,order):
    arr=[]; names=[]
    for t in order:
        if t in models:
            arr.append(xgb_predict(models[t],X)); names.append('latent_prob_'+t)
    return (np.column_stack(arr).astype(np.float32) if arr else np.zeros((len(X),0),np.float32)),names

def predict_burdens(models,X,order):
    arr=[]; names=[]
    for t in order:
        if t in models:
            arr.append(np.clip(xgb_predict(models[t],X),0,1)); names.append('latent_'+t)
    return (np.column_stack(arr).astype(np.float32) if arr else np.zeros((len(X),0),np.float32)),names

def _latent_col(Z,names,target,default=.5):
    nm='latent_prob_'+target
    if nm not in names: return np.full(len(Z),default,float)
    return np.nan_to_num(Z[:,names.index(nm)].astype(float),nan=default)

def structured_basis(Z,names,size_mm):
    mp={n:Z[:,i] for i,n in enumerate(names)}; n=len(size_mm)
    def g(nm,default=.5):
        v=mp.get(nm); return np.full(n,default,float) if v is None else np.nan_to_num(v,nan=default)
    pA=g('latent_prob_aphe'); pR=g('latent_prob_rim_aphe',.15)
    pWV=g('latent_prob_washout_venous'); pWD=g('latent_prob_washout_delayed')
    pCV=g('latent_prob_capsule_venous'); pCD=g('latent_prob_capsule_delayed')
    bA=g('latent_burden_aphe',.25); bWV=g('latent_burden_washout_venous',.20); bWD=g('latent_burden_washout_delayed',.20)
    bCV=g('latent_burden_capsule_venous',.10); bCD=g('latent_burden_capsule_delayed',.10)
    pW=1-(1-pWV)*(1-pWD); pC=1-(1-pCV)*(1-pCD)
    bW=np.maximum(bWV,bWD); bC=np.maximum(bCV,bCD)
    d=np.asarray(size_mm,float); med=np.nanmedian(d) if np.isfinite(d).any() else 30.; d=np.nan_to_num(d,nan=med)
    s10=sigmoid((d-10)/2.5); s20=sigmoid((d-20)/3.5); mid=s10*(1-s20)
    major_count=pW+pC
    lr5=np.clip(pA*(s20*(1-(1-pW)*(1-pC))+mid*pW),0,1)
    lr4=np.clip(pA*(1-lr5)+(1-pA)*.60*(pW+pC),0,1)
    lrm=np.clip(pR*(.55+.45*(1-pA)),0,1)
    bc=np.clip(.5*bA+.3*bW+.2*bC,0,1)
    return np.column_stack([pA,pR,pW,pC,major_count,s10,s20,pA*pW,pA*pC,pW*pC,
                            pA*s10,pA*s20,pR*(1-pA),pR*pW,lr4,lr5,lrm,
                            bA,bW,bC,bc,pA*bA,pW*bW,pC*bC]).astype(np.float32)

def direct_gate(models,X):
    return np.mean([xgb_predict(m,X) for m in models],axis=0)

def factor_gate(models,X):
    sp,tiv=models
    ps=np.mean([xgb_predict(m,X) for m in sp],axis=0)
    pt=np.mean([xgb_predict(m,X) for m in tiv],axis=0)
    out=np.column_stack([1-ps,ps*(1-pt),ps*pt]); return out/(out.sum(axis=1,keepdims=True)+EPS)

def all_threshold(model,X):
    X=np.asarray(X,np.float32); n=len(X)
    thresholds=np.tile(np.arange(1,5),n)
    Xrep=np.repeat(X,4,axis=0); t=(thresholds/4.).reshape(-1,1).astype(np.float32)
    Xexp=np.column_stack([Xrep,t,t*t]).astype(np.float32)
    q=xgb_predict(model,Xexp).reshape(n,4)
    q=np.minimum.accumulate(q,axis=1); q=np.clip(q,0,1)
    p=np.column_stack([1-q[:,0],q[:,0]-q[:,1],q[:,1]-q[:,2],q[:,2]-q[:,3],q[:,3]])
    p=np.clip(p,0,None); p/=p.sum(axis=1,keepdims=True)+EPS
    score=(p*np.arange(1,6)[None,:]).sum(axis=1)
    return score,p

def ordinal_ensemble(models,X):
    regs,ats=models
    reg=np.mean([np.clip(xgb_predict(m,X),.5,5.5) for m in regs],axis=0)
    ss=[]; pp=[]
    for m in ats:
        s,p=all_threshold(m,X); ss.append(s); pp.append(p)
    return reg,np.mean(ss,axis=0),np.mean(pp,axis=0)

def labels_router(gd,gf,reg,ats,p_aphe,p_rim,size_mm,params):
    beta,alpha,c1,c2,c3,c4,tM,tT,margin,tA,gammaR=params
    g=beta*gd+(1-beta)*gf; g=np.clip(g,EPS,None)
    if p_rim is not None: g[:,1]*=np.exp(gammaR*(np.asarray(p_rim,float)-.5))
    g/=g.sum(axis=1,keepdims=True)+EPS
    score=alpha*reg+(1-alpha)*ats; cps=np.array([c1,c2,c3,c4],float)
    k=1+(score[:,None]>cps[None,:]).sum(axis=1)
    out=np.array([f'LR-{int(v)}' for v in np.clip(k,1,5)],object)
    pord,pm,pt=g[:,0],g[:,1],g[:,2]
    chooseM=(pm>=pt)&(pm>=tM)&(pm>=pord+margin)
    chooseT=(pt>pm)&(pt>=tT)&(pt>=pord+margin)
    out[chooseM]='LR-M'; out[chooseT]='LR-TIV'
    ord_mask=np.isin(out,ORDINAL_LABELS); size=np.asarray(size_mm,float)
    too_small=ord_mask&(out=='LR-5')&np.isfinite(size)&(size<10.0); out[too_small]='LR-4'
    if p_aphe is not None:
        veto=ord_mask&(out=='LR-5')&(np.asarray(p_aphe,float)<tA); out[veto]='LR-4'
    return out

def load_portable(path):
    with gzip.open(path,'rb') as f: return pickle.load(f)

def predict_from_matrix(model,X,size):
    X=np.asarray(X,np.float32)
    Zc,cn=predict_concepts(model['concept_models'],X,model['concept_order'])
    Zb,bn=predict_burdens(model.get('burden_models',{}),X,model.get('burden_order',[]))
    Z=np.column_stack([Zc,Zb]).astype(np.float32); names=list(cn)+list(bn)
    B=structured_basis(Z,names,size); A=np.column_stack([X,Z,B]).astype(np.float32)
    gd=direct_gate(model['direct_gate'],A); gf=factor_gate(model['factor_gate'],A); reg,ats,_=ordinal_ensemble(model['ordinal'],A)
    pA=_latent_col(Z,names,'aphe',.5); pR=_latent_col(Z,names,'rim_aphe',.1)
    return labels_router(gd,gf,reg,ats,pA,pR,size,model['router'])

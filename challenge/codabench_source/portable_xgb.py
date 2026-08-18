from __future__ import annotations
import numpy as np

EPS = 1e-8


def _leaf_value(tree, row):
    l=tree['l']; r=tree['r']; f=tree['f']; c=tree['c']; d=tree['d']
    n=0
    while l[n] != -1:
        val=row[f[n]]
        if np.isnan(val):
            left = bool(d[n])
        else:
            left = val < c[n]
        n = l[n] if left else r[n]
    return c[n]


def predict_raw(model, X):
    X=np.asarray(X,dtype=np.float32)
    if X.ndim==1: X=X[None,:]
    obj=model['objective']
    if obj=='multi:softprob':
        k=max(int(model.get('num_class',0)), len(model['base_margin']))
        out=np.tile(np.asarray(model['base_margin'],dtype=np.float32).reshape(1,-1),(len(X),1))
        if out.shape[1]!=k:
            out=np.zeros((len(X),k),dtype=np.float32)+float(np.asarray(model['base_margin']).ravel()[0])
        for tree in model['trees']:
            cls=int(tree['k'])
            for i,row in enumerate(X):
                out[i,cls] += np.float32(_leaf_value(tree,row))
        return out
    out=np.full(len(X), float(np.asarray(model['base_margin']).ravel()[0]), dtype=np.float32)
    for tree in model['trees']:
        for i,row in enumerate(X):
            out[i] += np.float32(_leaf_value(tree,row))
    return out


def predict(model, X):
    raw=predict_raw(model,X)
    obj=model['objective']
    if obj=='binary:logistic':
        raw=np.clip(raw,-40,40)
        return 1.0/(1.0+np.exp(-raw))
    if obj=='multi:softprob':
        z=raw-np.max(raw,axis=1,keepdims=True)
        e=np.exp(z)
        return e/(e.sum(axis=1,keepdims=True)+EPS)
    if obj=='reg:squarederror':
        return raw
    raise ValueError('Unsupported objective: '+str(obj))

#!/usr/bin/env python3
"""AMPLIFAI Codabench entry point for PHORA+ v2.

Uses only packages already present in the official challenge image (NumPy,
Pandas, SciPy) plus a tiny bundled NIfTI reader and a portable XGBoost tree
evaluator. No network access or runtime installation is required.
"""
from __future__ import annotations

import os, sys, traceback
from pathlib import Path

# Prevent each worker from spawning its own BLAS thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("AMPLIFAI_DATA_ROOT", "/app/data/cases"))
MODEL_PATH = HERE / "phora_model.pkl.gz"
VALID_LABELS = {"LR-1","LR-2","LR-3","LR-4","LR-5","LR-M","LR-TIV"}

# Imported after HERE is available; local nibabel/ is our minimal compatible reader.
sys.path.insert(0, str(HERE))
from phora_portable_predict import load_portable, predict_from_matrix
from phora_features import extract_case_features
from handcrafted_radiomics_portable import extract_case_radiomics
from context_features import extract_context_ot_case

_MODEL = None


def _init_worker():
    global _MODEL
    _MODEL = load_portable(MODEL_PATH)["model"]


def _safe_number(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _extract_and_predict(case_id: str):
    global _MODEL
    if _MODEL is None:
        _init_worker()
    cid=str(case_id)
    case_dir=DATA_ROOT / cid
    errors=[]
    feat={}

    # Match development merge order: spatial -> handcrafted radiomics -> context/OT.
    extractors=(
        ("spatial", lambda: extract_case_features(case_dir, cid, 5)),
        ("radiomics", lambda: extract_case_radiomics(case_dir, cid)),
        ("context", lambda: extract_context_ot_case(case_dir, cid)),
    )
    for name,fn in extractors:
        try:
            d=fn() or {}
            for k,v in d.items():
                if k not in feat:
                    feat[k]=v
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    selected=_MODEL["selected"]
    X=np.asarray([[_safe_number(feat.get(c,np.nan)) for c in selected]],dtype=np.float32)
    size=np.asarray([_safe_number(feat.get(_MODEL["size_col"],np.nan))],dtype=float)
    try:
        pred=str(predict_from_matrix(_MODEL,X,size)[0])
    except Exception as exc:
        errors.append("model: "+traceback.format_exc(limit=2).replace("\n"," | "))
        # Even with failed feature extraction, the all-missing vector follows learned
        # XGBoost missing branches and preserves complete case coverage.
        X=np.full((1,len(selected)),np.nan,dtype=np.float32)
        pred=str(predict_from_matrix(_MODEL,X,np.asarray([np.nan]))[0])

    if pred not in VALID_LABELS:
        errors.append(f"invalid prediction {pred!r}; replaced by LR-5")
        pred="LR-5"
    return cid,pred,"; ".join(errors)


def _sample_csv(input_arg: str) -> Path:
    p=Path(input_arg)
    if p.is_file():
        return p
    q=p / "sample_cases.csv"
    if q.exists():
        return q
    return Path("/app/input_data/sample_cases.csv")


def main():
    input_arg=sys.argv[1] if len(sys.argv)>1 else "/app/input_data"
    output_arg=sys.argv[2] if len(sys.argv)>2 else "/app/output"
    outdir=Path(output_arg); outdir.mkdir(parents=True,exist_ok=True)
    sample_path=_sample_csv(input_arg)
    cases=pd.read_csv(sample_path)
    if "case_id" not in cases.columns:
        raise ValueError(f"sample_cases.csv must contain case_id: {sample_path}")
    ids=[str(x) for x in cases["case_id"].tolist()]
    print(f"PHORA+ v2: {len(ids)} cases | data={DATA_ROOT} | model={MODEL_PATH.name}",flush=True)

    # Four processes give useful speed-up for Python texture loops while keeping CT RAM bounded.
    requested=int(os.environ.get("PHORA_WORKERS","4"))
    workers=max(1,min(requested,4,len(ids))) if ids else 1
    rows=[]
    if workers==1:
        _init_worker()
        for i,cid in enumerate(ids,1):
            c,p,e=_extract_and_predict(cid); rows.append({"case_id":c,"prediction":p})
            print(f"[{i}/{len(ids)}] {c} -> {p}" + (f" | WARN {e}" if e else ""),flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker) as ex:
            fut={ex.submit(_extract_and_predict,cid):cid for cid in ids}
            done=0
            result_map={}
            for f in as_completed(fut):
                cid=fut[f]; done+=1
                try:
                    c,p,e=f.result()
                except Exception as exc:
                    # Last-resort in parent: do not lose coverage.
                    c=cid; p="LR-5"; e=f"worker failure: {type(exc).__name__}: {exc}"
                result_map[c]=(p,e)
                print(f"[{done}/{len(ids)}] {c} -> {p}" + (f" | WARN {e}" if e else ""),flush=True)
        rows=[{"case_id":cid,"prediction":result_map.get(cid,("LR-5","missing worker result"))[0]} for cid in ids]

    out=pd.DataFrame(rows,columns=["case_id","prediction"])
    # Exact one-to-one coverage in the order supplied by Codabench.
    if len(out)!=len(ids) or out["case_id"].duplicated().any():
        raise RuntimeError("Internal coverage/duplicate error while creating predictions.csv")
    bad=~out["prediction"].isin(VALID_LABELS)
    if bad.any():
        raise RuntimeError("Invalid labels generated: "+str(out.loc[bad,"prediction"].unique().tolist()))
    dest=outdir / "predictions.csv"
    out.to_csv(dest,index=False)
    print(f"Wrote {dest} ({len(out)} predictions)",flush=True)

if __name__=="__main__":
    main()

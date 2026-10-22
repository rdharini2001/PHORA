"""Derive signed, lesion-restricted, phase-to-phase change maps from provided CT series.

Color encodes signed Hounsfield-unit difference, not neural attribution.
Comparison: ART-DRY, VEN-ART, DEL-VEN. Uses original voxel arrays
and the source lesion annotation; the CT phases are assumed co-located
as supplied. The filter threshold is for visualization, not diagnosis.
"""
from pathlib import Path
import sys, json, zipfile
import numpy as np
from PIL import Image
from prepare_ct import nifti

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs'/'assets'
data=json.loads((OUT/'case_data.json').read_text())
phases=[('ART','DRY'),('VEN','ART'),('DEL','VEN')]
for cid in data:
    zpath=Path('/mnt/data')/(cid+'.zip')
    d=data[cid]
    with zipfile.ZipFile(zpath) as z:
        mpath=next(x for x in z.namelist() if x.endswith('/annotations/lesion.nii.gz'))
        mask=np.array(nifti(z,mpath)>0,copy=True)
        d['signed_phase_stats']={}
        for p,base in phases:
            pfile=next(x for x in z.namelist() if x.endswith(f'/ct/{cid}_{p}.nii.gz'))
            bfile=next(x for x in z.namelist() if x.endswith(f'/ct/{cid}_{base}.nii.gz'))
            a=np.asarray(nifti(z,pfile),dtype=np.float32)
            b=np.asarray(nifti(z,bfile),dtype=np.float32)
            if a.shape!=mask.shape or b.shape!=mask.shape: raise RuntimeError(f'Grid mismatch for {cid} {p} {base}')
            signed=np.subtract(a,b,dtype=np.float32)
            vox=signed[mask]; vox=vox[np.isfinite(vox)]
            d['signed_phase_stats'][p]={
               'reference':base,'mean_delta_hu':round(float(np.mean(vox)),1),
               'fraction_increase_10hu':round(float(np.mean(vox>=10)),3),
               'fraction_decrease_10hu':round(float(np.mean(vox<=-10)),3),
               'fraction_within_10hu':round(float(np.mean(np.abs(vox)<10)),3),
            }
            for i,zi in enumerate(d['slices']):
                slice_mask=np.uint8(mask[zi])*255
                resized_mask=np.asarray(Image.fromarray(slice_mask).resize((384,384),Image.Resampling.NEAREST))>0
                sl=np.asarray(Image.fromarray(signed[zi]).resize((384,384),Image.Resampling.BILINEAR),dtype=np.float32)
                activity=np.clip((np.abs(sl)-10)/90,0,1)
                alpha=np.where(np.abs(sl)>=10,55+150*activity,0).astype(np.uint8)
                alpha[~resized_mask]=0
                rgba=np.zeros((384,384,4),dtype=np.uint8)
                rgba[:,:,:3]=np.where((sl>=0)[...,None], [240,166,95],[80,192,230]).astype(np.uint8)
                rgba[:,:,3]=alpha
                Image.fromarray(rgba,'RGBA').save(OUT/'cases'/cid/f'{p}_signed_{i:02}.png',optimize=True)
            print(cid,p,d['signed_phase_stats'][p],flush=True)
            del a,b,signed
(OUT/'case_data.json').write_text(json.dumps(data,indent=2))
(OUT/'case_data.js').write_text('window.PHORA_CASES = '+json.dumps(data,separators=(',',':'))+';\n')

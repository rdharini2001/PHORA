"""Generate transparent, traced web assets directly from the supplied NIfTI series.
Python stdlib for NIfTI-1 read; numpy/scipy/Pillow for image rendering.
Displays are illustrative research visualizations, not model saliency/inference.
"""
from pathlib import Path
import os, zipfile, gzip, struct, io, json, math
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets/cases';OUT.mkdir(parents=True,exist_ok=True)
DATA={}
PHASES=['DRY','ART','VEN','DEL']
ANNOT={'lesion':'Lesion mask','aphe':'Arterial hyperenhancement','venous_washout':'Venous washout','delayed_washout':'Delayed washout'}

def nifti(z, path):
    with z.open(path) as s:
      with gzip.GzipFile(fileobj=s) as f:
        raw=f.read()
    hdr=raw[:352]; e='<' if struct.unpack('<i',hdr[:4])[0]==348 else '>'
    dim=struct.unpack(e+'8h',hdr[40:56]); dt=struct.unpack(e+'h',hdr[70:72])[0]
    off=round(struct.unpack(e+'f',hdr[108:112])[0])
    scl,inter=struct.unpack(e+'2f',hdr[112:120])
    dtype={2:'u1',4:'i2',8:'i4',16:'f4',64:'f8',512:'u2'}[dt]
    shp=(dim[3],dim[2],dim[1]);arr=np.frombuffer(raw,dtype=np.dtype(e+dtype),offset=off).reshape(shp)
    if scl and scl!=1: arr=arr.astype('f4')*scl
    if inter: arr=arr.astype('f4')+inter
    return arr

def sample_bbox(mask):
    coord=np.nonzero(mask); return [(int(c.min()),int(c.max())) for c in coord]

def window8(a,c=85,w=390):
    p=np.clip((a-(c-w/2))/(w),0,1)
    # CT window, slight S curve for nuanced detail
    return np.uint8(np.round(p*255))

def savew(arr,path):
    im=Image.fromarray(arr,'L'); im.save(path,'WEBP',quality=83,method=5)

def overlaymask(mask,color,path):
    # RGBA overlay in browser, exact per-voxel alignment at exported image resolution
    rgba=np.zeros((*mask.shape,4),dtype=np.uint8)
    rgba[:,:,:3]=color
    rgba[:,:,3]=mask.astype(np.uint8)*155
    Image.fromarray(rgba,'RGBA').save(path,'PNG',optimize=True)

def gradmap(phase,ref,lesion,path):
    # Dynamic enhancement *difference*, not trained attention
    diff=np.asarray(phase,dtype=np.float32)-np.asarray(ref,dtype=np.float32)
    mag=np.clip(np.abs(diff)/140.0,0,1)
    alpha=(np.uint8(200*mag))*(lesion.astype(np.uint8))
    rgba=np.zeros((*alpha.shape,4),dtype='uint8'); rgba[:,:,:3]=[245,168,90];rgba[:,:,3]=alpha
    Image.fromarray(rgba,'RGBA').save(path,'PNG',optimize=True)

def main():
 for case in ['CASE00001','CASE00002']:
   src=Path('/mnt/data')/(case+'.zip')
   o=OUT/case;o.mkdir(parents=True,exist_ok=True)
   with zipfile.ZipFile(src) as z:
     files=z.namelist(); masks={}
     for ann in ANNOT:
       f=next((p for p in files if p.endswith('/annotations/'+ann+'.nii.gz')),None)
       if f: masks[ann]=np.array(nifti(z,f)>0, dtype=bool, copy=True)
     les=masks['lesion']; bbox=sample_bbox(les); zmin,zmax=bbox[0]
     # sample evenly through lesion and neighbor z range, ensure peak slice included
     counts=les.sum(axis=(1,2)); peak=int(np.argmax(counts))
     low=max(0,zmin-3);high=min(les.shape[0]-1,zmax+3)
     slices=sorted(set([int(x) for x in np.round(np.linspace(low,high,min(high-low+1,21)))] +[peak] ))
     # Full 512-pixel axial slices reduced to 420 for website while preserving pixel alignment
     msize=384
     d={'id':case,'slices':slices,'primary_slice':peak,'shape':list(les.shape),'annotation_keys':list(masks),
       'voxel_count':int(les.sum()),'slice_bounds':[zmin,zmax], 'slice_count':len(slices), 'phase_stats':{}, 'slice_stats':[], 'overlays':{}}
     print(case,'shape',les.shape,'bbox',bbox,'peak',peak,'slices',len(slices),'voxels',int(les.sum()),flush=True)
     # overlay masks
     palette={'lesion':[90,238,203],'aphe':[255,177,112],'venous_washout':[157,157,255],'delayed_washout':[251,111,143]}
     for i,zi in enumerate(slices):
       for key,m in masks.items():
          mm=np.uint8(m[zi])*255
          if mm.max()==0: continue
          mt=np.asarray(Image.fromarray(mm).resize((msize,msize),resample=Image.Resampling.NEAREST))>0
          overlaymask(mt,palette[key],o/f'{key}_{i:02}.png')
     # Read all phases separately; stream data to avoid storing gigabytes.
     phase_means={}
     for phase in PHASES:
       f=next(p for p in files if p.endswith('/ct/'+case+'_'+phase+'.nii.gz'))
       arr=nifti(z,f)
       vals=np.asarray(arr[les],dtype=np.float64)
       phase_means[phase]=float(np.mean(vals))
       d['phase_stats'][phase]={'lesion_mean_hu':round(float(np.mean(vals)),1), 'lesion_std_hu':round(float(np.std(vals)),1)}
       for i,zi in enumerate(slices):
         im=window8(arr[zi])
         im=np.asarray(Image.fromarray(im).resize((msize,msize),resample=Image.Resampling.BILINEAR))
         savew(im,o/f'{phase}_{i:02}.webp')
       if phase=='DRY':
         # store reference cropped slices only (not volume)
         ref=[arr[zi].astype('f4',copy=True) for zi in slices]
       if phase!='DRY':
         for i,zi in enumerate(slices):
           v=np.asarray(arr[zi],dtype='f4');a=ref[i]
           # reflect enhancement within lesion only, fixed HU range across phases
           diff=window8(v-ref[i],c=0,w=320)
           im=np.asarray(Image.fromarray(diff).resize((msize,msize),resample=Image.Resampling.BILINEAR))
           Image.fromarray(im,'L').save(o/f'{phase}_difference_{i:02}.webp','WEBP',quality=78)
           # absolute difference overlay, 384x384
           mm=np.asarray(Image.fromarray(np.uint8(les[zi])*255).resize((msize,msize),resample=Image.Resampling.NEAREST))>0
           vv=np.asarray(Image.fromarray(v).resize((msize,msize),resample=Image.Resampling.BILINEAR))
           rr=np.asarray(Image.fromarray(a).resize((msize,msize),resample=Image.Resampling.BILINEAR))
           gradmap(vv,rr,mm,o/f'{phase}_heat_{i:02}.png')
       print(' ',phase, 'meanHU',round(phase_means[phase],2),flush=True)
       del arr
     d['trajectory']=[{'phase':p,'mean_hu':round(phase_means[p],1)} for p in PHASES]
     d['delta_art_dry']=round(phase_means['ART']-phase_means['DRY'],1)
     d['delta_ven_art']=round(phase_means['VEN']-phase_means['ART'],1)
     d['delta_del_art']=round(phase_means['DEL']-phase_means['ART'],1)
     d['delta_del_ven']=round(phase_means['DEL']-phase_means['VEN'],1)
     DATA[case]=d
 out=ROOT/'docs/assets/case_data.json'
 out.write_text(json.dumps(DATA,indent=2))
 print('json', out, os.stat(out).st_size)

if __name__=='__main__': main()

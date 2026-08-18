"""Tiny NIfTI-1 reader used by the PHORA AMPLIFAI submission.
Implements only the subset of NiBabel's API needed by phora_features.py.
No external dependency beyond NumPy/SciPy (provided by challenge image).
"""
from __future__ import annotations
import gzip, struct
from pathlib import Path
import numpy as np

_DTYPES = {
    2: np.dtype('u1'), 4: np.dtype('i2'), 8: np.dtype('i4'),
    16: np.dtype('f4'), 64: np.dtype('f8'), 256: np.dtype('i1'),
    512: np.dtype('u2'), 768: np.dtype('u4'), 1024: np.dtype('i8'), 1280: np.dtype('u8'),
}

class Nifti1Image:
    def __init__(self, data, affine):
        self._data = np.asarray(data)
        self.affine = np.asarray(affine, dtype=float)
        self.shape = self._data.shape
        self.dataobj = self._data
    def get_fdata(self, dtype=np.float64):
        return np.asarray(self._data, dtype=dtype)

def _open(path):
    p=str(path)
    return gzip.open(p,'rb') if p.lower().endswith('.gz') else open(p,'rb')

def _qform_affine(hdr, endian, pixdim):
    b,c,d=struct.unpack_from(endian+'3f',hdr,256)
    qx,qy,qz=struct.unpack_from(endian+'3f',hdr,268)
    aa=1.0-(b*b+c*c+d*d)
    if aa < 1e-7:
        n=(b*b+c*c+d*d)**0.5
        if n>0: b,c,d=b/n,c/n,d/n
        a=0.0
    else:
        a=aa**0.5
    R=np.array([
        [a*a+b*b-c*c-d*d, 2*b*c-2*a*d,       2*b*d+2*a*c],
        [2*b*c+2*a*d,       a*a+c*c-b*b-d*d, 2*c*d-2*a*b],
        [2*b*d-2*a*c,       2*c*d+2*a*b,       a*a+d*d-c*c-b*b],
    ],dtype=float)
    dx=float(pixdim[1]) or 1.0; dy=float(pixdim[2]) or 1.0; dz=float(pixdim[3]) or 1.0
    qfac=-1.0 if float(pixdim[0])<0 else 1.0
    A=np.eye(4,dtype=float); A[:3,:3]=R @ np.diag([dx,dy,dz*qfac]); A[:3,3]=[qx,qy,qz]
    return A

def load(path):
    with _open(path) as f:
        hdr=f.read(352)
        if len(hdr)<348: raise ValueError(f'Invalid NIfTI header: {path}')
        le=struct.unpack_from('<i',hdr,0)[0]; be=struct.unpack_from('>i',hdr,0)[0]
        if le==348: endian='<'
        elif be==348: endian='>'
        else: raise ValueError(f'Only NIfTI-1 (.nii/.nii.gz) is supported; sizeof_hdr={le}/{be}: {path}')
        dim=struct.unpack_from(endian+'8h',hdr,40); ndim=int(dim[0]); dims=[int(x) for x in dim[1:1+max(3,ndim)]]
        if ndim<3: dims=dims[:ndim]+[1]*(3-ndim)
        shape=tuple(dims[:3])
        if ndim>3 and any(int(x)!=1 for x in dim[4:1+ndim]):
            raise ValueError(f'Expected 3-D NIfTI, got dim={dim}: {path}')
        datatype=struct.unpack_from(endian+'h',hdr,70)[0]
        if datatype not in _DTYPES: raise ValueError(f'Unsupported NIfTI datatype code {datatype}: {path}')
        pixdim=np.asarray(struct.unpack_from(endian+'8f',hdr,76),dtype=float)
        vox_offset=float(struct.unpack_from(endian+'f',hdr,108)[0]); slope=float(struct.unpack_from(endian+'f',hdr,112)[0]); inter=float(struct.unpack_from(endian+'f',hdr,116)[0])
        qcode=struct.unpack_from(endian+'h',hdr,252)[0]; scode=struct.unpack_from(endian+'h',hdr,254)[0]
        if scode>0:
            A=np.eye(4,dtype=float)
            A[0,:]=struct.unpack_from(endian+'4f',hdr,280); A[1,:]=struct.unpack_from(endian+'4f',hdr,296); A[2,:]=struct.unpack_from(endian+'4f',hdr,312)
        elif qcode>0:
            A=_qform_affine(hdr,endian,pixdim)
        else:
            A=np.diag([pixdim[1] or 1.0,pixdim[2] or 1.0,pixdim[3] or 1.0,1.0]).astype(float)
        offset=max(352,int(round(vox_offset)))
        f.seek(offset)
        base=_DTYPES[datatype].newbyteorder(endian)
        n=int(np.prod(shape)); raw=f.read(n*base.itemsize)
        if len(raw)<n*base.itemsize: raise ValueError(f'Truncated NIfTI data: {path}')
        arr=np.frombuffer(raw,dtype=base,count=n).reshape(shape,order='F')
        if slope==0 or not np.isfinite(slope): slope=1.0
        if slope!=1.0 or inter!=0.0:
            arr=arr.astype(np.float32)*slope+inter
        # detach from compressed byte buffer and normalize native endian
        arr=np.asarray(arr).copy()
    return Nifti1Image(arr,A)

from . import processing

from __future__ import annotations
import numpy as np
from scipy import ndimage
from . import Nifti1Image

def resample_from_to(img, to_vox_map, order=3, mode='constant', cval=0.0):
    shape, affine=to_vox_map
    shape=tuple(int(v) for v in shape[:3]); affine=np.asarray(affine,float)
    if tuple(img.shape[:3])==shape and np.allclose(img.affine,affine,atol=1e-7):
        return Nifti1Image(np.asarray(img.dataobj).copy(),affine)
    T=np.linalg.inv(np.asarray(img.affine,float)) @ affine
    inp=np.asarray(img.dataobj)
    # SciPy's cubic spline matches the interpolation family used by the development extractor.
    out=ndimage.affine_transform(inp, T[:3,:3], offset=T[:3,3], output_shape=shape,
                                 order=int(order), mode=mode, cval=float(cval),
                                 prefilter=(int(order)>1))
    return Nifti1Image(out,affine)

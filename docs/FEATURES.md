# Feature representation

PHORA begins with approximately 900 inference-safe candidate imaging variables. The frozen challenge model retains 140 imaging variables after nested screening, then appends cross-fitted latent clinical variables and structured interaction terms.

## Candidate families

### Handcrafted radiomics
- 3-D shape and first-order statistics
- GLCM, GLRLM, GLSZM, GLDM and NGTDM textures
- phase-specific extraction after a 5-mm padded lesion crop and 1-mm isotropic resampling
- fixed 25-HU discretization
- first-order phase differences

### Spatial physiology
- whole-lesion geometry and spectral morphology
- rim / middle / core enhancement habitats
- graph total variation and Dirichlet energy
- directional semivariograms
- hotspot-centroid displacement and hotspot topology
- voxelwise ART-DRY, VEN-ART, DEL-ART and DEL-VEN difference fields
- enhancement trajectory range, path length, AUC, peak phase and curvature

### Context and distributional descriptors
- 3--12 mm local perilesional reference kinetics
- Moran's I and Geary's C
- hotspot connectivity
- vascular-contact surrogate
- 1-Wasserstein distances and Jensen-Shannon divergence between phase distributions

## Frozen imaging representation

The 140 selected imaging features in the submitted Codabench model comprise:

| Family | Selected features |
|---|---:|
| Handcrafted radiomics | 70 |
| Spatial physiology | 60 |
| Context / optimal transport | 10 |

The exact list is released in [`results/model/selected_imaging_features.csv`](../results/model/selected_imaging_features.csv).

## Unified representation

The final tabular vector is

```text
140 selected imaging features
+ 6 cross-fitted concept probabilities
+ 5 cross-fitted spatial-burden estimates
+ 24 soft clinical interaction features
= 175 dimensions
```

The 175-D vector is a direct concatenation; PHORA does not use PCA, a neural embedding, or an autoencoder for feature fusion. The exact latent/basis names and frozen router parameters are summarized in [`results/model/model_summary.json`](../results/model/model_summary.json).

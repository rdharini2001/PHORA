# Data

PHORA was developed for the **MICCAI 2026 AMPLIFAI Challenge**. The current challenge release contains 779 studies: 531 training, 59 public validation, and 189 private test cases. Only the 590 public train/validation cases are distributed to participants; the private test set remains on the challenge platform.

For the seven-class classification experiments, 70 training rows labelled `No lesion` are excluded because they are not legal challenge output labels. This leaves 461 training lesions and 59 public-validation lesions.

## Expected local layout

```text
data/
├── batch_001.zip
├── batch_002.zip
├── batch_003.zip
├── batch_004.zip
├── train_metadata.csv
└── val_metadata.csv
```

The notebook extracts archives recursively; repeated batch nesting is supported. A case is expected to contain:

```text
<case_id>/
├── ct/
│   ├── <case_id>_ART.nii.gz
│   ├── <case_id>_VEN.nii.gz       # if available
│   ├── <case_id>_DEL.nii.gz       # if available
│   └── <case_id>_DRY.nii.gz       # if available
└── annotations/
    └── lesion.nii.gz
```

Raw images are intentionally **not included** in this repository. The AMPLIFAI public training/validation data are released under the challenge's CC BY-NC-SA terms. See the official challenge data and rules pages before redistributing dataset-derived material.

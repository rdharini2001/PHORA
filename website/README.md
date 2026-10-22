# PHORA — Interactive Multiphase CT Research Website

**Website companion to the MICCAI 2026 PHORA paper.**

**Paper:** [PHORA: Physiology-Guided Hierarchical Ordinal Radiomics for LI-RADS Classification in Multiphase CT](https://papers.miccai.org/miccai-2026-sat/paper/AMPLIFAI_007.pdf)  
**Source repository:** [rdharini2001/PHORA](https://github.com/rdharini2001/PHORA)

The `docs/` directory is a responsive research exhibit built around the paper's actual physiology-guided decision logic. It includes an interactive **two-case, four-phase CT explorer** generated from the supplied NIfTI data, voxel-aligned reference masks, an image-derived phase-difference map, enhancement trajectories, hierarchical model explanations, and the paper's public-validation comparisons.

## Data integrity and interpretation

- The viewer is generated from `CASE00001.zip` (four clinical annotation masks) and `CASE00002.zip` (lesion mask only).
- The display assets are downsampled for online viewing; the full-size source CT volumes are **not included**, and should remain in the approved data distribution.
- PHORA's published composite score of 0.718, adjusted QWK of 0.738, and special-category recognition of 0.605 are sourced from the paper and the original repository result CSV.
- PHORA is a **tabular radiomics system** whose inference combines phases into one case-level prediction. The website shows *how observed imaging evidence changes*; it does not fabricate phase-level LI-RADS outputs.
- The amber overlay is a direct phase-difference image derived from HU values, **not PHORA attention**, Grad-CAM, SHAP, or model attribution.
- Please review [`REPRODUCIBILITY_AUDIT.md`](REPRODUCIBILITY_AUDIT.md): the archived router summary does not match the fixed router parameters printed in the paper; this requires version attribution in the original research code.

## Preview

Open `docs/index.html` in a modern browser. The data manifest is embedded as a JavaScript mirror to avoid fetch/CORS restrictions on local previews. For a hosted preview, run:

```bash
python -m http.server 8000 --directory docs
```

Then navigate to http://localhost:8000.

## Add this website to the existing PHORA code repository

Do not replace the computational source tree with this smaller website package. The existing GitHub repository retains the full model code and saved validation results.

**Using this patch ZIP** after cloning `rdharini2001/PHORA`:

```bash
# In the downloaded website patch folder
bash scripts/install_in_repo.sh /path/to/cloned/PHORA
cd /path/to/cloned/PHORA
python -m unittest discover -s tests -v  # original repository checks
# commit and push the new docs/ and workflow files
```

The installer preserves `docs/METHOD.md`, `docs/FEATURES.md`, `docs/REPRODUCIBILITY.md` and all original source files, and adds/updates only the website's named static assets, research notes, and Pages workflow. If GitHub Pages is enabled through Actions in Settings → Pages, deployment uses `.github/workflows/research-website.yml`.

Suggested URL after publishing: `https://rdharini2001.github.io/PHORA/`.

## Rebuild CT web assets from the original case ZIPs

With Python, NumPy, SciPy, and Pillow installed, run `python scripts/prepare_ct.py`. It expects both case ZIPs in `/mnt/data` unless you adjust its `src` path. It exports lesion-centered axial images, annotation PNG overlays, and HU-difference overlays. The original NIfTI full volumes are not redistributed through the website.

## Verification

```bash
node --check docs/assets/site.js
python -m unittest discover -s tests -v
python scripts/audit_publication.py /path/to/cloned/PHORA
```

Research demonstration only; not for clinical interpretation or medical decisions.

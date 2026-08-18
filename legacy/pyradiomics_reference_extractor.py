#!/usr/bin/env python3
"""Extract 3D PyRadiomics features from AMPLIFAI multi-phase CT data.

Expected per-case layout:
    <case_id>/
      ct/<case_id>_ART.nii.gz
      ct/<case_id>_VEN.nii.gz
      ct/<case_id>_DEL.nii.gz
      ct/<case_id>_DRY.nii.gz
      annotations/lesion.nii.gz

The script:
  * discovers cases recursively (works with batch_*/cases/<case_id> layouts),
  * computes lesion shape once from the binary mask,
  * computes first-order and 3D texture features separately for every available phase,
  * resamples the mask to each phase grid with nearest-neighbor interpolation,
  * records missing phases instead of failing,
  * optionally adds phase-difference first-order features,
  * merges train metadata when supplied,
  * writes a feature table and a failure log.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

PHASES = ("ART", "VEN", "DEL", "DRY")

# Initialized once per worker process.
_SHAPE_EXTRACTOR: featureextractor.RadiomicsFeatureExtractor | None = None
_PHASE_EXTRACTOR: featureextractor.RadiomicsFeatureExtractor | None = None


def build_extractors(
    spacing_mm: float = 1.0,
    bin_width_hu: float = 25.0,
    minimum_roi_voxels: int = 10,
) -> tuple[featureextractor.RadiomicsFeatureExtractor,
           featureextractor.RadiomicsFeatureExtractor]:
    """Create reproducible shape-only and phase-intensity extractors."""
    common_settings: dict[str, Any] = {
        "label": 1,
        "binWidth": float(bin_width_hu),
        "resampledPixelSpacing": [float(spacing_mm)] * 3,
        "interpolator": "sitkBSpline",  # masks are NN-resampled by PyRadiomics
        "normalize": False,              # preserve CT Hounsfield units
        "correctMask": False,            # mask is explicitly aligned below
        "minimumROIDimensions": 2,
        "minimumROISize": int(minimum_roi_voxels),
        "additionalInfo": False,
        "preCrop": True,
        "padDistance": 5,
    }

    shape = featureextractor.RadiomicsFeatureExtractor(**common_settings)
    shape.disableAllImageTypes()
    shape.enableImageTypeByName("Original")
    shape.disableAllFeatures()
    shape.enableFeatureClassByName("shape")

    phase = featureextractor.RadiomicsFeatureExtractor(**common_settings)
    phase.disableAllImageTypes()
    phase.enableImageTypeByName("Original")
    phase.disableAllFeatures()
    for feature_class in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        phase.enableFeatureClassByName(feature_class)

    return shape, phase


def init_worker(spacing_mm: float, bin_width_hu: float, minimum_roi_voxels: int) -> None:
    global _SHAPE_EXTRACTOR, _PHASE_EXTRACTOR
    _SHAPE_EXTRACTOR, _PHASE_EXTRACTOR = build_extractors(
        spacing_mm=spacing_mm,
        bin_width_hu=bin_width_hu,
        minimum_roi_voxels=minimum_roi_voxels,
    )


def discover_cases(data_root: Path) -> dict[str, Path]:
    """Return {case_id: case_directory}; duplicate IDs are rejected."""
    cases: dict[str, Path] = {}
    for pattern in ("annotations/lesion.nii.gz", "annotations/lesion.nii"):
        for lesion_path in data_root.rglob(pattern):
            case_dir = lesion_path.parent.parent
            if not (case_dir / "ct").is_dir():
                continue
            case_id = case_dir.name
            if case_id in cases and cases[case_id] != case_dir:
                raise RuntimeError(
                    f"Duplicate case_id {case_id!r}: {cases[case_id]} and {case_dir}"
                )
            cases[case_id] = case_dir
    return dict(sorted(cases.items()))


def to_binary_mask(mask: sitk.Image) -> sitk.Image:
    """Convert every positive label to one, preserving physical geometry."""
    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return binary


def align_mask_to_image(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    """Resample a registered mask to the exact image grid using nearest neighbor."""
    return sitk.Resample(
        mask,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def image_geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "size": list(image.GetSize()),
        "spacing": [float(x) for x in image.GetSpacing()],
        "origin": [float(x) for x in image.GetOrigin()],
        "direction": [float(x) for x in image.GetDirection()],
    }


def numeric_features(result: dict[str, Any]) -> dict[str, float]:
    """Keep only finite/scalar PyRadiomics feature values, excluding diagnostics."""
    output: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            array = np.asarray(value)
            if array.size != 1:
                continue
            number = float(array.reshape(-1)[0])
        except (TypeError, ValueError):
            continue
        output[key] = number if math.isfinite(number) else np.nan
    return output


def mask_qc(mask: sitk.Image) -> dict[str, float]:
    array = sitk.GetArrayViewFromImage(mask)
    voxel_count = int(np.count_nonzero(array))
    voxel_volume = float(np.prod(mask.GetSpacing()))
    return {
        "mask_voxel_count": voxel_count,
        "mask_volume_native_mm3": voxel_count * voxel_volume,
    }


def extract_case(case_id: str, case_dir_string: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Extract one row of features plus zero or more error records."""
    if _SHAPE_EXTRACTOR is None or _PHASE_EXTRACTOR is None:
        raise RuntimeError("Worker extractors were not initialized")

    case_dir = Path(case_dir_string)
    lesion_path = case_dir / "annotations" / "lesion.nii.gz"
    if not lesion_path.exists():
        lesion_path = case_dir / "annotations" / "lesion.nii"
    failures: list[dict[str, str]] = []
    row: dict[str, Any] = {"case_id": case_id}

    try:
        native_mask = to_binary_mask(sitk.ReadImage(str(lesion_path)))
        qc = mask_qc(native_mask)
        row.update(qc)
        row["mask_spacing_x"] = float(native_mask.GetSpacing()[0])
        row["mask_spacing_y"] = float(native_mask.GetSpacing()[1])
        row["mask_spacing_z"] = float(native_mask.GetSpacing()[2])

        if qc["mask_voxel_count"] == 0:
            raise ValueError("lesion mask is empty")

        # Shape is calculated once from the mask's own physical geometry.
        shape_image = sitk.Cast(native_mask, sitk.sitkFloat32)
        shape_result = numeric_features(_SHAPE_EXTRACTOR.execute(shape_image, native_mask, label=1))
        for key, value in shape_result.items():
            clean_key = key.replace("original_shape_", "shape_", 1)
            row[clean_key] = value
    except Exception as exc:  # keep the case row; phase extraction may still be informative
        failures.append({"case_id": case_id, "component": "lesion_mask/shape", "error": repr(exc)})
        for phase in PHASES:
            row[f"phase_present_{phase}"] = 0
        return row, failures

    for phase in PHASES:
        image_path = case_dir / "ct" / f"{case_id}_{phase}.nii.gz"
        if not image_path.exists():
            image_path = case_dir / "ct" / f"{case_id}_{phase}.nii"
        row[f"phase_present_{phase}"] = int(image_path.exists())
        if not image_path.exists():
            continue

        try:
            image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
            phase_mask = align_mask_to_image(native_mask, image)
            phase_voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(phase_mask)))
            if phase_voxels == 0:
                raise ValueError("mask became empty after alignment to phase grid")

            row[f"{phase}_mask_voxel_count"] = phase_voxels
            row[f"{phase}_spacing_x"] = float(image.GetSpacing()[0])
            row[f"{phase}_spacing_y"] = float(image.GetSpacing()[1])
            row[f"{phase}_spacing_z"] = float(image.GetSpacing()[2])

            result = numeric_features(_PHASE_EXTRACTOR.execute(image, phase_mask, label=1))
            for key, value in result.items():
                row[f"{phase}_{key}"] = value
        except Exception as exc:
            failures.append({"case_id": case_id, "component": phase, "error": repr(exc)})

    return row, failures


def add_firstorder_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically useful temporal differences for common first-order features."""
    pairs = (
        ("ART", "DRY"),  # arterial enhancement relative to native
        ("VEN", "ART"),  # arterial-to-portal change
        ("DEL", "ART"),  # arterial-to-delayed change
        ("DEL", "VEN"),  # portal-to-delayed change
    )

    suffixes: set[str] = set()
    for column in df.columns:
        for phase in PHASES:
            prefix = f"{phase}_original_firstorder_"
            if column.startswith(prefix):
                suffixes.add(column[len(phase) + 1 :])  # keep original_firstorder_...
                break

    new_columns: dict[str, pd.Series] = {}
    for left, right in pairs:
        for suffix in sorted(suffixes):
            left_col = f"{left}_{suffix}"
            right_col = f"{right}_{suffix}"
            if left_col in df.columns and right_col in df.columns:
                new_columns[f"DELTA_{left}_MINUS_{right}_{suffix}"] = (
                    pd.to_numeric(df[left_col], errors="coerce")
                    - pd.to_numeric(df[right_col], errors="coerce")
                )

    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def load_metadata(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    metadata = pd.read_csv(path)
    if "case_id" not in metadata.columns:
        raise ValueError(f"Metadata must contain a 'case_id' column: {path}")
    metadata = metadata.copy()
    metadata["case_id"] = metadata["case_id"].astype(str)
    if metadata["case_id"].duplicated().any():
        duplicates = metadata.loc[metadata["case_id"].duplicated(), "case_id"].tolist()[:10]
        raise ValueError(f"Duplicate case_id values in metadata: {duplicates}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root containing extracted batch folders/cases")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="Optional train_metadata.csv to merge by case_id")
    parser.add_argument("--failure-csv", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--spacing-mm", type=float, default=1.0)
    parser.add_argument("--bin-width-hu", type=float, default=25.0)
    parser.add_argument("--minimum-roi-voxels", type=int, default=10)
    parser.add_argument("--no-deltas", action="store_true",
                        help="Do not add phase-difference first-order features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cases = discover_cases(args.data_root)
    if not cases:
        raise FileNotFoundError(
            f"No lesion.nii(.gz) files found under {args.data_root}"
        )
    logging.info("Discovered %d cases", len(cases))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.spacing_mm, args.bin_width_hu, args.minimum_roi_voxels),
    ) as executor:
        futures = {
            executor.submit(extract_case, case_id, str(case_dir)): case_id
            for case_id, case_dir in cases.items()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                row, case_failures = future.result()
                rows.append(row)
                failures.extend(case_failures)
            except Exception as exc:
                failures.append({"case_id": case_id, "component": "worker", "error": repr(exc)})
                rows.append({"case_id": case_id})
            if index == 1 or index % 25 == 0 or index == len(futures):
                logging.info("Processed %d/%d cases", index, len(futures))

    features = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    if not args.no_deltas:
        features = add_firstorder_deltas(features)

    metadata = load_metadata(args.metadata_csv)
    if metadata is not None:
        features = metadata.merge(features, on="case_id", how="left", validate="one_to_one")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)

    failure_path = args.failure_csv or args.output_csv.with_name(
        args.output_csv.stem + "_failures.csv"
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["case_id", "component", "error"]).to_csv(
        failure_path, index=False
    )

    manifest = {
        "data_root": str(args.data_root.resolve()),
        "n_discovered_cases": len(cases),
        "n_output_rows": int(len(features)),
        "n_feature_columns": int(len(features.columns)),
        "n_failure_records": len(failures),
        "phases": list(PHASES),
        "spacing_mm": args.spacing_mm,
        "bin_width_hu": args.bin_width_hu,
        "minimum_roi_voxels": args.minimum_roi_voxels,
        "firstorder_deltas": not args.no_deltas,
    }
    manifest_path = args.output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Features: %s", args.output_csv)
    logging.info("Failures: %s", failure_path)
    logging.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()

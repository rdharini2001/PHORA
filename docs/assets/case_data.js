window.PHORA_CASES = {
  "CASE00001": {
    "id": "CASE00001",
    "slices": [
      39,
      41,
      42,
      44,
      45,
      47,
      49,
      50,
      52,
      53,
      55,
      57,
      58,
      60,
      61,
      63,
      65,
      66,
      68,
      69,
      71
    ],
    "primary_slice": 55,
    "shape": [
      91,
      512,
      512
    ],
    "annotation_keys": [
      "lesion",
      "aphe",
      "venous_washout",
      "delayed_washout"
    ],
    "voxel_count": 94878,
    "slice_bounds": [
      42,
      68
    ],
    "slice_count": 21,
    "phase_stats": {
      "DRY": {
        "lesion_mean_hu": 46.8,
        "lesion_std_hu": 37.6
      },
      "ART": {
        "lesion_mean_hu": 79.8,
        "lesion_std_hu": 24.7
      },
      "VEN": {
        "lesion_mean_hu": 94.4,
        "lesion_std_hu": 24.7
      },
      "DEL": {
        "lesion_mean_hu": 71.0,
        "lesion_std_hu": 20.7
      }
    },
    "slice_stats": [],
    "overlays": {},
    "trajectory": [
      {
        "phase": "DRY",
        "mean_hu": 46.8
      },
      {
        "phase": "ART",
        "mean_hu": 79.8
      },
      {
        "phase": "VEN",
        "mean_hu": 94.4
      },
      {
        "phase": "DEL",
        "mean_hu": 71.0
      }
    ],
    "delta_art_dry": 33.0,
    "delta_ven_art": 14.6,
    "delta_del_art": -8.8,
    "delta_del_ven": -23.4
  },
  "CASE00002": {
    "id": "CASE00002",
    "slices": [
      80,
      81,
      83,
      84,
      85,
      86,
      88,
      89,
      90,
      92,
      93,
      94,
      96,
      97,
      98,
      100,
      101,
      102,
      103,
      105,
      106
    ],
    "primary_slice": 96,
    "shape": [
      194,
      512,
      512
    ],
    "annotation_keys": [
      "lesion"
    ],
    "voxel_count": 9326,
    "slice_bounds": [
      83,
      103
    ],
    "slice_count": 21,
    "phase_stats": {
      "DRY": {
        "lesion_mean_hu": -35.5,
        "lesion_std_hu": 31.2
      },
      "ART": {
        "lesion_mean_hu": 11.1,
        "lesion_std_hu": 40.5
      },
      "VEN": {
        "lesion_mean_hu": 26.0,
        "lesion_std_hu": 54.1
      },
      "DEL": {
        "lesion_mean_hu": 2.7,
        "lesion_std_hu": 48.3
      }
    },
    "slice_stats": [],
    "overlays": {},
    "trajectory": [
      {
        "phase": "DRY",
        "mean_hu": -35.5
      },
      {
        "phase": "ART",
        "mean_hu": 11.1
      },
      {
        "phase": "VEN",
        "mean_hu": 26.0
      },
      {
        "phase": "DEL",
        "mean_hu": 2.7
      }
    ],
    "delta_art_dry": 46.6,
    "delta_ven_art": 14.9,
    "delta_del_art": -8.4,
    "delta_del_ven": -23.3
  }
};

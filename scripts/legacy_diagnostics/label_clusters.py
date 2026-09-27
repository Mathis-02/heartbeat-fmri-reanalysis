from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from nilearn.datasets import fetch_atlas_harvard_oxford


SUBJECT = "sub-09210"

CLUSTERS_FILE = Path(
    f"derivatives/{SUBJECT}/stats/"
    "heart_minus_sound_clusters_fdr05.tsv"
)

OUTPUT_FILE = Path(
    f"derivatives/{SUBJECT}/stats/"
    "heart_minus_sound_clusters_labeled.tsv"
)


# --- Load cluster table ---

clusters = pd.read_csv(
    CLUSTERS_FILE,
    sep="\t",
)


# --- Load Harvard-Oxford atlases ---

cortical = fetch_atlas_harvard_oxford(
    "cort-maxprob-thr25-2mm"
)

subcortical = fetch_atlas_harvard_oxford(
    "sub-maxprob-thr25-2mm"
)

def get_img(atlas_maps):
    if isinstance(
        atlas_maps,
        nib.spatialimages.SpatialImage,
    ):
        return atlas_maps

    return nib.load(atlas_maps)


cort_img = get_img(cortical.maps)
sub_img = get_img(subcortical.maps)

cort_data = cort_img.get_fdata().astype(int)
sub_data = sub_img.get_fdata().astype(int)

cort_inv_affine = np.linalg.inv(
    cort_img.affine
)

sub_inv_affine = np.linalg.inv(
    sub_img.affine
)


# --- LUT dictionaries ---

cort_lut = dict(
    zip(
        cortical.lut["index"],
        cortical.lut["name"],
    )
)

sub_lut = dict(
    zip(
        subcortical.lut["index"],
        subcortical.lut["name"],
    )
)


def atlas_label(
    coord,
    atlas_data,
    inverse_affine,
    lut,
):
    ijk_float = nib.affines.apply_affine(
        inverse_affine,
        coord,
    )

    ijk = np.round(
        ijk_float
    ).astype(int)

    shape = atlas_data.shape

    if np.any(ijk < 0) or np.any(ijk >= shape):
        return "Outside atlas"

    label_index = int(
        atlas_data[
            ijk[0],
            ijk[1],
            ijk[2],
        ]
    )

    if label_index == 0:
        return "Unlabelled"

    return lut.get(
        label_index,
        f"Unknown label {label_index}",
    )


# --- Label every reported peak ---

cortical_labels = []
subcortical_labels = []

for _, row in clusters.iterrows():

    coord = (
        float(row["X"]),
        float(row["Y"]),
        float(row["Z"]),
    )

    cortical_labels.append(
        atlas_label(
            coord,
            cort_data,
            cort_inv_affine,
            cort_lut,
        )
    )

    subcortical_labels.append(
        atlas_label(
            coord,
            sub_data,
            sub_inv_affine,
            sub_lut,
        )
    )


clusters["HarvardOxford_cortical"] = (
    cortical_labels
)

clusters["HarvardOxford_subcortical"] = (
    subcortical_labels
)


# --- Save ---

clusters.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
)

print(
    clusters[
        [
            "Cluster ID",
            "X",
            "Y",
            "Z",
            "Peak Stat",
            "Cluster Size (mm3)",
            "HarvardOxford_cortical",
            "HarvardOxford_subcortical",
        ]
    ].head(25)
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

# --- Keep only main cluster peaks ---

main_clusters = clusters[
    ~clusters["Cluster ID"]
    .astype(str)
    .str.contains("[a-zA-Z]", regex=True)
].copy()

positive = (
    main_clusters[
        main_clusters["Peak Stat"] > 0
    ]
    .sort_values(
        "Peak Stat",
        ascending=False,
    )
    .head(10)
)

negative = (
    main_clusters[
        main_clusters["Peak Stat"] < 0
    ]
    .sort_values(
        "Peak Stat",
        ascending=True,
    )
    .head(10)
)

print("\nTop Heart > Sound clusters:")
print(
    positive[
        [
            "Cluster ID",
            "X",
            "Y",
            "Z",
            "Peak Stat",
            "Cluster Size (mm3)",
            "HarvardOxford_cortical",
            "HarvardOxford_subcortical",
        ]
    ]
)

print("\nTop Sound > Heart clusters:")
print(
    negative[
        [
            "Cluster ID",
            "X",
            "Y",
            "Z",
            "Peak Stat",
            "Cluster Size (mm3)",
            "HarvardOxford_cortical",
            "HarvardOxford_subcortical",
        ]
    ]
)
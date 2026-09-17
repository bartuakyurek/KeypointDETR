import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import Isomap

from utils.data_utils import naive_read_pcd



ID2NAMES = {
    "02691156": "airplane",
    "02808440": "bathtub",
    "02818832": "bed",
    "02876657": "bottle",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03467517": "guitar",
    "03513137": "helmet",
    "03624134": "knife",
    "03642806": "laptop",
    "03790512": "motorcycle",
    "03797390": "mug",
    "04225987": "skateboard",
    "04379243": "table",
    "04530566": "vessel",
}

NAMES2ID = {v: k for k, v in ID2NAMES.items()}


def geo_distance_matrix(points):
    isomap = Isomap(
        n_components=2,
        n_neighbors=15, # WARNING: This was set to 5 previously but yields in unconnected components, and hence there was a warning print. Increasing it reduces efficiency warnings.
        path_method="auto",
    )
    isomap.fit_transform(points)
    return isomap.dist_matrix_


def write_keypoints_geo_distance_matrix(pcd_file, kp_idx, out_file):
    pcd = naive_read_pcd(pcd_file)[0]

    distance_matrix = geo_distance_matrix(pcd)
    distance_matrix = distance_matrix[kp_idx]

    np.savetxt(
        out_file,
        distance_matrix,
        fmt="%f",
        delimiter=",",
    )


def process_class(class_name, pcd_root, anno_file):
    print(f"[INFO] Processing class: {class_name}...")
    class_id = NAMES2ID[class_name]

    class_pcd_root = pcd_root / class_id

    with open(anno_file) as f:
        annots = json.load(f)

    annots = [
        annot
        for annot in annots
        if annot["class_id"] == class_id
    ]

    keypoints = {
        annot["model_id"]: [
            kp_info["pcd_info"]["point_index"]
            for kp_info in annot["keypoints"]
        ]
        for annot in annots
    }

    pcd_files = class_pcd_root.glob("*.pcd")

    for pcd_file in pcd_files:
        filename = pcd_file.stem
        write_file = class_pcd_root / f"{filename}.txt"

        if filename not in keypoints:
            print(f"WARNING: No keypoints found for {filename}")
            continue

        kp_idx = keypoints[filename]

        write_keypoints_geo_distance_metrix(pcd_file, kp_idx, write_file)
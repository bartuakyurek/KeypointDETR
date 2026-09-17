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
        n_neighbors=20, # WARNING: This was set to 5 previously but yields in unconnected components, and hence there was a warning print. Increasing it reduces efficiency warnings.
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


def process_class(class_name, pcd_root, anno_file, recompute=False):
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

        if write_file.exists() and not recompute:
            print(f"[SKIP] Already exists: {write_file}")
            continue

        kp_idx = keypoints[filename]

        write_keypoints_geo_distance_matrix(
            pcd_file,
            kp_idx,
            write_file,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute geodesic distance matrices for KeypointNet."
    )

    parser.add_argument(
        "--class",
        dest="classes",
        nargs="+",
        required=True,
        help="Classes to process. Use 'all' to process all classes.",
    )

    parser.add_argument(
        "--pcd-root",
        type=Path,
        default=Path("../data/keypointnet_data/pcds"),
        help="Root directory containing the PCDs.",
    )

    parser.add_argument(
        "--anno-file",
        type=Path,
        default=Path("../data/keypointnet_data/annotations/all.json"),
        help="Keypoint annotation JSON file.",
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute geodesic distances even if the output .txt already exists.",
    )


    return parser.parse_args()


def main():
    args = parse_args()

    if "all" in args.classes:
        classes = list(NAMES2ID.keys())
    else:
        classes = args.classes

    # Validate class names before doing any computation
    invalid_classes = [
        class_name
        for class_name in classes
        if class_name not in NAMES2ID
    ]

    if invalid_classes:
        raise ValueError(
            f"Unknown class(es): {invalid_classes}\n"
            f"Available classes: {list(NAMES2ID.keys())}"
        )

    for class_name in classes:
        print(f"\n{'=' * 60}")
        print(f"Processing class: {class_name}")
        print(f"{'=' * 60}")

        process_class(
            class_name,
            args.pcd_root,
            args.anno_file,
            args.recompute,
        )


if __name__ == "__main__":
    main()

import tqdm
import click
import torch
import numpy as np
import trimesh
import os 
from pathlib import Path
from datetime import datetime

from scipy.spatial.distance import cdist
from pl_model import LitModel
from data.st_data import KPS_Geodesic_Dataset, NAMES2ID
from utils.metrics import get_cd, hungary_iou

# TODO: Automatically detect the model version 

@click.command()
@click.option('--data_root', type=str, default='../data/keypointnet_data')
@click.option('--checkpoint', type=str, default='./runs/keypoint_saliency/version_XX/checkpoints/last.ckpt')   
@click.option('--gpus', default=1)
def run(checkpoint, gpus, data_root, visualize=False):
    model = LitModel.load_from_checkpoint(checkpoint).cuda()
    model.eval()

    args = model.hparams.args

    # Directories of input test data
    test_file = 'test.txt' #os.path.join(data_root,'splits', 'test.txt')
    mesh_root = os.path.join(data_root, 'ShapeNetCore.v2.ply')
    class_id = NAMES2ID[args.class_name]
    print(args, "\n -->", test_file)
    dataset = KPS_Geodesic_Dataset(args, test_file, False)

    # Create directories for saving the results
    results_dir = Path('./results')
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    results_dir = Path(os.path.join(results_dir, args.class_name + '_' + timestamp))
    results_dir.mkdir(parents=True, exist_ok=True)

    mcd = []
    hmiou = {}

    n_thresholds = 11 # For Hungarian mIoU metric
    for i in range(n_thresholds):
        key = i * 0.01
        hmiou[key] = []

    for i in tqdm.tqdm(range(len(dataset))):
        pc, heat, mesh_name = dataset[i]

        pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()
        heat = torch.tensor(heat, dtype=torch.float32).unsqueeze(0).cuda()
        # pred
        with torch.no_grad():
            # TODO: How to determine the size of the prediction? Can we change it via an argument (probably need to adjust the model's final layer)
            pts, gts = model.infer(pc, heat)
            pts = pts.cpu().numpy()
            gts = gts.cpu().numpy()

        dists = cdist(gts, pts, metric='euclidean')
        cd = get_cd(dists)
        mcd.append(cd)

        for j in range(n_thresholds): 
            key = j * 0.01
            hiou = hungary_iou(dists, dist_thresh=key)
            hmiou[key].append(hiou)

        # Save to results/{modelname} as:
        # 1- OK .txt of the predicted points
        # 2- OK snapshots of offline rendering
        # 3- OK (turn off the online visualization)
        # 4- TODO: Plot the errors for sanity check
        #print("[BARTU DEBUG] Number of predicted/ground truth keypoints per mesh: ", len(pts), "/", len(gts))

        # Save predicted checktpoint coordinates as "x y z" per row ---------------------------
        prediction_str = ""
        for pred_pt in pts:
            for coord in pred_pt:
                prediction_str += str(coord) + " "
            prediction_str += "\n"
        
        preds_path = Path(results_dir) / "control_pts" / f"preds_{i}.txt"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_path.write_text(prediction_str)

        # Save a snapshot of the predictions --------------------------------------------------
        mesh = trimesh.load(os.path.join(mesh_root, class_id, mesh_name + '.ply'))
        gt_pts = [trimesh.primitives.Sphere(radius=0.02, center=pt).to_mesh() for pt in gts]
        for pt in gt_pts:
            pt.visual.vertex_colors = (0, 255, 0, 255)
        pred_pts = [trimesh.primitives.Sphere(radius=0.02, center=pt).to_mesh() for pt in pts]
        for pt in pred_pts:
            pt.visual.vertex_colors = (0, 0, 255, 255)
        scene = trimesh.Scene([mesh] + gt_pts + pred_pts)

        # Render to an image
        png = scene.save_image(resolution=(600, 600), visible=False)
        png_dir = Path(results_dir) / "renders" / f"{args.class_name}_{i}.png"
        png_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(png_dir, "wb") as f:
            f.write(png)

        if visualize:
            scene.show()
        # ------------------------------------------------------------------------------------

    print("Threshold \t Average Hungarian mIoU")
    for j in range(n_thresholds): 
        key = j * 0.01 
        print( key, "\t", np.mean(hmiou[key]))

    print(np.mean(mcd))


if __name__ == "__main__":

    # Bartu edit: workaround for a pickle error
    # based on https://github.com/m-bain/whisperX/issues/1304
    _original_torch_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _trusted_load
    # End of bartu edit


    # Find the latest checkpoint file

    run()

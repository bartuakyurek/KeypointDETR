# KeypointDETR: an end-to-end 3d keypoint detector

> [!WARNING]
> This repository is forked only for research purposes.
> I have done some changes to fix the errors of the
> published implementation at the time. And the code might
> be further adapted for other unrelated research tasks. 
> Please refer to the original repo of KeypointDETR.

## Run

### 1. Extracting Geodesic Distance Maps

Geodesic distance maps are extracted during the data preprocessing phase for generating the ground truth heatmaps. 
Compute the shortest geodesic distance from points to the keypoints and save the results as '.txt' files.


For selected classes:
``` 
python geodesic_distance.py --class guitar table chair
``` 

Running for all classes:
```
python geodesic_distance.py --class all
```

>[!INFO]
> These arguments are added as a part of this fork for convenience.
> Make sure the dataset is under ../data/keypoints_data
> and it should contain /splits folder which is available 
> in the repository of the dataset [https://github.com/qq456cvb/KeypointNet/tree/master/splits](here).

### 2. Config

Modify the config [config/keypoint_saliency.yaml](https://github.com/bibi547/KeypointDETR/blob/master/config/keypoint_saliency.yaml) for your path, filename, and categories.

### 3. Requirement

> [!WARNING]
> These requirements are extracted for this forked repository,
> they do not necessarily correspond to the official implementation.

NOTE: These requirements are only tested under Linux Fedora Workstation 45.
`` pip install -r requirements.txt``` 

### 4. Train

```
python train.py
```


### 5. Test

```
python test.py
```

## Citation

```
@inproceedings{jin2024keypointdetr,
  title={KeypointDETR: an end-to-end 3d keypoint detector},
  author={Jin, Hairong and Shen, Yuefan and Lou, Jianwen and Zhou, Kun and Zheng, Youyi},
  booktitle={ECCV},
  year={2024}
}
```

## Acknowledgement

[DGCNN](https://github.com/WangYueFt/dgcnn)

[Point Transformer](https://github.com/qq456cvb/Point-Transformers)

[KeypointNet Dataset](https://github.com/qq456cvb/KeypointNet.git)


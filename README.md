# P2-Net
This paper focus on joint description and detection of local features for pixel and point matching:
[P2-Net: Joint Description and Detection of Local Features for Pixel and Point Matching](https://arxiv.org/abs/2103.01055) by Bing Wang, Changhao Chen, Zhaopeng Cui, Jie Qin, Chris Xiaoxuan Lu, Zhengdi Yu, Peijun Zhao, Zhen Dong, Fan Zhu, Niki Trigoni, Andrew Markham.
## Introduction
Accurate 2D and 3D keypoint detection and description are vital for establishing image-point cloud correspondences. We propose a dual fully convolutional framework to directly match pixels and points, enabling fine-grained correspondence establishment. Our approach, integrating an ultra-wide reception mechanism and novel loss function, mitigates information variations between pixel and point local regions.
### Network Architecture
![Network](https://github.com/BingCS/P2-Net/blob/main/figs/Network.png)
## Installation
Create the environment and install the required libaries:
```bibtex
conda create -n P2NET python==3.8
conda activate P2NET
conda install --file requirements.txt
```
Compile the C++ extension module for python located in **cpp_wrappers**. Open a terminal in this folder, and run:
```bibtex
sh compile_wrappers.sh
```
The code has been tested on Python 3.8, PyTorch 1.3.1, GCC 11.3 and CUDA 11.7, but it should work with other configurations.
## Dataset Download
### Indoor Datasets
#### (1)7Scenes
The dataset can be downloaded from [here](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

Let's take the '7-scenes-chess' scene as an example to explore its data format: Inside, there's a file named 'camera-intrinsics.txt' containing the internal parameters of the scene's camera.  Additionally, there are multiple folders named 'seq-xx' each containing a bunch of frames saved as '.color.png', '.depth.png', and '.pose.txt' files.
```bibtex
7-Scenes--7-scenes-chess--camera-intrinsics.txt
                       |--seq-01--.color.png
                               |--.depth.png
                               |--.pose.txt
                       |--seq-02
                       |--...
       |--7-scenes-fire
       |--...
```
                         
                          
        
Although the 7scenes dataset itself doesn't provide point cloud data, you can generate it yourself using these files. Just run the '7scenes_gen.py' script located in the 'data/tools' directory. This script is capable of generating point cloud data from the 7scenes dataset and producing pairs of '.pkl' format files required for training and validation.

```bibtex
python 7scenes_gen.py
```
#### (2)RGB-D Scenes V2
The dataset can be downloaded from [here](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/).

RGB-D Scenes V2 contains 14 scenes, ranging from `scene_01` to `scene_14`, with the first 10 scenes used for training and the last 4 scenes used for testing.

The format of this dataset is the same as that of the 7Scenes dataset. Simply run the 'RGB-D_gen.py' script located in the 'data/tools' directory. The script can generate point cloud data from the corresponding data set and generate the '.pkl' format file pairs required for training and validation.

```bibtex
python RGB-D_gen.py
```
#### (3)3DMatch
The dataset can be downloaded from [here](https://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets).

3DMatch dataset consists of data from 62 scenes, with 54 scenes used for training and 8 scenes used for evaluation. The specific scene names can be found in `train.txt` and `test.txt`.

Generate the '.pkl' format file pairs required for training and validation：

```bibtex
python 3DMatch_gen.py
```
#### (4)ScanNet
The dataset can be downloaded from [here](https://github.com/ScanNet/ScanNet).

ScanNet is an RGB-D video dataset. We use the smaller subset option provided by the authors, `scannet_frames_25k`, which includes both the training and test sets.

Organize the data format of this dataset in 7Scenes format.

Generate the '.pkl' format file pairs required for training and validation：

```bibtex
python ScanNet_gen.py
```
### Outdoor Datasets
#### Kitti-DC
Kitti-DC is an outdoor dataset with 342 RGB-3D point cloud pairs from 4 distinct urban scenes, collected using a 64-line LiDAR scanner mounted on a moving vehicle.

The dataset is accessible in Google Cloud:
[Kitti-DC](https://drive.google.com/file/d/1c1TcUV2fMmXKK_vyZstVLD9J4-pCVCRu/view).

Organize the data format of this dataset in 7Scenes format.

Generate the '.pkl' format file pairs required for training and validation：

```bibtex
python Kitti-DC_gen.py
```
## Trianing
(1)Take the 7Scenes dataset as an example. The training on 7Scenes dataset can be done by running:
```bibtex
python train_p2net.py  --opt_p2net data_dir="./7Scenes"  img_dir="./7Scenes"
```

(2)The training on RGB-D Scenes V2 dataset can be done by running:
```bibtex
python train_p2net.py  --opt_p2net data_dir="./RGB-D"  img_dir="./RGB-D"
```

(3)The training on 3DMatch dataset can be done by running:
```bibtex
python train_p2net.py  --opt_p2net data_dir="./3DMatch"  img_dir="./3DMatch"
```

(4)The training on ScanNet dataset can be done by running:
```bibtex
python train_p2net.py  --opt_p2net data_dir="./ScanNet"  img_dir="./ScanNet"
```

(5)The training on Kitti-DC dataset can be done by running:
```bibtex
python train_p2net.py  --opt_p2net data_dir="./Kitti-DC"  img_dir="./Kitti-DC"
```

## Testing
(1)7Scenes

Using the model trained in the previous step, extract keypoints, descriptors, and calculate scores for the dataset:
```bibtex
python test_p2net.py --run extractor  --opt_evaluation data_dir="./7Scenes"  img_dir="./7Scenes"
```
Finally, execute the evaluation function and pass in the paths to the extracted results and the data as parameters:
```bibtex
python test_p2net.py --run evaluator  --opt_evaluation data_dir="./7Scenes"  img_dir="./7Scenes"
```


(2)RGB-D Scenes V2

Using the model trained in the previous step, extract keypoints, descriptors, and calculate scores for the dataset:
```bibtex
python test_p2net.py --run extractor  --opt_evaluation data_dir="./RGB-D"  img_dir="./RGB-D"
```
Finally, execute the evaluation function and pass in the paths to the extracted results and the data as parameters:
```bibtex
python test_p2net.py --run evaluator  --opt_evaluation data_dir="./RGB-D"  img_dir="./RGB-D"
```


(3)3DMatch

Using the model trained in the previous step, extract keypoints, descriptors, and calculate scores for the dataset:
```bibtex
python test_p2net.py --run extractor  --opt_evaluation data_dir="./3DMatch"  img_dir="./3DMatch"
```
Finally, execute the evaluation function and pass in the paths to the extracted results and the data as parameters:
```bibtex
python test_p2net.py --run evaluator  --opt_evaluation data_dir="./3DMatch"  img_dir="./3DMatch"
```


(4)ScanNet

Using the model trained in the previous step, extract keypoints, descriptors, and calculate scores for the dataset:
```bibtex
python test_p2net.py --run extractor  --opt_evaluation data_dir="./ScanNet"  img_dir="./ScanNet"
```
Finally, execute the evaluation function and pass in the paths to the extracted results and the data as parameters:
```bibtex
python test_p2net.py --run evaluator  --opt_evaluation data_dir="./ScanNet"  img_dir="./ScanNet"
```


(5)Kitti-DC

Using the model trained in the previous step, extract keypoints, descriptors, and calculate scores for the dataset:
```bibtex
python test_p2net.py --run extractor  --opt_evaluation data_dir="./Kitti-DC"  img_dir="./Kitti-DC"
```
Finally, execute the evaluation function and pass in the paths to the extracted results and the data as parameters:
```bibtex
python test_p2net.py --run evaluator  --opt_evaluation data_dir="./Kitti-DC"  img_dir="./Kitti-DC"
```

## Results
Evaluation on Three Benchmarks:
| PnP | FMR  | IR  |  IN  | RR  |   
|------|------|------|------|------|
| 3DMatch | 99.4 | 47.9 | 196.2 | 66.9 | 
| ScanNet | 97.6 | 57.5 | 143.6 | 76.7 | 
| Kitti-DC | 100 | 67.9 | 175.1 | 98.8 |

| Kabsch | FMR  | IR  |  IN  | RR  |   
|------|------|------|------|------|
| 3DMatch | 100 | 51.1 | 196.2 | 94.9 | 
| ScanNet | 97.6 | 57.5 | 143.6 | 85.9 | 
| Kitti-DC | 100 | 67.9 | 175.1 | 97.7 |

Evaluation on the RGB-D Scenes V2 dataset:
|  | Scenes11  | Scenes12  |  Scenes13  | Scenes14  |   
|------|------|------|------|------|
| FMR | 94.6 | 96.7 | 88.1 | 69.4 | 
| RR | 79.5 | 76.2 | 56.3 | 45.6 | 
| IR | 38.8 | 47.2 | 38.2 | 26.6 | 

Evaluation on the 7Scenes dataset:
|  | Chess  | Fire  |  Heads  | Office  | Pumpkin  | Kitchen  |  Stairs  | 
|------|------|------|------|------|------|------|------|
| FMR | 100 | 99.2 | 88.1 | 91.8 | 89.5 | 92.5 | 64.3 |
| RR | 95.8 | 84.1 | 94.6 | 83.8 | 70.7 | 68.9 | 65.8 |
| IR | 72.1 | 69.2 | 59.2 | 69.0 | 62.0 | 62.8 | 44.8 |

## Visualization
Below are some sample visualization results of our method:
![Matching visualization1](https://github.com/BingCS/P2-Net/blob/main/figs/visualization1.png)
![Matching visualization2](https://github.com/BingCS/P2-Net/blob/main/figs/visualization2.png)
![Matching visualization3](https://github.com/BingCS/P2-Net/blob/main/figs/visualization3.png)

## Citation
If you find this project useful, please cite:
```bibtex
@article{wang2021p2net,
  title={P2-Net: Joint Description and Detection of Local Features for Pixel and Point Matching},
  author={Bing Wang and Changhao Chen and Zhaopeng Cui and Jie Qin and Chris Xiaoxuan Lu and Zhengdi Yu and Peijun Zhao and Zhen Dong and Fan Zhu and Niki Trigoni and Andrew Markham},
  journal={arXiv:2103.01055 [cs.CV]},
  year={2021}
}
```

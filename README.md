# GVF-TAPE -- Video Generation Models

### Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-Top Manipulation [CoRL 2025]

[[project page]](https://clearlab-sustech.github.io/gvf-tape/) [[Arxiv]](https://arxiv.org/abs/2509.00361)

[Chuye Zhang](https://zhangchuye.github.io)<sup>1</sup> [Xiaoxiong Zhang](https://xiaoxiongzzzz.github.io)<sup>1</sup> [Wei Pan](https://weisonweileen.github.io/#/)<sup>1</sup> [Linfang Zheng](https://lynne-zheng-linfang.github.io)<sup>†2,3</sup> [Wei Zhang](https://faculty.sustech.edu.cn/?tagid=zhangw3&go=2)<sup>†1,2</sup>

<sup>1</sup>Southern University of Science and Technology, <sup>2</sup>LimX Dynamics, <sup>3</sup>The University of HongKong

This codebase is official implementation of GVF-TAPE, it contains codes to train a flow-based video generation model.

## Setting Environment
You can set up the environment with conda

```
git clone https://github.com/Xiaoxiongzzzz/GVF-TAPE-Video-Generation.git
cd GVF-TAPE-Video-Generation
conda env create -f environment.yml
conda activate videogenerator
```
## Dataset Structure
The pytorch dataset class is defined in ./flowdiffusion/dataset.py.

This class is based on [LIBERO](https://libero-project.github.io/main.html) dataset with some modification:
* Agentview_rgb in LIBERO dataset is upside down, you need to invert it.
* If you need generate RGB-D videos, then augmenting this dataset with depth information is needed. It's can be achieved by using [VideoDepthAnything](https://github.com/DepthAnything/Video-Depth-Anything) or other model.
* Preprocess the task description to text embedding and save it to dataset. */flowdiffusion/preprocess_dataset.py script could do this.

You can also write your dataset class based on our implementation.

## Training
You can specify the dataset path and output path in */flowdiffusion/train_rectified_flow.py and train easily:
```
accelerate launch flowdiffusion/train_rectified_flow.py
```
In our experiments, sample step = 3 is enough for LIBERO.

## Ackownledgement
This codebase is based on [AVDC](https://github.com/flow-diffusion/AVDC), thanks for their excellent code.

Contact [Xiaoxiong Zhang](https://xiaoxiongzzzz.github.io) if you have any questions and suggestions.
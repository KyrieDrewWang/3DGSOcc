# Installation instructions

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n splattingocc python=3.8 -y
conda activate splattingocc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

**c. Install mmcv-full.**

```shell
pip install mmcv-full==1.6.0
```

**d. Install mmdet and mmseg.**

```shell
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```

**e. Install the splatting module.**
```shell
cd mmdet3d/models/gaussplating/submodules/diff-gaussian-rasterization_contrastive_f_1
pip install .
# under mmdet3d/models/gaussplating/submodules/:
# diff-gaussian-rasterization_contrastive_f_1 is for only rendering one feature map (can be colors, semantic, or depth)
# diff-gaussian-rasterization-d is for rendering semantic, depth, and colors
# diff-gaussian-rasterization-f is for rednering semantic and colors
# currently we try diff-gaussian-rasterization_contrastive_f_1 for the simplest situation and the others can be ignored
```

**f. Install splattingocc from source code.**

```shell
git clone git@github.com:KyrieDrewWang/3DGSOcc.git
cd 3DGSOcc
pip install -v -e .
# python setup.py install
```

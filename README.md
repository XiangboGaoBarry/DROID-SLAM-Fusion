This project is built upon DROID-SLAM. To run the code, please follow the [DROID-SLAM instructions](https://github.com/princeton-vl/DROID-SLAM), including dataset downloading, demo, and training. Our project does not require extra dependency.

We propose to use some feature-level fusion methods to enhance the feature extraction procedure. In this project, we implement feature concatenation, self-attention fuser, and deformable-attention fuser.

To train with our feature-level fusion modules, run
```
python train.py --datapath=<path to tartanair> --gpus=4 --lr=0.00025 --dual_backbone --fusion_method [concat/self_att/deform_att]
```

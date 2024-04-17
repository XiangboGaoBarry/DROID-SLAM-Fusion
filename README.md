This project is built upon DROID-SLAM. To run the code, please follow the [DROID-SLAM instructions](https://github.com/princeton-vl/DROID-SLAM), including dataset downloading, demo, and training. Our project does not require extra dependency.

If you would like to calculate the GFLOPs
```
pip install ptflops
cd droid_slam
python flop_cal.py --fusion_method [Optional] --net [Optional] --stereo
```
Run `python flop_cal.py --help` for more information

## Efficiency Comparsion

We propose to use some feature-level fusion methods to enhance the feature extraction procedure. In this project, we implement feature concatenation, self-attention fuser, and deformable-attention fuser.

To train with our feature-level fusion modules, run
```
python train.py --datapath=<path to tartanair> --gpus=4 --lr=0.00025 --dual_backbone --fusion_method [concat/self_att/deform_att]
```
Only TartanAir and EuRoC datasets are supported for our current implementation.


## Results Comparison

We trained the without feature-level fusion(DROID-SLAM) and with our feature-level fusion: feature concatenation, self-attention fuser, and deform-attention fuser. For the original DROID-SLAM implementation, we evaluated the mono image and stereo images. For our feature-level fusion, results are only evaluated on stereo images.

### EuRoC
|                       | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | Avg  |
|-----------------------|------|------|------|------|------|------|------|------|------|------|------|------|
| DROID-SLAM(mono)      | 0.018| 0.015| 0.070| 0.054| 0.074| 0.051| 0.013| 0.031| 0.023| 0.021| 0.041| 0.037|
| DROID-SLAM(stereo)    | 0.016| 0.015| 0.052| 0.044| 0.060| 0.041| 0.015| 0.022| 0.020| 0.018| 0.026| 0.030|
| Feat Concat(ours)     | 0.016| 0.013| 0.047| 0.036| 0.056| **0.036**| 0.015| 0.018| **0.017**| **0.017**| 0.026| 0.027|
| Self-Att Fuser(ours)  | **0.014**| **0.012**| **0.038**| **0.032**| 0.044| 0.040| 0.012| **0.017**| 0.020| **0.017**| 0.019| **0.024**|
| Deform-Att Fuser(ours)| **0.014**| **0.012**| 0.039| 0.034| 0.043| 0.040| **0.010**| **0.017**| **0.017**| 0.018| **0.017**| **0.024**|


### TartanAir
|                       | MH000| MH001| MH002| MH003| MH004| MH005| MH006| MH007| Avg  |
|-----------------------|------|------|------|------|------|------|------|------|------|
| DROID-SLAM(mono)      | 0.217| 0.152| 0.117| 0.070| **0.030**| **4.072**| 0.707| 0.184| 0.694|
| DROID-SLAM(stereo)    | 0.199| 0.136| **0.120**| 0.061| **0.030**| 4.230| 0.612| 0.155| 0.693|
| Feat Concat(ours)     | 0.187| 0.142| 0.131| 0.067| **0.031**| 4.611| 0.619| 0.169| 0.745|
| Self-Att Fuser(ours)  | 0.169| 0.125| 0.136| 0.062| **0.031**| 4.086| 0.605| 0.147| 0.670|
| Deform-Att Fuser(ours)| **0.164**| **0.122**| 0.137| 0.063| 0.031| 4.104| **0.603**| **0.140**| **0.671**|



# Anonymous-1275
Anonymous-1275 for acm mm2023

**Model Parameter Files Setup**
1. Download the model parameter files:

   - Parameter for pre-trained backbone: [mit_b4.pth](https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b)
    - backup: https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b (Code: 4l3b)
     
   - Parameter for our sota model: [The_best_Epoch.pth](https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b)
    - backup: https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b (Code: 4l3b)
     

2. Place the downloaded `mit_b4.pth` and `The_best_Epoch.pth` files in the appropriate paths:

   - Place `mit_b4.pth` in the `lib/backbone/` (recommended path).
   - Place `The_best_Epoch.pth` in the `experiment/exp_icformer_1/` (recommended path).

3. (Optional) You can update the file paths in the code for these parameters:

   - In `/lib/network/network_demo2.py`, set the path for `mit_b4.pth`.
   - In `test.py`, set the path for `The_best_Epoch.pth`.

## Train and Test Settings

### Datasets

Please follow the instructions in Section 4.1 of the paper to properly split your dataset. After splitting the dataset, update the corresponding arguments in `train.py` and `test.py` accordingly.

(Optional) You can refer to the work of MICCAI2020 https://github.com/DengPingFan/PraNet for related datasets.

### Train Settings

1. Update the dataset path in `train.py` according to your dataset location.

### Test Settings

1. Update the dataset path in `test.py` according to your dataset location.
2. Ensure that the trained model path is correctly set in `test.py` to load the appropriate model for testing.



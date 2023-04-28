# Anonymous-1275
Anonymous-1275 for acm mm2023

**Model Parameter Files Setup**
1. Download the model parameter files:

   - Parameter for pre-trained backbone: [mit_b4.pth](https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b)
   -- backup: https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b (Code: 4l3b)
     
   - Parameter for our sota model: [The_best_Epoch.pth](https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b)
   -- backup: https://pan.baidu.com/s/1ZPzha0T-crFndYjISoIj6w?pwd=4l3b (Code: 4l3b)
     

2. Place the downloaded `mit_b4.pth` and `The_best_Epoch.pth` files in the appropriate paths:

   - Place `mit_b4.pth` in the `lib/backbone/` directory (recommended path).
   - Place `The_best_Epoch.pth` in the `experiment/exp_icformer_1/` directory (recommended path).

3. (Optional) You can update the file paths in the code for these parameters:

   - In `/lib/network/network_demo2.py`, set the path for `mit_b4.pth`.
   - In `test`, set the path for `The_best_Epoch.pth`.

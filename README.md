ICFormer: Transformer with Inverse-Attention and Contrastive Learning for Polyp Segmentation

### Datasets
1. You can refer to the work of ACMMM2021 https://github.com/plemeri/UACANet for related datasets.

2. Please follow the Section 4.1 of the paper to properly split your dataset. 
3. After splitting the dataset, update the corresponding path in `train.py` and `test.py` accordingly.



### Model Parameter Files Setup
1. Download the model parameter files:
   - The parameter will be released as soon.

2. Place the downloaded `mit_b4.pth` and `The_best_Epoch.pth` files in the appropriate paths:
   - Place `mit_b4.pth` in the `lib/backbone/` (recommended path).
   - Place `The_best_Epoch.pth` in the `experiment/exp_icformer_1/` (recommended path).

3. (Optional) You can update the file paths in the code for these parameters:

   - In `/lib/network/network_demo2.py`, set the path for `mit_b4.pth`.
   - In `test.py`, set the path for `The_best_Epoch.pth`.




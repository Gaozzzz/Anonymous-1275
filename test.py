import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from lib.network.network_demo2 import Network as Net
from utils.dataloader import test_dataset
from utils.eval_functions import get_scores

if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--experiment_path', type=str, default='experiment', help='Set your experiment path')
    parser.add_argument('--testData_path', default='Set your test dataset path')
    opt = parser.parse_args()

    #   set model path
    Path = 'exp_icformer_1'
    Model = 'The_best_Epoch.pth'
    pth_path = os.path.join(opt.experiment_path, Path, Model)
    model = Net()
    model.load_state_dict(torch.load(pth_path))
    model.cuda()
    model.eval()

    for _data_name in ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300', ]:
        ##### put data_path here #####
        #   set your test dataset path
        data_path = f'{opt.testData_path}/{_data_name}'
        ##### save_path #####
        save_path = os.path.join(opt.experiment_path, Path, _data_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        prs = []
        gts = []
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            out = model(image)
            res = out[0]
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            prs.append(res)
            gts.append(gt)
        get_scores(gts, prs, _data_name)
        print(_data_name, 'Finish!')

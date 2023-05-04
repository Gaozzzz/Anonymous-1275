import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm

from lib.network.network_demo2 import Network
from utils.dataloader import PolypDataset, test_dataset
from utils.loss_function import structure_loss

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainData_path', default='set your train dataset path')
    parser.add_argument('--testData_path', default='set your test dataset path')
    parser.add_argument('--valData_path', default='set your val dataset path')
    parser.add_argument('--save_path', default='set your save path')

    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--batchsize', default=16)
    parser.add_argument('--epoch', default=50)
    parser.add_argument('--trainsize', default=352)
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def criterion(pred, target):
    loss = structure_loss(pred, target)
    return loss


def test(model, path, dataset, device):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        out = model(image)
        res = out[0]
        # eval Dice
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice
    return DSC / num1


def train(args):
    best = 0
    patience = 20
    early_stopping = 0
    best_epoch = 0
    train_loss = []
    train_loss_iter = []
    size_rates = [0.75, 1, 1.25]
    device = torch.device(args.device)
    os.makedirs(os.path.join(args.save_path), exist_ok=True)

    """dataloader"""
    image_root = os.path.join(args.trainData_path, 'images/')
    gt_root = os.path.join(args.trainData_path, 'masks/')

    train_dataset = PolypDataset(image_root, gt_root, args.trainsize, augmentations=args.augmentation)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batchsize,
                                   shuffle=True,
                                   num_workers=8,
                                   pin_memory=True)

    """model"""
    model = Network()
    model = model.to(device)

    """optimizer"""
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=1e-4, momentum=0.9)

    """train"""
    print("#" * 20, "Start Training", "#" * 20)
    Epoch = tqdm.tqdm(range(1, args.epoch + 1), desc='Epoch', total=args.epoch,
                      position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')

    for epoch in Epoch:
        model.train()
        iterator = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
            train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
        for _, pack in iterator:
            for rate in size_rates:
                image, gt = pack
                image = image.to(device)
                gt = gt.to(device)

                """rate"""
                trainsize = int(round(args.trainsize * rate / 32) * 32)
                if rate != 1:
                    image = F.upsample(image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gt = F.upsample(gt, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                optimizer.zero_grad()
                pred = model(image)
                res = F.interpolate(pred[0], size=gt.shape[-2:], mode='bilinear', align_corners=True)
                loss = criterion(res, gt) + pred[1] / 5
                train_loss_iter.append(loss.item())
                loss.backward()
                optimizer.step()

                """loss"""
                if rate == 1:
                    iterator.set_postfix({'loss': loss.item()})

        train_loss.append(np.mean(train_loss_iter))
        train_loss_iter = []

        """Val"""
        if (epoch + 1) % 1 == 0:
            valdice = test(model, args.valData_path, 'val', device)

            if valdice > best:
                best = valdice
                torch.save(model.state_dict(), f'{args.save_path}/The_best_Epoch.pth')
                print(f'Save the best model.The meandice is {best}.')
                early_stopping = 0
        early_stopping += 1
        if early_stopping > patience:
            print("Early_Stopping!")
            print("The best epoch: {}".format(best_epoch))
            print("The best mDice: {}".format(best))
            break

if __name__ == '__main__':
    args = _args()
    train(args)

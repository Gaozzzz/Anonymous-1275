import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def debug_tile(out, size=(100, 100)):
    debugs = []
    for debs in out:
        debug = []
        for i, deb in enumerate(debs):
            log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
            log = (log - log.min()) / (log.max() - log.min())
            log *= 255
            log = log.astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)


def plot_loss(train_loss, save_path):
    y_train_loss = train_loss
    x_train_loss = range(len(y_train_loss))

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('loss')

    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig(f'{save_path}/loss_curve.png')

import numpy as np

epsilon = 1e-7

def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def get_scores(gts, prs, _data_name):
    mean_precision = 0
    mean_recall = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_dice /= len(gts)

    print("dataset: {}, scores: mdice={}".format(_data_name, mean_dice))

    return mean_dice

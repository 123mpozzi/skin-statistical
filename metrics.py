import os, sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import math


## USAGE: python metrics.py <prediction-folder (predictions/HGR_small_on_ECU)>

# prevents zero division
smooth = 1e-20 #1e-07

# load images as numpy BOOL arrays (0-1)
def load_images(gt_path: str, pred_path: str, threshold: int = 128):
    # load as grayscale uint8
    gt_gray = np.array(Image.open(gt_path).convert('L')) # .convert('LA')
    pred_gray = np.array(Image.open(pred_path).convert('L')) #.convert('LA')
    # binarize and convert to bool
    gt_bool = gt_gray > threshold
    pred_bool = pred_gray > threshold

    # debug
    # print(f'Dims: gt-gray={gt_gray.shape} gt-bool={gt_bool.shape} p-gray={pred_gray.shape} p-bool={pred_bool.shape}')
    # display(Image.open(gt_path))
    # display(Image.open(gt_path).convert('L'))
    return gt_bool, pred_bool

# NEW AVERAGE: calculate average only of medium-scores (PRecision, REcall, SPecificity)
# y and p files must have the same filename
def pd_metrics_list(images: list, metric_fns: list, threshold: int = 128) -> list:
    out = []

    for entry in tqdm(images):
        y_pred = entry[0]
        y_true = entry[1]

        gt_gray = np.array(y_true.convert('L')) # .convert('LA')
        pred_gray = np.array(y_pred.convert('L')) #.convert('LA')
        # binarize and convert to bool
        gt_bool = gt_gray > threshold
        pred_bool = pred_gray > threshold

        # start adding item data into a structure
        idata = {}
        #idata['y'] = y_path
        #idata['p'] = p_path

        # load images from paths and apply threshold to binarize
        # the skin probabilities of predictions
        y_true = gt_bool
        y_pred = pred_bool
        confmat = confmat_scores(y_true, y_pred)

        # calculate metrics for the image couple loaded
        for metric_fn in metric_fns:
            f_name = metric_fn.__name__
            f_argcount = metric_fn.__code__.co_argcount # amount of argument in function definition

            if len(f_name.split('_')) > 1: # is a new-average metric, must not compute now
                continue

            # only one args: the metric only uses confusion matrix scores and is LUT-optimized
            if f_argcount == 1:
                idata[f_name] = metric_fn(confmat)
            # two args: confusion matrix scores aren't enough
            else:
                idata[f_name] = metric_fn(y_true, y_pred)

            # debug
            #display(Image.fromarray(y_true))
            #display(Image.fromarray(y_pred))
        
        # update the final data list
        out.append(idata)
    
    print(f'  Found {len(out)} matches')
    
    return out

# NEW AVERAGE: calculate average only of medium-scores (PRecision, REcall, SPecificity)
# y and p files must have the same filename
def pd_metrics(gt_dir: str, pred_dir: str, metric_fns: list, threshold: int = 128) -> list:
    out = []

    for y_filename in tqdm(os.listdir(gt_dir)):
        y_path = os.path.join(gt_dir, y_filename)
        p_filename = os.path.splitext(y_filename)[0] + '.png'
        p_path = os.path.join(pred_dir, p_filename) # pred are always PNG


        # start adding item data into a structure
        idata = {}
        idata['y'] = y_path
        idata['p'] = p_path

        # load images from paths and apply threshold to binarize
        # the skin probabilities of predictions
        y_true, y_pred = load_images(y_path, p_path, threshold)
        confmat = confmat_scores(y_true, y_pred)

        # calculate metrics for the image couple loaded
        for metric_fn in metric_fns:
            f_name = metric_fn.__name__
            f_argcount = metric_fn.__code__.co_argcount # amount of argument in function definition

            if len(f_name.split('_')) > 1: # is a new-average metric, must not compute now
                continue

            # only one args: the metric only uses confusion matrix scores and is LUT-optimized
            if f_argcount == 1:
                idata[f_name] = metric_fn(confmat)
            # two args: confusion matrix scores aren't enough
            else:
                idata[f_name] = metric_fn(y_true, y_pred)

            # debug
            #display(Image.fromarray(y_true))
            #display(Image.fromarray(y_pred))
        
        # update the final data list
        out.append(idata)
    
    print(f'  Found {len(out)} matches')
    
    return out

# print human-readable metrics results data
# 
# !!NOTE!!
# the mean F1 value calculated by summing all experiments F1 and divinding by N elements
# is different than the mean calculated by applying the F1 formula on average REcall and PRecision
def print_pd_mean(total: list, metric_fns: list, desc: str) -> None:
    print(f'{desc}')
    res = {}
    new_avg = []

    for metric_fn in metric_fns:
        f_name = metric_fn.__name__
        f_score = -99

        if len(f_name.split('_')) > 1: # is a new-average metric
            new_avg.append(metric_fn)
            continue
        else:
            f_score = sum(d[f_name] for d in total) / len(total)
            res[f_name] = f_score
    
    for metric_fn in new_avg:
        f_name = metric_fn.__name__.split('_')[0]
        f_score = metric_fn(res['precision'], res['recall'], res['specificity'])
        res[f_name] = f_score
    
    for key, value in res.items():
        print(f'{key}: {value}')
    
    return res

# returns a dict that can be used as a LUT-table of the confusion matrix scores
# for scores info, see quick graph https://en.wikipedia.org/wiki/Precision_and_recall
# dtype casting is used to prevent overflow long_scalars
def confmat_scores(y_true, y_pred) -> dict:
    data = {}
    cast_type = 'double'

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred

    AP = np.sum(y_true, dtype=cast_type) # TP + FN
    AN = np.sum(neg_y_true, dtype='double') # TN + FP
    SE = np.sum(y_pred, dtype='double') #TP + FP
    TP = np.sum(y_true * y_pred, dtype='double')
    FP = SE - TP
    TN = np.sum(neg_y_true * neg_y_pred, dtype='double')
    FN = AP - TP

    data['ap'] = AP
    data['an'] = AN
    data['se'] = SE
    data['tp'] = TP
    data['fp'] = FP
    data['tn'] = TN
    data['fn'] = FN

    return data

# TODO: y_true and y_pred already as np arrays ? don't use paths here
# TODO: rename components into y_pred y_true
def iou_old(y_true, y_pred) -> float:
    component1 = y_true
    component2 = y_pred

    overlap = component1*component2 # Logical AND (or use np.logical_and(target, prediction))
    union = component1 + component2 # Logical OR (or use np.logical_or(target, prediction))

    # (or iou_score = np.sum(intersection) / np.sum(union))
    IOU = overlap.sum() / (union.sum() + smooth) # Treats "True" as 1,
                                                      # sums number of Trues
                                                      # in overlap and union
                                                      # and divides
    return IOU

# Intersection over Union
# can be re-expressed in terms of precision and recall
# https://tomkwok.com/posts/iou-vs-f1/
def iou(cs):
    # precision_score = precision(cs)
    # recall_score = recall(cs)
    # return (precision_score * recall_score) / (precision_score + recall_score - precision_score * recall_score)
    return cs['tp'] / (cs['tp'] + cs['fp'] + cs['fn'] + smooth)

# Recall (aliases: TruePositiveRate, Sensitivity)
# how many relevant items are selected?
def recall(cs):
    return cs['tp'] / (cs['ap'] + smooth)

# Specificity (aliases: FalsePositiveRate)
# how many negative elements are truly negative?
def specificity(cs):
    return cs['tn'] / (cs['an'] + smooth)

# how many selected items are relevant?
def precision(cs):
    return cs['tp'] / (cs['se'] + smooth)

# Fb-measure: recall is considered Beta(b) times important as precision
# F2 weights recall higher than precision, F.5 weights precision higher than recall
# Beta(b) is a positive real factor
def fb(cs, b = 1):
    precision_score = precision(cs)
    recall_score = recall(cs)
    return (1 + b**2) * ((precision_score * recall_score) / ((b**2 * precision_score) + recall_score + smooth))

# F1-score (aliases: F1-measure, F-score with Beta=1)
def f1(cs):
    #precision_score = precision(cs)
    #recall_score = recall(cs)
    #return 2 * (float(precision_score * recall_score) / float(precision_score + recall_score + smooth))
    return fb(cs)

# F2-score
def f2(cs):
    return fb(cs, 2)

def dprs(cs):
    a = (1 - precision(cs))**2
    b = (1 - recall(cs))**2
    c = (1 - specificity(cs))**2
    
    return math.sqrt(a + b + c)

def f1_n(pr, re, sp):
    return 2 * ((pr * re) / (pr + re + smooth))

def dprs_n(pr, re, sp):
    a = (1 - pr)**2
    b = (1 - re)**2
    c = (1 - sp)**2
    return math.sqrt(a + b + c)

# todo: test
# range is [-1 1]
def mcc(cs):
    # explained in https://doi.org/10.1186/s12864-019-6413-7
    # the following fixes prevent where MCC could not be calculated normally
    M = np.matrix([[cs['tp'], cs['fn']], [cs['fp'], cs['tn']]]) # define confusion matrix
    nz = np.count_nonzero(M) # get non-zero elements of the matrix
    # fix 1
    if nz == 1: # 3 elements of M are 0
        # all samples of the dataset belong to 1 class
        if cs['tp'] != 0 or cs['tn'] != 0: # they either are all correctly classified
            return 1
        else:
            return -1 # or all uncorrectly classified
    
    # fix 2
    # where a row or a column of M are zero while the other true entries
    # are non zero, MCC takes the indefinite form 0/0
    if nz == 2 and np.sum(np.abs(M.diagonal())) != 0 and np.sum(np.abs(np.diag(np.fliplr(M)))) != 0:
        # replace the zero elements with an arbitrary small value 
        M[M == 0] = smooth
    
    # calculate MCC
    num = cs['tp'] * cs['tn'] - cs['fp'] * cs['fn']
    den = math.sqrt((cs['tp'] + cs['fp']) * (cs['tp'] + cs['fn']) * (cs['tn'] + cs['fp']) * (cs['tn'] + cs['fn']))

    # print(f'num={num} den={den} TP={TP} FP={FP} FN={FN} TN={TN}') # debug

    return num / (den + smooth)


if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    if n == 2: # predict over the same dataset of the model
        pred_root = sys.argv[1] # folder to use
    elif n == 3: # cross dataset predictions
        db_model = sys.argv[1]
        db_pred = sys.argv[2]
        pred_root = os.path.join('predictions', f'{db_model}_on_{db_pred}') # folder to use
    else:
        exit('''There must be 1 or 2 arguments
        With 1 arg specify the folder containing the predictions to evaluate (eg: predictions/HGR_small_on_ECU)
        With 2 args specify the model db and the predictions db (eg: HGR_small ECU)
        db examples: Schmugge, ECU, HGR''')

    metrics = [f1_n, dprs_n, recall, precision, specificity]

    y_path = os.path.join(pred_root, 'y') # 'predictions/HGR_small_on_ECU/y'
    p_path = os.path.join(pred_root, 'p') # 'predictions/HGR_small_on_ECU/p'

    rpd = pd_metrics(y_path, p_path, metrics)
    print_pd_mean(rpd, metrics, desc=pred_root)

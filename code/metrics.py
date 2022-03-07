import math
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# Measure the goodness of the classifier by comparing predictions with groundtruths


# Prevent zero division
smooth = 1e-20

def load_images(gt_path: str, pred_path: str, threshold: int = 128):
    '''Load images as numpy boolean arrays'''
    # Load as grayscale uint8
    gt_gray = np.array(Image.open(gt_path).convert('L'))
    pred_gray = np.array(Image.open(pred_path).convert('L'))
    # Binarize and convert to bool
    gt_bool = gt_gray > threshold
    pred_bool = pred_gray > threshold
    return gt_bool, pred_bool

# MEDIUM AVERAGE: calculate average only of medium-scores (PRecision, REcall, SPecificity)
# Note: y and p files must have the same filename
def pd_metrics(gt_dir: str, pred_dir: str, metric_fns: list, threshold: int = 128) -> list:
    out = []

    for y_filename in tqdm(os.listdir(gt_dir)):
        y_path = os.path.join(gt_dir, y_filename)
        p_filename = os.path.splitext(y_filename)[0] + '.png'
        p_path = os.path.join(pred_dir, p_filename) # pred are always PNG

        # Start adding item data into a structure
        idata = {}
        idata['y'] = y_path
        idata['p'] = p_path

        # Load images from paths and apply threshold to binarize
        # the skin probability maps obtained from predictions
        y_true, y_pred = load_images(y_path, p_path, threshold)
        confmat = confmat_scores(y_true, y_pred)

        # Calculate metrics for the image pair loaded
        for metric_fn in metric_fns:
            f_name = metric_fn.__name__
            f_argcount = metric_fn.__code__.co_argcount # amount of argument in function definition

            if f_name.endswith('_medium'): # is a medium-average metric, must not compute now
                continue

            # only one args: the metric only uses confusion matrix scores and is LUT-optimized
            if f_argcount == 1:
                idata[f_name] = metric_fn(confmat)
            # two args: confusion matrix scores aren't enough
            else:
                idata[f_name] = metric_fn(y_true, y_pred)
        
        # Update the final data list
        out.append(idata)
    
    print(f'  Found {len(out)} matches')
    
    return out

def print_pd_mean(total: list, metric_fns: list, desc: str) -> None:
    '''
    Print human-readable metrics results data
    
    
    PLEASE NOTE
    ---
    The mean F1 value calculated by summing all experiments F1 and dividing by N elements
    is different than the mean calculated by applying the F1 formula on average REcall and PRecision!

    #### Medium Averaging
    In the code I call 'medium average' the metrics in which I average only the
    medium-scores (PRecision, REcall, SPecificity)
    and in the end I calculate the functions of the 'final' metrics (F1, dprs) using these averages

    'Medium' as in their formulas they use the basic metrics (the ones in a confusion matrix:
    True Positives, False Negatives, ..), while 'final' metrics
    use the medium metrics themselves in their formulas.

    By following this logic, 'final' averaging means calculating the final metrics
    at the first step, along with the medium metrics, for each image and averaging
    these values on the batch of images.

    In a mathematical way:
    f1: 2 * precision * recall / (precision + recall)
    f1_finavg: avg(f1)
    f1_medavg: 2 * avg(precision) * avg(recall) / (avg(precision) + avg(recall))

    '''
    print(f'{desc}')
    res = {}
    medium_avg = []

    for metric_fn in metric_fns:
        f_name = metric_fn.__name__
        f_score = -99

        if f_name.endswith('_medium'): # is a medium-average metric
            medium_avg.append(metric_fn)
            continue
        else:
            f_score = sum(d[f_name] for d in total) / len(total)
            res[f_name] = f_score
    
    # The 'medium average' metrics average only the intermediate-scores (PRecision, REcall, SPecificity)
    # and then calculate the functions of the final metrics
    for metric_fn in medium_avg:
        f_name = metric_fn.__name__
        f_score = metric_fn(res['precision'], res['recall'], res['specificity'])
        res[f_name] = f_score
    
    for key, value in sorted(res.items()):
        print(f'{key}: {value}')
    
    return res

def confmat_scores(y_true, y_pred) -> dict:
    '''
    Return a dict that can be used as a LUT-table of the confusion matrix scores
    For info on each score, see https://en.wikipedia.org/wiki/Precision_and_recall
    '''
    data = {}
    cast_type = 'double'

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred

    # dtype casting is used to prevent overflow long_scalars
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

def iou_logical(y_true, y_pred) -> float:
    '''Intersection over Union'''
    overlap = y_true * y_pred # Logical AND
    union =   y_true + y_pred # Logical OR
    # Note that matrices are bool due to '> threshold' in load_images(),
    # it they were not, for union must to use bitwise OR '|'
    
    # Treats "True" as 1, sums number of Trues
    # in overlap and union and divides
    IOU = overlap.sum() / (union.sum() + smooth) 
    return IOU

def iou(cs):
    '''
    Intersection over Union can be re-expressed in terms of precision and recall
    Credit to https://tomkwok.com/posts/iou-vs-f1/
    '''
    return cs['tp'] / (cs['tp'] + cs['fp'] + cs['fn'] + smooth)

def recall(cs):
    '''
    Recall (aliases: TruePositiveRate, Sensitivity)

    How many relevant items are selected?
    '''
    return cs['tp'] / (cs['ap'] + smooth)

def specificity(cs):
    '''
    Specificity (aliases: FalsePositiveRate)

    How many negative elements are truly negative?
    '''
    return cs['tn'] / (cs['an'] + smooth)

def precision(cs):
    '''How many selected items are relevant?'''
    return cs['tp'] / (cs['se'] + smooth)

def fb(cs, b = 1):
    '''
    Fb-measure: recall is considered Beta(b) times important as precision.
    For example, F2 weights recall higher than precision, while
    F0.5 weights precision higher than recall.
    
    Beta(b) is a positive real factor
    '''
    precision_score = precision(cs)
    recall_score = recall(cs)
    return (1 + b**2) * ((precision_score * recall_score) / ((b**2 * precision_score) + recall_score + smooth))

def f1(cs):
    '''F1-score (aliases: F1-measure, F-score with Beta=1)'''
    return fb(cs)

def f2(cs):
    '''F2-score'''
    return fb(cs, 2)

def f1_medium(pr, re, sp):
    '''
    F1-score (aliases: F1-measure, F-score with Beta=1)
    ---
    Implementation suited for medium averaging
    '''
    return 2 * ((pr * re) / (pr + re + smooth))

def dprs(cs):
    '''
    Measures the Euclidean distance between the segmentation,
    represented by the point (PR, RE, SP), and the ground truth, the ideal point(1, 1, 1),
    hence lower values are better.
    Note: it considers all three of Precision, Recall, and Specificity.

    Can be higher than 1 in extremely bad cases

    ---
    Intawong, K., Scuturici, M., & Miguet, S. (2013). A New Pixel-Based Quality Measure
    for Segmentation Algorithms Integrating Precision, Recall and Specificity.
    Computer Analysis of Images and Patterns, 188-195.
    https://doi.org/10.1007/978-3-642-40261-6_22
    '''
    a = (1 - precision(cs))**2
    b = (1 - recall(cs))**2
    c = (1 - specificity(cs))**2
    
    return math.sqrt(a + b + c)

def dprs_medium(pr, re, sp):
    '''
    Measures the Euclidean distance between the segmentation,
    represented by the point (PR, RE, SP), and the ground truth, the ideal point(1, 1, 1),
    hence lower values are better.
    Note: it considers all three of Precision, Recall, and Specificity.

    Can be higher than 1 in extremely bad cases

    ---
    Intawong, K., Scuturici, M., & Miguet, S. (2013). A New Pixel-Based Quality Measure
    for Segmentation Algorithms Integrating Precision, Recall and Specificity.
    Computer Analysis of Images and Patterns, 188-195.
    https://doi.org/10.1007/978-3-642-40261-6_22

    ---
    Implementation suited for medium averaging
    '''
    a = (1 - pr)**2
    b = (1 - re)**2
    c = (1 - sp)**2
    return math.sqrt(a + b + c)

# Note: the function has not been tested thoroughly and needs to be verified
# range is [-1 1]
def mcc(cs):
    '''
    Common statistical measures can dangerously show overoptimistic inflated results,
    especially on imbalanced datasets.

    The Matthews correlation coefficient (MCC), instead, is a more reliable statistical
    rate which produces a high score only if the prediction obtained good results
    in all of the four confusion matrix categories (true positives, false negatives,
    true negatives, and false positives), proportionally both to the size of positive
    elements and the size of negative elements in the dataset.

    Range of values is [-1 1]

    ---
    Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation
    coefficient (MCC) over F1 score and accuracy in binary classification evaluation.
    BMC Genomics, 21(1).
    https://doi.org/10.1186/s12864-019-6413-7
    '''

    # The following fixes prevent where MCC could not be calculated normally
    M = np.matrix([[cs['tp'], cs['fn']], [cs['fp'], cs['tn']]]) # define confusion matrix
    nz = np.count_nonzero(M) # get non-zero elements of the matrix
    
    # Fix 1
    if nz == 1: # 3 elements of M are 0
        # all samples of the dataset belong to 1 class
        if cs['tp'] != 0 or cs['tn'] != 0: # they either are all correctly classified
            return 1
        else:
            return -1 # or all uncorrectly classified
    
    # Fix 2
    # Where a row or a column of M are zero while the other true entries
    # are non zero, MCC takes the indefinite form 0/0
    if nz == 2 and np.sum(np.abs(M.diagonal())) != 0 and np.sum(np.abs(np.diag(np.fliplr(M)))) != 0:
        # replace the zero elements with an arbitrary small value 
        M[M == 0] = smooth
    
    # Calculate MCC
    num = cs['tp'] * cs['tn'] - cs['fp'] * cs['fn']
    den = math.sqrt((cs['tp'] + cs['fp']) * (cs['tp'] + cs['fn']) * (cs['tn'] + cs['fp']) * (cs['tn'] + cs['fn']))

    return num / (den + smooth)

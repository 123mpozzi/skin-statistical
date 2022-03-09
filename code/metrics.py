import math

import numpy as np

# Measure the goodness of the classifier by comparing predictions with groundtruths

# In skin detection, false negatives may weight more as they cannot be fixed by post-processing
# whereas false positives can to a degree


# Prevent zero division
smooth = 1e-20


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

    ---
    Info on F1 vs MCC from the paper analysis is simulated in
    `tests/test_metrics.py -> mcc_unittest()`
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

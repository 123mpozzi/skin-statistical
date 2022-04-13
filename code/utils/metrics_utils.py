import os
from statistics import mean, pstdev

import numpy as np
from metrics import confmat_scores
from PIL import Image
from tqdm import tqdm

from utils.logmanager import *


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
def calc_metrics(gt_dir: str, pred_dir: str, metric_fns: list, threshold: int = 128) -> list:
    '''
    Compute all the given metric functions over all images in a folder
    by considering a single image at a time and comparing
    its groundtruth to its prediction map

    Return a list of dicts.
    Each dict represents the metrics measurement on a single image

    Medium-averaging metric functions get skipped as they cannot be computed on a single image
    '''
    out = []

    # Loop images
    for y_filename in tqdm(os.listdir(gt_dir)):
        y_path = os.path.join(gt_dir, y_filename)
        p_filename = os.path.splitext(y_filename)[0] + '.png'
        p_path = os.path.join(pred_dir, p_filename) # pred are always PNG

        # Start adding current image data into a dict structure
        idata = {}
        idata['y'] = y_path
        idata['p'] = p_path

        # Load images from paths and apply threshold to binarize
        # the skin probability maps obtained from predictions
        y_true, y_pred = load_images(y_path, p_path, threshold)
        # Calculate confusion matrix scores for current image
        confmat = confmat_scores(y_true, y_pred)

        # Calculate metrics for current image and add them to the dict structure
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
        
        # Update the final list with current image data
        out.append(idata)
    
    info(f'  Found {len(out)} matches')
    return out


def calc_mean_metrics(measurements_list: list, metric_fns: list, desc: str, method: str) -> None:
    '''
    Print human-readable metrics results data
    
    Process a list of single measurements and
    return a dict containing each metric mean and stdev values
    
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
    info(desc)
    res = {}
    # Insert datasets and method data into the resulting dict
    res['method'] = method
    try:
        desc = desc.split(' ')[0] # remove hash string
        desc = os.path.normpath(desc) # remove trailing slash
        desc = os.path.basename(desc) # get prediction folder name
        desc = desc.lower().replace('_small', '') # lower case and rename HGR_small to HGR
        dss = desc.split('_on_') # split training and predicting datasets
        ds_tr = dss[0]
        ds_te = dss[1]
        res['train'] = ds_tr
        res['test'] = ds_te
    except: # eg. for testing
        pass

    medium_avg = []

    for metric_fn in metric_fns:
        f_name = metric_fn.__name__
        f_score = -99

        if f_name.endswith('_medium'): # is a medium-average metric
            medium_avg.append(metric_fn)
            continue
        else:
            f_data = [ d[f_name] for d in measurements_list ]
            f_mean = mean(f_data)
            f_mean = '{:.4f}'.format(f_mean) # round to 4 decimals and zerofill
            f_std = pstdev(f_data)
            f_std = '{:.2f}'.format(f_std) # round to 2 decimals and zerofill
            #info(f'{f_name}: {f_mean} ± {f_std}')
            # add each metric data to the dict
            res[f_name] = f'{f_mean} ± {f_std}'
    
    # The 'medium average' metrics average only the intermediate-scores (PRecision, REcall, SPecificity)
    # and then calculate the functions of the final metrics
    for metric_fn in medium_avg:
        f_name = metric_fn.__name__
        # Calculate the medium-average score using medium-scores averages
        pr = float(res['precision'].split(' ')[0])
        re = float(res['recall'].split(' ')[0])
        sp = float(res['specificity'].split(' ')[0])
        f_score = metric_fn(pr, re, sp)
        f_score = '{:.4f}'.format(f_score)
        res[f_name] = f_score
    
    for key, value in sorted(res.items()):
        info(f'{key}: {value}')
    
    return res

def read_performance(perf_dir: str):
    '''Read inference time from performance benchmark files, and print it'''
    csv_sep = ','

    # will contain the final mean between each observation's mean
    observations_means = []

    # do the mean of each observation
    for i in range(5000):
        perf_filename = f'bench{i}.txt'
        perf_file = os.path.join(perf_dir, perf_filename)

        if not os.path.isfile(perf_file):
            break

        # read txt lines (as csv)
        file2c = open(perf_file)
        doubles = file2c.read().splitlines()
        file2c.close()

        intra_obs_timelist = []
        for entry in doubles: # ori_path, execution_time(s)
            ori_path = entry.split(csv_sep)[0]
            execution_time = entry.split(csv_sep)[1]
            intra_obs_timelist.append(float(execution_time))
        
        obs_mean = mean(intra_obs_timelist)
        obs_mean = '{:.6f}'.format(obs_mean) # round and zerofill
        obs_std = pstdev(intra_obs_timelist)
        obs_std = '{:.3f}'.format(obs_std) # round and zerofill

        obs_string = f'{obs_mean} ± {obs_std}'

        observations_means.append(obs_string)
        info(f'{perf_dir} at {i}: {obs_string}')
    
    # get the means from observation means, without the std
    obs_mean_values = []
    for entry in observations_means:
        obs_mean_values.append(float(entry.split(' ')[0]))
    
    # do the final mean of the observation means
    fin_mean = mean(obs_mean_values)
    fin_mean = '{:.6f}'.format(fin_mean) # round and zerofill
    fin_std = pstdev(obs_mean_values)
    fin_std = '{:.3f}'.format(fin_std) # round and zerofill

    fin_string = f'{fin_mean} ± {fin_std}'

    info(f'{perf_dir} at FIN: {fin_string}\n')
    return fin_string

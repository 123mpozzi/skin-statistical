import os
from statistics import mean, pstdev

import numpy as np
from metrics import confmat_scores, dprs, f1, iou
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


## Latex Utils

def print_latex(cross_preds: bool, skintones: bool, db_paths: list):
    if not skintones:
        db_list = ['ecu', 'hgr', 'schmugge']
    else:
        db_list = ['dark', 'medium', 'light']

    # paths resolving
    if not cross_preds:
        detectors = ['skinny', 'probabilistic', 'dyc']
        db_paths = []
        for db in db_list:
            for sd in detectors:
                if db == 'hgr' and sd == 'dyc':
                    db = 'hgr_small'
                if skintones:
                    sd += '_st'
                db_paths.append(f'dataset/{sd}/base/{db}')
    else:
        detectors = ['skinny', 'bayes']
        db_paths = []
        for db_tr in db_list:
            for db_te in db_list:
                if db_te != db_tr:
                    for sd in detectors:
                        if skintones:
                            sd += '_st'
                        db_paths.append(f'dataset/{sd}/cross/{db_tr}_on_{db_te}')

    metrics = [f1, iou, dprs]
    json_table = []

    # compute metrics
    for ds in db_paths:
        y_path = os.path.join(ds, 'y') # '{dataset}/y'
        p_path = os.path.join(ds, 'p') # '{dataset}/p'
        
        singles = calc_metrics(y_path, p_path, metrics)
        # 'dataset/skinny/...'
        skin_detector = ds.split('/')[1]

        if not cross_preds:
            ds = ds + '_on_' + os.path.basename(ds)

        table_item = calc_mean_metrics(singles, metrics, desc=ds, method=skin_detector)

        json_table.append(table_item)

    if not cross_preds:
        print(get_latex_base(json_table, db_list))
    else:
        print(get_latex_cross(json_table, db_list))


def is_better(value1, value2, mode: str):
    '''Return whether a value is better than another value'''
    if mode == 'upper':
        return value1 > value2
    else:
        return value1 < value2

def bold_best(data: list, datas: list, base = False):
    '''Make the best values bold'''
    maxv = {}
    # Save the best values between the METHODS (skinny/probabilistic)
    # for each metric and dataset combination
    for obj in data:
        o_m = obj['method']
        o_train = obj['train']
        o_test = obj['test']
        o_f1 = obj['f1']
        o_iou = obj['iou']
        o_dprs = obj['dprs']
        f1iou = float(o_f1.split(' ')[0]) - float(o_iou.split(' ')[0])
        obj['f1iou'] = round(f1iou, 4)

        print(obj)

        # They are cross predictions hence test != train
        if base == False and o_train == o_test:
            continue
        
        # For each table metric multirow group
        for f_name in ['f1', 'iou', 'dprs', 'f1iou']:
            fn_val = obj[f_name] # metric value of the current iteration
            fn_idformat = f'{f_name}_{o_train}_{o_test}' # ID format
            fnv = f'{fn_idformat}_v' # value of the max measurement
            fni = f'{fn_idformat}_i' # ID of the max measurement

            bmode = 'upper'
            if f_name == 'dprs' or f_name == 'f1iou':
                bmode = 'lower'

            # For each table column
            for trdata in datas:
                for tedata in datas:
                    if o_train == trdata and o_test == tedata:
                        # Save best between methods

                        # if max does not exist, add its value and ID
                        if fnv not in maxv:
                            maxv[fnv] =  fn_val
                            maxv[fni] =  o_m
                        # if new max, save the measurement and its ID
                        elif f_name != 'f1iou' and is_better(float(fn_val.split(' ')[0]), float(maxv[fnv].split(' ')[0]), bmode):
                            maxv[fnv] =  fn_val
                            maxv[fni] =  o_m
                        elif f_name == 'f1iou':
                            if is_better(float(fn_val), float(maxv[fnv]), bmode):
                                maxv[fnv] =  fn_val
                                maxv[fni] =  o_m
    newdata = []
    # And now make them bold
    for obj in data:
        o_m = obj['method']
        o_train = obj['train']
        o_test = obj['test']
        o_f1 = obj['f1']
        o_iou = obj['iou']
        o_dprs = obj['dprs']
        f1iou = float(o_f1.split(' ')[0]) - float(o_iou.split(' ')[0])
        obj['f1iou'] = '{:.4f}'.format(f1iou) # round and zerofill

        for f_name in ['f1', 'iou', 'dprs', 'f1iou']:
            fn_val = obj[f_name] # metric value of the current iteration
            fn_idformat = f'{f_name}_{o_train}_{o_test}' # ID format
            fnv = f'{fn_idformat}_v' # value of the max measurement
            fni = f'{fn_idformat}_i' # ID of the max measurement

            if fni in maxv and maxv[fni] == o_m:
                obj_formatted = '\\texttt{' + '\\textbf{' + str(obj[f_name]) + '}' + '}'
                obj[f_name] = obj_formatted # set bold and monospace
                newdata.append(obj)
            else:
                obj_formatted = '\\texttt{' + str(obj[f_name]) + '}'
                obj[f_name] = obj_formatted # set monospace
                newdata.append(obj)
    data = newdata
    print('newdata:')
    print(data)

    # Change JSON format into a standalone data structure containing all table variables
    ff = {}
    for obj in data:
        o_m = obj['method']
        o_train = obj['train']
        o_test = obj['test']
        o_f1 = obj['f1']
        o_iou = obj['iou']
        o_dprs = obj['dprs']
        f1iou = obj['f1iou']

        ff[f'f1_{o_m}_{o_train}_{o_test}'] = o_f1
        ff[f'iou_{o_m}_{o_train}_{o_test}'] = o_iou
        ff[f'dprs_{o_m}_{o_train}_{o_test}'] = o_dprs
        ff[f'f1iou_{o_m}_{o_train}_{o_test}'] = f1iou
    
    print(ff)
    return ff

# Data is a list of JSON items
# JSON item example: {"name":"ecu", "F1":".9123 +- 0.25", "IOU":".8744 +- 0.11"}
def get_latex_cross(data: list, datas = None):
    '''Return latex table containing cross-datasets metrics measurements'''
    tex_body = ''

    if datas == None:
        datas = ['ecu', 'hgr', 'schmugge']

    ff = bold_best(data, datas)

    # Start building the body string
    tex_ms = ['F1 $\\uparrow$', 'IOU $\\uparrow$', 'Dprs $\\downarrow$', 'F1 - IOU $\\downarrow$']
    i = 2
    for tm in tex_ms:
        mns = tm.split(' ')

        if len(mns) > 2: # f1 - iou
            mn = 'f1iou'
        else:
            mn = mns[0].lower()
        
        # For each metrics there are 2 lines(methods): Skinny and Probabilistic
        for j in range(2):
            pfix = ''

            if j == 0:
                met = 'skinny'
                mf = met[0].lower()
                metf = met.upper() + '\\rule{0pt}{14pt}' # spacing between multirows (metrics)
                metric_w_arrow = tm
                pfix = f'''\\multirow{{2}}{{*}}{{{{{metric_w_arrow}}}}}'''
            elif j == 1:
                met = 'bayes'
                mf = met[0].lower()
                metf = met.upper()
            
            if datas != ['ecu', 'hgr', 'schmugge']:
                met = f'{met}_st'

            # Another data struct to gather all items necessary for writing a table line
            tmp = {}
            datas_startletter = []
            for ds_tr in datas:
                datas_startletter.append(ds_tr[0].lower())
                for ds_te in datas:
                    if ds_tr == ds_te:
                        continue
                    tmp[f'{mf}_{ds_tr[0].lower()}{ds_te[0].lower()}'] = ff[f'{mn}_{met}_{ds_tr}_{ds_te}']

            tex_body += f'{pfix}& {metf}'
            for letter_tr in datas_startletter:
                for letter_te in datas_startletter:
                    if letter_tr != letter_te:
                        table_item = tmp[f'{mf}_{letter_tr}{letter_te}']
                        tex_body += f' & {table_item}'
            tex_body += '\\\\'
    
    # String header
    tex_header = r'''
    \begin{tabular}{clcccccc}
    \toprule
    \multicolumn{1}{c}{} & \multicolumn{1}{c}{\head{Training}} 
    '''
    
    # Add first row
    for dss in datas:
        tex_header += r'& \multicolumn{2}{c}{\head{' + dss.upper() + '}} '
    tex_header += r'\\'

    tex_header += r'\multicolumn{1}{c}{} & \multicolumn{1}{c}{\head{Testing}} '
    # Add second row
    for dss in datas:
        for dssd in datas:
            if dssd != dss:
                tex_header += r' & \multicolumn{1}{c}{\head{' + dssd.upper() + '}} '
    tex_header += r'\\'
    tex_header += r'\midrule'

    # String end
    tex_end = r'''
    \bottomrule
    \end{tabular}
    '''

    tex = tex_header + tex_body + tex_end
    return tex

# data is a list of JSON items
# JSON item example: {"name":"ecu", "F1":".9123 +- 0.25", "IOU":".8744 +- 0.11"}
def get_latex_base(data: list, datas = None):
    '''Return latex table containing base-datasets metrics measurements'''
    tex_body = ''

    if datas == None:
        datas = ['ecu', 'hgr', 'schmugge']

    ff = bold_best(data, datas, True)

    # Start building the body string
    metrics = ['f1', 'iou', 'dprs']
    
    # At first loop ROWS
    # for each metrics there are 2 lines(methods): Skinny and Bayes, DYC
    for j in range(3):
        if j == 0:
            met = 'skinny'
        elif j == 1:
            met = 'bayes'
        else:
            met = 'dyc'
        metf = met.upper()
        mf = met[0].lower()

        if datas != ['ecu', 'hgr', 'schmugge']:
            met = f'{met}_st'

        # Another data struct to gather all items necessary for writing a table line
        tmp = {}
        # Then loop COLUMNS
        datas_startletter = []
        for ds in datas:
            datas_startletter.append(ds[0].lower())
            for tm in metrics:
                tmp[f'{mf}_{ds[0].lower()}{tm}'] = ff[f'{tm}_{met}_{ds}_{ds}']

        # m is method
        # eh = ecu_on_hgr, es = ecu_on_schmugge, ...
        tex_body += f'{metf}'
        for letter in datas_startletter:
            for tmm in metrics:
                table_item = tmp[f'{mf}_{letter}{tmm}']
                tex_body += f' & {table_item}'
        tex_body += '\\\\'
    
    # String header
    tex_header = r'''
    \begin{tabular}{lccccccccc}
    \toprule
    '''
    
    # Add first row
    for dss in datas:
        tex_header += r'& \multicolumn{3}{c}{\head{' + dss.upper() + '}} '
    tex_header += r'\\'
    
    tex_header += r'''
    & \multicolumn{1}{c}{\head{F1 $\uparrow$}} & \multicolumn{1}{c}{\head{IOU $\uparrow$}} & \multicolumn{1}{c}{\head{Dprs $\downarrow$}}
    & \multicolumn{1}{c}{\head{F1 $\uparrow$}} & \multicolumn{1}{c}{\head{IOU $\uparrow$}} & \multicolumn{1}{c}{\head{Dprs $\downarrow$}}
    & \multicolumn{1}{c}{\head{F1 $\uparrow$}} & \multicolumn{1}{c}{\head{IOU $\uparrow$}} & \multicolumn{1}{c}{\head{Dprs $\downarrow$}}\\
    \midrule
    '''

    # String end
    tex_end = r'''
    \bottomrule
    \end{tabular}
    '''

    tex = tex_header + tex_body + tex_end
    return tex

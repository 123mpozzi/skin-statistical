import traceback
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os, time
from shutil import copyfile
from utils.hash_utils import hash_dir
from utils.db_utils import get_model_filename, get_datasets
from utils.logmanager import *


method_name = 'probabilistic'
predictions_dir = os.path.join('..', 'predictions')

def get_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def pred_dir(type: str, timestr: str, name: str) -> str:
    '''
    Return a proper directory to store predictions given
    prediction type, timestamp string, and predictions name/title
    '''
    if type in ('base', 'cross'):
        return os.path.join(predictions_dir, timestr, method_name, type, name)
    elif type == 'bench':
        return os.path.join(predictions_dir, type, timestr, name)
    else: # default
        return os.path.join(predictions_dir, name)

def open_image(src):
    # Convert to RGB as some image may be read as RGBA: https://stackoverflow.com/a/54713582
    return Image.open(src,'r').convert('RGB')

def pred_out(path_x: str, out_dir: str) -> list:
    # use the x filename for all saved images filenames (x, y, p)
    filename, x_ext = os.path.splitext(os.path.basename(path_x))
    # the masks and predictions will be saved LOSSLESS as PNG
    out_p = os.path.join(out_dir, 'p', filename + '.png')
    out_y = os.path.join(out_dir, 'y', filename + '.png')
    out_x = os.path.join(out_dir, 'x', filename + x_ext)
    return (out_p, out_y, out_x)

# out_bench is the file in which append inference performance data
def predict(probability, path_x, path_y, out_dir, out_bench: str = ''):
    im = open_image(path_x)
    temp = im.copy()
    im.close()

    out_p, out_y, out_x = pred_out(path_x, out_dir)

    # Save p
    t_elapsed = create_image(temp, probability, out_p)

    # Close file and free memory
    temp.close()

    # Copy x and y
    copyfile(path_x, out_x)
    #if path_y is not None: # may also predict images without a groundtruth # TODO: future work
    copyfile(path_y, out_y)

    # Save inference performance to file
    if out_bench: # empty strings are falsy
        with open(out_bench, 'a') as out:
            out.write(f'{path_x},{t_elapsed}\n')

# Credit to https://stackoverflow.com/a/36469395 for image management with Pillow
def create_image(im: Image, probability, out_p) -> float:
    '''
    Infer on an image and save the prediction

    Return inference time
    '''
    im.load()

    t_start = time.time()
    # ALGO
    newimdata = []
    
    for r,g,b in im.getdata():
        row_num = (r*256*256) + (g*256) + b # calculating the serial row number 
        if(probability['Probability'][row_num] <0.555555):
            newimdata.append((0,0,0))
        else:
            newimdata.append((255,255,255))
    
    im.putdata(newimdata)
    t_elapsed = time.time() - t_start
    
    im.save(out_p)
    return t_elapsed

def make_predictions(image_paths, in_model, out_dir, out_bench: str = '', pbar_position: int = -1):
    assert os.path.isfile(in_model), critical('Model file not existing: ' + in_model)

    info("Reading CSV...")
    probability = pd.read_csv(in_model) # getting the rows from csv
    if pbar_position == -1: # on multiprocessing do not clog console
        info('Data collection completed')

    # make dirs
    for basedir in ('p', 'y', 'x'):
        os.makedirs(os.path.join(out_dir, basedir), exist_ok=True)
    
    if pbar_position == -1: # default bar position
        image_paths_tqdm = tqdm(image_paths)
    else: # set bar position
        image_paths_tqdm = tqdm(image_paths, position=pbar_position)
    
    for i in image_paths_tqdm:
        im_abspath = os.path.abspath(i[0]) 
        y_abspath = os.path.abspath(i[1])

        # Try predicting
        try:
            predict(probability, im_abspath, y_abspath, out_dir, out_bench) # this tests the data
        # File not found, prediction algo fail, ..
        except Exception:
            error(f'Failed to infer on image: {im_abspath}')
            print(traceback.format_exc())

    predictions_hash = hash_dir(out_dir)

    if pbar_position == -1: # on multiprocessing do not clog console
        print(predictions_hash)
    return predictions_hash

def base_preds(timestr: str, models: list):
    '''
    Base predictions
    For each dataset: the model trained from the training set is used
    and predictions are performed on self test set
    '''
    # Load each model
    for in_model in models:
        model_name = get_model_filename(in_model)
        assert os.path.isdir(in_model.dir), 'Dataset has no directory: ' + in_model.name

        # Make predictions
        image_paths = in_model.get_test_paths() # predict on testing set
        out_dir = pred_dir('base', timestr, in_model.name)
        make_predictions(image_paths, model_name, out_dir)

def cross_preds(timestr: str, train_databases: list, predict_databases: list = None):
    '''
    Cross predictions
    For each dataset: the model trained from the training set is used
    and predictions are performed on all the images of every other datasets
    '''
    if predict_databases is None:
        predict_databases = get_datasets() #train_databases

    # Load each model
    for train_db in train_databases:
        model_name = get_model_filename(train_db)

        # Load each target
        for predict_db in predict_databases:
            # do not predict on self
            if train_db == predict_db:
                continue
            
            assert os.path.isdir(predict_db.dir), 'Dataset has no directory: ' + predict_db.name
            # Make predictions
            image_paths = predict_db.get_all_paths() # predict the whole dataset
            out_dir = pred_dir('cross', timestr, f'{train_db.name}_on_{predict_db.name}')
            make_predictions(image_paths, model_name, out_dir)

from PIL import Image
import pandas as pd
from tqdm import tqdm
import os, time
from shutil import copyfile
from utils.hash_utils import hash_dir
from utils.db_utils import *
from utils.ECU import ECU
from utils.Schmugge import Schmugge
from utils.HGR import HGR
from utils.dark import dark
from utils.medium import medium
from utils.light import light
import click

# TODO: clean root folder: leave only README, train,predict,metrics,gitignore,requirements
# delete prepare_dataset (move in utils/db py files), delete augment(move in utils/schmugge py file)
# Move CSV models in models/

# TODO: call db.reset() before prediction to reset the original CSV

# TODO: final checks: -check dir hashes -check metrics result

method_name = 'probabilistic'

# Return a proper prediction dir given prediction type, timestamp string, and predictions name/title
def pred_dir(type: str, timestr: str, name: str) -> str:
    if type in ('base', 'cross'):
        return os.path.join('.', 'predictions', timestr, method_name, type, name)
    elif type == 'bench':
        return os.path.join('.', 'predictions', type, timestr)
    else: # default
        return os.path.join('.', 'predictions', name)

def open_image(src):
    return Image.open(src,'r')

def pred_out(path_x: str, out_dir: str) -> list:
    # use the x filename for all saved images filenames (x, y, p)
    filename, x_ext = os.path.splitext(os.path.basename(path_x))
    # the masks and predictions will be saved LOSSLESS as PNG
    out_p = os.path.join(out_dir, 'p', filename + '.png')
    out_y = os.path.join(out_dir, 'y', filename + '.png')
    out_x = os.path.join(out_dir, 'x', filename + x_ext)
    return (out_p, out_y, out_x)

# measure_time is the file to open and append performance data
def predict(probability, path_x, path_y, out_dir, out_bench: str = None):
    im = open_image(path_x)
    temp = im.copy()
    im.close()

    out_p, out_y, out_x = pred_out(path_x, out_dir)

    # save p
    t_elapsed = create_image(temp, probability, out_p)
    # close file and free memory
    temp.close()

    # copy x and y
    copyfile(path_x, out_x)
    copyfile(path_y, out_y)

    # save inference performance to file
    if out_bench:
        with open(out_bench, 'a') as out:
            out.write(f'{path_x},{t_elapsed}\n')

# Credit to https://stackoverflow.com/a/36469395
def create_image(im: Image, probability, out_p) -> float: 
    im.load()

    t_start = time.time()
    # ALGO
    newimdata = []
    for r,g,b in im.getdata():
        row_num = (r*256*256) + (g*256) + b #calculating the serial row number 
        if(probability['Probability'][row_num] <0.555555):
            newimdata.append((0,0,0))
        else:
            newimdata.append((255,255,255))
    
    im.putdata(newimdata)
    t_elapsed = time.time() - t_start
    
    im.save(out_p)
    return t_elapsed

def make_predictions(image_paths, in_model, out_dir, out_bench: str = None):
    print("Reading CSV...")
    probability = pd.read_csv(in_model) # getting the rows from csv    
    print('Data collection completed')

    # make dirs
    for basedir in ('p', 'y', 'x'):
        os.makedirs(os.path.join(out_dir, basedir), exist_ok=True)

    for i in tqdm(image_paths):
        im_abspath = os.path.abspath(i[0]) 
        y_abspath = os.path.abspath(i[1])
        predict(probability, im_abspath, y_abspath, out_dir, out_bench) # this tests the data

    predictions_hash = print(hash_dir(out_dir))
    return predictions_hash

def get_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

# Base predictions
# For each dataset: the model is trained on the training set
# and then predictions are performed on the dataset's test set
def base_preds(timestr: str, models: list):
    # base preds: based on splits defined by me and only predict on self
    # timestr/bayes/base/{ecu,hgr,schmugge}/{p/y/x}
    for in_model in models: # load each model
        model_name = get_model_filename(in_model)

        image_paths = in_model.get_test_paths() # predict on testing set
        out_dir = pred_dir('base', timestr, in_model)
        make_predictions(image_paths, model_name, out_dir)


# Cross predictions
# For each dataset: the model is trained on the training set
# and then predictions are performed on all the images of every other datasets
def cross_preds(timestr: str, train_databases: list, predict_databases: list = None):
    if predict_databases is None:
        predict_databases = train_databases

    for train_db in train_databases: # load each model
        model_name = get_model_filename(train_db)

        for predict_db in predict_databases:
            if train_db == predict_db: # do not predict on self
                continue

            image_paths = predict_db.get_all_paths() # predict the whole dataset
            out_dir = pred_dir('cross', timestr, f'{train_db.name}_on_{predict_db.name}')
            make_predictions(image_paths, model_name, out_dir)

def filter_schmugge(skintone: str):
    # re-import Schmugge
    Schmugge().gen_csv(predefined=False)

    # Filter a dataset CSV by given skintone, the other entries will be deleted from the CSV file
    Schmugge().update_notes(skintone, train_mode=False)
    Schmugge().count_skintones(skintone)
    Schmugge().count_notes(mode='test')


# Main command which groups the subcommands: single, batch, bench
@click.group()
def cli():
    pass

# BATCH: N-on-M datasets predictions
@cli.command()
@click.option('--type', '-t', type=click.Choice(['base', 'cross', 'all']), required=True)
@click.option('--skintones/--no-skintones', 'skintones', default=False,
              help = 'Whether to predict on skintone sub-datasets')
def batch(type, skintones): # TODO: test but first fix dark()
    timestr = get_timestamp()

    models = skin_databases_names()
    #if skintones == True:
    #    models = skin_databases_skintones # TODO: fix dark.py

    if type == 'base':
        base_preds(timestr, models)
        pass
    elif type == 'cross':
        cross_preds(timestr, models)
        pass
    else: # 'all' does either base+cross or skinbase+skincross, depending on --skintone
        base_preds(timestr, models)
        cross_preds(timestr, models)
        pass

# BENCHMARK: measure inference time
@cli.command()
@click.option('--size', '-s', type=int, default = 15, show_default=True,
              help='Benchmark set size, in images (-1 is whole db)')
@click.option('--observations', '-o', type=int, default = 5, show_default=True,
              help='Observations to register for the benchmark set')
def bench(size, observations): # TODO: test
    timestr = get_timestamp()

    # Use first 15 ECU images as test set        
    ECU().prepare_benchmark_set(count=size)
    image_paths = ECU().get_test_paths()
    out_dir = pred_dir('bench', timestr, None)

    # Do multiple observations
    # The predictions will be the same but performance will be logged 5 times
    for k in range(observations):
        make_predictions(image_paths, get_model_filename(ECU()), out_dir, out_bench=f'bench{k}.txt')

# SINGLE: 1-on-1 datasets prediction. Can be on self too
@cli.command()
@click.option('--model', '-m',
              type=click.Choice(skin_databases_names(), case_sensitive=True), required=True)
@click.option('--predict', '-p', 'predict_',
              type=click.Choice(skin_databases_names(), case_sensitive=True))
@click.option('--from', '-f', 'from_', type=int, default = 0, help = 'Slice start')
@click.option('--to', '-t', type=int, default = -1, help='Slice end (index excluded)')
@click.option('--set', '-s', 'set_', type=click.Choice(['test', 'all']), help='Force prediction set')
def single(model, predict_, from_, to, set_):
    pred_set = set_

    # prediction on self
    if predict_ is None:
        predict_ = model

    # Determine whether to use test ser ot whole dataset
    if pred_set is None: # no forced prediction set, use normal behaviour
        if predict_ == model:
            pred_set = 'test'
        elif predict_ != model:
            pred_set = 'all'
    
    if pred_set == 'test':
        image_paths = get_db_by_name(predict_).get_test_paths() # on same dataset, use test paths
    else:
        image_paths = get_db_by_name(predict_).get_all_paths() # on cross datasets, use all paths

    # to=-1 means till dataset end
    if to == -1:
        to = len(image_paths) # end index is not included in python slicing operator

    model_name = get_model_filename(get_db_by_name(model))
    out_dir = pred_dir(None, None, name = f'{model}_on_{predict_}')
    make_predictions(image_paths[from_:to], model_name, out_dir)

    # TODO: bench
    # TODO: also add batch modes (like predict_multiprocess.py?) - but here they are all singleprocessing
    # split main command into more commands
    # modes: single|batch|bench
    # !! TODO: change the single_cmd values in pred_multiprocess.py

    if True:
        return

    ## ALL: predict using all the default datasets
    if db_model == 'all': # TODO: misleading name
        timestr = get_timestamp()
        models = (ECU(), HGR(), Schmugge())

        base_preds(timestr, models)
        cross_preds(timestr, models)
    

    ## SKINTONES: predict all on Schmugge skintones
    elif db_model == 'skintones':
        timestr = get_timestamp()
        models = (dark(), medium(), light())

        base_preds(timestr, models)
        cross_preds(timestr, models)
    

    ## BENCHMARK: measure the execution time
    elif db_model == 'bench':
        timestr = get_timestamp()

        # Use first 15 ECU images as test set        
        ECU().prepare_benchmark_set(count=15)
        image_paths = ECU().get_test_paths()
        out_dir = pred_dir('bench', timestr, None)

        # do 5 observations
        # the predictions will be the same
        # but the performance will be logged 5 times
        observations = 5
        for k in range(observations):
            make_predictions(image_paths, get_model_filename(ECU()), out_dir, out_bench=f'bench{k}.txt')
    

    ## DEFAULT: normal case
    else:
        train_db = get_db_by_name(db_model)

        image_paths = get_db_by_name(db_pred).get_test_paths() # TODO: all paths if cross?
        out_dir = pred_dir(None, None, name = f'{db_model}_on_{db_pred}')
        make_predictions(image_paths, get_model_filename(train_db), out_dir)


if __name__ == "__main__":
    #print(hash_dir('./predictions/ECU_on_Schmugge'))
    #print(hash_dir('./predictions/ECU_on_Schmugge_single'))
    #print(hash_dir('./predictions/ECU_on_Schmugge_new'))

    cli()

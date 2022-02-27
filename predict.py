from prepare_dataset import process_schmugge, read_schmugge
from PIL import Image
import pandas as pd
from utils import *
from tqdm import tqdm
import sys, os, time
from shutil import copyfile

skintones = ('dark', 'medium', 'light')
method_name = 'probabilistic'

# Return a proper database dir given a database name
def db_dir(dbname: str) -> str:
    return os.path.join('.', 'dataset', dbname)

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


def base_preds(timestr: str, models: list):
    # base preds: based on splits defined by me and only predict on self
    # timestr/bayes/base/{ecu,hgr,schmugge}/{p/y/x}
    for in_model in models: # load each model
        model_name = in_model + '.csv'
        
        in_dir = db_dir(in_model)
        out_dir = pred_dir('base', timestr, in_model)

        # check if predicting skintones
        if in_model in skintones:
            in_dir = db_dir('Schmugge')
            load_skintone_split(in_model)   # TODO: what if I first predict on light then on Schmugge?? is the data.csv reset back?? I think not

        image_paths = get_test_paths(os.path.join(in_dir, 'data.csv'))
        make_predictions(image_paths, model_name, out_dir)


def cross_preds(timestr: str, models: list, datasets: list):
    # cross preds: use a dataset whole as the testing set
    # timestr/bayes/cross/ecu_on_ecu/{p/y/x}
    for in_model in models: # load each model
        model_name = in_model + '.csv'

        for db in datasets:
            db_name = os.path.basename(db)#.lower()

            # check if predicting skintones
            if in_model in skintones:
                filter_schmugge(db)
                db_name = db
                in_dir = db_dir('Schmugge')
                image_paths = get_test_paths(os.path.join(in_dir, 'data.csv')) # TODO ?? why test instead of all??
            else:
                image_paths = get_all_paths(os.path.join(db, 'data.csv'))
            
            out_dir = pred_dir('cross', timestr, f'{in_model}_on_{db_name}')
            make_predictions(image_paths, model_name, out_dir)

def filter_schmugge(skintone: str):
    # re-import Schmugge
    schm = read_schmugge('./dataset/Schmugge/data/.config.SkinImManager', './dataset/Schmugge/data/data')
    process_schmugge(schm, './dataset/Schmugge/data.csv', ori_out_dir='./dataset/Schmugge/newdata/ori', gt_out_dir='./dataset/Schmugge/newdata/gt')

    # Filter a dataset CSV by given skintone, the other entries will be deleted from the CSV file
    csv_file = os.path.join(db_dir('Schmugge'), 'data.csv') # dataset to process
    update_mode = 'test'
    update_schmugge(csv_file, skintone, mode = update_mode)
    count_skintones(csv_file, skintone)
    count_notes(csv_file, update_mode)


if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    db_model = ''
    db_pred = ''
    if n == 2: # predict over the same dataset of the model
        db_model = sys.argv[1] # first argument, argv[0] is the name of the script
        db_pred = db_model
    elif n == 3: # cross dataset predictions
        db_model = sys.argv[1]
        db_pred = sys.argv[2]
    else:
        exit('''There must be 1 or 2 arguments!
        Usage: python predict.py db-model [db-predict]
        Available DB values:\tSchmugge, ECU, HGR
        Special DB values:\tall, skintone, bench''')


    ## ALL: predict using all the default datasets
    if db_model == 'all': # TODO: misleading name
        timestr = get_timestamp()
        models = ['ECU', 'HGR_small', 'Schmugge']
        datasets = []
        for model_name in models:
            datasets.append(db_dir(model_name))

        base_preds(timestr, models)
        cross_preds(timestr, models, datasets)
    

    ## SKINTONES: predict all on Schmugge skintones
    elif db_model == 'skintones':
        timestr = get_timestamp()
        models = skintones

        base_preds(timestr, models)
        cross_preds(timestr, models, models)
    

    ## BENCHMARK: measure the execution time
    elif db_model == 'bench':
        timestr = get_timestamp()

        # use first 15 ECU images as test set
        in_dir = db_dir('ECU')
        db_csv = os.path.join(in_dir, 'data.csv')
        in_model = 'ECU.csv'
        out_dir = pred_dir('bench', timestr, None)

        # set only the first 15 ECU images as test
        prepare_benchmark_set(db_csv, count=15)
        image_paths = get_test_paths(db_csv)

        # do 5 observations
        # the predictions will be the same
        # but the performance will be logged 5 times
        observations = 5
        for k in range(observations):
            make_predictions(image_paths, in_model, out_dir, out_bench=f'bench{k}.txt')
    

    ## DEFAULT: normal case
    else: 
        in_model = f'./{db_model}.csv'
        in_dir = db_dir(db_pred)
        out_dir = pred_dir(None, None, name = f'{db_model}_on_{db_pred}')

        # special case: predict on Schmugge skintones sub-splits
        if db_pred in skintones:
            filter_schmugge(skintone = db_pred)
            in_dir = db_dir('Schmugge')

        image_paths = get_test_paths(os.path.join(in_dir, 'data.csv'))
        make_predictions(image_paths, in_model, out_dir)


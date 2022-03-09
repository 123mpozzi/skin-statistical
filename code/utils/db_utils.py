import os

import cv2

from utils.abd import abd
from utils.ECU import ECU
from utils.HGR import HGR
from utils.Pratheepan import Pratheepan
from utils.Schmugge import Schmugge, dark, light, medium
from utils.skin_dataset import skin_dataset
from utils.UChile import UChile
from utils.VPU import VPU

# NOTE: method-specific (probabilistic)
models_dir = os.path.join('..', 'models')

skin_databases_skintones = (dark(), medium(), light())
skin_databases = (ECU(), Schmugge(), HGR(), dark(), medium(), light(),
    VPU(), UChile(), abd(), Pratheepan())


# NOTE: method-specific (probabilistic)
def get_model_filename(database: skin_dataset) -> str:
    return os.path.join(models_dir, database.name + '.csv')

def get_db_by_name(name: str) -> skin_dataset:
    for database in skin_databases:
        if database.name == name:
            return database
    
    exit(f'Invalid skin database: {name}')

def skin_databases_names(db_list: list = skin_databases) -> list:
    return [x.name for x in db_list]

def get_trainable() -> list:
    '''Return the list of datasets with already defined splits'''
    result = [x for x in skin_databases if len(x.get_train_paths()) > 0]
    return result

def get_models() -> list:
    '''Return the list of skin datasets having a trained model file'''
    result = [x for x in skin_databases if os.path.isfile(get_model_filename(x))]
    return result

def get_datasets() -> list:
    '''Return the list of actual existing skin dataset in file manager'''
    result = [x for x in skin_databases if os.path.isdir(x.dir)]
    return result

def gen_pred_folders(models: list, batch_type: str) -> list:
    '''Generate folders filenames given a type of batch prediction and model list'''
    base_folders = [f'{x.name}_on_{x.name}' for x in models]
    cross_folders = [f'{x.name}_on_{y.name}' for x in models for y in models if x.name != y.name]

    if batch_type == 'base':
        return base_folders
    elif batch_type == 'cross':
        return cross_folders
    else:
        return base_folders + cross_folders

def bin2vdm(db: skin_dataset, out_dir):
    '''
    Convert boolean binary grountruths in the VDM grountruth format: 
    red pixels where there is skin, overlayed over the original images

    And update CSV file with the new gt path
    '''
    # read the images CSV
    file_content = db.read_csv()

    # rewrite csv file
    with open(db.csv, 'w') as out:
        for entry in file_content:
            csv_fields = db.split_csv_fields(entry)
            ori_path = csv_fields[0]
            gt_path = csv_fields[1]
            
            # Process images

            # load images
            ori_im = cv2.imread(ori_path)
            gt_im = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # everything but skin
            butsk = cv2.copyTo(ori_im, cv2.bitwise_not(gt_im))
            # split the resulting channels
            b,g,r = cv2.split(butsk)
            # utilizzo la maschera come canale red (rossa al posto che bianca)
            r = cv2.bitwise_or(r, gt_im)
            # riunisco gli split con il nuovo canale red
            res = cv2.merge([b,g,r])

            cv2.imwrite(out_dir, res)
            # Update gt path
            csv_fields[1] = out_dir
            
            # Update db CSV
            out.write(db.to_csv_row(*csv_fields))

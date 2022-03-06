#from skin_dataset import skin_dataset
from utils.skin_dataset import skin_dataset
from utils.ECU import ECU
from utils.Schmugge import Schmugge, dark, medium, light
from utils.HGR import HGR
from utils.VPU import VPU
from utils.UChile import UChile
from utils.abd import abd
from utils.Pratheepan import Pratheepan
import os

# db_utils è specifico per questo metodo probabilistico, in Skinny cambiano le utils (model_name, ..) ?
models_dir = os.path.join('..', 'models')

skin_databases_skintones = (dark(), medium(), light())
skin_databases = (ECU(), Schmugge(), HGR(), dark(), medium(), light(),
    VPU(), UChile(), abd(), Pratheepan())

def get_model_filename(database: skin_dataset) -> str:
    return os.path.join(models_dir, database.name + '.csv')

def get_db_by_name(name: str) -> skin_dataset:
    for database in skin_databases:
        if database.name == name:
            return database
    
    exit(f'Invalid skin database: {name}')

def skin_databases_names(db_list: list = skin_databases) -> list:
    #names = []
    #for db in db_list:
    #    names.append(db.name)
    #return names
    return [x.name for x in skin_databases]

def get_models() -> list:
    '''Return the list of skin datasets having a trained model file'''
    result = [x for x in skin_databases if os.path.isfile(get_model_filename(x))]
    return result

def get_datasets() -> list:
    '''Return the list of actual existing skin dataset in file manager'''
    result = [x for x in skin_databases if os.path.isdir(x.dir)]
    return result

# Return list of stacktraces occured
# May be empty it there are no errors
def reset_datasets() -> list:
    stacktraces = []
    for database in skin_databases:
        stacktraces.append(database.reset())
    
    return stacktraces

def gen_pred_folders(models: list, batch_type: str) -> list:
    '''Generate folders filenames given a type of batch prediction and model list'''
    base_folders = [f'{x.name}_on_{x.name}' for x in models]
    cross_folders = [f'{x.name}_on_{y.name}' for x in models for y in models if x.name != y.name]

    if batch_type == 'base':
        return base_folders
    elif batch_type == 'cross':
        return cross_folders
    else:
        return base_folders.extend(cross_folders)

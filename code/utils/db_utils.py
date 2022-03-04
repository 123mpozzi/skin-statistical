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

# db_utils è specifico per questo metodo probabilistico, in Skinny cambiano le utils (model_name, ..)
models_dir = os.path.join('..', 'models')

#skin_databases = (ECU(), Schmugge(), HGR(), dark(), medium(), light())
skin_databases_normal = (ECU(), Schmugge(), HGR())
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
    names = []
    for db in db_list:
        names.append(db.name)
    return names

# Return list of stacktraces occured
# May be empty it there are no errors
def reset_datasets() -> list:
    stacktraces = []
    for database in skin_databases:
        stacktraces.append(database.reset())
    
    return stacktraces

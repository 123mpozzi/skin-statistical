#from skin_dataset import skin_dataset
from utils.skin_dataset import skin_dataset
from utils.ECU import ECU
from utils.Schmugge import Schmugge
from utils.HGR import HGR

# db_utils è specifico per questo metodo probabilistico, in Skinny cambiano le utils (model_name, ..)

skin_databases = (ECU(), Schmugge(), HGR())

def get_model_filename(database: skin_dataset) -> str:
    return database.name + '.csv'

def get_db_by_name(name: str) -> skin_dataset:
    for database in skin_databases:
        if database.name == name:
            return database
    
    exit(f'Invalid skin database: {name}')


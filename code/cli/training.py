import click
from train import *
from utils.db_utils import (get_datasets, get_db_by_name, get_model_filename,
                            skin_databases_names)


@click.group()
def cli_training():
    pass

@cli_training.command(short_help='Generate the model CSV file from a given training dataset')
@click.option('--dataset', '-d', type=click.Choice(skin_databases_names(get_datasets()),
                case_sensitive=False), required=True)
def train(dataset):
    db = get_db_by_name(dataset)
    out = get_model_filename(db)
    image_paths = db.get_train_paths()
    do_training(image_paths, out)

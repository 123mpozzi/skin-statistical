import os
import subprocess

import click
from predict import get_timestamp, method_name, predictions_dir
from utils.db_utils import get_datasets, get_models
from utils.ECU import ECU
from utils.HGR import HGR
from utils.Schmugge import Schmugge, dark, light, medium

# Run the necessary predictions to get tables featured in the thesis


def gen_cmd(skintones: bool, cross_dataset: bool, multiprocessing: bool, timestr: str) -> str:
    targets_args = '-t dark -t medium -t light' if skintones else '-t ECU -t HGR_small -t Schmugge'
    batch_mode = 'cross' if cross_dataset else 'base'
    cmd = 'batchm' if multiprocessing else 'batch'

    method = method_name + '_st' if skintones else method_name
    out_dir = os.path.join(predictions_dir, timestr, method, batch_mode)

    return 'python main.py {} -m {} {} -o {}'.format(cmd, batch_mode, targets_args, out_dir)

@click.group()
def cli_thesis():
    pass

@cli_thesis.command(name='thesis', short_help='Reproduce tables featured in the thesis')
@click.option('--multiprocessing/--no-multiprocessing', '-m', 'multiprocessing', default=True, show_default=True,
              help = 'Whether to enable multiprocessing or not')
def thesis(multiprocessing):
    models = ECU(), HGR(), Schmugge(), dark(), medium(), light()

    for m in models:
        assert m in get_datasets(), f'Necessary dataset not found: {m}'
        assert m in get_models(), f'Necessary model not found: {m}'

    # Reset datasets with predefined splits
    #for m in models:
    #    m.reset(predefined=True)

    timestr = get_timestamp()
    # Call each command synchronously with subprocess.call(), wait for it to end
    commands = []
    for pred_type in [True, False]:
        for batch_mode in [True, False]:
            commands.append(gen_cmd(skintones=pred_type, cross_dataset=batch_mode, multiprocessing=multiprocessing, timestr=timestr))
    
    for cmd in commands:
        subprocess.call(cmd.split(), shell=True, stdin=None, stdout=None, stderr=None)

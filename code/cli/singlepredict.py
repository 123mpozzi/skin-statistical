import os

import click
import pandas as pd
from predict import (base_preds, cross_preds, get_timestamp, make_predictions,
                     pred_dir, predict, pred_name)
from utils.db_utils import *
from utils.ECU import ECU, ECU_bench
from utils.logmanager import *
from utils.metrics_utils import read_performance


@click.group()
def cli_predict():
    pass

@cli_predict.command(short_help='N-on-M datasets predictions')
@click.option('--mode', '-m', type=click.Choice(['base', 'cross', 'all']), required=True)
@click.option('--dataset' , '-d',  multiple=True,
              type=click.Choice(skin_databases_names(get_models_with_datasets()), case_sensitive=False), required = True,
              help = 'Datasets to use (eg. -d ECU -d HGR_small -d medium)')
def batch(mode, dataset):
    '''
    BATCH: N-on-M datasets predictions
    N are models, M are datasets
    '''
    timestr = get_timestamp()

    models = dataset
    models = skin_databases_names(models)

    if mode == 'base':
        base_preds(timestr, models)
    elif mode == 'cross':
        cross_preds(timestr, models)
    else: # 'all' does either base+cross or skinbase+skincross, depending on --skintone
        base_preds(timestr, models)
        cross_preds(timestr, models)

@cli_predict.command(short_help='Measure inference time')
@click.option('--size', '-s', type=int, default = 15, show_default=True,
              help='Benchmark set size, in images (-1 is whole db)')
@click.option('--observations', '-o', type=int, default = 5, show_default=True,
              help='Observations to register for the benchmark set')
def bench(size, observations):
    '''BENCHMARK: measure inference time'''
    timestr = get_timestamp()

    # Use first 15 ECU images as test set        
    ECU_bench().reset(amount=size)
    image_paths = ECU_bench().get_test_paths()
    out_dir = pred_dir('bench', timestr, 'observation{}')

    # Do multiple observations
    # The predictions will be the same but performance will be logged 5 different times
    for k in range(observations):
        assert os.path.isdir(ECU_bench().dir), 'Dataset has no directory: ' + ECU_bench().name
        make_predictions(image_paths, get_model_filename(ECU()),
            out_dir.format(k), out_bench=os.path.join(out_dir.format(k), '..', f'bench{k}.txt'))
    
    # Print inference times
    read_performance(os.path.join(out_dir.format(0), '..'))


@cli_predict.command(short_help='1-on-1 datasets prediction')
@click.option('--model', '-m',
              type=click.Choice(skin_databases_names(get_models()), case_sensitive=False), required=True)
@click.option('--predict', '-p', 'predict_',
              type=click.Choice(skin_databases_names(get_datasets()), case_sensitive=False))
@click.option('--from', '-f', 'from_', type=int, default = 0, help = 'Slice start')
@click.option('--to', '-t', type=int, default = -1, help='Slice end (index excluded)')
@click.option('--bar', '-b', type=int, default = -1, help='Progress bar position (for multiprocessing)')
@click.option('--output', '-o', default = '',
              type=click.Path(exists=False),
              help = 'Define the directory in which to save predictions')
def single(model, predict_, from_, to, bar, output):
    '''SINGLE: 1-on-1 datasets prediction. Can be on self too'''
    # prediction on self
    if predict_ is None:
        predict_ = model
    
    target_dataset = get_db_by_name(predict_)
    if predict_ == model:
        image_paths = target_dataset.get_test_paths() # on same dataset, use test paths
    else:
        image_paths = target_dataset.get_all_paths() # on cross datasets, use all paths

    # to=-1 means till dataset end
    if to == -1:
        to = len(image_paths) # end index is not included in python slicing operator

    assert os.path.isdir(target_dataset.dir), 'Dataset has no directory: ' + target_dataset.name
    # Make predictions
    model_name = get_model_filename(get_db_by_name(model))
    if output == '':
        out_dir = pred_dir(None, None, name = f'{model}_on_{predict_}')
    else:
        name = pred_name(f'{model}_on_{predict_}')
        out_dir = os.path.join(output, name)
        os.makedirs(output, exist_ok=True)
    make_predictions(image_paths[from_:to], model_name, out_dir, pbar_position=bar)

@cli_predict.command(
    short_help='Single image prediction')
@click.option('--model', '-m',
              type=click.Choice(skin_databases_names(get_models()), case_sensitive=False), required=True)
@click.option('--path', '-p',
              type=click.Path(exists=True), required=True,
              help = 'Path to the image to predict on')
def image(model, path):
    '''
    IMAGE: 1 model on 1 image prediction.
    Image may not have a grountruth.
    Result will be placed in the same directory as the input image.
    '''
    ori_name = os.path.basename(path)
    ori_filename, _ = os.path.splitext(ori_name)

    im_abspath = os.path.abspath(path)
    im_dir = os.path.dirname(path)
    p_out = os.path.join(im_dir, ori_filename + '_p.png')

    assert os.path.isfile(path), 'Image file not existing: ' + path
    # Make predictions
    model_name = get_model_filename(get_db_by_name(model))
    info("Reading CSV...")
    probability = pd.read_csv(model_name) # getting the rows from csv
    predict(probability, im_abspath, None, p_out)

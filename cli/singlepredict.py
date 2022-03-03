import click
from predict import make_predictions, get_timestamp, base_preds, cross_preds, pred_dir
from utils.db_utils import *
from utils.ECU import ECU, ECU_bench


@click.group()
def cli_predict():
    pass

# BATCH: N-on-M datasets predictions
# Note: will use only thesis datasets!
@cli_predict.command(short_help='N-on-M datasets predictions')
@click.option('--type', '-t', type=click.Choice(['base', 'cross', 'all']), required=True)
@click.option('--skintones/--no-skintones', 'skintones', default=False,
              help = 'Whether to predict on skintone sub-datasets')
def batch(type, skintones):
    timestr = get_timestamp()

    models = skin_databases_normal
    if skintones == True:
        models = skin_databases_skintones

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
@cli_predict.command(short_help='Measure inference time')
@click.option('--size', '-s', type=int, default = 15, show_default=True,
              help='Benchmark set size, in images (-1 is whole db)')
@click.option('--observations', '-o', type=int, default = 5, show_default=True,
              help='Observations to register for the benchmark set')
def bench(size, observations):
    timestr = get_timestamp()

    # Use first 15 ECU images as test set        
    ECU_bench().reset(amount=size)
    image_paths = ECU_bench().get_test_paths()
    out_dir = pred_dir('bench', timestr, 'observation{}')

    # Do multiple observations
    # The predictions will be the same but performance will be logged 5 different times
    for k in range(observations):
        make_predictions(image_paths, get_model_filename(ECU()),
            out_dir.format(k), out_bench=os.path.join(out_dir.format(k), '..', f'bench{k}.txt'))

# SINGLE: 1-on-1 datasets prediction. Can be on self too
@cli_predict.command(short_help='1-on-1 datasets prediction')
@click.option('--model', '-m',
              type=click.Choice(skin_databases_names(), case_sensitive=False), required=True)
@click.option('--predict', '-p', 'predict_',
              type=click.Choice(skin_databases_names(), case_sensitive=False))
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

    # Make predictions
    model_name = get_model_filename(get_db_by_name(model))
    out_dir = pred_dir(None, None, name = f'{model}_on_{predict_}')
    make_predictions(image_paths[from_:to], model_name, out_dir)
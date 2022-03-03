import click
from metrics import *
from utils.hash_utils import hash_dir

# TODO: validation() --


@click.group()
def measure():
    pass

@measure.command(short_help='Evaluate skin detector performance')
@click.option('--path', '-p',
              type=click.Path(exists=True), required=True,
              help = 'Path to the folder containing the predictions dir (eg. ECU_on_Schmugge)')
def eval(path):
    # Define metric functions used to evaluate
    metrics = [f1_m, f2, iou, dprs_m, mcc, recall, precision, specificity]

    # Get folders containing grountruth and prediction IMAGES
    y_path = os.path.join(path, 'y') # Path eg. 'predictions/HGR_small_on_ECU/y'
    p_path = os.path.join(path, 'p') # Path eg. 'predictions/HGR_small_on_ECU/p'

    rpd = pd_metrics(y_path, p_path, metrics)
    print_pd_mean(rpd, metrics, desc=path + ' with hash=' + hash_dir(path))

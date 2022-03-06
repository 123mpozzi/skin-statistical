import click
from metrics import *
from utils.hash_utils import hash_dir
import json

# TODO: validation() --


dump_dir = os.path.join('..', 'dumps')
dump_filename = os.path.join(dump_dir, 'metrics_{}_{}.json')

@click.group()
def measure():
    pass

@measure.command(short_help='Evaluate skin detector performance')
@click.option('--path', '-p',
              type=click.Path(exists=True), required=True,
              help = 'Path to the folder containing the predictions dir (eg. ECU_on_Schmugge)')
@click.option('--dump/--no-dump', '-d', default=False, help = 'Whether to dump results to files')
def eval(path, dump):
    # Define metric functions used to evaluate
    #metrics = [f1_m, f1fin, f2fin, iou, dprs_m, mcc, recall, precision, specificity]
    metrics = [f1_medium, f1, f2, iou, iou_logical, dprs_medium, dprs, mcc, recall, precision, specificity]

    # Get folders containing grountruth and prediction IMAGES
    y_path = os.path.join(path, 'y') # Path eg. 'predictions/HGR_small_on_ECU/y'
    p_path = os.path.join(path, 'p') # Path eg. 'predictions/HGR_small_on_ECU/p'

    rpd = pd_metrics(y_path, p_path, metrics)
    res = print_pd_mean(rpd, metrics, desc=path + ' with hash=' + hash_dir(path))

    if dump:
        path_bn = os.path.basename(path)
        os.makedirs(dump_dir, exist_ok=True)

        with open(dump_filename.format(path_bn, 'average'), 'w') as f:
            json.dump(res, f, sort_keys = True, indent = 4)
        with open(dump_filename.format(path_bn, 'singles'), 'w') as f:
            json.dump(rpd, f, sort_keys = True, indent = 4)

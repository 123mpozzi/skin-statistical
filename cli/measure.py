import click
from utils.db_utils import skin_databases_names, get_db_by_name
from predict import pred_dir

# TODO: validation() --


@click.group()
def measure():
    pass

@measure.command(short_help='Evaluate skin detector performance')
@click.option('--model', '-m',
              type=click.Choice(skin_databases_names(), case_sensitive=False), required=True)
@click.option('--predict', '-p', 'predict_',
              type=click.Choice(skin_databases_names(), case_sensitive=False))
def eval(model, predict_):
    pred_type = 'base' if model == predict_ else 'cross'
    pred_dir(type=pred_type)
    pass


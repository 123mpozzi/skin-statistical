import click
from utils.db_utils import get_db_by_name, skin_databases_names


@click.group()
def cli_manage():
    pass

@cli_manage.command(short_help='Reset CSV files of datasets by reprocessing them')
@click.option('--dataset', '-d', type=click.Choice(skin_databases_names(), case_sensitive=False), required=True)
@click.option('--predefined/--no-predefined', '-p', 'predefined', default=False,
              help = 'Whether to generate a new data csv or import thesis configuration')
def reset(dataset, predefined):
    if True:
        trace = get_db_by_name(dataset).reset(predefined=predefined)
        if trace: # if there are been errors of some kind
            print(trace)
    else:
        for dataset in skin_databases_names():
            get_db_by_name(dataset).reset()


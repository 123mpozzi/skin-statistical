import click
from utils.db_utils import skin_databases_names, get_db_by_name


@click.group()
def manage():
    pass

@manage.command(short_help='Reset CSV files of datasets by reprocessing them')
@click.option('--db', '-d', type=click.Choice(skin_databases_names(), case_sensitive=False), required=True)
@click.option('--predefined/--no-predefined', '-p', 'predefined', default=False,
              help = 'Whether to generate a new data csv or import thesis configuration')
def reset(db, predefined):
    if True:
        trace = get_db_by_name(db).reset(predefined=predefined)
        if trace: # if there are been errors of some kind
            print(trace)
    else:
        for db in skin_databases_names():
            get_db_by_name(db).reset()


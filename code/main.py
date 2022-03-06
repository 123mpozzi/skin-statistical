import click
from cli.multipredict import cli_multipredict
from cli.singlepredict import cli_predict
from cli.manage import manage
from cli.measure import measure


# Collect command groups
cli = click.CommandCollection(sources=[cli_multipredict, cli_predict, manage, measure])

if __name__ == "__main__":
    # Setup Command Line Interface
    # Register commands
    cli()

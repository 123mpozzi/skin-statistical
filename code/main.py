import click

from cli.manage import cli_manage
from cli.measure import cli_measure
from cli.multipredict import cli_multipredict
from cli.singlepredict import cli_predict
from cli.training import cli_training

# Collect command groups
cli = click.CommandCollection(sources=[cli_multipredict, cli_predict,
        cli_manage, cli_measure, cli_training])

if __name__ == "__main__":
    # Setup Command Line Interface
    # Register commands
    cli()

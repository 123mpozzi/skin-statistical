import click
from logging import basicConfig, getLogger, INFO, FileHandler, StreamHandler
from cli.multipredict import cli_multipredict
from cli.singlepredict import cli_predict
from cli.manage import manage
from cli.measure import measure


# Collect command groups
cli = click.CommandCollection(sources=[cli_multipredict, cli_predict, manage, measure])

if __name__ == "__main__":
    # Setup Command Line Interface and Logger
    basicConfig(
        level=INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            FileHandler("debug.log", mode='a'),
            StreamHandler()
        ]
    )
    getLogger().setLevel(INFO)

    # Register commands
    cli()

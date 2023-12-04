import click

from rnn import rnn
from rnn_scratch import rnn_scratch


@click.group()
def cli():
    pass


cli.add_command(rnn)
cli.add_command(rnn_scratch)

if __name__ == "__main__":
    cli()

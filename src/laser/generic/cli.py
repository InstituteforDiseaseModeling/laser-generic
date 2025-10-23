"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mlaser.generic` python will execute
    ``__main__.py`` as a script. That means there will not be any
    ``laser.generic.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there"s no ``laser.generic.__main__`` in ``sys.modules``.

  Also see (1) from https://click.palletsprojects.com/en/stable/setuptools/
"""

import click

from .core import compute


@click.command()
@click.argument("names", nargs=-1)
def run(names):
    click.echo(compute(names))

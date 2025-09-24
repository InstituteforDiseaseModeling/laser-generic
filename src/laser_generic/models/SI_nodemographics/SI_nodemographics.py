r"""
SI_nodemographics.py

This module defines a basic SI Model and provides a command-line interface (CLI) to run the simulation.

Classes:

    None

Functions:

    run(\*\*kwargs)

        Runs the measles model simulation with the specified parameters.

        Parameters:

            - nticks (int): Number of ticks to run the simulation. Default is 365.
            - seed (int): Random seed for the simulation. Default is 20241107.
            - verbose (bool): If True, print verbose output. Default is False.
            - viz (bool): If True, display visualizations to help validate the model. Default is True.
            - pdf (bool): If True, output visualization results as a PDF. Default is False.
            - output (str): Output file for results. Default is None.
            - params (str): JSON file with parameters. Default is None.
            - param (tuple): Additional parameter overrides in the form of (param:value or param=value). Default is an empty tuple.

Usage:

    To run the simulation from the command line (3650 ticks, 20241107 seed, show visualizations):

"""

import click

from laser_generic import Model
from laser_generic import Susceptibility_SI
from laser_generic import Transmission_SI
from laser_generic.models.SI_nodemographics import get_parameters
from laser_generic.models.SI_nodemographics import get_scenario
from laser_generic.utils import seed_infections_randomly_SI


@click.command()
@click.option("--nticks", default=3650, help="Number of ticks to run the simulation")
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--viz", is_flag=True, default=True, help="Display visualizations  to help validate the model")
@click.option("--pdf", is_flag=True, help="Output visualization results as a PDF")
@click.option("--output", default=None, help="Output file for results")
@click.option("--param", "-p", multiple=True, help="Additional parameter overrides (param:value or param=value)")
def run(**kwargs):
    """
    Run the SI simulation with the given parameters.

    This function initializes the model with the specified parameters, sets up the
    components of the model, seeds initial infections, runs the simulation, and
    optionally visualizes the results.

    Parameters:

        **kwargs: Arbitrary keyword arguments containing the parameters for the simulation.

            Expected keys include:

                - "verbose": (bool) Whether to print verbose output.
                - "viz": (bool) Whether to visualize the results.
                - "pdf": (str) The file path to save the visualization as a PDF.

    Returns:

        None
    """

    parameters = get_parameters(kwargs)
    scenario = get_scenario(parameters)
    model = Model(scenario, parameters)

    # infection dynamics come _before_ incubation dynamics so newly set itimers
    # don't immediately expire
    model.components = [Susceptibility_SI, Transmission_SI]

    seed_infections_randomly_SI(model, ninfections=10)
    # seed_infections_in_patch(model, ipatch=0, ninfections=100)

    model.run()

    if parameters["viz"]:
        model.visualize(pdf=parameters["pdf"])

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, verbose=True, viz=True, pdf=False)

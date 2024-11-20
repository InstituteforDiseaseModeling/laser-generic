"""
Define the parameters and scenario for the generic single-node SI model

Functions:

    get_parameters(kwargs) -> PropertySet:

        Initializes and returns a `PropertySet` object with default parameters,
        optionally overridden by command line arguments.
    
    get_scenario(params, verbose:boolean=False) -> pd.DataFrame:

        Initializes and returns a single node scenario DataFrame

"""
import re
import click
import numpy as np
import geopandas as gpd
import pandas as pd

from laser_core.propertyset import PropertySet


def get_parameters(kwargs) -> PropertySet:
    """
    Generate a set of parameters for the generic SI simulation.

    Args: 
        kwargs (dict): A dictionary containing the command line arguments.

    Returns:

        PropertySet: A PropertySet object containing all the parameters for the simulation.
    """

    meta_params = PropertySet(
        {
            "nticks": 3650,
            "verbose": False,
        }
    )

    SI_params = PropertySet(
        {
            "beta": np.float32(2.0),
        }
    )

    Scenario_params = PropertySet(
        {
            "initial_population": 100000,
        }
    )

    params = PropertySet(meta_params, SI_params, Scenario_params)

    # Finally, overwrite any parameters with those from the command line (optional)
    for key, value in kwargs.items():
        if key in params.keys():
            click.echo(f"Using `{value}` for parameter `{key}` from the command line…")
            params[key] = value
        else:  # arbitrary param:value pairs from the command line
            for kvp in kwargs["param"]:
                key, value = re.split("[=:]+", kvp)
                if key not in params:
                    click.echo(f"Unknown parameter `{key}` ({value=}). Skipping…")
                    continue
                value = type(params[key])(value)  # Cast the value to the same type as the existing parameter
                click.echo(f"Using `{value}` for parameter `{key}` from the command line…")
                params[key] = value

    return params



def get_scenario(params) -> pd.DataFrame:
    """
    get_scenario(params, verbose: bool = False) -> pd.DataFrame:

        Set up scenario from input parameters
        
        Parameters:
            "initial_population": int: The initial population size for the simulation.
        
        Returns:
            pd.DataFrame: A GeoDataFrame containing the merged population and geographical data.
    """

    #Almost completely unnecessary here but following the design pattern.
    return pd.DataFrame.from_dict({"population": params.initial_population})
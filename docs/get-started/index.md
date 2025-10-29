# Get started

LASER is intended for modeling diseases with straightforward and predictable transmission dynamics, such as influenza or measles.

## Installation

See the Home page for [installation instructions](../index.md#installation).

## Usage

To use the project:

```python
import laser_generic
laser_generic.compute(...)
```

## Build a model

LASER is designed to be modular and flexible to accommodate a variety of modeling needs. The example below will demonstrate the basic set up of a LASER model, using a simple, 25-patch SEIR-like model with vital dynamics (births).

1. Define the scenario.

    You must define a `scenario` DataFrame with one row per patch, each including a `population`, and optionally `latitude` and `longitude` for plotting or spatial coupling.

    This example defines a 5×5 grid of patches with larger center and smaller outer nodes.

    ``` python
    import pandas as pd

    def make_city_scenario(n=5, spacing=1.0):
        coords = [(i*spacing, j*spacing) for i in range(-(n//2), n//2+1)
                                        for j in range(-(n//2), n//2+1)]
        pops = []
        for (x, y) in coords:
            dist = abs(x) + abs(y)
            if dist == 0:
                pops.append(5000)    # city center
            elif dist == 1:
                pops.append(2000)    # suburbs
            else:
                pops.append(500)     # rural
        return pd.DataFrame({
            "population": pops,
            "latitude": [y for (x, y) in coords],
            "longitude": [x for (x, y) in coords],
        })

    scenario = make_city_scenario()
    ```

1. Define the simulation parameters.

    LASER uses a `PropertySet` (like a dict) to define model-wide parameters. Start with defaults and override as needed.

    ``` python
    from laser_generic.utils import get_default_parameters

    params = get_default_parameters() | {
        "seed": 42,
        "beta": 0.3,         # transmission rate
        "inf_mean": 5.0,     # mean infectious period
        "inf_sigma": 1.0,    # stddev infectious period
    }

    # Optional importation settings
    # params |= {
    #     "importation_period": 10,
    #     "importation_count": 5,
    #     "importation_start": 0,
    #     "importation_end": 50,
    # }
    ```

1. Create the model.

    This initializes patch-level arrays and allocates the population frame.

    ``` python
    from laser_generic.model import Model

    model = Model(scenario, params)
    ```

1. Attach components.

    Components are step functions called every tick. Attach those you want in order.

    ``` python
    from laser_generic.transmission import Transmission
    from laser_generic.susceptibility import Susceptibility
    from laser_generic.exposure import Exposure
    from laser_generic.infection import Infection
    from laser_generic.importation import Infect_Random_Agents
    from laser_generic.births import Births

    model.components = [
        Births,                # assigns dob
        Susceptibility,        # creates .susceptibility property
        Transmission,          # simulates infection pressure
        Exposure,              # tracks latent infections
        Infection,             # resolves infectious → recovered
        Infect_Random_Agents,  # seeds infections over time
    ]

    # Optional: ensure safe integer arithmetic
    import numpy as np
    model.patches.populations = model.patches.populations.astype(np.int64)
    ```

1. Run the simulation.

    Once components are attached, run the simulation.

    ``` python
    model.run()
    ```

1. Export patch-level time series.

    Use this helper to export all `(tick, patch, variable)` values into a long-format CSV or HDF file.

    ``` python
    def export_patch_timeseries(model, filename="report.csv", format="csv"):
        import pandas as pd
        import numpy as np

        npatches = len(model.patches)
        vars = {}
        for name in dir(model.patches):
            arr = getattr(model.patches, name, None)
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == npatches:
                vars[name] = arr

        records = []
        maxtime = max(arr.shape[0] for arr in vars.values())
        for t in range(maxtime):
            for p in range(npatches):
                row = {"tick": t, "patch": p}
                for v, arr in vars.items():
                    if t < arr.shape[0]:
                        row[v] = arr[t, p]
                records.append(row)

        df = pd.DataFrame(records)

        if format == "csv":
            df.to_csv(filename, index=False)
        elif format == "h5":
            df.to_hdf(filename, key="laser", mode="w")
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Exported patch-level time series to {filename}")
        return df

    # Export to disk
    df = export_patch_timeseries(model, "patch_report.csv", format="csv")
    ```

1. Inspect or visualize output.

    LASER provides built-in visualizations: scenario maps, birth distributions, and timing pie charts.

    ``` python
    model.visualize(pdf=False)

    print("Simulation complete.")
    print("Patch populations over time (first 10 ticks):")
    print(model.patches.populations[:10, :])
    print("Cases test over time (first 100 ticks):")
    print(model.patches.cases_test[:100, :])
    ```

## Tutorials

The [Tutorials](../tutorials/sir.md) section begins with example code to demonstrate setting up simple SIR models and gradually adding complexity. For an interactive experience, begin with the first Jupyter notebook tutorial [SI model with no demographics](../tutorials/notebooks/01_SI_nobirths_logistic_growth.ipynb) for a very simple implementation of the LASER model.
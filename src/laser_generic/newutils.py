import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

__all__ = ["RateMap", "draw_vital_dynamics", "grid", "linear"]


class RateMap:
    def __init__(self, npatches: int, nsteps: int):
        self._npatches = npatches
        self._nsteps = nsteps

        return

    @staticmethod
    def from_scalar(scalar: float, npatches: int, nsteps: int) -> "RateMap":
        assert scalar >= 0.0, "scalar must be non-negative"
        assert npatches > 0, "npatches must be greater than 0"
        assert nsteps > 0, "nsteps must be greater than 0"
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        tmp = np.array([[scalar]], dtype=np.float32)
        instance._data = np.broadcast_to(tmp, (nsteps, npatches))

        return instance

    @staticmethod
    def from_timeseries(data: np.ndarray, npatches: int) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert npatches > 0, "npatches must be greater than 0"
        nsteps = data.shape[0]
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        instance._data = np.broadcast_to(data[:, None], (nsteps, npatches))

        return instance

    @staticmethod
    def from_patches(data: np.ndarray, nsteps: int) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert nsteps > 0, "nsteps must be greater than 0"
        npatches = data.shape[0]
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        instance._data = np.broadcast_to(data[None, :], (nsteps, npatches))

        return instance

    @staticmethod
    def from_array(data: np.ndarray, writeable: bool = False) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 2, "data must be a 2D array"
        assert data.shape[0] > 0, "data must have at least one row"
        assert data.shape[1] > 0, "data must have at least one column"
        nsteps, npatches = data.shape
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        instance._data = data.astype(np.float32)
        instance._data.flags.writeable = writeable

        return instance

    @property
    def rates(self):
        return self._data

    @property
    def npatches(self):
        return self._npatches

    @property
    def nsteps(self):
        return self._nsteps


def grid(M=5, N=5, grid_size=10, population_fn=None, origin_x=0, origin_y=0):
    """
    Create an MxN grid of cells anchored at (0, 0) with populations and geometries.

    Args:
        M (int): Number of rows (north-south).
        N (int): Number of columns (east-west).
        grid_size (float): Size of each cell in kilometers (default 10).
        population (callable): Function returning population for a cell.
        origin_x (float): longitude of the origin (bottom-left corner) -180 <= origin_x < 180.
        origin_y (float): latitude of the origin (bottom-left corner) -90 <= origin_y < 90.

    Returns:
        GeoDataFrame: Columns are nodeid, population, geometry.
    """
    if population_fn is None:

        def population_fn(x: int, y: int) -> int:
            return int(np.random.uniform(1000, 100000))

    # Convert grid_size from kilometers to degrees (approximate)
    km_per_degree = 111.320
    grid_size_deg = grid_size / km_per_degree

    cells = []
    nodeid = 0
    for i in range(M):
        for j in range(N):
            x0 = origin_x + j * grid_size_deg
            y0 = origin_y + i * grid_size_deg
            x1 = x0 + grid_size_deg
            y1 = y0 + grid_size_deg
            poly = Polygon(
                [
                    (x0, y0),  # NW
                    (x1, y0),  # NE
                    (x1, y1),  # SE
                    (x0, y1),  # SW
                    (x0, y0),  # Close polygon
                ]
            )
            cells.append({"nodeid": nodeid, "population": population_fn(j, i), "geometry": poly})
            nodeid += 1

    gdf = gpd.GeoDataFrame(cells, columns=["nodeid", "population", "geometry"], crs="EPSG:4326")

    return gdf


def linear(N=10, node_size_km=10, population_fn=None, origin_x=0, origin_y=0):
    """
    Create a linear set of population nodes as rectangular cells.

    Args:
        N (int): Number of nodes.
        node_size_km (float): Size of each node in kilometers (width).
        population_fn (callable): Function taking node index and returning population.
        origin_x (float): Longitude of the starting node (westmost).
        origin_y (float): Latitude of the starting node (southmost).

    Returns:
        GeoDataFrame: Columns are nodeid, population, geometry.
    """
    if population_fn is None:

        def population_fn(idx: int) -> int:
            return int(np.random.uniform(1000, 100000))

    # Convert node size from km to degrees (approximate)
    # meters_per_degree = 111_320
    # node_size_deg = (node_size_km * 1000) / meters_per_degree
    km_per_degree = 111.320
    node_size_deg = node_size_km / km_per_degree

    cells = []
    for idx in range(N):
        x0 = origin_x + idx * node_size_deg
        y0 = origin_y
        x1 = x0 + node_size_deg
        y1 = y0 + node_size_deg
        poly = Polygon(
            [
                (x0, y0),  # SW
                (x1, y0),  # SE
                (x1, y1),  # NE
                (x0, y1),  # NW
                (x0, y0),  # Close polygon
            ]
        )
        cells.append({"nodeid": idx, "population": population_fn(idx), "geometry": poly})

    gdf = gpd.GeoDataFrame(cells, columns=["nodeid", "population", "geometry"], crs="EPSG:4326")

    return gdf


def draw_vital_dynamics(birthrates: RateMap, mortality: RateMap, initial_pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert birthrates.npatches == mortality.npatches == initial_pop.shape[0], (
        "birthrates, mortality, and initial_pop must have the same number of patches"
    )
    assert birthrates.nsteps == mortality.nsteps, "birthrates and mortality must have the same number of steps"

    current_pop = initial_pop.copy()
    births = np.zeros_like(birthrates.rates, dtype=np.uint32)
    deaths = np.zeros_like(mortality.rates, dtype=np.uint32)

    for t in range(birthrates.nsteps):
        # Poisson draw for births per patch
        births[t] = np.random.poisson(birthrates.rates[t] * current_pop / 1000)  # CBR is per 1,000 population
        # Binomial draw for deaths per patch
        # np.expm1(x) computes exp(x) - 1 accurately for small x
        # -np.expm1(x) computes 1 - exp(x) accurately for small x
        # -np.expm1(-mortality.rates[t]) gives the probability of death in a time step
        deaths[t] = np.random.binomial(current_pop, -np.expm1(-mortality.rates[t]))
        # Update population
        current_pop += births[t]
        current_pop -= deaths[t]

    return births, deaths

import time
from typing import ClassVar

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry import Polygon

__all__ = ["TimingStats", "ValuesMap", "estimate_capacity", "get_centroids", "grid"]


class ValuesMap:
    """
    A class to efficiently represent values mapped over nodes and time steps.

    Arguments:
        nnodes (int): Number of nodes.
        nsteps (int): Number of time steps.

    Methods to create ValuesMap from different data sources:
        - from_scalar(scalar: float, nnodes: int, nsteps: int)
        - from_timeseries(data: np.ndarray, nnodes: int)
        - from_nodes(data: np.ndarray, nsteps: int)
        - from_array(data: np.ndarray, writeable: bool = False)
    """

    def __init__(self, nnodes: int, nsteps: int):
        self._nnodes = nnodes
        self._nsteps = nsteps

        return

    @staticmethod
    def from_scalar(scalar: float, nnodes: int, nsteps: int) -> "ValuesMap":
        """
        Create a ValuesMap with the same scalar value for all nodes and time steps.

        Args:
            scalar (float): The scalar value to fill the map.
            nnodes (int): Number of nodes.
            nsteps (int): Number of time steps.

        Returns:
            ValuesMap: The created ValuesMap instance.
        """
        assert scalar >= 0.0, "scalar must be non-negative"
        assert nnodes > 0, "nnodes must be greater than 0"
        assert nsteps > 0, "nsteps must be greater than 0"
        instance = ValuesMap(nnodes=nnodes, nsteps=nsteps)
        tmp = np.array([[scalar]], dtype=np.float32)
        instance._data = np.broadcast_to(tmp, (nsteps, nnodes))

        return instance

    @staticmethod
    def from_timeseries(data: np.ndarray, nnodes: int) -> "ValuesMap":
        """
        Create a ValuesMap from a time series array for all nodes.

        All nodes have the same time series data.

        nsteps is inferred from the length of data.

        Args:
            data (np.ndarray): 1D array of time series data.
            nnodes (int): Number of nodes.

        Returns:
            ValuesMap: The created ValuesMap instance.
        """
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert nnodes > 0, "nnodes must be greater than 0"
        nsteps = data.shape[0]
        instance = ValuesMap(nnodes=nnodes, nsteps=nsteps)
        instance._data = np.broadcast_to(data[:, None], (nsteps, nnodes))

        return instance

    @staticmethod
    def from_nodes(data: np.ndarray, nsteps: int) -> "ValuesMap":
        """
        Create a ValuesMap from a nodes array for all time steps.

        All time steps have the same node data.

        nnodes is inferred from the length of data.

        Args:
            data (np.ndarray): 1D array of node data.
            nsteps (int): Number of time steps.

        Returns:
            ValuesMap: The created ValuesMap instance.
        """
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert nsteps > 0, "nsteps must be greater than 0"
        nnodes = data.shape[0]
        instance = ValuesMap(nnodes=nnodes, nsteps=nsteps)
        instance._data = np.broadcast_to(data[None, :], (nsteps, nnodes))

        return instance

    @staticmethod
    def from_array(data: np.ndarray, writeable: bool = False) -> "ValuesMap":
        """
        Create a ValuesMap from a 2D array of data.

        Args:
            data (np.ndarray): 2D array of shape (nsteps, nnodes).
            writeable (bool): If True, the underlying data array is writeable and can be modified during simulation. Default is False.

        Returns:
            ValuesMap: The created ValuesMap instance.
        """
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 2, "data must be a 2D array"
        assert data.shape[0] > 0, "data must have at least one row"
        assert data.shape[1] > 0, "data must have at least one column"
        nsteps, nnodes = data.shape
        instance = ValuesMap(nnodes=nnodes, nsteps=nsteps)
        instance._data = data.astype(np.float32)
        instance._data.flags.writeable = writeable

        return instance

    @property
    def nnodes(self):
        """Number of nodes."""
        return self._nnodes

    @property
    def nsteps(self):
        """Number of time steps."""
        return self._nsteps

    @property
    def shape(self):
        """Shape of the underlying data array (nsteps, nnodes)."""
        return self._data.shape

    @property
    def values(self):
        """Underlying data array of shape (nsteps, nnodes)."""
        return self._data


def grid(M=5, N=5, node_size_km=10, population_fn=None, origin_x=0, origin_y=0):
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

    # Convert node_size_km from kilometers to degrees (approximate)
    km_per_degree = 111.320
    node_size_deg = node_size_km / km_per_degree

    cells = []
    nodeid = 0
    for i in range(M):
        for j in range(N):
            x0 = origin_x + j * node_size_deg
            y0 = origin_y + i * node_size_deg
            x1 = x0 + node_size_deg
            y1 = y0 + node_size_deg
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


def estimate_capacity(birthrates: np.ndarray, initial_pop: np.ndarray) -> np.ndarray:
    """
    Estimate the carrying capacity of each node given birthrates and initial population.

    Args:
        birthrates (np.ndarray): 2D array of shape (nticks, nnodes) with CBR (birthrates per 1000 individuals per year) in effect at each tick.
        initial_pop (np.ndarray): 1D array of shape (nnodes,) with initial population per node.

    Returns:
        np.ndarray: 1D array of shape (nnodes,) with estimated final population count per node.
    """
    nticks, nnodes = birthrates.shape
    assert len(initial_pop) == nnodes, "initial_pop length must match number of nodes in birthrates_map"
    estimate = initial_pop.copy()

    for t in range(nticks):
        # Poisson draw for births per patch

        # naive = rates[t] / 1000 / 365
        # _ = np.random.poisson(naive * estimate)
        accurate = (1.0 + birthrates[t] / 1000) ** (1.0 / 365) - 1.0
        delta = np.random.poisson(accurate * estimate)
        estimate += delta

    return estimate


class TimingContext:
    """Internal class for timing context management."""

    def __init__(self, label: str, stats: "TimingStats", parent: dict) -> None:  # type: ignore
        self.label = label
        self.stats = stats
        self.parent = parent
        self.children = {}
        self.ncalls = 0
        self.elapsed = 0
        self.start = 0
        self.end = 0

        return

    def __enter__(self):
        self.ncalls += 1
        self.stats._enter(self)
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter_ns()
        self.elapsed += self.end - self.start
        self.stats._exit(self)

        return

    @property
    def inclusive(self) -> int:
        return self.elapsed

    @property
    def exclusive(self) -> int:
        excl = self.elapsed
        for child in self.children.values():
            excl -= child.elapsed

        return excl


class _TimingStats:
    """
    Internal class for managing timing statistics.
    """

    def __init__(self) -> None:
        self.frozen = False
        self.context = {}
        self.root = self.start("root")
        self.root.__enter__()

        return

    def start(self, label: str) -> TimingContext:
        """Create a timing context with the given label."""
        assert self.frozen is False

        if label not in self.context:
            self.context[label] = TimingContext(label, self, self.context)

        return self.context[label]

    def _enter(self, context: TimingContext) -> None:
        self.context = context.children
        return

    def _exit(self, context: TimingContext) -> None:
        assert self.context is context.children
        self.context = context.parent
        return

    def freeze(self) -> None:
        """Freeze the timing statistics."""
        assert self.frozen is False
        self.root.__exit__(None, None, None)
        self.frozen = True

        return

    _scale_factors: ClassVar[dict[str, float]] = {
        "ns": 1,
        "nanoseconds": 1,
        "us": 1e3,
        "µs": 1e3,
        "microseconds": 1e3,
        "ms": 1e6,
        "milliseconds": 1e6,
        "s": 1e9,
        "sec": 1e9,
        "seconds": 1e9,
    }

    def to_string(self, scale: str = "ms") -> str:
        assert self.frozen is True

        assert scale in self._scale_factors
        factor = self._scale_factors[scale]

        lines = []

        def _recurse(node: TimingContext, depth: int) -> None:
            indent = "    " * depth
            tot_time = node.elapsed / factor
            avg_time = node.elapsed / node.ncalls / factor if node.ncalls > 0 else 0
            exc_time = node.exclusive / factor
            lines.append(
                f"{indent}{node.label}: {node.ncalls} calls, total {tot_time:.3f} {scale}, avg {avg_time:.3f} {scale}, excl {exc_time:.3f} {scale}"
            )
            for child in node.children.values():
                _recurse(child, depth + 1)

            return

        _recurse(self.root, 0)
        return "\n".join(lines)

    def to_dict(self, scale: str = "ms") -> dict:
        assert self.frozen is True

        assert scale in self._scale_factors
        factor = self._scale_factors[scale]

        def _recurse(node: TimingContext) -> dict:
            result = {
                "label": node.label,
                "ncalls": node.ncalls,
                "inclusive_ns": node.inclusive / factor,
                # "exclusive_ns": node.exclusive / factor,
                "children": [],
            }
            for child in node.children.values():
                result["children"].append(_recurse(child))

            return result

        return _recurse(self.root)


TimingStats = _TimingStats()


def validate(pre, post):
    """
    Decorator to add pre- and post-validation to a method.

    Calls the given pre- and post-validation methods if the model or component is in validating mode.
    """

    def decorator(func):
        def wrapper(self, tick: int, *args, **kwargs):
            if pre and (getattr(self.model, "validating", False) or getattr(self, "validating", False)):
                with TimingStats.start(pre.__name__):
                    getattr(self, pre.__name__)(tick)
            result = func(self, tick, *args, **kwargs)
            if post and (getattr(self.model, "validating", False) or getattr(self, "validating", False)):
                with TimingStats.start(post.__name__):
                    getattr(self, post.__name__)(tick)
            return result

        return wrapper

    return decorator


def get_centroids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Get centroids of geometries in gdf in degrees (EPSG:4326)."""

    gdf_3857 = gdf.to_crs(epsg=3857)
    centroids_3857 = gdf_3857.geometry.centroid

    # centroids_3857.to_crs(epsg=4326) emits a Warning is there is only one point (one node)
    if len(centroids_3857) > 1:
        centroids_deg = centroids_3857.to_crs(epsg=4326)
    else:
        # Explicitly transform the single centroid
        transformer = Transformer.from_crs(3857, 4326, always_xy=True)
        x, y = centroids_3857.x.values[0], centroids_3857.y.values[0]
        lon, lat = transformer.transform(x, y)
        centroids_deg = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")

    return centroids_deg

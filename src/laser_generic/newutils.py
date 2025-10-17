import time
import warnings

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

__all__ = ["PubSub", "RateMap", "TimingStats", "estimate_capacity", "grid"]


class RateMap:
    def __init__(self, nnodes: int, nsteps: int):
        self._nnodes = nnodes
        self._nsteps = nsteps

        return

    @staticmethod
    def from_scalar(scalar: float, nnodes: int, nsteps: int) -> "RateMap":
        assert scalar >= 0.0, "scalar must be non-negative"
        assert nnodes > 0, "nnodes must be greater than 0"
        assert nsteps > 0, "nsteps must be greater than 0"
        instance = RateMap(nnodes=nnodes, nsteps=nsteps)
        tmp = np.array([[scalar]], dtype=np.float32)
        instance._data = np.broadcast_to(tmp, (nsteps, nnodes))

        return instance

    @staticmethod
    def from_timeseries(data: np.ndarray, nnodes: int) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert nnodes > 0, "nnodes must be greater than 0"
        nsteps = data.shape[0]
        instance = RateMap(nnodes=nnodes, nsteps=nsteps)
        instance._data = np.broadcast_to(data[:, None], (nsteps, nnodes))

        return instance

    @staticmethod
    def from_nodes(data: np.ndarray, nsteps: int) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert nsteps > 0, "nsteps must be greater than 0"
        nnodes = data.shape[0]
        instance = RateMap(nnodes=nnodes, nsteps=nsteps)
        instance._data = np.broadcast_to(data[None, :], (nsteps, nnodes))

        return instance

    @staticmethod
    def from_array(data: np.ndarray, writeable: bool = False) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 2, "data must be a 2D array"
        assert data.shape[0] > 0, "data must have at least one row"
        assert data.shape[1] > 0, "data must have at least one column"
        nsteps, nnodes = data.shape
        instance = RateMap(nnodes=nnodes, nsteps=nsteps)
        instance._data = data.astype(np.float32)
        instance._data.flags.writeable = writeable

        return instance

    @property
    def rates(self):
        return self._data

    @property
    def nnodes(self):
        return self._nnodes

    @property
    def nsteps(self):
        return self._nsteps


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
    def __init__(self, label: str, stats: "TimingStats", parent: dict) -> None:
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
    def __init__(self) -> None:
        self.frozen = False
        self.context = {}
        self.root = self.start("root")
        self.root.__enter__()

        return

    def start(self, label: str) -> TimingContext:
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
        assert self.frozen is False
        self.root.__exit__(None, None, None)
        self.frozen = True

        return

    def to_string(self, scale: str = "ms") -> str:
        assert self.frozen is True

        scale_factors = {
            "ns": 1,
            "nanoseconds": 1,
            "us": 1e3,
            "µs": 1e3,
            "microseconds": 1e3,
            "ms": 1e6,
            "milliseonds": 1e6,
            "s": 1e9,
            "sec": 1e9,
            "seconds": 1e9,
        }
        assert scale in scale_factors
        factor = scale_factors[scale]

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


TimingStats = _TimingStats()


def validate(pre, post):
    def decorator(func):
        def wrapper(self, tick: int, *args, **kwargs):
            if pre and self.model.validating:
                with TimingStats.start(pre.__name__):
                    getattr(self, pre.__name__)(tick)
            result = func(self, tick, *args, **kwargs)
            if post and self.model.validating:
                with TimingStats.start(post.__name__):
                    getattr(self, post.__name__)(tick)
            return result

        return wrapper

    return decorator


class PubSub:
    def __init__(self, fn_name):
        self._subscriptions = []
        self._fn_name = fn_name
        self.ts = TimingStats
        return

    def __add__(self, component):
        if component in self._subscriptions:
            warnings.warn(f"{component.__class__.__name__} already subscribed.", stacklevel=2)
            return self

        callback = getattr(component, self._fn_name) if hasattr(component, self._fn_name) else None

        if callback is None:
            warnings.warn(f"{component.__class__.__name__}.{self._fn_name}() missing.", stacklevel=2)
            return self

        if not callable(callback):
            warnings.warn(f"{component.__class__.__name__}.{self._fn_name} is not callable.", stacklevel=2)
            return self

        self._subscriptions += [component]

        return self

    def __sub__(self, component):
        if component not in self._subscriptions:
            warnings.warn(f"{component.__class__.__name__} not subscribed.", stacklevel=2)
            return

        self._subscriptions.remove(component)

        return self

    @property
    def subscriptions(self):
        return [getattr(component, self._fn_name) for component in self._subscriptions]

    def trigger(self, *args):
        for component in self._subscriptions:
            callback = getattr(component, self._fn_name)
            with self.ts.start(f"{component.__class__.__name__}.{self._fn_name}()"):
                callback(*args)

        return

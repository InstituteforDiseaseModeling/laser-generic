import time
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon

__all__ = ["RateMap", "TimingStats", "draw_vital_dynamics", "grid", "linear"]


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


def validate(pre, post):
    def decorator(func):
        def wrapper(self, tick: int, *args, **kwargs):
            if pre:
                getattr(self, pre.__name__)(tick)
            result = func(self, tick, *args, **kwargs)
            if post:
                getattr(self, post.__name__)(tick)
            return result

        return wrapper

    return decorator


class _TimerContext:
    def __init__(self, timer_stats: "TimingStats", label: str):
        self._timer_stats = timer_stats
        self._label = label
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter_ns()
        self._timer_stats._enter_context(self._label, self._start_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter_ns()
        self._timer_stats._exit_context(self._label, self._start_time, end_time)


class TimingStats:
    def __init__(self):
        self._global_start_time = time.perf_counter_ns()
        self._frozen = False
        self._timer_stack: list[str] = []
        self._timing_data: dict[str, dict[str, Any]] = {}

    def freeze(self) -> None:
        if self._frozen:
            return

        self._frozen = True
        global_end_time = time.perf_counter_ns()

        if "__global__" not in self._timing_data:
            self._timing_data["__global__"] = {
                "total_time": global_end_time - self._global_start_time,
                "call_count": 1,
                "children": set(),
                "parent": None,
                "self_time": global_end_time - self._global_start_time,
            }

    def start(self, label: str) -> "_TimerContext":
        if self._frozen:
            raise RuntimeError("Cannot start new timers after freeze() has been called")

        return _TimerContext(self, label)

    def _enter_context(self, label: str, start_time: int) -> None:
        current_parent = self._timer_stack[-1] if self._timer_stack else None
        self._timer_stack.append(label)

        if label not in self._timing_data:
            self._timing_data[label] = {"total_time": 0, "call_count": 0, "children": set(), "parent": current_parent, "self_time": 0}

        if current_parent:
            self._timing_data[current_parent]["children"].add(label)

        self._timing_data[label]["call_count"] += 1

    def _exit_context(self, label: str, start_time: int, end_time: int) -> None:
        elapsed = end_time - start_time
        self._timing_data[label]["total_time"] += elapsed

        if self._timer_stack and self._timer_stack[-1] == label:
            self._timer_stack.pop()

        self._compute_self_time()

    def _compute_self_time(self) -> None:
        for data in self._timing_data.values():
            children_time = sum(self._timing_data[child]["total_time"] for child in data["children"] if child in self._timing_data)
            data["self_time"] = data["total_time"] - children_time

    def to_string(self, scale: str = "ms") -> str:
        if not self._frozen:
            raise RuntimeError("Must call freeze() before generating string representation")

        scale_factors = {
            "ns": (1, "ns"),
            "microseconds": (1_000, "μs"),
            "μs": (1_000, "μs"),
            "milliseconds": (1_000_000, "ms"),
            "ms": (1_000_000, "ms"),
            "seconds": (1_000_000_000, "s"),
            "s": (1_000_000_000, "s"),
        }

        if scale not in scale_factors:
            raise ValueError(f"Invalid scale '{scale}'. Valid options: {list(scale_factors.keys())}")

        scale_factor, scale_unit = scale_factors[scale]

        root_entries = [label for label, data in self._timing_data.items() if data["parent"] is None]

        result_lines = []
        # for root in sorted(root_entries):
        for root in root_entries:
            self._format_timing_entry(root, result_lines, 0, scale_factor, scale_unit)

        return "\n".join(result_lines)

    def _format_timing_entry(self, label: str, result_lines: list[str], indent_level: int, scale_factor: int, scale_unit: str) -> None:
        data = self._timing_data[label]
        indent = "  " * indent_level
        total_time = data["total_time"] / scale_factor
        self_time = data["self_time"] / scale_factor
        count = data["call_count"]

        if len(data["children"]) > 0:
            result_lines.append(f"{indent}{label}: {total_time:.2f}{scale_unit} (self: {self_time:.2f}{scale_unit}, calls: {count})")
        else:
            result_lines.append(f"{indent}{label}: {total_time:.2f}{scale_unit} (calls: {count})")

        # for child in sorted(data["children"]):
        for child in data["children"]:
            if child in self._timing_data:
                self._format_timing_entry(child, result_lines, indent_level + 1, scale_factor, scale_unit)

    def plot_treemap(self, title: str = "Timing Treemap", scale: str = "ms", figsize: tuple = (12, 8)) -> None:
        if not self._frozen:
            raise RuntimeError("Must call freeze() before generating treemap")

        scale_factors = {
            "ns": (1, "ns"),
            "microseconds": (1_000, "μs"),
            "μs": (1_000, "μs"),
            "milliseconds": (1_000_000, "ms"),
            "ms": (1_000_000, "ms"),
            "seconds": (1_000_000_000, "s"),
            "s": (1_000_000_000, "s"),
        }

        if scale not in scale_factors:
            raise ValueError(f"Invalid scale '{scale}'. Valid options: {list(scale_factors.keys())}")

        scale_factor, scale_unit = scale_factors[scale]

        _fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        total_time = sum(data["total_time"] for data in self._timing_data.values() if data["parent"] is None)

        if total_time > 0:
            y_pos = 0
            # for label in sorted([l for l, d in self._timing_data.items() if d["parent"] is None]):
            for label in [lbl for lbl, dta in self._timing_data.items() if dta["parent"] is None]:
                data = self._timing_data[label]
                height = data["total_time"] / total_time
                self._draw_treemap_rect(ax, label, data, 0, y_pos, 1, height, scale_factor, scale_unit, 0)
                y_pos += height

        plt.tight_layout()
        plt.show()

    def _draw_treemap_rect(
        self, ax, label: str, data: dict, x: float, y: float, width: float, height: float, scale_factor: int, scale_unit: str, depth: int
    ) -> None:
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc", "#c2c2f0", "#ffb3e6", "#c4e17f", "#76d7c4", "#f7dc6f"]
        color = colors[depth % len(colors)]

        rect = Rectangle((x, y), width, height, linewidth=1, edgecolor="black", facecolor=color, alpha=0.7)
        ax.add_patch(rect)

        time_value = data["total_time"] / scale_factor
        text = f"{label}\n{time_value:.1f}{scale_unit}"

        if width > 0.1 and height > 0.05:
            ax.text(
                x + width / 2,
                y + height / 2,
                text,
                ha="center",
                va="center",
                fontsize=max(6, int(8 - depth)),
                wrap=True,
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.8},
            )

        children = [child for child in data["children"] if child in self._timing_data]
        if children and height > 0.05:
            total_children_time = sum(self._timing_data[child]["total_time"] for child in children)

            if total_children_time > 0:
                child_y = y
                child_height = height * 0.8
                child_start_x = x + width * 0.1

                # for child in sorted(children):
                for child in children:
                    child_data = self._timing_data[child]
                    child_width = (width * 0.8) * (child_data["total_time"] / total_children_time)

                    if child_width > 0.01:
                        self._draw_treemap_rect(
                            ax, child, child_data, child_start_x, child_y, child_width, child_height, scale_factor, scale_unit, depth + 1
                        )
                        child_start_x += child_width

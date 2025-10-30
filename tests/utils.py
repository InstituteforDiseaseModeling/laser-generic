import warnings

import contextily as ctx
import numpy as np

from laser.generic.newutils import grid

__all__ = ["base_maps", "stdgrid"]


def POP_FN(x, y):
    return int(np.random.uniform(10_000, 1_000_000))


# Black Rock Desert, NV = 40°47'13"N 119°12'15"W (40.786944, -119.204167)
_latitude = 40.786944
_longitude = -119.204167


def stdgrid(M=10, N=10, node_size_km=10, population_fn=POP_FN, origin_x=_longitude, origin_y=_latitude):
    return grid(M, N, node_size_km, population_fn, origin_x, origin_y)


try:
    base_maps = [
        ctx.providers.Esri.NatGeoWorldMap,
        ctx.providers.Esri.WorldGrayCanvas,
        ctx.providers.Esri.WorldImagery,
        ctx.providers.Esri.WorldPhysical,
        ctx.providers.Esri.WorldShadedRelief,
        ctx.providers.Esri.WorldStreetMap,
        ctx.providers.Esri.WorldTerrain,
        ctx.providers.Esri.WorldTopoMap,
        # ctx.providers.NASAGIBS.ModisTerraTrueColorCR,
    ]
except Exception:
    warnings.warn("Couldn't load basemaps.", stacklevel=2)
    base_maps = []

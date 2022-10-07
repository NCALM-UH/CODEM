import os
import tempfile
import warnings
from typing import NamedTuple
from typing import Optional

import pytest
from osgeo import gdal


class Translation(NamedTuple):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Rotation(NamedTuple):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def manipulate_raster(
    dem_aoi: str,
    *,
    rotation: Optional[Rotation] = None,
    translation: Optional[Translation] = None,
) -> str:

    if rotation is None and translation is None:
        warnings.warn(
            UserWarning(
                "manipulate_raster was called but with no rotation or "
                "translation option specified."
            )
        )
        return dem_aoi
    temp_location = os.path.dirname(dem_aoi)
    output_file = tempfile.NamedTemporaryFile(
        dir=temp_location, suffix=".tif", delete=False
    ).name
    raise NotImplementedError


def dem_aoi(tmp_location: str, dem_foundation: str, aoi_shapefile: str) -> str:
    output_file = tempfile.NamedTemporaryFile(
        dir=tmp_location, suffix=".tif", delete=False
    ).name
    gdal.Warp(
        destNameOrDestDS=output_file,
        srcDSOrSrcDSTab=dem_foundation,
        cutlineDSName=aoi_shapefile,
        cropToCutline=True,
        copyMetadata=True,
        dstNodata=0,
    )
    return output_file

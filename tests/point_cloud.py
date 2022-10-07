import dataclasses
import json
import os
import tempfile
import warnings
from math import cos
from math import isclose
from math import sin
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import codem
import pdal
import pytest


class Translation(NamedTuple):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Rotation(NamedTuple):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def manipulate_pc(
    pc_aoi: str,
    *,
    rotation: Optional[Rotation] = None,
    translation: Optional[Translation] = None,
) -> str:

    if rotation is None and translation is None:
        warnings.warn(
            UserWarning(
                "manipulate_pc was called but with no rotation or "
                "translation option specified."
            )
        )

        return pc_aoi

    transforms = []
    if translation is not None:
        matrix = (
            f"1 0 0 {translation.x} 0 1 0 {translation.y} 0 0 1 {translation.z} 0 0 0 1"
        )
        transforms.append(matrix)
    if rotation is not None:
        if not isclose(rotation.x, 0):
            sinx = sin(rotation.x)
            cosx = cos(rotation.x)
            matrix = f"1 0 0 0 0 {cosx} {-sinx} 0 0 {sinx} {cosx} 0 0 0 0 1"
            transforms.append(matrix)
        if not isclose(rotation.y, 0):
            siny = sin(rotation.y)
            cosy = cos(rotation.y)
            matrix = f"{cosy} 0 {siny} 0 0 1 0 0 {-siny} 0 {cosy} 0 0 0 0 1"
            transforms.append(matrix)
        if not isclose(rotation.z, 0):
            sinz = sin(rotation.z)
            cosz = cos(rotation.z)
            matrix = f"{cosz} {-sinz} 0 0 {sinz} {cosz} 0 0 0 0 1 0 0 0 0 1"
            transforms.append(matrix)

    pipeline: List[Union[str, Dict[str, str]]] = [
        {"type": "filters.transformation", "matrix": matrix} for matrix in transforms
    ]
    pipeline.insert(0, pc_aoi)

    temp_location = os.path.dirname(pc_aoi)
    output_file = tempfile.NamedTemporaryFile(
        dir=temp_location, suffix=".laz", delete=False
    ).name
    pipeline.append({"type": "writers.las", "filename": output_file, "forward": "all"})
    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()
    return output_file


def pc_aoi(tmp_location: str, pc_foundation: str, aoi_shapefile: str) -> str:
    output_file = tempfile.NamedTemporaryFile(
        dir=tmp_location, suffix=".laz", delete=False
    ).name
    pipeline = pdal.Reader(pc_foundation)
    pipeline |= pdal.Filter.ferry(dimensions="=>AOIDimension")
    pipeline |= pdal.Filter.assign(assignment="AOIDimension[:]=1")
    pipeline |= pdal.Filter.overlay(
        dimension="AOIDimension", datasource=aoi_shapefile, where="AOIDimension == 1"
    )
    pipeline |= pdal.Filter.range(limits="AOIDimension[0:0]")
    pipeline |= pdal.Writer.las(filename=output_file, minor_version=4, forward="all")
    pipeline.execute()
    return output_file

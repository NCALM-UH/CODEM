import dataclasses
import json
import os
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


def translate_pc(
    pc_aoi: str,
    rotation: Optional[Rotation] = None,
    translation: Optional[Translation] = None,
) -> str:

    if rotation is None and translation is None:
        return pc_aoi

    output_file = os.path.abspath("tests/data/transformed.laz")
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
    pipeline.append({"type": "writers.las", "filename": output_file, "forward": "all"})
    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()
    return output_file


def registration(fnd_path: str, aoi_path: str) -> bool:
    codem_run_config = codem.CodemRunConfig(fnd_path, aoi_path)
    config = dataclasses.asdict(codem_run_config)

    assert os.path.realpath(fnd_path) == os.path.realpath(config["FND_FILE"])
    assert os.path.realpath(aoi_path) == os.path.realpath(config["AOI_FILE"])

    fnd_obj, aoi_obj = codem.preprocess(config)

    assert not fnd_obj.processed
    assert not aoi_obj.processed

    # make sure I can't set to a negative value
    with pytest.raises(ValueError):
        fnd_obj.resolution = -1

    # make sure I can't set to a negative non-finite value
    with pytest.raises(ValueError):
        aoi_obj.resolution = float("-inf")

    fnd_obj.prep()
    aoi_obj.prep()

    assert fnd_obj.processed
    assert aoi_obj.processed

    # perform dsm registration
    dsm_reg = codem.coarse_registration(fnd_obj, aoi_obj, config)

    # perform fine registration
    icp_reg = codem.fine_registration(fnd_obj, aoi_obj, dsm_reg, config)

    # apply registration
    reg_file = codem.apply_registration(fnd_obj, aoi_obj, icp_reg, config)

    assert os.path.exists(reg_file)
    return True


def test_pc_registration(pc_foundation: str, pc_aoi: str) -> None:

    # test registration where AOI should be right in-place
    assert registration(pc_foundation, pc_aoi)

    # translate 1_000 units and try again
    shifted = Translation(x=1000.0, y=0, z=0)
    modified_aoi = translate_pc(pc_aoi, translation=shifted)
    assert registration(pc_foundation, modified_aoi)

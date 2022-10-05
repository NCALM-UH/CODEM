import dataclasses
import itertools
import math
import os
import pathlib
import tempfile

import codem
import pytest
from point_cloud import manipulate_pc
from point_cloud import pc_aoi
from point_cloud import Rotation
from point_cloud import Translation
from raster import dem_aoi


aoi_shapefile = os.path.abspath("tests/data/aoi_shapefile/aoi.shp")
dem_foundation = os.path.abspath("tests/data/dem.tif")
pc_foundation = os.path.abspath("tests/data/pc.laz")
temporary_directory = os.path.abspath("tests/data/temporary")


def make_pc_aoi(aoi_temp_directory: str = temporary_directory) -> str:
    return pc_aoi(aoi_temp_directory, pc_foundation, aoi_shapefile)


def make_raster_aoi(aoi_temp_directory: str = temporary_directory) -> str:
    return dem_aoi(aoi_temp_directory, dem_foundation, aoi_shapefile)


pc_aoi_file = make_pc_aoi()
raster_aoi_file = make_raster_aoi()

pc_aoi_alterations = [
    pytest.param(pc_aoi_file, id="PC AOI Original"),
    pytest.param(
        manipulate_pc(pc_aoi_file, rotation=Rotation(z=2 * math.pi)),
        id="PC AOI Rotate 360 degrees",
    ),
    pytest.param(
        manipulate_pc(pc_aoi_file, translation=Translation(x=10.0)),
        id="PC AOI Translate x=10",
    ),
    pytest.param(
        manipulate_pc(pc_aoi_file, rotation=Rotation(z=math.pi)),
        id="PC AOI Rotate 180 degrees",
    ),
    pytest.param(
        manipulate_pc(
            pc_aoi_file,
            rotation=Rotation(z=math.pi / 2),
            translation=Translation(x=10_000.0, y=500.0),
        ),
        id="PC AOI Rotate 90 degrees and Translate x=10,000, y=500",
    ),
]

dem_aoi_alterations = [pytest.param(raster_aoi_file, id="DEM AOI Original")]


@pytest.mark.parametrize(
    "aoi", itertools.chain(pc_aoi_alterations, dem_aoi_alterations)
)
@pytest.mark.parametrize(
    "foundation",
    [
        pytest.param(pc_foundation, id="PC Foundation"),
        pytest.param(dem_foundation, id="DEM Foudnation"),
    ],
)
def test_registration(foundation: str, aoi: str, tmp_path: pathlib.Path) -> None:
    output_directory = tmp_path.resolve().as_posix()
    codem_run_config = codem.CodemRunConfig(
        foundation, aoi, OUTPUT_DIR=output_directory
    )
    config = dataclasses.asdict(codem_run_config)

    assert os.path.realpath(foundation) == os.path.realpath(config["FND_FILE"])
    assert os.path.realpath(aoi) == os.path.realpath(config["AOI_FILE"])

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

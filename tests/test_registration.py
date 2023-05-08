import dataclasses
import itertools
import math
import os
import pathlib

import codem
import numpy as np
import pytest
from osgeo import gdal
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
        pytest.param(dem_foundation, id="DEM Foundation"),
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


@pytest.mark.parametrize("foundation,compliment", [(dem_foundation, raster_aoi_file)])
def test_area_or_point(
    foundation: str, compliment: str, tmp_path: pathlib.Path
) -> None:
    """Rasters can have their elevation data represented as the position in the
    top-left corner (RasterPixelIsArea) or in the middle of the pixel
    (RasterPixelIsPoint)


    Pixel is Area:

       (0)    (1)
    (0)_|_______|
        |       |
        |       |
    (1)_|_______|

    Pixel is Point:

           (0)     (1)
         ___|___ ___|___
        |   |   |   |   |
    (0)-|---|---|---|---|
        |___|___|___|___|
            |       |

    Through the CODEM registration pipeline, data is shifted such that pixel is
    point is the default representation.


    # to change a raster:
    $ gdal_translate
    -mo AREA_OR_POINT=POINT
    --config GTIFF_POINT_GEO_IGNORE True
    input_pixel_is_area.tif output_pixel_is_point.tif
    """

    # compliment by default has AREA_OR_PIXEL=AREA
    compliment_base, extension = os.path.splitext(compliment)
    alternate = f"{compliment_base}_pixel_is_point{extension}"

    compliment_info = gdal.Info(compliment, format="json")
    compliment_pixel = compliment_info["metadata"][""]["AREA_OR_POINT"]
    # create the alternative compliment with the different setup
    alternate_pixel = "Point" if compliment_pixel.lower() == "area" else "Area"

    # need to set the config option accordingly
    gdal.SetConfigOption("GTIFF_POINT_GEO_IGNORE", "YES")
    gdal.Translate(
        alternate, compliment, options=f"-mo AREA_OR_POINT={alternate_pixel.upper()}"
    )
    gdal.SetConfigOption("GTIFF_POINT_GEO_IGNORE", "NO")

    alternate_info = gdal.Info(alternate, format="json")
    alternate_pixel = alternate_info["metadata"][""]["AREA_OR_POINT"]

    # make sure metadata actually is different
    assert compliment_pixel.lower() != alternate_pixel.lower()

    # compare bounds with gdal looking at corners
    assert compliment_info["cornerCoordinates"] != alternate_info["cornerCoordinates"]

    # now we run two registration processes alongside checking if both AOIs are equivalent...
    output_directory = tmp_path.resolve().as_posix()
    compliment_config = dataclasses.asdict(
        codem.CodemRunConfig(foundation, compliment, OUTPUT_DIR=output_directory)
    )
    alternate_config = dataclasses.asdict(
        codem.CodemRunConfig(foundation, alternate, OUTPUT_DIR=output_directory)
    )

    foundation_comp, compliment_obj = codem.preprocess(compliment_config)
    foundation_alt, alternate_obj = codem.preprocess(alternate_config)

    # resolutions should be the same
    assert math.isclose(compliment_obj.resolution, alternate_obj.resolution)

    # transforms should have different offsets
    assert compliment_obj.transform.xoff != alternate_obj.transform.xoff  # type: ignore
    assert compliment_obj.transform.yoff != alternate_obj.transform.yoff  # type: ignore

    # let's prep
    foundation_comp.prep()
    foundation_alt.prep()
    compliment_obj.prep()
    alternate_obj.prep()

    # make sure we actually have the correct attribute set
    assert compliment_obj.area_or_point.lower() != alternate_obj.area_or_point.lower()

    # GeoData.point_cloud compensates for this offset...
    assert np.allclose(compliment_obj.point_cloud, alternate_obj.point_cloud)

    dsm_reg_compliment = codem.coarse_registration(
        foundation_comp, compliment_obj, compliment_config
    )

    dsm_reg_alternate = codem.coarse_registration(
        foundation_alt, alternate_obj, alternate_config
    )

    # we should have the same number of putative matches
    assert len(dsm_reg_compliment.putative_matches) == len(
        dsm_reg_alternate.putative_matches
    )

    # ensure that coarse transformations are different by resolution
    if compliment_obj.area_or_point.lower() == "area":
        transform_to_offset = dsm_reg_compliment.transformation
        transform_to_fix = dsm_reg_alternate.transformation
    else:
        transform_to_offset = dsm_reg_alternate.transformation
        transform_to_fix = dsm_reg_compliment.transformation

    # resolution should be the same
    pos_offset = transform_to_offset[0:2, -1] + np.array(
        [compliment_obj.resolution, -compliment_obj.resolution]
    )
    assert np.allclose(pos_offset, transform_to_fix[0:2, -1])

    # perform fine registration
    icp_reg_compliment = codem.fine_registration(
        foundation_comp, compliment_obj, dsm_reg_compliment, compliment_config
    )
    icp_reg_alternate = codem.fine_registration(
        foundation_alt, alternate_obj, dsm_reg_alternate, alternate_config
    )

    # ensure that coarse transformations are different by resolution
    if compliment_obj.area_or_point.lower() == "area":
        transform_to_offset = icp_reg_compliment.transformation
        transform_to_fix = icp_reg_alternate.transformation
    else:
        transform_to_offset = icp_reg_alternate.transformation
        transform_to_fix = icp_reg_compliment.transformation

    pos_offset = transform_to_offset[0:2, -1] + np.array(
        [compliment_obj.resolution, -compliment_obj.resolution]
    )
    assert np.allclose(pos_offset, transform_to_fix[0:2, -1])

    # do the registration
    registered_compliment = codem.apply_registration(
        foundation_comp, compliment_obj, icp_reg_compliment, compliment_config
    )
    registered_alternate = codem.apply_registration(
        foundation_alt, alternate_obj, icp_reg_alternate, alternate_config
    )

    registered_compliment_info = gdal.Info(registered_compliment, format="json")
    registered_alternate_info = gdal.Info(registered_alternate, format="json")

    # bounds between the two datasets should be different
    assert (
        registered_compliment_info["cornerCoordinates"]
        != registered_alternate_info["cornerCoordinates"]
    )

    # metadata should be preserved between input and registered output
    assert (
        registered_compliment_info["metadata"][""]["AREA_OR_POINT"]
        == compliment_info["metadata"][""]["AREA_OR_POINT"]
    )
    assert (
        registered_alternate_info["metadata"][""]["AREA_OR_POINT"]
        == alternate_info["metadata"][""]["AREA_OR_POINT"]
    )

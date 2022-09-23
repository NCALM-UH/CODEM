import json
import os

import pdal
import pytest


@pytest.fixture(scope="session")
def tif_fnd() -> str:
    foundation_file = os.path.abspath("tests/data/0_smallfnd.tif")
    assert os.path.exists(foundation_file)
    return foundation_file


@pytest.fixture(scope="session")
def tif_aoi() -> str:
    aoi_file = os.path.abspath("tests/data/1_smallAOI.tif")
    assert os.path.exists(aoi_file)
    return aoi_file


@pytest.fixture(scope="session")
def dem_foundation() -> str:
    foundation = os.path.abspath("tests/data/dem.tif")
    return foundation


@pytest.fixture(scope="session")
def pc_foundation() -> str:
    foundation = os.path.abspath("tests/data/pc.laz")
    return foundation


# @pytest.fixture(scope="session")
# def dem_aoi(dem_foundation) -> str:
#     from osgeo import gdal
#     dataset = gdal.Open(dem_foundation)


@pytest.fixture(scope="session")
def pc_aoi(pc_foundation: str) -> str:

    output_file = os.path.abspath("tests/data/pc_aoi.laz")

    reader = pdal.Reader(type="readers.las", filename=pc_foundation)
    pipeline = reader.pipeline()
    info = pipeline.quickinfo
    bounds = info["readers.las"]["bounds"]

    ratio = 0.25  # closer to 0.5, smaller the AOI
    x_range = bounds["maxx"] - bounds["minx"]
    xmin = bounds["minx"] + ratio * x_range
    xmax = bounds["maxx"] - ratio * x_range

    y_range = bounds["maxy"] - bounds["miny"]
    ymin = bounds["miny"] + ratio * y_range
    ymax = bounds["maxy"] - ratio * y_range

    pipeline = [
        pc_foundation,
        {
            "type": "filters.crop",
            "bounds": f"([{xmin}, {xmax}], [{ymin}, {ymax}])",
        },
        {"type": "writers.las", "filename": output_file, "forward": "all"},
    ]
    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()
    return output_file

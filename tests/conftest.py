import os

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

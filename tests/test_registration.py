import dataclasses
import os
from typing import TYPE_CHECKING

import codem
import pytest

if TYPE_CHECKING:
    from codem.preprocessing.preprocess import GeoData


def test_basic_tif_registration(tif_fnd: GeoData, tif_aoi: GeoData) -> None:
    codem_run_config = codem.CodemRunConfig(tif_fnd, tif_aoi)
    config = dataclasses.asdict(codem_run_config)

    assert os.path.realpath(tif_fnd) == os.path.realpath(config["FND_FILE"])
    assert os.path.realpath(tif_aoi) == os.path.realpath(config["AOI_FILE"])

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

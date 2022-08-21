"""
main.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

The main script for running a two-step co-registration. A feature-based
global registration operating on DSMs is followed by a local ICP point
to plane registration. Logs for each registration run can be found in
the logs/ directory, and the outputs (text and image) can be found in
the relevant run directory within outputs/. 
"""
from distutils.util import strtobool
import os
import time
import yaml
import argparse
import pathlib
import enlighten
import logging
from modem.lib.log import Log
from modem.preprocessing.preprocess import instantiate
from modem.registration import DsmRegistration as D
from modem.registration import IcpRegistration as I
from modem.registration import ApplyRegistration as A


def str2bool(v):
    return bool(strtobool(v))


def get_args():
    ap = argparse.ArgumentParser(
        description="MODEM: Multi-Modal Digital Elevation Model Registration"
    )
    ap.add_argument(
        "foundation_file",
        type=pathlib.Path,
        help="path to the foundation file",
    )
    ap.add_argument(
        "aoi_file",
        type=pathlib.Path,
        help="path to the area of interest file",
    )
    ap.add_argument(
        "--min_resolution",
        "-min",
        type=float,
        default=1.0,
        help="minimum pipeline data resolution",
    )
    ap.add_argument(
        "--dsm_akaze_threshold",
        "-dat",
        type=float,
        default=0.0001,
        help="AKAZE feature detection response threshold",
    )
    ap.add_argument(
        "--dsm_lowes_ratio",
        "-dlr",
        type=float,
        default=0.9,
        help="feature matching relative strength control",
    )
    ap.add_argument(
        "--dsm_ransac_max_iter",
        "-drmi",
        type=int,
        default=10000,
        help="max iterations for the RANSAC algorithm",
    )
    ap.add_argument(
        "--dsm_ransac_threshold",
        "-drt",
        type=float,
        default=10,
        help="maximum residual error for a feature matched pair to be included in RANSAC solution",
    )
    ap.add_argument(
        "--dsm_solve_scale",
        "-dss",
        type=str2bool,
        default=True,
        help="boolean to include or exclude scale from the solved registration transformation",
    )
    ap.add_argument(
        "--dsm_strong_filter",
        "-dsf",
        type=float,
        default=10,
        help="stddev of the large Gaussian filter used to normalize DSM prior to feature extraction",
    )
    ap.add_argument(
        "--dsm_weak_filter",
        "-dwf",
        type=float,
        default=1,
        help="stddev of the small Gaussian filter used to normalize the DSM prior to feature extraction",
    )
    ap.add_argument(
        "--icp_angle_threshold",
        "-iat",
        type=float,
        default=0.001,
        help="minimum change in Euler angle between ICP iterations",
    )
    ap.add_argument(
        "--icp_distance_threshold",
        "-idt",
        type=float,
        default=0.001,
        help="minimum change in translation between ICP iterations",
    )
    ap.add_argument(
        "--icp_max_iter",
        "-imi",
        type=int,
        default=100,
        help="max iterations of the ICP algorithm",
    )
    ap.add_argument(
        "--icp_rmse_threshold",
        "-irt",
        type=float,
        default=0.0001,
        help="minimum relative change between iterations in the RMSE",
    )
    ap.add_argument(
        "--icp_robust",
        "-ir",
        type=str2bool,
        default=True,
        help="boolean to include or exclude robust weighting in registration solution",
    )
    ap.add_argument(
        "--icp_solve_scale",
        "-iss",
        type=str2bool,
        default=True,
        help="boolean to include or exclude scale from the solved registration",
    )
    ap.add_argument(
        "--verbose", "-v", type=str2bool, default=False, help="turn on verbose logging"
    )
    args = ap.parse_args()

    return args


def create_config(args):
    config = {}
    config["FND_FILE"] = str(args.foundation_file)
    config["AOI_FILE"] = str(args.aoi_file)
    config["MIN_RESOLUTION"] = float(args.min_resolution)
    config["DSM_AKAZE_THRESHOLD"] = float(args.dsm_akaze_threshold)
    config["DSM_LOWES_RATIO"] = float(args.dsm_lowes_ratio)
    config["DSM_RANSAC_MAX_ITER"] = int(args.dsm_ransac_max_iter)
    config["DSM_RANSAC_THRESHOLD"] = float(args.dsm_ransac_threshold)
    config["DSM_SOLVE_SCALE"] = args.dsm_solve_scale
    config["DSM_STRONG_FILTER_SIZE"] = float(args.dsm_strong_filter)
    config["DSM_WEAK_FILTER_SIZE"] = float(args.dsm_weak_filter)
    config["ICP_ANGLE_THRESHOLD"] = float(args.icp_angle_threshold)
    config["ICP_DISTANCE_THRESHOLD"] = float(args.icp_distance_threshold)
    config["ICP_MAX_ITER"] = int(args.icp_max_iter)
    config["ICP_RMSE_THRESHOLD"] = float(args.icp_rmse_threshold)
    config["ICP_ROBUST"] = args.icp_robust
    config["ICP_SOLVE_SCALE"] = args.icp_solve_scale
    config["VERBOSE"] = args.verbose
    config["ICP_SAVE_RESIDUALS"] = False

    current_time = time.localtime(time.time())
    timestamp = "%d-%02d-%02d_%02d-%02d-%02d" % (
        current_time.tm_year,
        current_time.tm_mon,
        current_time.tm_mday,
        current_time.tm_hour,
        current_time.tm_min,
        current_time.tm_sec,
    )
    output_dir = os.path.join(os.path.dirname(args.aoi_file), "registration_" + timestamp)
    os.mkdir(output_dir)
    config["OUTPUT_DIR"] = os.path.abspath(output_dir)

    config_path = os.path.join(config["OUTPUT_DIR"], "config.yml")
    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            explicit_start=True,
        )

    return config


def validate_config(config):
    assert os.path.exists(config["FND_FILE"]), "Invalid path to foundation data file"
    assert os.path.exists(config["AOI_FILE"]), "Invalid path to area of interest file"
    assert (
        isinstance(config["MIN_RESOLUTION"], float) and config["MIN_RESOLUTION"] > 0
    ), "Minimum pipeline resolution must be greater than 0"
    assert (
        isinstance(config["DSM_AKAZE_THRESHOLD"], float)
        and config["DSM_AKAZE_THRESHOLD"] > 0
    ), "AKAZE threshold must be greater than 0"
    assert (
        isinstance(config["DSM_LOWES_RATIO"], float)
        and config["DSM_LOWES_RATIO"] >= 0.01
        and config["DSM_LOWES_RATIO"] <= 1.0
    ), "Lowes ratio must be between 0.01 and 1.0"
    assert (
        isinstance(config["DSM_RANSAC_MAX_ITER"], int)
        and config["DSM_RANSAC_MAX_ITER"] > 0
    ), "Maximum number of RANSAC interations must be a positive integer"
    assert (
        isinstance(config["DSM_RANSAC_THRESHOLD"], float)
        and config["DSM_RANSAC_THRESHOLD"] > 0
    ), "RANSAC threshold must be a positive number"
    assert isinstance(
        config["DSM_SOLVE_SCALE"], bool
    ), "Flag for solving scale in DSM feature registration must be boolean"
    assert (
        isinstance(config["DSM_STRONG_FILTER_SIZE"], float)
        and config["DSM_STRONG_FILTER_SIZE"] > 0
    ), "DSM strong filter size must be greater than 0"
    assert (
        isinstance(config["DSM_WEAK_FILTER_SIZE"], float)
        and config["DSM_WEAK_FILTER_SIZE"] > 0
    ), "DSM weak filter size must be greater than 0"
    assert (
        isinstance(config["ICP_ANGLE_THRESHOLD"], float)
        and config["ICP_ANGLE_THRESHOLD"] > 0
    ), "ICP minimum angle convergence threshold must be greater than 0"
    assert (
        isinstance(config["ICP_DISTANCE_THRESHOLD"], float)
        and config["ICP_DISTANCE_THRESHOLD"] > 0
    ), "ICP minimum distance convergence threshold must be greater than 0"
    assert (
        isinstance(config["ICP_MAX_ITER"], int) and config["ICP_MAX_ITER"] > 0
    ), "Maximum number of ICP interations must be a positive integer"
    assert (
        isinstance(config["ICP_RMSE_THRESHOLD"], float)
        and config["ICP_RMSE_THRESHOLD"] > 0
    ), "ICP minimum change in RMSE convergence threshold must be greater than 0"
    assert isinstance(
        config["ICP_ROBUST"], bool
    ), "Flag for attempting to make ICP robust to outliers must be boolean"
    assert isinstance(
        config["ICP_SOLVE_SCALE"], bool
    ), "Flag for solving scale in ICP registration must be boolean"
    assert isinstance(
        config["VERBOSE"], bool
    ), "Flag for verbose output must be boolean"
    assert isinstance(
        config["ICP_SAVE_RESIDUALS"], bool
    ), "Flag for exporting ICP residuals must be boolean"
    assert os.path.exists(
        config["OUTPUT_DIR"]
    ), "An invalid path for algorithm output was created"


def run(config):
    """
    Preprocesses and registers the provided data

    Parameters
    ----------
    config: dict
        Dictionary of configuration parameters
    """
    logger = logging.getLogger(__name__)

    manager = enlighten.get_manager()
    status = manager.status_bar(
        status_format="MODEM{fill}Stage: {stage}{fill}{elapsed}",
        color="bold_underline_bold_bright_blue",
        justify=enlighten.Justify.CENTER,
        stage="Initializing",
    )
    run_bar = manager.counter(
        count=0,
        total=100,
        desc="Registration Process",
        color="bright_blue",
        justify=enlighten.Justify.CENTER,
        position=1,
    )
    run_bar.refresh()
    status.refresh()

    print("╔════════════════════════════════════╗")
    print("║               MODEM                ║")
    print("╚════════════════════════════════════╝")
    print("║     AUTHORS: Preston Hartzell  &   ║")
    print("║     Jesse Shanahan                 ║")
    print("║     DEVELOPED FOR: CRREL/NEGGS     ║")
    print("╚════════════════════════════════════╝")
    print()
    print("══════════════PARAMETERS══════════════")
    keys = list(config.keys())
    for key in keys:
        logger.info(f"{key} = {config[key]}")
    run_bar.update(1, force=True)

    print("══════════PREPROCESSING DATA══════════")
    status.update(stage="Preprocessing Inputs", force=True)
    fnd_obj = instantiate(config, fnd=True)
    aoi_obj = instantiate(config, fnd=False)
    resolution = max(
        fnd_obj.native_resolution,
        aoi_obj.native_resolution,
        config["MIN_RESOLUTION"],
    )
    fnd_obj.resolution = resolution
    aoi_obj.resolution = resolution
    run_bar.update(7, force=True)
    fnd_obj.prep()
    run_bar.update(45)
    aoi_obj.prep()
    run_bar.update(4, force=True)
    logger.info(f"Registration resolution has been set to: {fnd_obj.resolution} meters")

    print("═════BEGINNING COARSE REGISTRATION═════")
    status.update(stage="Performing Coarse Registration", force=True)
    dsm_reg = D.DsmRegistration(fnd_obj, aoi_obj, config)
    dsm_reg.register()
    run_bar.update(22)

    print("══════BEGINNING FINE REGISTRATION══════")
    status.update(stage="Performing Fine Registration", force=True)
    icp_reg = I.IcpRegistration(fnd_obj, aoi_obj, dsm_reg, config)
    icp_reg.register()
    run_bar.update(16)

    print("═════════APPLYING REGISTRATION═════════")
    app_reg = A.ApplyRegistration(
        fnd_obj,
        aoi_obj,
        icp_reg.registration_parameters,
        icp_reg.residual_vectors,
        icp_reg.residual_origins,
        config,
    )
    app_reg.apply()
    run_bar.update(5, force=True)


def main():
    args = get_args()
    config = create_config(args)
    validate_config(config)
    Log(config)
    run(config)


if __name__ == "__main__":
    main()

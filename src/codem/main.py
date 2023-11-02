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
import argparse
import dataclasses
import math
import os
import time
import warnings
from contextlib import ContextDecorator
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import yaml
from codem import __version__
from codem.lib.log import Log
from codem.preprocessing.preprocess import clip_data
from codem.preprocessing.preprocess import CodemParameters
from codem.preprocessing.preprocess import GeoData
from codem.preprocessing.preprocess import instantiate
from codem.registration import ApplyRegistration
from codem.registration import DsmRegistration
from codem.registration import IcpRegistration
from distutils.util import strtobool


class DummyProgress(ContextDecorator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def __enter__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_task(self, *args: Any, **kwargs: Any) -> None:
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def get_default_columns() -> List:
        return []


@dataclasses.dataclass
class CodemRunConfig:
    FND_FILE: str
    AOI_FILE: str
    MIN_RESOLUTION: float = float("nan")
    DSM_AKAZE_THRESHOLD: float = 0.0001
    DSM_LOWES_RATIO: float = 0.9
    DSM_RANSAC_MAX_ITER: int = 10000
    DSM_RANSAC_THRESHOLD: float = 10.0
    DSM_SOLVE_SCALE: bool = True
    DSM_STRONG_FILTER: float = 10.0
    DSM_WEAK_FILTER: float = 1.0
    ICP_ANGLE_THRESHOLD: float = 0.001
    ICP_DISTANCE_THRESHOLD: float = 0.001
    ICP_MAX_ITER: int = 100
    ICP_RMSE_THRESHOLD: float = 0.0001
    ICP_ROBUST: bool = True
    ICP_SOLVE_SCALE: bool = True
    VERBOSE: bool = False
    ICP_SAVE_RESIDUALS: bool = False
    OUTPUT_DIR: Optional[str] = None
    TIGHT_SEARCH: bool = False
    LOG_TYPE: str = "rich"
    WEBSOCKET_URL: str = "127.0.0.1:8889"

    def __post_init__(self) -> None:
        # set output directory
        if self.OUTPUT_DIR is None:
            current_time = time.localtime(time.time())
            timestamp = "%d-%02d-%02d_%02d-%02d-%02d" % (
                current_time.tm_year,
                current_time.tm_mon,
                current_time.tm_mday,
                current_time.tm_hour,
                current_time.tm_min,
                current_time.tm_sec,
            )

            output_dir = os.path.join(
                os.path.dirname(self.AOI_FILE), f"registration_{timestamp}"
            )
            os.mkdir(output_dir)
            self.OUTPUT_DIR = os.path.abspath(output_dir)

        # validate attributes
        if not os.path.exists(self.FND_FILE):
            raise FileNotFoundError(f"Foundation file {self.FND_FILE} not found.")
        if not os.path.exists(self.AOI_FILE):
            raise FileNotFoundError(f"AOI file {self.AOI_FILE} not found.")
        if self.MIN_RESOLUTION <= 0:
            raise ValueError("Minimum pipeline resolution must be a greater than 0.")
        if self.DSM_AKAZE_THRESHOLD <= 0:
            raise ValueError("Minmum AKAZE threshold must be greater than 0.")
        if self.DSM_LOWES_RATIO < 0.01 or self.DSM_LOWES_RATIO >= 1.0:
            raise ValueError("Lowes ratio must be between 0.01 and 1.0.")
        if self.DSM_RANSAC_MAX_ITER < 1:
            raise ValueError(
                "Maximum number of RANSAC iterations must be a positive integer."
            )
        if self.DSM_RANSAC_THRESHOLD <= 0:
            raise ValueError("RANSAC threshold must be a positive number.")
        if self.DSM_STRONG_FILTER <= 0:
            raise ValueError("DSM strong filter size must be greater than 0.")
        if self.DSM_WEAK_FILTER <= 0:
            raise ValueError("DSM weak filter size must be greater than 0.")
        if self.ICP_ANGLE_THRESHOLD <= 0:
            raise ValueError(
                "ICP minimum angle convergence threshold must be greater than 0."
            )
        if self.ICP_DISTANCE_THRESHOLD <= 0:
            raise ValueError(
                "ICP minimum distance convergence threshold must be greater than 0."
            )
        if self.ICP_MAX_ITER < 1:
            raise ValueError(
                "Maximum number of ICP iterations must be a positive integer."
            )
        if self.ICP_RMSE_THRESHOLD <= 0:
            raise ValueError(
                "ICP minimum change in RMSE convergence threshold must be greater than 0."
            )

        # dump config
        config_path = os.path.join(self.OUTPUT_DIR, "config.yml")
        with open(config_path, "w") as f:
            yaml.safe_dump(
                dataclasses.asdict(self),
                f,
                default_flow_style=False,
                sort_keys=False,
                explicit_start=True,
            )
        return None


def str2bool(v: str) -> bool:
    return bool(strtobool(v))


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CODEM: Multi-Modal Digital Elevation Model Registration"
    )
    ap.add_argument(
        "foundation_file",
        type=str,
        help="path to the foundation file",
    )
    ap.add_argument(
        "aoi_file",
        type=str,
        help="path to the area of interest file",
    )
    ap.add_argument(
        "--min-resolution",
        "-min",
        type=float,
        default=CodemRunConfig.MIN_RESOLUTION,
        help="minimum pipeline data resolution",
    )
    ap.add_argument(
        "--dsm-akaze-threshold",
        "-dat",
        type=float,
        default=0.0001,
        help="AKAZE feature detection response threshold",
    )
    ap.add_argument(
        "--dsm-lowes-ratio",
        "-dlr",
        type=float,
        default=0.9,
        help="feature matching relative strength control",
    )
    ap.add_argument(
        "--dsm-ransac-max-iter",
        "-drmi",
        type=int,
        default=10000,
        help="max iterations for the RANSAC algorithm",
    )
    ap.add_argument(
        "--dsm-ransac-threshold",
        "-drt",
        type=float,
        default=10,
        help="maximum residual error for a feature matched pair to be included in RANSAC solution",
    )
    ap.add_argument(
        "--dsm-solve-scale",
        "-dss",
        type=str2bool,
        default=True,
        help="boolean to include or exclude scale from the solved registration transformation",
    )
    ap.add_argument(
        "--dsm-strong-filter",
        "-dsf",
        type=float,
        default=10,
        help="stddev of the large Gaussian filter used to normalize DSM prior to feature extraction",
    )
    ap.add_argument(
        "--dsm-weak-filter",
        "-dwf",
        type=float,
        default=1,
        help="stddev of the small Gaussian filter used to normalize the DSM prior to feature extraction",
    )
    ap.add_argument(
        "--icp-angle-threshold",
        "-iat",
        type=float,
        default=0.001,
        help="minimum change in Euler angle between ICP iterations",
    )
    ap.add_argument(
        "--icp-distance-threshold",
        "-idt",
        type=float,
        default=0.001,
        help="minimum change in translation between ICP iterations",
    )
    ap.add_argument(
        "--icp-max-iter",
        "-imi",
        type=int,
        default=100,
        help="max iterations of the ICP algorithm",
    )
    ap.add_argument(
        "--icp-rmse-threshold",
        "-irt",
        type=float,
        default=0.0001,
        help="minimum relative change between iterations in the RMSE",
    )
    ap.add_argument(
        "--icp-robust",
        "-ir",
        type=str2bool,
        default=True,
        help="boolean to include or exclude robust weighting in registration solution",
    )
    ap.add_argument(
        "--icp-solve-scale",
        "-iss",
        type=str2bool,
        default=True,
        help="boolean to include or exclude scale from the solved registration",
    )
    ap.add_argument(
        "--icp-save-residuals",
        action="store_true",
        help="Write ICP residual information",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="turn on verbose logging"
    )
    ap.add_argument(
        "--tight-search",
        "-ts",
        action="store_true",
        help=(
            "Limits the registration search to the region of overlap. Both datasets "
            "must have the same CRS defined."
        ),
    )
    ap.add_argument(
        "--output-dir", "-o", type=str, help="Directory to place registered output."
    )
    ap.add_argument(
        "--version",
        action="version",
        version=f"{__version__}",
        help="Display codem version information",
    )
    ap.add_argument(
        "--log-type",
        "-l",
        type=str,
        default=CodemRunConfig.LOG_TYPE,
        help="Specify how to log codem output, options include websockets, rich or console",
    )
    ap.add_argument(
        "--websocket-url",
        type=str,
        default=CodemRunConfig.WEBSOCKET_URL,
        help="Url to websocket receiver to connect to",
    )
    return ap.parse_args()


def create_config(args: argparse.Namespace) -> CodemParameters:
    config = CodemRunConfig(
        os.fsdecode(os.path.abspath(args.foundation_file)),
        os.fsdecode(os.path.abspath(args.aoi_file)),
        MIN_RESOLUTION=float(args.min_resolution),
        DSM_AKAZE_THRESHOLD=float(args.dsm_akaze_threshold),
        DSM_LOWES_RATIO=float(args.dsm_lowes_ratio),
        DSM_RANSAC_MAX_ITER=int(args.dsm_ransac_max_iter),
        DSM_RANSAC_THRESHOLD=float(args.dsm_ransac_threshold),
        DSM_SOLVE_SCALE=args.dsm_solve_scale,
        DSM_STRONG_FILTER=float(args.dsm_strong_filter),
        DSM_WEAK_FILTER=float(args.dsm_weak_filter),
        ICP_ANGLE_THRESHOLD=float(args.icp_angle_threshold),
        ICP_DISTANCE_THRESHOLD=float(args.icp_distance_threshold),
        ICP_MAX_ITER=int(args.icp_max_iter),
        ICP_RMSE_THRESHOLD=float(args.icp_rmse_threshold),
        ICP_ROBUST=args.icp_robust,
        ICP_SOLVE_SCALE=args.icp_solve_scale,
        VERBOSE=args.verbose,
        ICP_SAVE_RESIDUALS=args.icp_save_residuals,
        TIGHT_SEARCH=args.tight_search,
        OUTPUT_DIR=args.output_dir,
        LOG_TYPE=args.log_type,
        WEBSOCKET_URL=args.websocket_url
    )
    config_dict = dataclasses.asdict(config)
    log = Log(config_dict)
    config_dict["log"] = log
    return config_dict  # type: ignore


def run_rich_console(
    config: CodemParameters,
) -> None:
    """
    Preprocess and register the provided data

    Parameters
    ----------
    config
        Dictionary of configuration parameters
    """
    from rich.console import Console  # type: ignore
    from rich.progress import Progress  # type: ignore
    from rich.progress import SpinnerColumn  # type: ignore
    from rich.progress import TimeElapsedColumn  # type: ignore

    console = Console()
    logger = config["log"].logger

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        registration = progress.add_task("Registration...", total=100)

        console.print("/************************************\\", justify="center")
        console.print("*               CODEM                *", justify="center")
        console.print("**************************************", justify="center")
        console.print("*     AUTHORS: Preston Hartzell  &   *", justify="center")
        console.print("*     Jesse Shanahan                 *", justify="center")
        console.print("*     DEVELOPED FOR: CRREL/NEGGS     *", justify="center")
        console.print("\\************************************/", justify="center")
        console.print()
        console.print("===========PARAMETERS===========", justify="center")
        for key, value in config.items():
            logger.info(f"{key} = {value}")
        progress.advance(registration, 1)

        console.print("===========PREPROCESSING DATA===========", justify="center")
        fnd_obj, aoi_obj = preprocess(config)
        clip_data(fnd_obj, aoi_obj, config)
        progress.advance(registration, 7)
        fnd_obj.prep()
        progress.advance(registration, 45)
        aoi_obj.prep()
        progress.advance(registration, 4)
        logger.info(
            f"Registration resolution has been set to: {fnd_obj.resolution} meters"
        )

        console.print(
            "===========BEGINNING COARSE REGISTRATION===========", justify="center"
        )
        dsm_reg = coarse_registration(fnd_obj, aoi_obj, config)
        progress.advance(registration, 22)

        console.print(
            "===========BEGINNING FINE REGISTRATION===========", justify="center"
        )
        icp_reg = fine_registration(fnd_obj, aoi_obj, dsm_reg, config)
        progress.advance(registration, 16)

        console.print("===========APPLYING REGISTRATION===========", justify="center")
        apply_registration(fnd_obj, aoi_obj, icp_reg, config)
        progress.advance(registration, 5)


def run_stdout_console(config: CodemParameters) -> None:
    """
    Preprocess and register the provided data

    Parameters
    ----------
    config: dict
        Dictionary of configuration parameters
    """

    # registration = progress.add_task("Registration...", total=100)

    logger = config["log"].logger

    print("/************************************\\")
    print("*               CODEM                *")
    print("**************************************")
    print("*     AUTHORS: Preston Hartzell  &   *")
    print("*     Jesse Shanahan                 *")
    print("*     DEVELOPED FOR: CRREL/NEGGS     *")
    print("\\************************************/")
    print()
    print("===========PARAMETERS===========")
    for key, value in config.items():
        logger.info(f"{key} = {value}")

    print("===========PREPROCESSING DATA===========")
    fnd_obj, aoi_obj = preprocess(config)
    clip_data(fnd_obj, aoi_obj, config)
    fnd_obj.prep()
    aoi_obj.prep()
    logger.info(f"Registration resolution has been set to: {fnd_obj.resolution} meters")

    print("===========BEGINNING COARSE REGISTRATION===========")
    dsm_reg = coarse_registration(fnd_obj, aoi_obj, config)

    print("===========BEGINNING FINE REGISTRATION===========")
    icp_reg = fine_registration(fnd_obj, aoi_obj, dsm_reg, config)

    print("===========APPLYING REGISTRATION===========")
    apply_registration(fnd_obj, aoi_obj, icp_reg, config)


def run_no_console(
    config: CodemParameters,
) -> None:
    """
    Preprocess and register the provided data

    Parameters
    ----------
    config: dict
        Dictionary of configuration parameters
    """

    from codem.lib.progress import WebSocketProgress

    logger = config["log"].logger

    with WebSocketProgress(config["WEBSOCKET_URL"]) as progress:
        registration = progress.add_task("Registration...", total=100)

        for key, value in config.items():
            logger.info(f"{key} = {value}")
        progress.advance(registration, 1)

        fnd_obj, aoi_obj = preprocess(config)
        clip_data(fnd_obj, aoi_obj, config)
        progress.advance(registration, 7)
        fnd_obj.prep()
        progress.advance(registration, 45)
        aoi_obj.prep()
        progress.advance(registration, 4)
        logger.info(
            f"Registration resolution has been set to: {fnd_obj.resolution} meters"
        )

        dsm_reg = coarse_registration(fnd_obj, aoi_obj, config)
        progress.advance(registration, 22)

        icp_reg = fine_registration(fnd_obj, aoi_obj, dsm_reg, config)
        progress.advance(registration, 16)

        apply_registration(fnd_obj, aoi_obj, icp_reg, config)
        progress.advance(registration, 5)


def preprocess(config: CodemParameters) -> Tuple[GeoData, GeoData]:
    fnd_obj = instantiate(config, fnd=True)
    aoi_obj = instantiate(config, fnd=False)
    if not math.isnan(config["MIN_RESOLUTION"]):
        resolution = config["MIN_RESOLUTION"]
        if resolution > max(fnd_obj.native_resolution, aoi_obj.native_resolution):
            warnings.warn(
                "Specified resolution is a coarser value in than either the "
                "foundation or AOI, registration may fail as a result. Consider "
                "leaving the min_resolution parameter to default value.",
                UserWarning,
                stacklevel=2,
            )
    else:
        resolution = max(fnd_obj.native_resolution, aoi_obj.native_resolution)
    fnd_obj.resolution = aoi_obj.resolution = resolution

    # create DSM, but if doing tight-search do not resample
    resample = not config["TIGHT_SEARCH"]
    fnd_obj._create_dsm(resample=resample)
    aoi_obj._create_dsm(resample=resample, fallback_crs=fnd_obj.crs)
    return fnd_obj, aoi_obj


def coarse_registration(
    fnd_obj: GeoData, aoi_obj: GeoData, config: CodemParameters
) -> DsmRegistration:
    dsm_reg = DsmRegistration(fnd_obj, aoi_obj, config)
    dsm_reg.register()
    return dsm_reg


def fine_registration(
    fnd_obj: GeoData,
    aoi_obj: GeoData,
    dsm_reg: DsmRegistration,
    config: CodemParameters,
) -> IcpRegistration:
    icp_reg = IcpRegistration(fnd_obj, aoi_obj, dsm_reg, config)
    icp_reg.register()
    return icp_reg


def apply_registration(
    fnd_obj: GeoData,
    aoi_obj: GeoData,
    icp_reg: IcpRegistration,
    config: CodemParameters,
    output_format: Optional[str] = None,
) -> str:
    app_reg = ApplyRegistration(
        fnd_obj,
        aoi_obj,
        icp_reg.registration_parameters,
        icp_reg.residual_vectors,
        icp_reg.residual_origins,
        config,
        output_format,
    )
    app_reg.apply()
    return app_reg.out_name


def main() -> None:
    args = get_args()
    config = create_config(args)

    if config["LOG_TYPE"] == "rich":
        run_rich_console(config)
    elif config["LOG_TYPE"] == "websocket":
        run_no_console(config)
        config["log"].logger.info("run no console has finished")
    else:
        run_stdout_console(config)


if __name__ == "__main__":
    main()

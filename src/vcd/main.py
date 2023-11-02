"""
main.py
"""
import argparse
import dataclasses
import os
import time
from typing import Tuple

import yaml
from codem import __version__
from codem.lib.log import Log
from distutils.util import strtobool
from vcd.meshing.mesh import Mesh
from vcd.preprocessing.preprocess import PointCloud
from vcd.preprocessing.preprocess import VCD
from vcd.preprocessing.preprocess import VCDParameters


@dataclasses.dataclass
class VcdRunConfig:
    BEFORE: str
    AFTER: str
    SPACING: float = 0.43
    GROUNDHEIGHT: float = 1.0
    RESOLUTION: float = 2.0
    VERBOSE: bool = False
    MIN_POINTS: int = 30
    CLUSTER_TOLERANCE: float = 2.0
    CULL_CLUSTER_IDS: Tuple[int, ...] = (-1, 0)
    CLASS_LABELS: Tuple[int, ...] = (2, 6)
    OUTPUT_DIR: str = "."
    COLORMAP: str = "RdBu"
    TRUST_LABELS: bool = False
    COMPUTE_HAG: bool = False
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

            output_dir = os.path.join(os.path.dirname(self.AFTER), f"vcd_{timestamp}")
            os.mkdir(output_dir)
            self.OUTPUT_DIR = os.path.abspath(output_dir)

        # validate attributes
        if not os.path.exists(self.BEFORE):
            raise FileNotFoundError(f"Before file {self.BEFORE} not found.")
        if not os.path.exists(self.AFTER):
            raise FileNotFoundError(f"After file {self.AFTER} not found.")

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
        description="CODEM-VCD: LiDAR Vertical Change Detection"
    )
    ap.add_argument(
        "before",
        type=str,
        help="Before LiDAR scan",
    )
    ap.add_argument(
        "after",
        type=str,
        help="After LiDAR scan",
    )
    ap.add_argument(
        "--spacing-override",
        type=float,
        default=VcdRunConfig.SPACING,
        help="Use specified spacing instead of computing from data",
    )
    ap.add_argument(
        "--ground-height",
        type=float,
        default=VcdRunConfig.GROUNDHEIGHT,
        help="Ground filtering height",
    )
    ap.add_argument(
        "--resolution",
        type=float,
        default=VcdRunConfig.RESOLUTION,
        help="Raster output resolution",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=VcdRunConfig.MIN_POINTS,
        help="Minimum points to cluster around",
    )
    ap.add_argument(
        "--cluster-tolerance",
        type=float,
        default=VcdRunConfig.CLUSTER_TOLERANCE,
        help="Cluster tolerance used by pdal.Filter.cluster",
    )
    ap.add_argument(
        "--cull-cluster-ids",
        type=str,
        default=",".join(map(str, VcdRunConfig.CULL_CLUSTER_IDS)),
        help="Comma separated list of cluster IDs to cull when producing the meshes",
    )
    ap.add_argument(
        "--class-labels",
        type=str,
        default=",".join(map(str, VcdRunConfig.CLASS_LABELS)),
        help="Comma separated list of classification labels to use when producing the meshes",
    )
    ap.add_argument(
        "-v", "--verbose", action="count", default=0, help="turn on verbose logging"
    )
    ap.add_argument(
        "--colormap",
        type=str,
        default=VcdRunConfig.COLORMAP,
        help=(
            "Colormap to apply to generated output files where supported.  Name has "
            "to align with a matplotlib named colormap.  See "
            "https://matplotlib.org/stable/tutorials/colors/colormaps.html#diverging "
            "for list of options."
        ),
    )
    ap.add_argument(
        "--trust-labels",
        action="store_true",
        help=(
            "Trusts existing classification labels in the removal of vegetation/noise, "
            "otherwise return information is used to approximate vegetation/noise "
            "detection."
        ),
    )
    ap.add_argument(
        "--compute-hag",
        action="store_true",
        help=(
            "Compute height above ground between after scan (non-ground) and before "
            "scan (ground), otherwise compute to nearest neighbor from after to before."
        ),
    )
    ap.add_argument(
        "--output-dir", "-o", type=str, help="Directory to place VCD output"
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
        default=VcdRunConfig.LOG_TYPE,
        help="Specify how to log codem output, options include websocket, rich or console",
    )
    ap.add_argument(
        "--websocket-url",
        type=str,
        default=VcdRunConfig.WEBSOCKET_URL,
        help="Url to websocket receiver to connect to"
    )
    return ap.parse_args()


def create_config(args: argparse.Namespace) -> VCDParameters:
    config = VcdRunConfig(
        os.fsdecode(os.path.abspath(args.before)),
        os.fsdecode(os.path.abspath(args.after)),
        SPACING=float(args.spacing_override),
        VERBOSE=args.verbose,
        GROUNDHEIGHT=float(args.ground_height),
        RESOLUTION=float(args.resolution),
        MIN_POINTS=int(args.min_points),
        CLUSTER_TOLERANCE=float(args.cluster_tolerance),
        CULL_CLUSTER_IDS=tuple(map(int, args.cull_cluster_ids.split(","))),
        CLASS_LABELS=tuple(map(int, args.class_labels.split(","))),
        TRUST_LABELS=args.trust_labels,
        COMPUTE_HAG=args.compute_hag,
        OUTPUT_DIR=args.output_dir,
        LOG_TYPE=args.log_type,
        WEBSOCKET_URL=args.websocket_url
    )
    config_dict = dataclasses.asdict(config)
    log = Log(config_dict)
    config_dict["log"] = log
    return config_dict  # type: ignore


def run_stdout_console(config: VCDParameters) -> None:
    print("/************************************\\")
    print("*               VCD                  *")
    print("**************************************")
    print("*     AUTHORS: Brad Chambers &       *")
    print("*     Howard Butler                  *")
    print("*     DEVELOPED FOR: CRREL/NEGGS     *")
    print("\\************************************/")
    print()
    print("==============PARAMETERS==============")

    logger = config["log"].logger
    for key, value in config.items():
        logger.info(f"{key} = {value}")
    before = PointCloud(config, "BEFORE")
    after = PointCloud(config, "AFTER")
    v = VCD(before, after)
    v.compute_indexes()
    v.make_products()
    v.cluster()
    v.rasterize()
    m = Mesh(v)
    m.write("cluster", m.cluster(v.clusters))
    v.save()


def run_no_console(config: VCDParameters) -> None:
    from codem.lib.progress import WebSocketProgress

    logger = config["log"].logger
    

    with WebSocketProgress(config["WEBSOCKET_URL"]) as progress:
        change_detection = progress.add_task("Vertical Change Detection...", total=100)
        
        for key, value in config.items():
            logger.info(f"{key} = {value}")
            
        before = PointCloud(config, "BEFORE")
        progress.advance(change_detection, 14)

        after = PointCloud(config, "AFTER")
        progress.advance(change_detection, 15)

        v = VCD(before, after)
        progress.advance(change_detection, 15)

        v.compute_indexes()
        progress.advance(change_detection, 15)

        v.make_products()
        progress.advance(change_detection, 15)

        v.cluster()
        progress.advance(change_detection, 15)

        v.rasterize()
        progress.advance(change_detection, 15)

        m = Mesh(v)
        m.write("cluster", m.cluster(v.clusters))
        v.save()

        progress.advance(change_detection, 10)


def run_rich_console(config: VCDParameters) -> None:
    """
    Preprocess and register the provided data

    Parameters
    ----------
    config: dict
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
        change_detection = progress.add_task("Vertical Change Detection...", total=100)

        # characters are problematic on a windows console
        console.print("/************************************\\", justify="center")
        console.print("*               VCD                  *", justify="center")
        console.print("**************************************", justify="center")
        console.print("*     AUTHORS: Brad Chambers &       *", justify="center")
        console.print("*     Howard Butler                  *", justify="center")
        console.print("*     DEVELOPED FOR: CRREL/NEGGS     *", justify="center")
        console.print("\\************************************/", justify="center")
        console.print()
        console.print("==============PARAMETERS==============", justify="center")
        for key, value in config.items():
            logger.info(f"{key} = {value}")
        progress.advance(change_detection, 1)

        console.print("==========PREPROCESSING DATA==========", justify="center")
        console.print("==========Filtering 'before' data ====", justify="center")
        before = PointCloud(config, "BEFORE")
        progress.advance(change_detection, 14)
        console.print("==========Filtering 'after' data =====", justify="center")
        after = PointCloud(config, "AFTER")
        progress.advance(change_detection, 15)
        console.print(
            "==========Computing indexes for comparison =====", justify="center"
        )
        v = VCD(before, after)
        v.compute_indexes()
        progress.advance(change_detection, 15)
        console.print("========== Extracting differences ", justify="center")
        v.make_products()
        progress.advance(change_detection, 15)
        console.print("========== Clustering ", justify="center")
        v.cluster()
        progress.advance(change_detection, 15)
        console.print("========== Rasterizing products ", justify="center")
        v.rasterize()
        progress.advance(change_detection, 15)
        console.print("========== Meshing products ", justify="center")

        m = Mesh(v)
        m.write("cluster", m.cluster(v.clusters))

        v.save()
        progress.advance(change_detection, 10)


def main() -> None:
    args = get_args()
    config = create_config(args)
    if config["LOG_TYPE"] == "rich":
        run_rich_console(config)
    elif config["LOG_TYPE"] == "websocket":
        run_no_console(config)
    else:
        run_stdout_console(config)  # type: ignore
    return None


if __name__ == "__main__":
    main()

"""
main.py
"""
import argparse
import dataclasses
import logging
import os
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import yaml
from codem.lib.log import Log
from distutils.util import strtobool
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TimeElapsedColumn
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
    CULL_CLUSTER_IDS: Tuple[int, ...] = (-1, 0, 1)
    OUTPUT_DIR: Optional[str] = None

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
        "--min_points",
        type=int,
        default=VcdRunConfig.MIN_POINTS,
        help="Minimum points to cluster around",
    )
    ap.add_argument(
        "--cluster_tolerance",
        type=float,
        default=VcdRunConfig.CLUSTER_TOLERANCE,
        help="Cluster tolerance used by pdal.Filter.cluster",
    )
    ap.add_argument(
        "--cull_cluster_ids",
        type=str,
        default=",".join(map(str, VcdRunConfig.CULL_CLUSTER_IDS)),
        help="Coma separated list of cluster IDs to cull when producing the meshes",
    )
    ap.add_argument(
        "-v", "--verbose", action="count", default=0, help="turn on verbose logging"
    )

    args = ap.parse_args()
    return args


def create_config(args: argparse.Namespace) -> Dict[str, Any]:
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
    )
    return dataclasses.asdict(config)


def run_console(
    config: VCDParameters, logger: logging.Logger, console: Console
) -> None:
    """
    Preprocess and register the provided data

    Parameters
    ----------
    config: dict
        Dictionary of configuration parameters
    """

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        registration = progress.add_task("Vertical Change Detection...", total=100)

        # characters are problematic on a windows console
        console.print("╔════════════════════════════════════╗", justify="center")
        console.print("║               VCD                  ║", justify="center")
        console.print("╚════════════════════════════════════╝", justify="center")
        console.print("║     AUTHORS: Brad Chambers &       ║", justify="center")
        console.print("║     Howard Butler                  ║", justify="center")
        console.print("║     DEVELOPED FOR: CRREL/NEGGS     ║", justify="center")
        console.print("╚════════════════════════════════════╝", justify="center")
        console.print()
        console.print("══════════════PARAMETERS══════════════", justify="center")
        for key in config:
            logger.info(f"{key} = {config[key]}")  # type: ignore
        progress.advance(registration, 1)

        console.print("══════════PREPROCESSING DATA══════════", justify="center")
        console.print("══════════Filtering 'before' data ====", justify="center")
        before = PointCloud(config, "BEFORE")
        progress.advance(registration, 14)
        console.print("══════════Filtering 'after' data =====", justify="center")
        after = PointCloud(config, "AFTER")
        progress.advance(registration, 15)
        console.print(
            "══════════Computing indexes for comparison =====", justify="center"
        )
        v = VCD(before, after)
        v.compute_indexes()
        progress.advance(registration, 15)
        console.print("══════════ Extracting differences ", justify="center")
        v.make_products()
        progress.advance(registration, 15)
        console.print("══════════ Clustering ", justify="center")
        v.cluster()
        progress.advance(registration, 15)
        console.print("══════════ Rasterizing products ", justify="center")
        v.rasterize()
        progress.advance(registration, 15)
        console.print("══════════ Meshing products ", justify="center")

        m = Mesh(v)
        m.write("non-ground", m.cluster(v.ng_clusters))
        m.write("ground", m.cluster(v.ground_clusters))

        v.save()
        progress.advance(registration, 10)


def main() -> None:
    args = get_args()
    config = create_config(args)
    console = Console()
    rich_handler = RichHandler(level="DEBUG", console=console, markup=False)
    codem_logger = Log(config)
    codem_logger.logger.addHandler(rich_handler)
    run_console(config, codem_logger.logger, console)  # type: ignore


if __name__ == "__main__":
    main()

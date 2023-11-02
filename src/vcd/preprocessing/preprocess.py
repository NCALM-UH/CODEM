"""
preprocess.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

"""
import contextlib
import os
import re
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import pdal
from codem import __version__
from codem.lib.log import Log
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info  # type: ignore
from pyproj.transformer import TransformerGroup
from scipy.spatial import cKDTree
from typing_extensions import TypedDict


class VCDParameters(TypedDict):
    GROUNDHEIGHT: np.ndarray
    RESOLUTION: np.float64
    OUTPUT_DIR: str
    BEFORE: str
    AFTER: str
    MIN_POINTS: int
    CLUSTER_TOLERANCE: float
    CULL_CLUSTER_IDS: Tuple[int, ...]
    CLASS_LABELS: Tuple[int, ...]
    COLORMAP: str
    TRUST_LABELS: bool
    COMPUTE_HAG: bool
    LOG_TYPE: str
    WEBSOCKET_URL: str
    log: Log


class Product(NamedTuple):
    df: pd.DataFrame
    z_name: str
    description: str = ""

    @property
    def slug(self) -> str:
        """Adapted from https://stackoverflow.com/a/8366771/498396"""
        return re.sub(r"[^0-9A-Za-z.]", "-", self.description.lower())


def get_json(filename: str) -> str:
    with contextlib.suppress(IOError):
        with open(filename, "r") as f:
            content = f.read()
    return content


class PointCloud:
    """
    A class for storing and preparing geospatial data

    Parameters
    ----------
    config: VCDParameters
        Dictionary of configuration options
    fnd: bool
        Whether the file is foundation data

    """

    def __init__(self, config: VCDParameters, key: str) -> None:
        self.logger = config["log"]
        if key in {"BEFORE", "AFTER"}:
            # see https://github.com/python/mypy/issues/7178
            self.filename = config[key]  # type: ignore
        else:
            raise ValueError
        self.config = config
        self.crs: Optional[CRS] = None
        self.utm = ""
        self.pipeline = self.open()

        if len(self.pipeline.arrays) > 1:
            raise NotImplementedError("VCD between multiple views is not supported")
        self.df = pd.DataFrame(self.pipeline.arrays[0])

        # drop the color information if it is present
        self.df = self.df.drop(columns=["Red", "Green", "Blue"], errors="ignore")

    def open(self) -> pdal.Pipeline:
        def _get_utm(pipeline: pdal.Pipeline) -> pdal.Pipeline:
            data = pipeline.quickinfo
            is_reader = [[k.split(".")[0] == "readers", k] for k in data.keys()]
            for k in is_reader:
                if k[0]:  # we are a reader
                    reader_info = data[k[1]]
                    bounds = reader_info["bounds"]
                    srs = CRS.from_user_input(reader_info["srs"]["compoundwkt"])

                    # we just take the first one. If there's more we are screwed
                    break

            tg = TransformerGroup(srs, 4326)
            for transformer in tg.transformers:
                dd = transformer.transform(
                    (bounds["minx"], bounds["maxx"]), (bounds["miny"], bounds["maxy"])
                )

                # stolen from Alan https://gis.stackexchange.com/a/423614/350
                # dd now in the form ((41.469221251843926, 41.47258675464548), (-93.68979255724548, -93.68530098082489))
                aoi = AreaOfInterest(
                    west_lon_degree=dd[1][0],
                    south_lat_degree=dd[0][0],
                    east_lon_degree=dd[1][1],
                    north_lat_degree=dd[0][1],
                )

                utm_crs_list = query_utm_crs_info(
                    area_of_interest=aoi, datum_name="WGS 84"
                )
                # when we get a list that has at least one element, we can break
                if utm_crs_list:
                    break
            else:
                raise ValueError(
                    "Unable to find transform not resulting in all finite values"
                )

            crs = CRS.from_epsg(utm_crs_list[0].code)

            utm = f"EPSG:{crs.to_epsg()}"
            pipeline |= pdal.Filter.reprojection(out_srs=utm)

            pipeline.crs = crs
            pipeline.utm = utm
            return pipeline

        filters: pdal.Pipeline
        if os.path.splitext(self.filename)[-1] == ".json":
            pipeline = get_json(self.filename)
            filters = pdal.Pipeline(pipeline)
            self.logger.logger.info("Loaded JSON pipeline ")
        else:
            filters = pdal.Reader(self.filename).pipeline()
            self.logger.logger.info(f"Loaded {self.filename}")

        filters = _get_utm(filters)

        self.crs = filters.crs
        self.utm = filters.utm

        # Do not (fully) trust original classifications -- original workflow.
        if not self.config["TRUST_LABELS"]:
            filters |= pdal.Filter.range(limits="Classification![7:7]")
            filters |= pdal.Filter.range(limits="Classification![18:)")
            filters |= pdal.Filter.range(limits="Classification![9:9]")
            filters |= pdal.Filter.returns(groups="only")
            filters |= pdal.Filter.elm(cell=20.0)
            filters |= pdal.Filter.outlier(where="Classification!=7")
            filters |= pdal.Filter.range(limits="Classification![7:7]")
            filters |= pdal.Filter.assign(assignment="Classification[:]=1")
            filters |= pdal.Filter.smrf()

        else:
            filters |= pdal.Filter.returns(groups="only")

        filters.execute()
        self.pipeline = filters
        return filters


class VCD:
    def __init__(self, before: PointCloud, after: PointCloud) -> None:
        self.before = before
        self.after = after
        self.products: List[Product] = []
        self.gh = before.config["GROUNDHEIGHT"]
        self.resolution = before.config["RESOLUTION"]
        self.trust_labels = before.config["TRUST_LABELS"]
        self.compute_hag = before.config["COMPUTE_HAG"]

    def compute_indexes(self) -> None:
        after = self.after.df
        before = self.before.df

        # Compute height as delta Z between nearest point in before cloud from the after cloud -- original workflow.
        if not self.before.config["COMPUTE_HAG"]:
            tree3d = cKDTree(before[["X", "Y", "Z"]].to_numpy())
            _, i3d = tree3d.query(after[["X", "Y", "Z"]].to_numpy(), k=1)
            after["dZ3d"] = after.Z - before.iloc[i3d].Z.values

        # Compute height as HAG, treating after as non-ground and before as ground -- new workflow.
        else:
            # Assing after non-ground, before ground.
            after["TempClassification"] = 1
            before["TempClassification"] = 2

            # Merge clouds.
            allpoints = pd.concat([after, before])

            # Stash original classifications, then compute HAG using TempClassification. Pop the original classifications.
            pipeline = pdal.Pipeline(dataframes=[allpoints])
            pipeline |= pdal.Filter.ferry(
                dimensions="TempClassification=>Classification"
            )
            pipeline |= pdal.Filter.hag_delaunay()
            pipeline.execute()

            # Assign HAG as dZ3d and d3 in keeping with the original approach.
            result = pipeline.get_dataframe(0)
            after["dZ3d"] = result["HeightAboveGround"]

    def cluster(self) -> None:
        after = self.after.df
        gh = self.gh

        thresholdFilter = pdal.Filter.range(limits="dZ3d![-{gh}:{gh}]".format(gh=gh))

        conditions = [
            f"Classification=={id}" for id in self.after.config["CLASS_LABELS"]
        ]
        expression = " || ".join(conditions)
        rangeFilter = pdal.Filter.expression(expression=expression)

        clusterFilter = pdal.Filter.cluster(
            min_points=self.after.config["MIN_POINTS"],
            tolerance=self.after.config["CLUSTER_TOLERANCE"],
        )

        conditions = [
            f"ClusterID!={id}" for id in self.after.config["CULL_CLUSTER_IDS"]
        ]
        expression = " && ".join(conditions)
        clusterIdFilter = pdal.Filter.expression(expression=expression)

        array = after.to_records()
        self.clusters = pdal.Pipeline(
            [thresholdFilter, rangeFilter, clusterFilter, clusterIdFilter], [array]
        )
        self.clusters.execute()
        cluster_df = pd.DataFrame(self.clusters.arrays[0])

        # Encode the size of each cluster as a new dimension for analysis.
        cluster_df["ClusterSize"] = cluster_df.groupby(["ClusterID"])[
            "ClusterID"
        ].transform("count")
        self.cluster_sizes = cluster_df["ClusterSize"].to_numpy()

        p = self.make_product(
            cluster_df.X,
            cluster_df.Y,
            cluster_df.ClusterID,
            description=f"Clusters greater than +/-{gh:.2f} height",
        )
        self.products.append(p)

    def make_products(self) -> None:
        after = self.after.df
        p = self.make_product(
            after.X, after.Y, after.dZ3d, description="Before minus after"
        )
        self.products.append(p)

    def make_product(
        self,
        x: pd.Series,
        y: pd.Series,
        z: pd.Series,
        description: str = "",
    ) -> pd.DataFrame:
        df = x.to_frame().join(y.to_frame()).join(z.to_frame())
        return Product(df=df, z_name=z.name, description=description)

    def rasterize(self) -> None:
        resolution = self.before.config["RESOLUTION"]
        rasters_dir = os.path.join(self.before.config["OUTPUT_DIR"], "rasters")
        summary_dir = os.path.join(
            self.before.config["OUTPUT_DIR"], "rasters", "summary"
        )
        products_dir = os.path.join(
            self.before.config["OUTPUT_DIR"], "rasters", "products"
        )

        os.makedirs(rasters_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(products_dir, exist_ok=True)

        def _rasterize(product: Product, utm: str) -> str:
            array = product.df.to_records()
            array = rfn.rename_fields(array, {product.z_name: "Z"})

            outfile = os.path.join(products_dir, product.slug) + ".tif"

            metadata = (
                f"TIFFTAG_XRESOLUTION={resolution},"
                f"TIFFTAG_YRESOLUTION={resolution},"
                f"TIFFTAG_IMAGEDESCRIPTION={product.description},"
                f"CODEM_VERSION={__version__}"
            )
            gdalopts = (
                "COMPRESS=LZW," "PREDICTOR=2," "OVERVIEW_COMPRESS=LZW," "BIGTIFF=YES"
            )

            pipeline = pdal.Writer.gdal(
                filename=outfile,
                metadata=metadata,
                gdalopts=gdalopts,
                override_srs=utm,
                resolution=resolution,
                output_type="idw",
            ).pipeline(array)
            pipeline.execute()
            return outfile

        _ = [_rasterize(p, self.before.utm) for p in self.products]
        return None

    def save(self, format: str = ".las") -> None:
        with contextlib.suppress(FileExistsError):
            os.mkdir(os.path.join(self.before.config["OUTPUT_DIR"], "points"))

        # Determine Colormap
        flex_max = self.clusters.arrays[0]["dZ3d"].min()

        new_max = self.clusters.arrays[0]["dZ3d"].max()

        divnorm = colors.TwoSlopeNorm(vmin=flex_max, vcenter=0, vmax=new_max)
        # we are only writing the first point-clouds
        colormap = plt.colormaps[self.before.config["COLORMAP"]]

        # write point cloud output
        path = "clusters"
        array = self.clusters.arrays[0]
        sizes = self.cluster_sizes
        filename = os.path.join(
            self.before.config["OUTPUT_DIR"], "points", f"{path}{format}"
        )

        # convert colors from [0. 1] floats to [0, 65535] per LAS spec
        rgb = np.array(
            [
                colors.to_rgba_array(colormap(divnorm(array["dZ3d"])))
                * np.iinfo(np.uint16).max
            ],
            dtype=np.uint16,
        )[0, :, :-1]

        array = rfn.append_fields(
            array,
            ["Red", "Green", "Blue", "ClusterSize"],
            [rgb[:, 0], rgb[:, 1], rgb[:, 2], sizes],
        )

        crs = self.after.crs
        pipeline = pdal.Writer.las(
            filename=filename,
            extra_dims="all",
            a_srs=crs.to_string() if crs is not None else crs,
        ).pipeline(array)
        pipeline.execute()

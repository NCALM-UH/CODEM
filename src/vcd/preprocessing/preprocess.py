"""
preprocess.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

"""
import contextlib
import os
import re
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import pdal
import rasterio
from codem.lib.log import Log
from pyproj import CRS
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info  # type: ignore
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
    log: Log


def get_json(filename: str) -> str:
    with contextlib.suppress(IOError):
        with open(filename, "r") as f:
            content = f.read()
    return content


def slugify(text: str) -> str:
    """Adapted from https://stackoverflow.com/a/8366771/498396"""
    return re.sub(r"[^0-9A-Za-z.]", "-", text.lower())


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

            transformer = Transformer.from_crs(srs, 4326)
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

            utm_crs_list = query_utm_crs_info(area_of_interest=aoi, datum_name="WGS 84")

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

        filters |= pdal.Filter.range(limits="Classification![7:7]")
        filters |= pdal.Filter.range(limits="Classification![18:)")
        filters |= pdal.Filter.range(limits="Classification![9:9]")
        filters |= pdal.Filter.returns(groups="only")
        filters |= pdal.Filter.elm(cell=20.0)
        filters |= pdal.Filter.outlier(where="Classification!=7")
        filters |= pdal.Filter.range(limits="Classification![7:7]")
        filters |= pdal.Filter.assign(assignment="Classification[:]=1")
        filters |= pdal.Filter.smrf()
        self.pipeline = filters
        filters.execute()
        return filters


class VCD:
    def __init__(self, before: PointCloud, after: PointCloud) -> None:
        self.before = before
        self.after = after
        self.products: List[pd.DataFrame] = []
        self.gh = before.config["GROUNDHEIGHT"]
        self.resolution = before.config["RESOLUTION"]

    def compute_indexes(self) -> None:

        after = self.after.df
        before = self.before.df

        tree2d = cKDTree(before[["X", "Y"]])
        d2d, i2d = tree2d.query(after[["X", "Y"]], k=1)
        tree3d = cKDTree(before[["X", "Y", "Z"]])
        d3d, i3d = tree3d.query(after[["X", "Y", "Z"]], k=1)

        after["dX2d"] = after.X - before.iloc[i2d].X.values
        after["dY2d"] = after.Y - before.iloc[i2d].Y.values
        after["dZ2d"] = after.Z - before.iloc[i2d].Z.values
        after["dX3d"] = after.X - before.iloc[i3d].X.values
        after["dY3d"] = after.Y - before.iloc[i3d].Y.values
        after["dZ3d"] = after.Z - before.iloc[i3d].Z.values

        after["d2"] = d2d
        after["d3"] = d3d

    def cluster(self) -> None:

        after = self.after.df
        gh = self.gh

        array = after[(after.Classification != 2) & (after.d3 > gh)].to_records()
        self.ng_clusters = pdal.Filter.cluster(
            min_points=self.after.config["MIN_POINTS"],
            tolerance=self.after.config["CLUSTER_TOLERANCE"],
        ).pipeline(array)
        self.ng_clusters.execute()
        ng_cluster_df = pd.DataFrame(self.ng_clusters.arrays[0])

        p = self.make_product(
            ng_cluster_df.X,
            ng_cluster_df.Y,
            ng_cluster_df.ClusterID,
            description=f"Non-ground clusters greater than {gh:.2f} height",
            colorscale="IceFire",
        )
        self.products.append(p)

        array = after[(after.Classification == 2) & (after.d3 > gh)].to_records()
        self.ground_clusters = pdal.Filter.cluster(
            min_points=self.after.config["MIN_POINTS"],
            tolerance=self.after.config["CLUSTER_TOLERANCE"],
        ).pipeline(array)
        self.ground_clusters.execute()
        ground_cluster_df = pd.DataFrame(self.ground_clusters.arrays[0])

        p = self.make_product(
            ground_cluster_df.X,
            ground_cluster_df.Y,
            ground_cluster_df.ClusterID,
            description=f"Ground clusters greater than {gh:.2f} height",
            colorscale="IceFire",
        )
        self.products.append(p)

    def make_products(self) -> None:
        after = self.after.df
        gh = self.gh
        resolution = self.resolution

        p = self.make_product(
            after.X, after.Y, after.dZ3d, description="Before minus after"
        )
        self.products.append(p)

        p = self.make_product(
            after[after.d3 < gh].X,
            after[after.d3 < gh].Y,
            after[after.d3 < gh].dZ3d,
            f"Points within {resolution:.2f}m difference",
        )
        self.products.append(p)

        p = self.make_product(
            after[after.d3 > gh].X,
            after[after.d3 > gh].Y,
            after[after.d3 > gh].dZ3d,
            f"Points more than {resolution:.2f}m difference",
        )
        self.products.append(p)

        p = self.make_product(
            after[(after.Classification == 2) & (after.d3 > gh)].X,
            after[(after.Classification == 2) & (after.d3 > gh)].Y,
            after[(after.Classification == 2) & (after.d3 > gh)].dZ3d,
            f"Ground points more than {resolution:.2f}m difference",
        )
        self.products.append(p)

        p = self.make_product(
            after[(after.Classification != 2) & (after.d3 > gh)].X,
            after[(after.Classification != 2) & (after.d3 > gh)].Y,
            after[(after.Classification != 2) & (after.d3 > gh)].dZ3d,
            f"Non-ground points more than {resolution:.2f}m difference",
        )
        self.products.append(p)

    def make_product(
        self,
        x: pd.Series,
        y: pd.Series,
        z: pd.Series,
        description: str = "",
        colorscale: str = "RdBu",
    ) -> pd.DataFrame:
        product = x.to_frame().join(y.to_frame()).join(z.to_frame())
        product.z = z.name
        product.slug = slugify(description)
        product.description = description
        product.colorscale = colorscale
        return product

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

        def _rasterize(product: pd.DataFrame, utm: str) -> str:

            array = product.to_records()
            array = rfn.rename_fields(array, {product.z: "Z"})

            outfile = os.path.join(products_dir, product.slug) + ".tif"

            metadata = f"TIFFTAG_XRESOLUTION={resolution},TIFFTAG_YRESOLUTION={resolution},TIFFTAG_IMAGEDESCRIPTION={product.description}"
            gdalopts = "MAX_Z_ERROR=0.01,COMPRESS=LERC_ZSTD,OVERVIEW_COMPRESS=LERC_ZSTD,BIGTIFF=YES"

            pipeline = pdal.Writer.gdal(
                filename=outfile,
                metadata=metadata,
                gdalopts=gdalopts,
                override_srs=utm,
                resolution=resolution,
            ).pipeline(array)
            pipeline.execute()
            return outfile

        def _merge(rasters: List[str], output_type: str) -> None:

            with rasterio.open(rasters[0]) as src0:
                meta = src0.meta
                descriptions = src0.descriptions

            meta.update(count=len(rasters))
            meta.update(
                compress="LERC_ZSTD",
                max_z_error=0.01,
                bigtiff="YES",
                overview_compress="LERC_ZSTD",
            )

            band_id = descriptions.index(output_type) + 1  # bands count from 1
            outfile = os.path.join(summary_dir, output_type) + ".tif"

            with rasterio.open(outfile, "w", **meta) as dst:

                for index, layer in enumerate(rasters, start=1):
                    with rasterio.open(layer) as src:

                        band_description = src.tags()["TIFFTAG_IMAGEDESCRIPTION"]
                        band = src.read(band_id)

                        dst.write_band(index, band)
                        dst.update_tags(band_id)
                        dst.set_band_description(index, band_description)

        rasters = [_rasterize(p, self.before.utm) for p in self.products]
        for feature in ("idw", "min", "max", "mean", "count"):
            _merge(rasters, feature)
        return None

    def save(self, format: str = ".las") -> None:
        with contextlib.suppress(FileExistsError):
            os.mkdir(os.path.join(self.before.config["OUTPUT_DIR"], "points"))
        new_ground = (
            os.path.join(self.before.config["OUTPUT_DIR"], "points", "ng-clusters")
            + format
        )
        ground = (
            os.path.join(self.before.config["OUTPUT_DIR"], "points", "gnd-clusters")
            + format
        )
        pipeline = pdal.Writer.las(
            minor_version=4, filename=new_ground, extra_dims="all"
        ).pipeline(self.ng_clusters.arrays[0])
        pipeline.execute()

        pipeline = pdal.Writer.las(
            minor_version=4,
            filename=ground,
            extra_dims="all",
        ).pipeline(self.ground_clusters.arrays[0])
        pipeline.execute()

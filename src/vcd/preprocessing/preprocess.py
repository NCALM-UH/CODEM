"""
preprocess.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

"""
import json
import os

import pdal
import rasterio

from typing import Optional

import codem.lib.resources as r
from typing_extensions import TypedDict
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

import numpy.lib.recfunctions as rfn

from pyproj import CRS
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj import Transformer, transform

from scipy.spatial import cKDTree

import re



def get_json(filename):
    try:
        with open(filename,'r') as f:
            return f.read()
    except:
        return None


def slugify(text):
    """Adapted from https://stackoverflow.com/a/8366771/498396"""
    text = text.lower()
    return re.sub(r'[^0-9A-Za-z.]', '-', text)



class PointCloud:
    """
    A class for storing and preparing geospatial data

    Parameters
    ----------
    config: dict
        Dictionary of configuration options
    fnd: bool
        Whether the file is foundation data

    """

    def __init__(self, config: dict, key: str) -> None:
        self.logger = config['log']
        self.filename = config[key]
        self.config = config
        self.crs = None
        self.utm = None
        self.pipeline = self.open()

        if len(self.pipeline.arrays) > 1:
            raise  NotImplementedError("VCD between multiple views is not supported")
        self.df = pd.DataFrame(self.pipeline.arrays[0])

    def open(self):

        def _get_utm(pipeline):

            data = pipeline.quickinfo
            is_reader = [[k.split('.')[0] == 'readers', k] for k in data.keys()]
            srs = {}
            bounds = {}
            for id, k in enumerate(is_reader):
                if k[0]: # we are a reader
                    reader_info = data[k[1]]
                    bounds = reader_info['bounds']
                    srs = CRS.from_user_input(reader_info['srs']['compoundwkt'])

                    # we just take the first one. If there's more we are screwed
                    break

            transformer = Transformer.from_crs(srs, 4326)
            dd = transformer.transform((bounds['minx'], bounds['maxx']),
                                       (bounds['miny'], bounds['maxy']))

            # stolen from Alan https://gis.stackexchange.com/a/423614/350

            # dd now in the form ((41.469221251843926, 41.47258675464548), (-93.68979255724548, -93.68530098082489))

            aoi = area_of_interest=AreaOfInterest( west_lon_degree=dd[1][0],
                                                   south_lat_degree=dd[0][0],
                                                   east_lon_degree=dd[1][1],
                                                   north_lat_degree=dd[0][1])

            utm_crs_list = query_utm_crs_info( area_of_interest = aoi,
                                               datum_name="WGS 84" )

            crs = CRS.from_epsg(utm_crs_list[0].code)

            utm = f'EPSG:{crs.to_epsg()}'
            pipeline |= pdal.Filter.reprojection(out_srs = utm)

            pipeline.crs = crs
            pipeline.utm = utm
            return pipeline

        filters = None

        pipeline = get_json(self.filename)
        if pipeline:
            filters = pdal.Pipeline(pipeline)
            self.logger.logger.info(f'Loaded JSON pipeline ')
        else:
            filters = pdal.Pipeline(self.filename)
            self.logger.logger.info(f'Loaded {self.filename}')

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


    def extract_crs(self):
        """Extract CRS from a PDAL pipeline for readers.las output as ESRI WKT1 for shapefile output"""

        output = self.srs.to_wkt(WktVersion.WKT1_ESRI)
        return output


class VCD:

    def __init__(self, before: PointCloud, after: PointCloud) -> None:
        self.before = before
        self.after = after
        self.products = []
        self.gh = before.config['GROUNDHEIGHT']
        self.resolution = before.config['RESOLUTION']

    def compute_indexes(self):

        after = self.after.df
        before = self.before.df
        gh = self.gh

        tree2d = cKDTree(before[['X','Y']])
        d2d, i2d = tree2d.query(after[['X','Y']], k=1)
        tree3d = cKDTree(before[['X','Y','Z']])
        d3d, i3d = tree3d.query(after[['X','Y','Z']], k=1)

        after['dX2d'] = after.X - before.iloc[i2d].X.values
        after['dY2d'] = after.Y - before.iloc[i2d].Y.values
        after['dZ2d'] = after.Z - before.iloc[i2d].Z.values
        after['dX3d'] = after.X - before.iloc[i3d].X.values
        after['dY3d'] = after.Y - before.iloc[i3d].Y.values
        after['dZ3d'] = after.Z - before.iloc[i3d].Z.values

        after['d2'] = d2d
        after['d3'] = d3d

    def cluster(self):

        after = self.after.df
        before = self.before.df
        gh = self.gh

        array = after[(after.Classification != 2) & (after.d3 > gh)].to_records()
        self.ng_clusters = pdal.Filter.cluster(min_points=30, tolerance=2.0).pipeline(array)
        self.ng_clusters.execute()
        ng_cluster_df = pd.DataFrame(self.ng_clusters.arrays[0])

        p = self.make_product(ng_cluster_df.X,
                              ng_cluster_df.Y,
                              ng_cluster_df.ClusterID,
                              description = f"Non-ground clusters greater than {gh:.2f} height",
                              colorscale="IceFire")
        self.products.append(p)


        array = after[(after.Classification==2) & (after.d3 > gh)].to_records()
        self.ground_clusters = pdal.Filter.cluster(min_points=30, tolerance=2.0).pipeline(array)
        self.ground_clusters.execute()
        ground_cluster_df = pd.DataFrame(self.ground_clusters.arrays[0])

        p = self.make_product(ground_cluster_df.X,
                              ground_cluster_df.Y,
                              ground_cluster_df.ClusterID,
                              description = f"Ground clusters greater than {gh:.2f} height",
                              colorscale="IceFire")
        self.products.append(p)

    def make_products(self):
        after = self.after.df
        before = self.before.df
        gh = self.gh
        resolution = self.resolution


        p = self.make_product(after.X, after.Y, after.dZ3d, description ="Before minus after")
        self.products.append(p)

        p = self.make_product(after[after.d3<gh].X,
                              after[after.d3<gh].Y,
                              after[after.d3<gh].dZ3d,
                              f"Points within {resolution:.2f}m difference")
        self.products.append(p)

        p = self.make_product(after[after.d3>gh].X,
                              after[after.d3>gh].Y,
                              after[after.d3>gh].dZ3d,
                              f"Points more than {resolution:.2f}m difference")
        self.products.append(p)

        p = self.make_product(after[(after.Classification==2)&(after.d3>gh)].X,
                              after[(after.Classification==2)&(after.d3>gh)].Y,
                              after[(after.Classification==2)&(after.d3>gh)].dZ3d,
                              f"Ground points more than {resolution:.2f}m difference")
        self.products.append(p)

        p = self.make_product(after[(after.Classification!=2)&(after.d3>gh)].X,
             after[(after.Classification!=2)&(after.d3>gh)].Y,
             after[(after.Classification!=2)&(after.d3>gh)].dZ3d,
             f'Non-ground points more than {resolution:.2f}m difference')
        self.products.append(p)

    def make_product(self, x, y, z, description="", colorscale='RdBu'):
        after = self.after.df
        before = self.before.df

        product = x.to_frame().join(y.to_frame()).join( z.to_frame())
        product.z = z.name
        product.slug = slugify(description)
        product.description = description
        product.colorscale = colorscale

        return product

    def plot(self):

        def _plot(product):

            try:
                os.mkdir(os.path.join(self.before.config['OUTPUT_DIR'], "plots"))
            except FileExistsError:
                pass

            outfile = os.path.join(self.before.config['OUTPUT_DIR'], "plots", product.slug) + '.png'

            fig = go.Figure(layout_title_text = product.description,
                            data=go.Scattergl(
                                              x = product.X,
                                              y = product.Y,
                                              mode = 'markers',
                                              marker=dict(
                                                          color=product[product.z],
                                                          colorscale=product.colorscale,
                                                          colorbar=dict(thickness=20),
                                              size=1) ))
            fig.update_yaxes( scaleanchor = "x", scaleratio = 1,)
            fig.update_layout( autosize=False, width=700, height=700,)
            img = fig.to_image('png')
            with open(outfile,'wb') as f:
                f.write(img)

        for p in self.products:
            _plot(p)


    def rasterize(self):
        resolution = self.before.config['RESOLUTION']
        rasters_dir = os.path.join(self.before.config['OUTPUT_DIR'], "rasters")
        summary_dir = os.path.join(self.before.config['OUTPUT_DIR'], "rasters", "summary")
        products_dir = os.path.join(self.before.config['OUTPUT_DIR'], "rasters", "products")

        try:
            os.mkdir(rasters_dir)
            os.mkdir(summary_dir)
            os.mkdir(products_dir)
        except FileExistsError:
            pass



        def _rasterize(product, utm, output_type="mean"):

            array = product.to_records()
            array = rfn.rename_fields(array, {product.z: 'Z'})

            outfile = os.path.join(products_dir, product.slug) + '.tif'

            metadata = f"TIFFTAG_XRESOLUTION={resolution},TIFFTAG_YRESOLUTION={resolution},TIFFTAG_IMAGEDESCRIPTION={product.description}"
            gdalopts = "MAX_Z_ERROR=0.01,COMPRESS=LERC_ZSTD,OVERVIEW_COMPRESS=LERC_ZSTD,BIGTIFF=YES"

            pipeline = pdal.Writer.gdal(filename = outfile,
                                        metadata = metadata,
                                        gdalopts = gdalopts,
                                        override_srs = utm,
                                        resolution = resolution).pipeline(array)
            pipeline.execute()
            return outfile

        def _merge(rasters, output_type):

            with rasterio.open(rasters[0]) as src0:
                meta = src0.meta
                descriptions = src0.descriptions


            meta.update(count = len(rasters))
            meta.update( compress="LERC_ZSTD",
                         max_z_error=0.01,
                         bigtiff="YES",
                         overview_compress="LERC_ZSTD")


            band_id = descriptions.index(output_type) + 1 # bands count from 1
            outfile = os.path.join(summary_dir, output_type) + '.tif'

            with rasterio.open(outfile, 'w', **meta) as dst:

                for id, layer in enumerate(rasters, start=1):
                    with rasterio.open(layer) as src:

                        band_description = src.tags()['TIFFTAG_IMAGEDESCRIPTION']
                        band = src.read(band_id)

                        dst.write_band(id, band)
                        dst.update_tags(band_id)
                        dst.set_band_description(id, band_description)

        rasters = []
        for p in self.products:
            rasters.append(_rasterize(p, self.before.utm))

        _merge(rasters, "idw")
        _merge(rasters, "min")
        _merge(rasters, "max")
        _merge(rasters, "mean")
        _merge(rasters, "count")





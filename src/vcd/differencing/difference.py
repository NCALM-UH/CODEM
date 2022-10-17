"""
preprocess.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

"""
import json
import logging
import os
from typing import Optional

import codem.lib.resources as r
from vcd.preprocessing.preprocess import PointCloud
from typing_extensions import TypedDict



logger = logging.getLogger(__name__)



class VCD:

    def __init__(self, before: PointCloud, after: PointCloud) -> None:
        self.logger = config['log']
        self.filename = config[key]
        self.compute_indexes()

    def compute_indexes(self):

        tree2d = cKDTree(self.before.df[['X','Y']])
        d2d, i2d = tree2d.query(self.after.df[['X','Y']], k=1)
        tree3d = cKDTree(self.before.df[['X','Y','Z']])
        d3d, i3d = tree3d.query(self.after.df[['X','Y','Z']], k=1)

        self.after.df['dX2d'] = self.after.df.X - self.after.df.iloc[i2d].X.values
        self.after.df['dY2d'] = self.after.df.Y - self.after.df.iloc[i2d].Y.values
        self.after.df['dZ2d'] = self.after.df.Z - self.after.df.iloc[i2d].Z.values
        self.after.df['dX3d'] = self.after.df.X - self.after.df.iloc[i3d].X.values
        self.after.df['dY3d'] = self.after.df.Y - self.after.df.iloc[i3d].Y.values
        self.after.df['dZ3d'] = self.after.df.Z - self.after.df.iloc[i3d].Z.values

        self.after.df['d2'] = d2d
        self.after.df['d3'] = d3d

    def plot(self):
        gh = self.config.ground_height
        def _plot(x, y, color, filename, colorscale='RdBu'):

            os.mkdir(self.config['OUTPUT_DIR'], "plots")
            outfile = os.path.join(self.config['OUTPUT_DIR'], "plots", filename)

            fig = go.Figure(data=go.Scattergl(
                                              x = x,
                                              y = y,
                                              mode = 'markers',
                                              marker=dict(
                                                          color=color,
                                                          colorscale=colorscale,
                                                          colorbar=dict(thickness=20),
                                              size=1) ))
            fig.update_yaxes( scaleanchor = "x", scaleratio = 1,)
            fig.update_layout( autosize=False, width=700, height=700,)
            img = fig.to_image('png')
            with open(outfile,'wb') as f:
                f.write(img)

        _plot(self.after.df.X, self.after.df.Y, self.after.df.dZ3d, 'before-after')
        _plot(self.after.df[self.after.df.d3<gh].X, self.after.df[self.after.df.d3<gh].Y, self.after.df[self.after.df.d3<gh].dZ3d, 'within-1m-difference')
        _plot(self.after.df[self.after.df.d3>gh].X, self.after.df[self.after.df.d3>gh].Y, self.after.df[self.after.df.d3>gh].dZ3d, 'more-than-1m-difference')
        _plot(self.after.df[(self.after.df.Classification==2)&(self.after.df.d3>gh)].X,
             self.after.df[(self.after.df.Classification==2)&(self.after.df.d3>gh)].Y,
             self.after.df[(self.after.df.Classification==2)&(self.after.df.d3>gh)].dZ3d,
             'ground-more-than-1m-differences')

        _plot(self.after.df[(self.after.df.Classification!=2)&(self.after.df.d3>gh)].X,
             self.after.df[(self.after.df.Classification!=2)&(self.after.df.d3>gh)].Y,
             self.after.df[(self.after.df.Classification!=2)&(self.after.df.d3>gh)].dZ3d,
             'nonground-more-than-1m-differences')

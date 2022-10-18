

from ..preprocessing.preprocess import PointCloud, VCD

import trimesh

# https://github.com/GeospatialPython/pyshp
import shapefile
from shapefile import TRIANGLE_STRIP, TRIANGLE_FAN, RING, OUTER_RING, FIRST_RING

import pdal

from pyproj.enums import WktVersion

import numpy as np

import os


class Mesh:
    def __init__(self, vcd: VCD) -> None:
        self.vcd = vcd

    def cluster(self, dataset):

        clusters = []
        dimension = 'ClusterID'
        pipeline = """
        {
            "pipeline": [
                {
                    "type":"filters.groupby",
                    "dimension":"ClusterID"
                }
            ]
        } """
        reader = pdal.Pipeline(pipeline, arrays = dataset.arrays)
        reader.execute()

        for arr in reader.arrays:

            if len(arr) < 5:
                if (len(arr)):
                    cluster_id = arr[0][dimension]
                    print (f"Not enough points to cluster {cluster_id}. We have {len(arr)} and need 5")
                else:
                    print (f"Cluster has no points!")

            x = arr['X']
            y = arr['Y']
            z = arr['Z']
            cluster_id = arr[0][dimension]

            points = np.vstack((x,y,z)).T

            self.vcd.before.logger.logger.info (f'computing Delaunay of {len(points)} points')

            pc = trimesh.points.PointCloud(points)

            hull = pc.convex_hull
            hull.cluster_id = cluster_id

            # cull out some specific cluster IDs
            culls = [-1,0,1]

            if cluster_id not in culls:
                clusters.append(hull)

        return clusters


    def write(self, filename, clusters):
        try:
            os.mkdir(os.path.join(self.vcd.before.config['OUTPUT_DIR'], "meshes"))
        except FileExistsError:
            pass

        outfile = os.path.join(self.vcd.before.config['OUTPUT_DIR'], "meshes", filename)
        wkt = self.vcd.before.crs.to_wkt(WktVersion.WKT1_ESRI)

        self.vcd.before.logger.logger.info (f'Saving mesh data to {filename}')


        with shapefile.Writer(outfile) as w:
            w.field('volume', 'N', decimal=2)
            w.field('area', 'N', decimal=2)
            w.field('clusterid', 'N')

            # Save CRS WKT
            with open(outfile+'.prj', 'w') as f:
                f.write(wkt)

            for cluster in clusters:
                w.multipatch(cluster.triangles, partTypes=[TRIANGLE_STRIP]* len(cluster.triangles)) # one type for each part
                w.record(cluster.volume, cluster.area, cluster.cluster_id)

            w.close()








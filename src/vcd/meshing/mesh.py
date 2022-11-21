import contextlib
import os
from typing import List

import numpy as np
import pdal
import shapefile
import trimesh
from pyproj.enums import WktVersion
from shapefile import TRIANGLE_STRIP
from vcd.preprocessing.preprocess import VCD


class Mesh:
    def __init__(self, vcd: VCD) -> None:
        self.vcd = vcd

    def cluster(self, dataset: pdal.Filter.cluster) -> List[trimesh.Trimesh]:

        clusters = []
        dimension = "ClusterID"
        pipeline = """
        {
            "pipeline": [
                {
                    "type":"filters.groupby",
                    "dimension":"ClusterID"
                }
            ]
        } """
        reader = pdal.Pipeline(pipeline, arrays=dataset.arrays)
        reader.execute()

        for arr in reader.arrays:

            if len(arr) < 5:
                if len(arr):
                    cluster_id = arr[0][dimension]
                    print(
                        f"Not enough points to cluster {cluster_id}. We have {len(arr)} and need 5"
                    )
                else:
                    print("Cluster has no points!")

            x = arr["X"]
            y = arr["Y"]
            z = arr["Z"]
            cluster_id = arr[0][dimension]
            classification = arr[0]["Classification"]

            points = np.vstack((x, y, z)).T

            self.vcd.before.logger.logger.info(
                f"computing Delaunay of {len(points)} points"
            )

            pc = trimesh.points.PointCloud(points)

            hull = pc.convex_hull
            hull.cluster_id = cluster_id
            hull.classification = classification

            # cull out some specific cluster IDs
            culls = self.vcd.before.config["CULL_CLUSTER_IDS"]

            if cluster_id not in culls:
                clusters.append(hull)

        return clusters

    def write(self, filename: str, clusters: List[trimesh.Trimesh]) -> None:
        with contextlib.suppress(FileExistsError):
            os.mkdir(os.path.join(self.vcd.before.config["OUTPUT_DIR"], "meshes"))
        outfile = os.path.join(self.vcd.before.config["OUTPUT_DIR"], "meshes", filename)
        if self.vcd.before.crs is None:
            # mypy must be satisfied with its optional!
            raise RuntimeError
        wkt = self.vcd.before.crs.to_wkt(WktVersion.WKT1_ESRI)
        self.vcd.before.logger.logger.info(f"Saving mesh data to {filename}")

        with shapefile.Writer(outfile) as w:
            w.field("volume", "N", decimal=2)
            w.field("area", "N", decimal=2)
            w.field("clusterid", "N")
            w.field("ground", "L")

            # Save CRS WKT
            with open(f"{outfile}.prj", "w") as f:
                f.write(wkt)

            for cluster in clusters:
                w.multipatch(
                    cluster.triangles,
                    partTypes=[TRIANGLE_STRIP] * len(cluster.triangles),
                )  # one type for each part
                is_ground = cluster.classification == 2
                w.record(cluster.volume, cluster.area, cluster.cluster_id, is_ground)

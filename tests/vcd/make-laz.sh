#!/bin/bash
#
#

bounds="([-10429500, -10429000], [5081800, 5082300])"

pdal translate https://s3-us-west-2.amazonaws.com/usgs-lidar-public/IA_SouthCentral_1_2020/ept.json \
    after.copc.laz \
    --readers.ept.bounds="$bounds"

pdal translate https://s3-us-west-2.amazonaws.com/usgs-lidar-public/IA_FullState/ept.json \
    before.copc.laz \
    --readers.ept.bounds="$bounds"

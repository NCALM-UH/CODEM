# -*- coding: utf-8 -*-

import os
import json
import arcpy
import docker


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "3D Registration"
        self.alias = "3d_registration"

        # List of tool classes associated with this toolbox
        self.tools = [Register_DEM2DEM, Register_MultiType]


class Register_DEM2DEM(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Register DEM to DEM"
        self.description = "Co-Register Digital Elevation Models"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        fnd = arcpy.Parameter(
            displayName="Foundation DEM",
            name="foundation_file",
            datatype=["GPRasterLayer", "DEFile"],
            parameterType="Required",
            direction="Input",
        )
        aoi = arcpy.Parameter(
            displayName="Area of Interest (AOI) DEM",
            name="aoi_file",
            datatype=["GPRasterLayer", "DEFile"],
            parameterType="Required",
            direction="Input"
        )

        min = arcpy.Parameter(
            displayName="Minimum Resolution (m)",
            name="min_resolution",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Resolution",
        )

        dss = arcpy.Parameter(
            displayName="DSM Registration",
            name="dsm_solve_scale",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="Solve Scale"
        )
        iss = arcpy.Parameter(
            displayName="ICP Registration",
            name="icp_solve_scale",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="Solve Scale"
        )

        dsf = arcpy.Parameter(
            displayName="Normalization Strong Filter Size (m)",
            name="dsm_strong_filter",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        dwf = arcpy.Parameter(
            displayName="Normalization Weak Filter Size (m)",
            name="dsm_weak_filter",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        dat = arcpy.Parameter(
            displayName="AKAZE Detection Threshold",
            name="dsm_akaze_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        dlr = arcpy.Parameter(
            displayName="Lowe's Ratio",
            name="dsm_lowes_ratio",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        drmi = arcpy.Parameter(
            displayName="RANSAC Maximum Iterations",
            name="dsm_ransac_max_iter",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        drt = arcpy.Parameter(
            displayName="RANSAC Error Threshold (m)",
            name="dsm_ransac_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )

        imi = arcpy.Parameter(
            displayName="ICP Maximum Iterations",
            name="icp_max_iter",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        iat = arcpy.Parameter(
            displayName="ICP Angle Threshold (degrees)",
            name="icp_angle_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        idt = arcpy.Parameter(
            displayName="ICP Distance Threshold (m)",
            name="icp_distance_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        irt = arcpy.Parameter(
            displayName="ICP RMSE Relative Change Threshold",
            name="icp_rmse_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        ir = arcpy.Parameter(
            displayName="Robust ICP",
            name="icp_robust",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )

        # Minimum pipeline resolution
        min.value = 2.0
        min.filter.type = "Range"
        min.filter.list = [0.01, 100]

        # DSM registration - solve scale option
        dss.value = True
        # ICP registration - solve scale option
        iss.value = True
        # DSM normalization strong filter size
        dsf.value = 10.0
        dsf.filter.type = "Range"
        dsf.filter.list = [0.01, 1000]
        # DSM normalization weak filter size
        dwf.value = 1.0
        dwf.filter.type = "Range"
        dwf.filter.list = [0.01, 1000]
        # AKAZE feature detection threshold
        dat.value = 0.0001
        dat.filter.type = "Range"
        dat.filter.list = [0.0000001, 1]
        # Lowe's ratio
        dlr.value = 0.9
        dlr.filter.type = "Range"
        dlr.filter.list = [0.01, 0.99]
        # Maximum RANSAC iterations in feature matching
        drmi.value = 10000
        drmi.filter.type = "Range"
        drmi.filter.list = [1, 1000000]
        # RANSAC feature location transformation error threshold
        drt.value = 10
        drt.filter.type = "Range"
        drt.filter.list = [0.01, 100]

        # Maximum ICP iterations
        imi.value = 100
        imi.filter.type = "Range"
        imi.filter.list = [1, 1000]
        # ICP convergence threshold - minimum angle change
        iat.value = 0.001
        iat.filter.type = "Range"
        iat.filter.list = [0.00001, 10]
        # ICP convergence threshold - minimum distance change
        idt.value = 0.001
        idt.filter.type = "Range"
        idt.filter.list = [0.00001, 10]
        # ICP convergence threshold - minimum relative change in RMSE
        irt.value = 0.0001
        irt.filter.type = "Range"
        irt.filter.list = [0.0000001, 1]
        # Robust ICP option
        ir.value = True

        #         0    1    2    3    4    5    6    7    8    9     10   11   12   13   14   15
        params = [fnd, aoi, min, dss, iss, dsf, dwf, dat, dlr, drmi, drt, imi, iat, idt, irt, ir]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        if parameters[5].value and parameters[6].value:
            if parameters[5].value <= parameters[6].value:
                parameters[5].setErrorMessage("Strong filter size must be larger than weak filter size")
                parameters[6].setErrorMessage("Weak filter size must be smaller than large filter size")
        return

    def getLayerPath(self, layer):
        if os.path.isabs(layer):
            layer_source = layer
        else:
            desc = arcpy.Describe(layer)
            path = desc.path
            layer_source = str(path) + "\\" + layer
        return layer_source

    def execute(self, parameters, messages):
        """The source code of the tool."""
        fnd_source = self.getLayerPath(parameters[0].valueAsText)
        aoi_source = self.getLayerPath(parameters[1].valueAsText)

        fnd_dir, fnd_file = os.path.split(fnd_source)
        aoi_dir, aoi_file = os.path.split(aoi_source)

        # Binding multiple volumes to the same directory is not good
        if fnd_dir == aoi_dir:
            volume = "/data"

            cmd = "modem"
            cmd += f' "{volume}/{fnd_file}"'
            cmd += f' "{volume}/{aoi_file}"'
            for parameter in parameters[2:]:
                cmd += f" --{parameter.name} {parameter.valueAsText}"

            volumes = {fnd_dir: {"bind": volume, "mode": "rw"}}
        else:
            fnd_volume = "/data/fnd"
            aoi_volume = "/data/aoi"

            cmd = "modem"
            cmd += f' "{fnd_volume}/{fnd_file}"'
            cmd += f' "{aoi_volume}/{aoi_file}"'
            for parameter in parameters[2:]:
                cmd += f" --{parameter.name} {parameter.valueAsText}"

            volumes={
                fnd_dir: {"bind": fnd_volume, "mode": "ro"},
                aoi_dir: {"bind": aoi_volume, "mode": "rw"}
            }

        arcpy.SetProgressor("step", "Registering AOI to Foundation", 0, 5)
        client = docker.from_env()
        container = client.containers.run(
            # "docker-registry.rsgiscx.net:443/rsgis/modem:0.2",
            "modem:0.21",
            command=["bash", "-c", cmd],
            volumes=volumes,
            detach=True,
            remove=True,
        )

        # Hack the Docker stdout and stderr for progress bar and messaging
        regfile = None
        output = container.attach(stdout=True, stream=True, logs=True)
        for line in output:
            temp = line.decode(errors="ignore")
            # Check for error
            if ("Traceback" in temp) or ("AssertionError" in temp):
                arcpy.AddError(temp)
            else:
                # Handle multiple lines
                for temp_line in temp.splitlines():
                    # Progress bar
                    if ("PREPROCESSING DATA" in temp_line):
                        arcpy.SetProgressorLabel("Step 1/4: Prepping AOI and Foundation Data")
                        arcpy.SetProgressorPosition()
                    if ("BEGINNING COARSE REGISTRATION" in temp_line):
                        arcpy.SetProgressorLabel("Step 2/4: Solving Coarse Registration")
                        arcpy.SetProgressorPosition()
                    if ("BEGINNING FINE REGISTRATION" in temp_line):
                        arcpy.SetProgressorLabel("Step 3/4: Solving Fine Registration")
                        arcpy.SetProgressorPosition()
                    if ("APPLYING REGISTRATION" in temp_line):
                        arcpy.SetProgressorLabel("Step 4/4: Applying Registration to AOI Data")
                        arcpy.SetProgressorPosition()

                    # Messaging
                    if ("- INFO -" in temp_line):
                        # Change paths to local
                        if ("/data/fnd" in temp_line):
                            idx = temp_line.find("/data/fnd")
                            temp_path = os.path.join(fnd_dir, temp_line[idx+10:])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        elif ("/data/aoi" in temp_line):
                            idx = temp_line.find("/data/aoi")
                            temp_path = os.path.join(aoi_dir, temp_line[idx+10:])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        elif ("/data" in temp_line):
                            idx = temp_line.find("/data")
                            temp_path = os.path.join(fnd_dir, temp_line[idx+6:])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        arcpy.AddMessage(temp_line)
                    if ("- WARNING -" in temp_line):
                        arcpy.AddWarning(temp_line)

                    # Output file location
                    if ("Registration has been applied to" in temp_line):
                        idx = temp_line.find("registration_")
                        reg_file = os.path.join(aoi_dir, temp_line[idx:])
                        reg_file = os.path.normpath(reg_file)

        if reg_file is not None:
            _, ext = os.path.splitext(reg_file)
            if (ext == ".tif"):
                try:
                    aprx = arcpy.mp.ArcGISProject('CURRENT')
                    activeMap = aprx.activeMap
                    arcpy.env.addOutputsToMap = True
                    if activeMap is not None:
                        activeMap.addDataFromPath(reg_file)
                except:
                    # arcpy.AddWarning("Unable to create registered DEM layer in ArcGIS.")
                    pass # We'll let this silently fail for now

        return

class Register_MultiType(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Register Multi-Type"
        self.description = "Co-Register 3D Spatial Data"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        fnd = arcpy.Parameter(
            displayName="Foundation Data File",
            name="foundation_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input",
        )
        aoi = arcpy.Parameter(
            displayName="Area of Interest (AOI) Data File",
            name="aoi_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )

        min = arcpy.Parameter(
            displayName="Minimum Resolution (m)",
            name="min_resolution",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Resolution",
        )

        dss = arcpy.Parameter(
            displayName="DSM Registration",
            name="dsm_solve_scale",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="Solve Scale"
        )
        iss = arcpy.Parameter(
            displayName="ICP Registration",
            name="icp_solve_scale",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="Solve Scale"
        )

        dsf = arcpy.Parameter(
            displayName="Normalization Strong Filter Size (m)",
            name="dsm_strong_filter",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        dwf = arcpy.Parameter(
            displayName="Normalization Weak Filter Size (m)",
            name="dsm_weak_filter",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        dat = arcpy.Parameter(
            displayName="AKAZE Detection Threshold",
            name="dsm_akaze_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        dlr = arcpy.Parameter(
            displayName="Lowe's Ratio",
            name="dsm_lowes_ratio",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        drmi = arcpy.Parameter(
            displayName="RANSAC Maximum Iterations",
            name="dsm_ransac_max_iter",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )
        drt = arcpy.Parameter(
            displayName="RANSAC Error Threshold (m)",
            name="dsm_ransac_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="DSM Registration Options",
        )

        imi = arcpy.Parameter(
            displayName="ICP Maximum Iterations",
            name="icp_max_iter",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        iat = arcpy.Parameter(
            displayName="ICP Angle Threshold (degrees)",
            name="icp_angle_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        idt = arcpy.Parameter(
            displayName="ICP Distance Threshold (m)",
            name="icp_distance_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        irt = arcpy.Parameter(
            displayName="ICP RMSE Relative Change Threshold",
            name="icp_rmse_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )
        ir = arcpy.Parameter(
            displayName="Robust ICP",
            name="icp_robust",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="ICP Registration Options",
        )

        # Foundation data file
        # fnd.value = "E:\dev\modem\demo\Foundation-PointCloud.laz"
        fnd.filter.list = ['las', 'laz', 'bpf', 'ply', 'obj', 'tif', 'tiff']
        # AOI data file
        # aoi.value = "E:\dev\modem\demo\AOI-Mesh.ply"
        aoi.filter.list = ['las', 'laz', 'bpf', 'ply', 'obj', 'tif', 'tiff']

        # Minimum pipeline resolution
        min.value = 2.0
        min.filter.type = "Range"
        min.filter.list = [0.01, 100]

        # DSM registration - solve scale option
        dss.value = True
        # ICP registration - solve scale option
        iss.value = True
        # DSM normalization strong filter size
        dsf.value = 10.0
        dsf.filter.type = "Range"
        dsf.filter.list = [0.01, 1000]
        # DSM normalization weak filter size
        dwf.value = 1.0
        dwf.filter.type = "Range"
        dwf.filter.list = [0.01, 1000]
        # AKAZE feature detection threshold
        dat.value = 0.0001
        dat.filter.type = "Range"
        dat.filter.list = [0.0000001, 1]
        # Lowe's ratio
        dlr.value = 0.9
        dlr.filter.type = "Range"
        dlr.filter.list = [0.01, 0.99]
        # Maximum RANSAC iterations in feature matching
        drmi.value = 10000
        drmi.filter.type = "Range"
        drmi.filter.list = [1, 1000000]
        # RANSAC feature location transformation error threshold
        drt.value = 10
        drt.filter.type = "Range"
        drt.filter.list = [0.01, 100]

        # Maximum ICP iterations
        imi.value = 100
        imi.filter.type = "Range"
        imi.filter.list = [1, 1000]
        # ICP convergence threshold - minimum angle change
        iat.value = 0.001
        iat.filter.type = "Range"
        iat.filter.list = [0.00001, 10]
        # ICP convergence threshold - minimum distance change
        idt.value = 0.001
        idt.filter.type = "Range"
        idt.filter.list = [0.00001, 10]
        # ICP convergence threshold - minimum relative change in RMSE
        irt.value = 0.0001
        irt.filter.type = "Range"
        irt.filter.list = [0.0000001, 1]
        # Robust ICP option
        ir.value = True

        #         0    1    2    3    4    5    6    7    8    9     10   11   12   13   14   15
        params = [fnd, aoi, min, dss, iss, dsf, dwf, dat, dlr, drmi, drt, imi, iat, idt, irt, ir]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        if parameters[5].value and parameters[6].value:
            if parameters[5].value <= parameters[6].value:
                parameters[5].setErrorMessage("Strong filter size must be larger than weak filter size")
                parameters[6].setErrorMessage("Weak filter size must be smaller than large filter size")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        fnd_dir, fnd_file = os.path.split(parameters[0].valueAsText)
        aoi_dir, aoi_file = os.path.split(parameters[1].valueAsText)

        # Binding multiple volumes to the same directory is not good
        if fnd_dir == aoi_dir:
            volume = "/data"

            cmd = "modem"
            cmd += f' "{volume}/{fnd_file}"'
            cmd += f' "{volume}/{aoi_file}"'
            for parameter in parameters[2:]:
                cmd += f" --{parameter.name} {parameter.valueAsText}"

            volumes = {fnd_dir: {"bind": volume, "mode": "rw"}}
        else:
            fnd_volume = "/data/fnd"
            aoi_volume = "/data/aoi"

            cmd = "modem"
            cmd += f' "{fnd_volume}/{fnd_file}"'
            cmd += f' "{aoi_volume}/{aoi_file}"'
            for parameter in parameters[2:]:
                cmd += f" --{parameter.name} {parameter.valueAsText}"

            volumes={
                fnd_dir: {"bind": fnd_volume, "mode": "ro"},
                aoi_dir: {"bind": aoi_volume, "mode": "rw"}
            }

        arcpy.SetProgressor("step", "Registering AOI to Foundation", 0, 5)
        client = docker.from_env()
        container = client.containers.run(
            "modem:0.21",
            command=["bash", "-c", cmd],
            volumes=volumes,
            detach=True,
            remove=True,
        )

        # Hack the Docker stdout and stderr for progress bar and messaging
        output = container.attach(stdout=True, stream=True, logs=True)
        for line in output:
            temp = line.decode(errors="ignore")
            # Check for error
            if ("Traceback" in temp) or ("AssertionError" in temp):
                arcpy.AddError(temp)
            else:
                # Handle multiple lines
                for temp_line in temp.splitlines():
                    # Progress bar
                    if ("PREPROCESSING DATA" in temp_line):
                        arcpy.SetProgressorLabel("Step 1/4: Prepping AOI and Foundation Data")
                        arcpy.SetProgressorPosition()
                    if ("BEGINNING COARSE REGISTRATION" in temp_line):
                        arcpy.SetProgressorLabel("Step 2/4: Solving Coarse Registration")
                        arcpy.SetProgressorPosition()
                    if ("BEGINNING FINE REGISTRATION" in temp_line):
                        arcpy.SetProgressorLabel("Step 3/4: Solving Fine Registration")
                        arcpy.SetProgressorPosition()
                    if ("APPLYING REGISTRATION" in temp_line):
                        arcpy.SetProgressorLabel("Step 4/4: Applying Registration to AOI Data")
                        arcpy.SetProgressorPosition()

                    # Messaging
                    if ("- INFO -" in temp_line):
                        # Change paths to local
                        if ("/data/fnd" in temp_line):
                            idx = temp_line.find("/data/fnd")
                            temp_path = os.path.join(fnd_dir, temp_line[idx+10:])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        elif ("/data/aoi" in temp_line):
                            idx = temp_line.find("/data/aoi")
                            temp_path = os.path.join(aoi_dir, temp_line[idx+10:])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        elif ("/data" in temp_line):
                            idx = temp_line.find("/data")
                            temp_path = os.path.join(fnd_dir, temp_line[idx+6:])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        arcpy.AddMessage(temp_line)
                    if ("- WARNING -" in temp_line):
                        arcpy.AddWarning(temp_line)

        return
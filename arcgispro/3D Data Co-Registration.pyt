import os
import json
import arcpy
import itertools
import subprocess
import pdal
import codem
import dataclasses
import math


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "3D Data Co-Registration"
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
            direction="Input",
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
            category="Solve Scale",
        )
        iss = arcpy.Parameter(
            displayName="ICP Registration",
            name="icp_solve_scale",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="Solve Scale",
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
        params = [
            fnd,
            aoi,
            min,
            dss,
            iss,
            dsf,
            dwf,
            dat,
            dlr,
            drmi,
            drt,
            imi,
            iat,
            idt,
            irt,
            ir,
        ]
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
                parameters[5].setErrorMessage(
                    "Strong filter size must be larger than weak filter size"
                )
                parameters[6].setErrorMessage(
                    "Weak filter size must be smaller than large filter size"
                )
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

        arcpy.SetProgressor("step", "Registering AOI to Foundation", 0, 5)

        cmd = ["codem"]
        cmd.append(os.fsdecode(f"{parameters[0].valueAsText}").replace(os.sep, "/"))
        cmd.append(os.fsdecode(f"{parameters[1].valueAsText}").replace(os.sep, "/"))
        for parameter in parameters[2:]:
            cmd.append(f"--{parameter.name}")
            cmd.append(f"{parameter.valueAsText}")

        try:
            completed_process = subprocess.run(cmd, text=True, capture_output=True)
        except Exception:
            arcpy.AddError(f"{cmd.join(' ')} failed")
            raise

        output = completed_process.stdout.split("\n")
        reg_file = None
        for line in output:
            temp = line
            # Check for error
            if ("Traceback" in temp) or ("AssertionError" in temp):
                arcpy.AddError(temp)
            else:
                # Handle multiple lines
                for temp_line in temp.splitlines():
                    # Progress bar
                    if "PREPROCESSING DATA" in temp_line:
                        arcpy.SetProgressorLabel(
                            "Step 1/4: Prepping AOI and Foundation Data"
                        )
                        arcpy.SetProgressorPosition()
                    if "BEGINNING COARSE REGISTRATION" in temp_line:
                        arcpy.SetProgressorLabel(
                            "Step 2/4: Solving Coarse Registration"
                        )
                        arcpy.SetProgressorPosition()
                    if "BEGINNING FINE REGISTRATION" in temp_line:
                        arcpy.SetProgressorLabel("Step 3/4: Solving Fine Registration")
                        arcpy.SetProgressorPosition()
                    if "APPLYING REGISTRATION" in temp_line:
                        arcpy.SetProgressorLabel(
                            "Step 4/4: Applying Registration to AOI Data"
                        )
                        arcpy.SetProgressorPosition()

                    # Messaging
                    if "- INFO -" in temp_line:
                        # Change paths to local
                        if f"{fnd_dir}/fnd" in temp_line:
                            idx = temp_line.find(f"{fnd_dir}/fnd")
                            temp_path = os.path.join(fnd_dir, temp_line[idx + 10 :])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        elif f"{aoi_dir}/aoi" in temp_line:
                            idx = temp_line.find(f"{aoi_dir}/aoi")
                            temp_path = os.path.join(aoi_dir, temp_line[idx + 10 :])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        elif "/data" in temp_line:
                            idx = temp_line.find("/data")
                            temp_path = os.path.join(fnd_dir, temp_line[idx + 6 :])
                            temp_path = os.path.normpath(temp_path)
                            temp_line = temp_line[0:idx] + temp_path
                        arcpy.AddMessage(temp_line)
                    if "- WARNING -" in temp_line:
                        arcpy.AddWarning(temp_line)

                    # Output file location
                    if "Registration has been applied to" in temp_line:
                        idx = temp_line.find("registration_")
                        reg_file = os.path.join(aoi_dir, temp_line[idx:])
                        reg_file = os.path.normpath(reg_file)

        if reg_file is not None:
            _, ext = os.path.splitext(reg_file)
            if ext == ".tif":
                aprx = arcpy.mp.ArcGISProject("CURRENT")
                activeMap = aprx.activeMap
                arcpy.env.addOutputsToMap = True
                if activeMap is not None:
                    activeMap.addDataFromPath(reg_file)
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
            direction="Input",
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
            category="Solve Scale",
        )
        iss = arcpy.Parameter(
            displayName="ICP Registration",
            name="icp_solve_scale",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
            category="Solve Scale",
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
        # fnd.value = "E:\dev\codem\demo\Foundation-PointCloud.laz"
        fnd.filter.list = ["las", "laz", "bpf", "ply", "obj", "tif", "tiff"]
        # AOI data file
        # aoi.value = "E:\dev\codem\demo\AOI-Mesh.ply"
        aoi.filter.list = ["las", "laz", "bpf", "ply", "obj", "tif", "tiff"]

        # Minimum pipeline resolution
        # min.value = 2.0
        min.value = 1.0
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

        params = [
            fnd,  # 0
            aoi,  # 1
            min,  # 2
            dss,  # 3
            iss,  # 4
            dsf,  # 5
            dwf,  # 6
            dat,  # 7
            dlr,  # 8
            drmi,  # 9
            drt,  # 10
            imi,  # 11
            iat,  # 12
            idt,  # 13
            irt,  # 14
            ir,  # 15
        ]
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

        # Strong and weak filter size check
        if (
            parameters[5].value
            and parameters[6].value
            and parameters[5].value <= parameters[6].value
        ):
            parameters[5].setErrorMessage(
                "Strong filter size must be larger than weak filter size"
            )
            parameters[6].setErrorMessage(
                "Weak filter size must be smaller than large filter size"
            )

        # Check if input DEMs have equal X and Y cell size values

        # first make sure both params are input
        if parameters[0].value and parameters[1].value:
            # get path
            fnd_full_path = os.fsdecode(f"{parameters[0].valueAsText}").replace(
                os.sep, "/"
            )
            aoi_full_path = os.fsdecode(f"{parameters[1].valueAsText}").replace(
                os.sep, "/"
            )

            inputs_list = [fnd_full_path, aoi_full_path]
            # check for both FND and AOI
            for index, input_file in enumerate(inputs_list):
                # analysis can only be done with raster/DEM input
                if os.path.splitext(input_file)[-1] in {".tif", ".tiff"}:
                    # see number of bands in raster (Valid DEMs only have 1)
                    raster_description = arcpy.Describe(input_file)

                    # set warning (tool can still be run) if more than one band
                    if raster_description.bandCount != 1:
                        parameters[index].setWarningMessage(
                            "Warning: Input DEM has more than one band in "
                            f"{os.path.basename(input_file)}. "
                            "The tool will not run properly with the "
                            "input data as is. Consider regenerating input DEM"
                        )

                    # need to access detail of Band1 (or only band for DEMs)

                    # first get band name
                    arcpy.env.workspace = input_file
                    bands_list = arcpy.ListRasters()
                    # join only band to get band description
                    # Refer to code sample for accessing Raster Band Properties: https://pro.arcgis.com/en/pro-app/2.9/arcpy/functions/raster-band-properties.htm
                    band_description = arcpy.Describe(
                        os.path.join(input_file, bands_list[0])
                    )

                    # if Y does not equal X, it can't happen!
                    if not math.isclose(
                        band_description.meanCellHeight,
                        band_description.meanCellWidth,
                        abs_tol=1e-5,
                    ):
                        parameters[index].setErrorMessage(
                            "Error: X and Y cell sizes are not equal in "
                            f"{os.path.basename(input_file)}. "
                            "The tool will not run with the input data as is. "
                            "Consider reprojecting input DEM"
                            f" X = {band_description.meanCellWidth}, Y = {band_description.meanCellHeight}"
                        )

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        fnd_dir, fnd_file = os.path.split(parameters[0].valueAsText)
        aoi_dir, aoi_file = os.path.split(parameters[1].valueAsText)

        fnd_full_path = os.fsdecode(f"{parameters[0].valueAsText}").replace(os.sep, "/")
        aoi_full_path = os.fsdecode(f"{parameters[1].valueAsText}").replace(os.sep, "/")
        aoi_file_extension = os.path.splitext(aoi_full_path)[-1]

        dsm_filetypes = codem.lib.resources.dsm_filetypes
        pcloud_filetypes = codem.lib.resources.pcloud_filetypes
        mesh_filetypes = codem.lib.resources.mesh_filetypes

        # create mapping of codem supported inputs vs. ArcGIS supported outputs
        mapping = {dsm_filetype: ".tif" for dsm_filetype in dsm_filetypes}
        for pcloud_filetype in pcloud_filetypes:
            mapping[pcloud_filetype] = ".las"
        for mesh_filetype in mesh_filetypes:
            mapping[mesh_filetype] = ".obj"

        arcpy.SetProgressor("step", "Registering AOI to Foundation", 0, 5)

        kwargs = {
            parameter.name.upper(): parameter.value for parameter in parameters[2:]
        }
        codem_run_config = codem.CodemRunConfig(fnd_full_path, aoi_full_path, **kwargs)
        config = dataclasses.asdict(codem_run_config)

        arcpy.SetProgressorLabel("Step 1/4: Prepping AOI and Foundation Data")
        arcpy.SetProgressorPosition()
        fnd_obj, aoi_obj = codem.preprocess(config)

        fnd_obj.prep()
        aoi_obj.prep()

        arcpy.SetProgressorLabel("Step 2/4: Solving Coarse Registration")
        arcpy.SetProgressorPosition()
        dsm_reg = codem.coarse_registration(fnd_obj, aoi_obj, config)

        arcpy.SetProgressorLabel("Step 3/4: Solving Fine Registration")
        arcpy.SetProgressorPosition()
        icp_reg = codem.fine_registration(fnd_obj, aoi_obj, dsm_reg, config)

        arcpy.SetProgressorLabel("Step 4/4: Applying Registration to AOI Data")
        arcpy.SetProgressorPosition()
        reg_file = codem.apply_registration(
            fnd_obj, aoi_obj, icp_reg, config, output_format=mapping[aoi_file_extension]
        )

        if not os.path.exists(reg_file):
            arcpy.AddError("Registration file not generated")
            return None
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        activeMap = aprx.activeMap
        arcpy.env.addOutputsToMap = True
        if activeMap is None:
            arcpy.AddWarning("activeMap is None")
        elif aoi_file_extension in mesh_filetypes:
            arcpy.AddWarning(
                f"File type {aoi_file_extension} cannot be visualized in ArcGIS Pro. "
                "Consider converting AOI or visualizing in other software."
            )
        else:
            activeMap.addDataFromPath(reg_file)
            arcpy.AddMessage(f"ActiveMap added {aoi_file_extension} file")
        return None

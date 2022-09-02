# -*- coding: utf-8 -*-
r""""""
__all__ = ["Register_DEM2DEM", "Register_MultiType"]
__alias__ = "3d_registration"
from arcpy.geoprocessing._base import gptooldoc, gp, gp_fixargs
from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject

# Tools
@gptooldoc("Register_DEM2DEM_3d_registration", None)
def Register_DEM2DEM(
    foundation_file=None,
    aoi_file=None,
    min_resolution=None,
    dsm_solve_scale=None,
    icp_solve_scale=None,
    dsm_strong_filter=None,
    dsm_weak_filter=None,
    dsm_akaze_threshold=None,
    dsm_lowes_ratio=None,
    dsm_ransac_max_iter=None,
    dsm_ransac_threshold=None,
    icp_max_iter=None,
    icp_angle_threshold=None,
    icp_distance_threshold=None,
    icp_rmse_threshold=None,
    icp_robust=None,
):
    """Register_DEM2DEM(foundation_file, aoi_file, min_resolution, dsm_solve_scale, icp_solve_scale, dsm_strong_filter, dsm_weak_filter, dsm_akaze_threshold, dsm_lowes_ratio, dsm_ransac_max_iter, dsm_ransac_threshold, icp_max_iter, icp_angle_threshold, icp_distance_threshold, icp_rmse_threshold, icp_robust)

    INPUTS:
     foundation_file (Raster Layer / File):
         Foundation DEM layer or path to foundation data file.
     aoi_file (Raster Layer / File):
         Area of interest DEM layer or path to area of interest data file.
     min_resolution (Double):
         Minimum allowable registration pipeline data resolution. Smaller
         values increase accuracy, but also computation time.
     dsm_solve_scale (Boolean):
         Flag to include or exclude scale from the solved coarse
         registration transformation.
     icp_solve_scale (Boolean):
         Flag to include or exclude scale from the solved fine registration
         transformation.
     dsm_strong_filter (Double):
         Standard deviation of the large Gaussian filter used to normalize
         the DSM prior to feature extraction. Larger values allow longer
         wavelength vertical features to pass through into the normalized DSM.
     dsm_weak_filter (Double):
         Standard deviation of the small Gaussian filter used to normalize
         the DSM prior to feature extraction. Larger values increasingly blur
         short wavelength vertical features in the normalized DSM.
     dsm_akaze_threshold (Double):
         AKAZE feature detection response threshold. Larger values require
         increasingly distinctive local geometry for a feature to be detected.
     dsm_lowes_ratio (Double):
         Feature matching relative strength control. Larger values allow
         weaker matches relative to the next best match.
     dsm_ransac_max_iter (Long):
         Maximum number of iterations allowed in the RANSAC algorithm that
         filters the putative feature matches.
     dsm_ransac_threshold (Double):
         Maximum residual error for a matched feature pair to be included in
         a RANSAC solution to a 3D registration transformation. Larger values
         include matched feature pairs with increasingly greater disagreement
         with the solution.
     icp_max_iter (Long):
         Maximum number of iterations allowed in the iterative closest point
         (ICP) algorithm
     icp_angle_threshold (Double):
         ICP convergence criterion: minimum change in Euler angle between
         iterations.
     icp_distance_threshold (Double):
         ICP convergence criterion: minimum change in translation between
         iterations.
     icp_rmse_threshold (Double):
         ICP convergence criterion: minimum relative change between
         iterations in the root mean squared error.
     icp_robust (Boolean):
         Flag to include or exclude robust weighting in the fine registration
         solution."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject

    try:
        retval = convertArcObjectToPythonObject(
            gp.Register_DEM2DEM_3d_registration(
                *gp_fixargs(
                    (
                        foundation_file,
                        aoi_file,
                        min_resolution,
                        dsm_solve_scale,
                        icp_solve_scale,
                        dsm_strong_filter,
                        dsm_weak_filter,
                        dsm_akaze_threshold,
                        dsm_lowes_ratio,
                        dsm_ransac_max_iter,
                        dsm_ransac_threshold,
                        icp_max_iter,
                        icp_angle_threshold,
                        icp_distance_threshold,
                        icp_rmse_threshold,
                        icp_robust,
                    ),
                    True,
                )
            )
        )
        return retval
    except Exception as e:
        raise e


@gptooldoc("Register_MultiType_3d_registration", None)
def Register_MultiType(
    foundation_file=None,
    aoi_file=None,
    min_resolution=None,
    dsm_solve_scale=None,
    icp_solve_scale=None,
    dsm_strong_filter=None,
    dsm_weak_filter=None,
    dsm_akaze_threshold=None,
    dsm_lowes_ratio=None,
    dsm_ransac_max_iter=None,
    dsm_ransac_threshold=None,
    icp_max_iter=None,
    icp_angle_threshold=None,
    icp_distance_threshold=None,
    icp_rmse_threshold=None,
    icp_robust=None,
):
    """Register_MultiType(foundation_file, aoi_file, min_resolution, dsm_solve_scale, icp_solve_scale, dsm_strong_filter, dsm_weak_filter, dsm_akaze_threshold, dsm_lowes_ratio, dsm_ransac_max_iter, dsm_ransac_threshold, icp_max_iter, icp_angle_threshold, icp_distance_threshold, icp_rmse_threshold, icp_robust)

    INPUTS:
     foundation_file (File):
         Path to foundation data file.
     aoi_file (File):
         Path to area of interest data file.
     min_resolution (Double):
         Minimum allowable registration pipeline data resolution. Smaller
         values increase computation time.
     dsm_solve_scale (Boolean):
         Flag to include or exclude scale from the solved coarse
         registration transformation.
     icp_solve_scale (Boolean):
         Flag to include or exclude scale from the solved fine registration
         transformation.
     dsm_strong_filter (Double):
         Standard deviation of the large Gaussian filter used to normalize
         the DSM prior to feature extraction. Larger values allow longer
         wavelength vertical features to pass through into the normalized DSM.
     dsm_weak_filter (Double):
         Standard deviation of the small Gaussian filter used to normalize
         the DSM prior to feature extraction. Larger values increasingly blur
         short wavelength vertical features in the normalized DSM.
     dsm_akaze_threshold (Double):
         AKAZE feature detection response threshold. Larger values require
         increasingly distinctive local geometry for a feature to be detected.
     dsm_lowes_ratio (Double):
         Feature matching relative strength control. Larger values allow
         weaker matches relative to the next best match.
     dsm_ransac_max_iter (Long):
         Maximum number of iterations allowed in the RANSAC algorithm that
         filters the putative feature matches.
     dsm_ransac_threshold (Double):
         Maximum residual error for a matched feature pair to be included in
         a RANSAC solution to a 3D registration transformation. Larger values
         include matched feature pairs with increasingly greater disagreement
         with the solution.
     icp_max_iter (Long):
         Maximum number of iterations allowed in the iterative closest point
         (ICP) algorithm
     icp_angle_threshold (Double):
         ICP convergence criterion: minimum change in Euler angle between
         iterations.
     icp_distance_threshold (Double):
         ICP convergence criterion: minimum change in translation between
         iterations.
     icp_rmse_threshold (Double):
         ICP convergence criterion: minimum relative change between
         iterations in the root mean squared error.
     icp_robust (Boolean):
         Flag to include or exclude robust weighting in the fine registration
         solution."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject

    try:
        retval = convertArcObjectToPythonObject(
            gp.Register_MultiType_3d_registration(
                *gp_fixargs(
                    (
                        foundation_file,
                        aoi_file,
                        min_resolution,
                        dsm_solve_scale,
                        icp_solve_scale,
                        dsm_strong_filter,
                        dsm_weak_filter,
                        dsm_akaze_threshold,
                        dsm_lowes_ratio,
                        dsm_ransac_max_iter,
                        dsm_ransac_threshold,
                        icp_max_iter,
                        icp_angle_threshold,
                        icp_distance_threshold,
                        icp_rmse_threshold,
                        icp_robust,
                    ),
                    True,
                )
            )
        )
        return retval
    except Exception as e:
        raise e


# End of generated toolbox code
del gptooldoc, gp, gp_fixargs, convertArcObjectToPythonObject

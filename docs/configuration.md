# Configuring CODEM

## Overview
Numerous algorithm parameters can be tuned when running `CODEM` by specifying option names and values. The available options can be viewed by running `codem --help`:

```
$ codem --help
usage: codem [-h] [--min_resolution MIN_RESOLUTION] [--dsm_akaze_threshold DSM_AKAZE_THRESHOLD]
             [--dsm_lowes_ratio DSM_LOWES_RATIO] [--dsm_ransac_max_iter DSM_RANSAC_MAX_ITER]
             [--dsm_ransac_threshold DSM_RANSAC_THRESHOLD] [--dsm_solve_scale DSM_SOLVE_SCALE]
             [--dsm_strong_filter DSM_STRONG_FILTER] [--dsm_weak_filter DSM_WEAK_FILTER]
             [--icp_angle_threshold ICP_ANGLE_THRESHOLD] [--icp_distance_threshold ICP_DISTANCE_THRESHOLD]
             [--icp_max_iter ICP_MAX_ITER] [--icp_rmse_threshold ICP_RMSE_THRESHOLD]
             [--icp_robust ICP_ROBUST] [--icp_solve_scale ICP_SOLVE_SCALE] [--verbose VERBOSE]
             foundation_file aoi_file
```

In most cases, the default parameter values (see below) are sufficient. 


## Configuration Parameters
Option values must adhere to the following limits and data types:

**Coarse, Feature-Based Registration Parameters:**
* `DSM_STRONG_FILTER`
    * description: standard deviation of the large Gaussian filter used to normalize the DSM prior to feature extraction; larger values allow longer wavelength vertical features to pass through into the normalized DSM
    * command line argument: `-dst` or `--dsm_strong_filter`
    * units: meters
    * dtype: `float`
    * limits: `x > 0.0`
    * default: `10`
* `DSM_WEAK_FILTER`
    * description: standard deviation of the small Gaussian filter used to normalize the DSM prior to feature extraction; larger values increasingly blur short wavelength vertical features in the normalized DSM
    * command line argument: `dwf` or `--dsm_weak_filter`
    * units: meters
    * dtype: `float`
    * limits: `x > 0.0`
    * default: `1`
* `DSM_AKAZE_THRESHOLD`
    * description: [Accelerated-KAZE](http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf) feature detection response threshold; larger values require increasingly distinctive local geometry for a feature to be detected
    * command line argument: `-dat` or `--dsm_akaze_threshold`
    * units: none
    * dtype: `float`
    * limits: `x > 0.0`
    * default: `0.0001`
* `DSM_LOWES_RATIO`
    * description: feature matching relative strength control; larger values allow weaker matches relative to the next best match
    * command line argument: `-dlr` or `--dsm_lowes_ratio`
    * units: none
    * dtype: `float`
    * limits: `0.0 < x < 1.0`
    * default: `0.9`
* `DSM_RANSAC_THRESHOLD`
    * description: maximum residual error for a matched feature pair to be included in a random sample consensus (RANSAC) solution to a 3D registration transformation; larger values include matched feature pairs with increasingly greater disagreement with the solution
    * command line argument: `-drt` or `--dsm_ransac_threshold`
    * units: meters
    * dtype: `float`
    * limits: `x > 0`
    * default: `10`
* `DSM_RANSAC_MAX_ITER`
    * description: the max iterations for the RANSAC algorithm
    * command line argument: `-drmi` or `--dsm_ransac_max_iter`
    * units: iterations
    * dtype: `int`
    * limits: `x > 0`
    * default: `10000`
* `DSM_SOLVE_SCALE`
    * description: flag to include or exclude scale from the solved coarse registration transformation
    * command line argument: `-dss` or `--dsm_solve_scale`
    * units: N/A
    * dtype: `bool`
    * limits: `True` or `False`
    * default: `True`

**Fine, ICP-Based Registration Parameters:**
* `ICP_MAX_ITER`
    * description: the max iterations for the iterative closest point algorithm
    * command line argument: `-imi` or `--icp_max_iter`
    * units: iterations
    * dtype: `int`
    * limits: `x > 0`
    * default: `100`
* `ICP_RMSE_THRESHOLD`
    * description: ICP convergence criterion; minimum relative change between iterations in the root mean squared error
    * command line argument: `-irt` or `--icp_rmse_threshold`
    * units: meters
    * dtype: `float`
    * limits: `x > 0`
    * default: `0.0001`
* `ICP_ANGLE_THRESHOLD`
    * description: ICP convergence criterion; minimum change in Euler angle between iterations
    * command line argument: `-iat` or `--icp_angle_threshold`
    * units: degrees
    * dtype: `float`
    * limits: `x > 0`
    * default: `0.001`
* `ICP_DISTANCE_THRESHOLD`
    * description: ICP convergence criterion; minimum change in translation between iterations
    * command line argument: `-idt` or `--icp_distance_threshold`
    * units: meters
    * dtype: `float`
    * limits: `x > 0`
    * default: `0.001`
* `ICP_SOLVE_SCALE`
    * description: flag to include or exclude scale from the solved fine registration transformation
    * command line argument: `-iss` or `--icp_solve_scale`
    * units: N/A
    * dtype: `bool`
    * limits: `True` or `False`
    * default: `True`
* `ICP_ROBUST`
    * description: flag to include or exclude robust weighting in the fine registration solution
    * command line argument: `-ir` or `--icp_robust`
    * units: N/A
    * dtype: `bool`
    * limits: `True` or `False`
    * default: `True`

**Other Parameters:**
* `MIN_RESOLUTION`
    * description: the minimum pipeline data resolution; smaller values increase computation time
    * command line argument: `-min` or `--min_resolution`
    * units: meters
    * dtype: `float`
    * limits: `x > 0`
    * default: `1.0`
* `VERBOSE`
    * description: flag to output verbose logging information to the console
    * command line argument: `-v` or `--verbose`
    * units: N/A
    * dtype: `bool`
    * limits: `True` or `False`
    * default: `False`



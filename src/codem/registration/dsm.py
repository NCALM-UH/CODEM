"""
DsmRegistration.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

This module contains classes to co-register pre-processed DSM data. Features are
extracted from two DSMs, matched, and the registration transformation solved
from the matched features.

This module contains the following classes:

* DsmRegistration: a class for DSM to DSM registration
* Unscaled3dSimilarityTransform: a class for solving the 6-parameter
  transformation (no scale factor) between two sets of 3D points.
* Scaled3dSimilarityTransform: a class for solving the 7-parameter
  transformation (includes a scale factor) between two sets of 3D points.
"""
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import numpy as np
from codem.preprocessing.preprocess import GeoData
from rasterio import Affine
from skimage.measure import ransac


class DsmRegistration:
    """
    A class to solve the transformation between two Digital Surface Models.
    Uses a feature matching approach.

    Parameters
    ----------
    fnd_obj: DSM object
        the foundation DSM
    aoi_obj: DSM object
        the area of interest DSM
    config: Dictionary
        dictionary of configuration parameters

    Methods
    --------
    register
    _get_kp
    _get_putative
    _filter_putative
    _save_match_img
    _get_geo_coords
    _get_rmse
    _output
    """

    def __init__(
        self, fnd_obj: GeoData, aoi_obj: GeoData, config: Dict[str, Any]
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.fnd_obj = fnd_obj
        self.aoi_obj = aoi_obj
        self._putative_matches: List[cv2.DMatch] = []

        if not aoi_obj.processed:
            raise RuntimeError(
                "AOI data has not been pre-processed, did you call the prep method?"
            )
        if not fnd_obj.processed:
            raise RuntimeError(
                "Foundation data has not been pre-processed, did you call the prep method?"
            )

    @property
    def putative_matches(self) -> List[cv2.DMatch]:
        return self._putative_matches

    @putative_matches.setter
    def putative_matches(self, matches: List[cv2.DMatch]) -> None:
        if len(matches) < 4:
            raise RuntimeError(
                (
                    f"{len(matches)} putative keypoint matches found (4 required). "
                    "Consider modifying DSM_LOWES_RATIO, current value: "
                    f"{self.config['DSM_LOWES_RATIO']}"
                )
            )
        self._putative_matches = matches

    def register(self) -> None:
        """
        Performs DSM co-registration via the following steps:
        * Extract features from foundation and AOI DSMs
        * Generate putative matches via nearest neighbor in descriptor space
        * Filter putative matches for those that conform to a common
        transformation

        After registration, RMSEs of matched features are computed and
        transformation details added as a class attribute.
        """
        self.logger.info("Solving DSM feature registration.")

        self.fnd_kp, self.fnd_desc = self._get_kp(
            self.fnd_obj.normed, self.fnd_obj.nodata_mask
        )
        self.logger.debug(f"{len(self.fnd_kp)} keypoints detected in foundation")
        if len(self.fnd_kp) < 4:
            raise RuntimeError(
                (
                    f"{len(self.fnd_kp)} keypoints were identiifed in the Foundation data"
                    " (4 required).  Consider modifying the DSM_AKAZE_THRESHOLD parameter,"
                    f" current value is {self.config['DSM_AKAZE_THRESHOLD']}"
                )
            )

        self.aoi_kp, self.aoi_desc = self._get_kp(
            self.aoi_obj.normed, self.aoi_obj.nodata_mask
        )
        self.logger.debug(f"{len(self.aoi_kp)} keypoints detected in area of interest")
        if len(self.aoi_kp) < 4:
            raise RuntimeError(
                (
                    f"{len(self.aoi_kp)} keypoints were identiifed in the "
                    "AOI data (4 required).  Consider modifying the DSM_AKAZE_THRESHOLD"
                    f"parameter, current value is {self.config['DSM_AKAZE_THRESHOLD']}"
                )
            )
        self._get_putative()
        self._filter_putative()
        self._save_match_img()

        self._get_rmse()
        self._output()

    def _get_kp(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[Tuple[cv2.KeyPoint, ...], np.ndarray]:
        """
        Extracts AKAZE features, in the form of keypoints and descriptors,
        from an 8-bit grayscale image.

        Parameters
        ----------
        img: np.array
            Normalized 8-bit grayscale image
        mask: np.array
            Mask of valid locations for img feature extraction

        Returns
        ----------
        kp: tuple(cv2.KeyPoint,...)
            OpenCV keypoints
        desc: np.array
            OpenCV AKAZE descriptors
        """
        detector = cv2.AKAZE_create(threshold=self.config["DSM_AKAZE_THRESHOLD"])
        kp, desc = detector.detectAndCompute(img, np.ones(mask.shape, dtype=np.uint8))
        return kp, desc

    def _get_putative(self) -> None:
        """
        Identifies putative matches for DSM co-registration via a nearest
        neighbor search in descriptor space. When very large numbers of
        descriptors exist, an approximate matching method is used to prevent
        failure. References regarding this problem:
        * https://answers.opencv.org/question/85003/why-there-is-a-hardcoded-maximum-numbers-of-descriptors-that-can-be-matched-with-bfmatcher/
        * https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        * https://luckytaylor.top/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html
        """
        if self.aoi_desc.shape[0] > 2**17 or self.fnd_desc.shape[0] > 2**17:
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,  # 12
                key_size=12,  # 20
                multi_probe_level=1,  # 2
            )
            desc_matcher = cv2.FlannBasedMatcher(index_params, {})
        else:
            desc_matcher = cv2.DescriptorMatcher_create(
                cv2.DescriptorMatcher_BRUTEFORCE_HAMMING
            )

        # Fnd = train; AOI = query; knnMatch parameter order is query, train
        knn_matches = desc_matcher.knnMatch(self.aoi_desc, self.fnd_desc, k=2)

        # Lowe's ratio test to filter weak matches
        good_matches = [
            m
            for m, n in knn_matches
            if m.distance < self.config["DSM_LOWES_RATIO"] * n.distance
        ]

        self.logger.debug(f"{len(good_matches)} putative keypoint matches found.")
        self.putative_matches = good_matches

    def _filter_putative(self) -> None:
        """
        Filters putative matches via conformance to a 3D similarity transform.
        Note that the fnd_uv and aoi_uv coordinates are relative to OpenCV's
        AKAZE feature detection origin at the center of the upper left pixel.
        https://github.com/opencv/opencv/commit/e646f9d2f1b276991a59edf01bc87dcdf28e2b8f
        """
        # Get 2D image space coords of putative keypoint matches
        fnd_uv = np.array(
            [self.fnd_kp[m.trainIdx].pt for m in self.putative_matches],
            dtype=np.float32,
        )
        aoi_uv = np.array(
            [self.aoi_kp[m.queryIdx].pt for m in self.putative_matches],
            dtype=np.float32,
        )

        # Get 3D object space coords of putative keypoint matches
        fnd_xyz = self._get_geo_coords(
            fnd_uv,
            self.fnd_obj.transform,
            self.fnd_obj.area_or_point,
            self.fnd_obj.infilled,
        )
        aoi_xyz = self._get_geo_coords(
            aoi_uv,
            self.aoi_obj.transform,
            self.aoi_obj.area_or_point,
            self.aoi_obj.infilled,
        )
        # Find 3D similarity transform conforming to max number of matches
        if self.config["DSM_SOLVE_SCALE"]:
            model, inliers = ransac(
                (aoi_xyz, fnd_xyz),
                Scaled3dSimilarityTransform,
                min_samples=3,
                residual_threshold=self.config["DSM_RANSAC_THRESHOLD"],
                max_trials=self.config["DSM_RANSAC_MAX_ITER"],
            )
        else:
            model, inliers = ransac(
                (aoi_xyz, fnd_xyz),
                Unscaled3dSimilarityTransform,
                min_samples=3,
                residual_threshold=self.config["DSM_RANSAC_THRESHOLD"],
                max_trials=self.config["DSM_RANSAC_MAX_ITER"],
            )
        if model is None:
            raise ValueError(
                "ransac model not fitted, no inliers found. Consider tuning "
                "DSM_RANSAC_THRESHOLD or DSM_RANSAC_MAX_ITER"
            )
        self.logger.info(f"{np.sum(inliers)} keypoint matches found.")
        assert np.sum(inliers) >= 4, "Less than four keypoint matches found."

        T: np.ndarray = model.transform
        c = np.linalg.norm(T[:, 0])
        assert (
            c > 0.67 and c < 1.5
        ), "Solved scale difference between foundation and AOI data exceeds 50%"

        self.transformation = T
        self.inliers = inliers
        self.fnd_inliers_xyz = fnd_xyz[inliers]
        self.aoi_inliers_xyz = aoi_xyz[inliers]

    def _save_match_img(self) -> None:
        """
        Save image of matched features with connecting lines on the
        foundation and AOI DSM images. The outline of the transformed AOI
        boundary is also plotted on the foundation DSM image.
        """
        if self.fnd_obj.transform is None:
            raise RuntimeError(
                "Foundation Object transform has not been set, did you run the prep() method?"
            )
        h, w = self.aoi_obj.normed.shape
        aoi_uv = np.array(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32
        )

        aoi_xyz = self._get_geo_coords(
            aoi_uv,
            self.aoi_obj.transform,
            self.aoi_obj.area_or_point,
            self.aoi_obj.infilled,
        )
        aoi_xyz = np.vstack((aoi_xyz.T, np.ones((1, 4)))).T

        fnd_xyz = (self.transformation @ aoi_xyz.T).T
        fnd_xy = np.vstack((fnd_xyz[:, 0:2].T, np.ones((1, 4)))).T

        F_inverse = np.reshape(np.asarray(~self.fnd_obj.transform), (3, 3))
        fnd_uv = (F_inverse @ fnd_xy.T).T
        fnd_uv = fnd_uv[:, 0:2]

        # Draw AOI outline onto foundation DSM
        fnd_prep = self.fnd_obj.normed.copy()
        fnd_prep = cv2.polylines(
            fnd_prep,
            np.expand_dims(fnd_uv, axis=0).astype(np.int32),
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Draw keypoints and match lines onto AOI and foundation DSMs
        kp_lines = cv2.drawMatches(
            self.aoi_obj.normed,
            self.aoi_kp,
            fnd_prep,
            self.fnd_kp,
            self.putative_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=self.inliers.astype(int).tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        output_file = os.path.join(self.config["OUTPUT_DIR"], "dsm_feature_matches.png")
        self.logger.info(f"Saving DSM feature match visualization to: {output_file}")
        cv2.imwrite(output_file, kp_lines)

    def _get_geo_coords(
        self, uv: np.ndarray, transform: Affine, area_or_point: str, dsm: np.ndarray
    ) -> np.ndarray:
        """
        Converts image space coordinates to object space coordinates. The passed
        uv coordinates must be relative to an origin at the center of the upper
        left pixel. These coordinates are correct for querying the DSM array for
        elevations. However, they need to be adjusted by 0.5 pixel when
        generating horizontal coordinates if the DSM transform is defined with
        an origin at the upper left of the upper left pixel.

        Parameters
        ----------
        uv: np.array
            Image space coordinates relative to an origin at the center of the
            upper left pixel (u=column, v=row)
        transform: affine.Affine
            Image space to object space 2D homogenous transformation matrix
        area_or_point: str
            "Area" or "Point" string indicating if the transform assumes the
            coordinate origin is at the upper left of the upper left pixel or at
            the center of the upper left pixel
        dsm: np.array
            The DSM elevation data

        Returns
        -------
        xyz: np.array
            Geospatial coordinates
        """
        z = []
        for cr in uv:
            cr_int = np.int32(np.rint(cr))
            z.append(dsm[cr_int[1], cr_int[0]])

        # When using the transform to convert from pixels to object space
        # coordinates, we need to increment the pixel coordinates by 0.5 if the
        # transform origin is defined at  the upper left corner of the upper
        # left pixel
        if area_or_point == "Area":
            uv += 0.5
        xy = []
        for cr in uv:
            temp = transform * (cr)
            xy.append([temp[0], temp[1]])

        z_ = np.asarray(z)
        xy_ = np.asarray(xy)
        return np.vstack((xy_.T, z_)).T

    def _get_rmse(self) -> None:
        """
        Calculates root mean squared error between the geospatial locations of
        the matched foundation and aoi features used in the registration.
        """
        fnd_xyz = self.fnd_inliers_xyz
        aoi_xyz = self.aoi_inliers_xyz
        X = self.transformation

        aoi_xyz = np.hstack((aoi_xyz, np.ones((aoi_xyz.shape[0], 1))))
        aoi_xyz_transformed = (X @ aoi_xyz.T).T
        aoi_xyz_transformed = aoi_xyz_transformed[:, 0:3]

        self.rmse_xyz = np.sqrt(
            np.sum((fnd_xyz - aoi_xyz_transformed) ** 2, axis=0) / fnd_xyz.shape[0]
        )
        self.rmse_3d = np.sqrt(np.sum(self.rmse_xyz**2))

    def _output(self) -> None:
        """
        Stores registration results in a dictionary and writes them to a file
        """
        X = self.transformation
        R = X[0:3, 0:3]
        c = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2 + R[2, 0] ** 2)
        omega = np.rad2deg(np.arctan2(R[2, 1] / c, R[2, 2] / c))
        phi = np.rad2deg(-np.arcsin(R[2, 0] / c))
        kappa = np.rad2deg(np.arctan2(R[1, 0] / c, R[0, 0] / c))
        tx = X[0, 3]
        ty = X[1, 3]
        tz = X[2, 3]

        self.registration_parameters = {
            "matrix": X,
            "omega": omega,
            "phi": phi,
            "kappa": kappa,
            "trans_x": tx,
            "trans_y": ty,
            "trans_z": tz,
            "scale": c,
            "n_pairs": self.fnd_inliers_xyz.shape[0],
            "rmse_x": self.rmse_xyz[0],
            "rmse_y": self.rmse_xyz[1],
            "rmse_z": self.rmse_xyz[2],
            "rmse_3d": self.rmse_3d,
        }

        output_file = os.path.join(self.config["OUTPUT_DIR"], "registration.txt")

        self.logger.info(
            f"Saving DSM feature registration parameters to: {output_file}"
        )
        with open(output_file, "w") as f:

            f.write("DSM FEATURE REGISTRATION")
            f.write("\n------------------------")
            f.write(f"\nTransformation matrix: \n {X}")
            f.write("\nTransformation Parameters:")
            f.write(f"\nOmega = {omega:.3f} degrees")
            f.write(f"\nPhi = {phi:.3f} degrees")
            f.write(f"\nKappa = {kappa:.3f} degrees")
            f.write(f"\nX Translation = {tx:.3f}")
            f.write(f"\nY Translation = {ty:.3f}")
            f.write(f"\nZ Translation = {tz:.3f}")
            f.write(f"\nScale = {c:.6f}")
            f.write(f"\nNumber of pairs = {self.registration_parameters['n_pairs']}")
            f.write("\nRMSEs:")
            f.write(
                f"\nX = +/-{self.rmse_xyz[0]:.3f},"
                f"\nY = +/-{self.rmse_xyz[1]:.3f},"
                f"\nZ = +/-{self.rmse_xyz[2]:.3f},"
                f"\n3D = +/-{self.rmse_3d:.3f}"
            )
            f.write("\n\n")


class Scaled3dSimilarityTransform:
    """
    A class for solving a 3D similarity transformation between two sets of 3D
    points. For use with scikit-image's RANSAC routing. The _umeyama method is
    from: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py
    """

    def __init__(self) -> None:
        self.solve_scale = True

    def estimate(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """
        Function to estimate the least squares transformation between the source
        (src) and destination (dst) 3D point pairs.

        Parameters
        ----------
        src: np.array
            Array of points to be transformed to the fixed dst point locations
        dst: np.array
            Array of fixed points ordered to correspond to the src points

        Returns
        -------
        success: bool
            True if transformation was able to be solved
        """

        self.transform = self._umeyama(src, dst, self.solve_scale)
        return True

    def residuals(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Function to compute residual distance between each point pair after
        registration

        Parameters
        ----------
        src: np.array
            Array of points transformed to the fixed dst point locations
        dst: np.array
            Array of fixed points ordered to correspond to the src points

        Returns
        -------
        residuals: np.array
            Residual for each point pair
        """
        src = np.hstack((src, np.ones((src.shape[0], 1))))
        src_transformed = (self.transform @ src.T).T
        src_transformed = src_transformed[:, 0:3]
        residuals: np.ndarray = np.sqrt(np.sum((src_transformed - dst) ** 2, axis=1))
        return residuals

    def _umeyama(
        self, src: np.ndarray, dst: np.ndarray, estimate_scale: bool
    ) -> np.ndarray:
        """
        Estimate N-D similarity transformation with or without scaling.

        Parameters
        ----------
        src : (M, N) array
            Source coordinates.
        dst : (M, N) array
            Destination coordinates.
        estimate_scale : bool
            Whether to estimate scaling factor.

        Returns
        -------
        T : (N + 1, N + 1)
            The homogeneous similarity transformation matrix. The matrix contains
            NaN values only if the problem is not well-conditioned.

        References
        ----------
        .. [1] "Least-squares estimation of transformation parameters between two
                point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
        """

        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38).
        A = dst_demean.T @ src_demean / num

        # Eq. (39).
        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T: np.ndarray = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.full_like(T, np.nan)
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d) if estimate_scale else 1.0
        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale
        return T


class Unscaled3dSimilarityTransform:
    """
    A class for solving a 3D similarity transformation between two sets of 3D
    points. For use with scikit-image's RANSAC routing. The _umeyama method is
    from: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py
    """

    def __init__(self) -> None:
        self.solve_scale = False

    def estimate(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """
        Function to estimate the least squares transformation between the source
        (src) and destination (dst) 3D point pairs.

        Parameters
        ----------
        src: np.array
            Array of points to be transformed to the fixed dst point locations
        dst: np.array
            Array of fixed points ordered to correspond to the src points

        Returns
        -------
        success: bool
            True if transformation was able to be solved
        """

        self.transform = self._umeyama(src, dst, self.solve_scale)
        return True

    def residuals(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Function to compute residual distance between each point pair after
        registration

        Parameters
        ----------
        src: np.array
            Array of points transformed to the fixed dst point locations
        dst: np.array
            Array of fixed points ordered to correspond to the src points

        Returns
        -------
        residuals: np.array
            Residual for each point pair
        """
        src = np.hstack((src, np.ones((src.shape[0], 1))))
        src_transformed = (self.transform @ src.T).T
        src_transformed = src_transformed[:, 0:3]
        residuals: np.ndarray = np.sqrt(np.sum((src_transformed - dst) ** 2, axis=1))
        return residuals

    def _umeyama(
        self, src: np.ndarray, dst: np.ndarray, estimate_scale: bool
    ) -> np.ndarray:
        """
        Estimate N-D similarity transformation with or without scaling.

        Parameters
        ----------
        src : (M, N) array
            Source coordinates.
        dst : (M, N) array
            Destination coordinates.
        estimate_scale : bool
            Whether to estimate scaling factor.

        Returns
        -------
        T : (N + 1, N + 1)
            The homogeneous similarity transformation matrix. The matrix contains
            NaN values only if the problem is not well-conditioned.

        References
        ----------
        .. [1] "Least-squares estimation of transformation parameters between two
                point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
        """

        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38).
        A = dst_demean.T @ src_demean / num

        # Eq. (39).
        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T: np.ndarray = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.full_like(T, np.nan)
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d) if estimate_scale else 1.0
        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale

        return T

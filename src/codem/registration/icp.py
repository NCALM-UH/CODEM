"""
IcpRegistration.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

This module contains a class to co-register two point clouds using a robust
point-to-plane ICP method.

This module contains the following class:

* IcpRegistration: a class for point cloud to point cloud registration
"""
from __future__ import annotations

import logging
import os
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
from codem.preprocessing.preprocess import GeoData
from codem.preprocessing.preprocess import RegistrationParameters
from scipy import spatial
from scipy.sparse import diags

if TYPE_CHECKING:
    from codem.registration import DsmRegistration


class IcpRegistration:
    """
    A class to solve the transformation between two point clouds. Uses point-to-
    plane ICP with robust weighting.

    Parameters
    ----------
    fnd_obj: DSM object
        the foundation DSM
    aoi_obj: DSM object
        the area of interest DSM
    dsm_reg: DsmRegistration object
        object holding the dsm registration data
    config: Dictionary
        dictionary of configuration parameters

    Methods
    --------
    register
    _residuals
    _get_weights
    _apply_transform
    _scaled
    _unscaled
    _output
    """

    def __init__(
        self,
        fnd_obj: GeoData,
        aoi_obj: GeoData,
        dsm_reg: DsmRegistration,
        config: Dict[str, Any],
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.fixed = fnd_obj.point_cloud
        self.normals = fnd_obj.normal_vectors
        self.moving = aoi_obj.point_cloud
        self.resolution = aoi_obj.resolution
        self.initial_transform = dsm_reg.registration_parameters["matrix"]
        self.outlier_thresh = dsm_reg.registration_parameters["rmse_3d"]
        self.config = config
        self.residual_origins = np.empty((0, 0), np.double)
        self.residual_vectors = np.empty((0, 0), np.double)

        assert (
            self.fixed.shape[1] == 3
            and self.moving.shape[1] == 3
            and self.normals.shape[1] == 3
        ), "Point and normal vector data must be 3D."
        assert (
            self.fixed.shape[0] >= 7 and self.moving.shape[0] >= 7
        ), "At least 7 points required for the point to plane ICP algorithm."
        assert (
            self.normals.shape == self.fixed.shape
        ), "Normal vector array must be same size as fixed points array."

    def register(self) -> None:
        """
        Executes ICP by minimizing point-to-plane distances:
        * Find fixed point closest to each moving point
        * Find the transform that minimizes the projected distance between each
        paired fixed and moving point, where the projection is to the fixed
        point normal direction
        * Apply the transform to the moving points
        * Repeat above steps until a convergence criteria or maximum iteration
        threshold is reached
        * Assign final transformation as attribute
        """
        self.logger.info("Solving ICP registration.")

        # Apply transform from previous feature-matching registration
        moving = self._apply_transform(self.moving, self.initial_transform)

        # Remove fixed mean to decorrelate rotation and translation
        fixed_mean = np.mean(self.fixed, axis=0)
        fixed = self.fixed - fixed_mean
        moving = moving - fixed_mean

        fixed_tree = spatial.cKDTree(fixed)

        cumulative_transform = np.eye(4)
        moving_transformed = moving
        rmse = np.float64(0.0)
        previous_rmse = np.float64(1e-12)

        alpha = 2.0
        beta = (self.resolution) / 2 + 0.5
        tau = 0.2

        for i in range(self.config["ICP_MAX_ITER"]):
            _, idx = fixed_tree.query(
                moving_transformed, k=1, distance_upper_bound=self.outlier_thresh
            )

            include_fixed = idx[idx < fixed.shape[0]]
            include_moving = idx < fixed.shape[0]
            temp_fixed = fixed[include_fixed]
            temp_normals = self.normals[include_fixed]
            temp_moving_transformed = moving_transformed[include_moving]

            assert (
                temp_fixed.shape[0] >= 7
            ), "At least 7 points within the ICP outlier threshold are required."

            weights = self._get_weights(
                temp_fixed, temp_normals, temp_moving_transformed, alpha, beta
            )
            alpha -= tau

            if self.config["ICP_SOLVE_SCALE"]:
                current_transform, euler, distance = self._scaled(
                    temp_fixed, temp_normals, temp_moving_transformed, weights
                )
            else:
                current_transform, euler, distance = self._unscaled(
                    temp_fixed, temp_normals, temp_moving_transformed, weights
                )

            cumulative_transform = current_transform @ cumulative_transform
            moving_transformed = self._apply_transform(moving, cumulative_transform)

            temp_moving_transformed = self._apply_transform(
                temp_moving_transformed, current_transform
            )
            squared_error = (temp_fixed - temp_moving_transformed) ** 2
            rmse = np.sqrt(
                np.sum(np.sum(squared_error, axis=1)) / temp_moving_transformed.shape[0]
            )

            relative_change_rmse = np.abs((rmse - previous_rmse) / previous_rmse)
            previous_rmse = rmse

            if relative_change_rmse < self.config["ICP_RMSE_THRESHOLD"]:
                self.logger.debug("ICP converged via minimum relative change in RMSE.")
                break

            if (
                euler < self.config["ICP_ANGLE_THRESHOLD"]
                and distance < self.config["ICP_DISTANCE_THRESHOLD"]
            ):
                self.logger.debug("ICP converged via angle and distance thresholds.")
                break

        self.rmse_3d = rmse
        self.rmse_xyz = np.sqrt(
            np.sum((temp_fixed - temp_moving_transformed) ** 2, axis=0)
            / temp_moving_transformed.shape[0]
        )
        self.number_points = temp_moving_transformed.shape[0]
        self.logger.debug(f"ICP number of iterations = {i+1}, RMSE = {rmse}")

        # The mean removal must be accommodated to generate the actual transform
        pre_transform = np.eye(4)
        pre_transform[:3, 3] = -fixed_mean
        post_transform = np.eye(4)
        post_transform[:3, 3] = fixed_mean
        icp_transform = post_transform @ cumulative_transform @ pre_transform

        T = icp_transform @ self.initial_transform
        c = np.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2 + T[2, 0] ** 2)
        assert (
            c > 0.67 and c < 1.5
        ), "Solved scale difference between foundation and AOI data exceeds 50%."
        if self.config["ICP_SAVE_RESIDUALS"]:
            self.residual_origins = self._apply_transform(self.moving, T)
            self.residual_vectors = self._residuals(
                fixed_tree, fixed, self.normals, moving_transformed
            )

        self.transformation = T
        self._output()

    def _residuals(
        self,
        fixed_tree: spatial.cKDTree,
        fixed: np.ndarray,
        normals: np.ndarray,
        moving: np.ndarray,
    ) -> np.ndarray:
        """
        Generates residual vectors for visualization purposes to illustrate the
        approximate orthogonal difference between the foundation and AOI
        surfaces that remains after registration. Note that these residuals will
        always be in meters.
        """
        _, idx = fixed_tree.query(moving, k=1)
        include_fixed = idx[idx < fixed.shape[0]]
        include_moving = idx < fixed.shape[0]
        temp_fixed = fixed[include_fixed]
        temp_normals = normals[include_fixed]
        temp_moving = moving[include_moving]

        residuals = np.sum((temp_moving - temp_fixed) * temp_normals, axis=1)
        residual_vectors: np.ndarray = (temp_normals.T * residuals).T
        return residual_vectors

    def _get_weights(
        self,
        fixed: np.ndarray,
        normals: np.ndarray,
        moving: np.ndarray,
        alpha: float,
        beta: float,
    ) -> diags:
        """
        A dynamic weight function from an as yet unpublished manuscript. Details
        will be inserted once the manuscript is published. Traditional robust
        least squares weight functions rescale the errors on each iteration;
        this method essentially rescales the weight funtion on each iteration
        instead. It appears to work slightly better than traditional methods.

        Parameters
        ----------
        fixed: np.array
            Array of fixed points ordered to correspond to the moving points
        normals: np.array
            Array of normal vectors corresponding to the fixed points
        moving: np.array
            Array of points to be transformed to the fixed point locations
        alpha: float
            Scalar value that controls how the weight function changes
        beta: float
            Scalar value that loosely represents the random noise in the data

        Returns
        -------
        weights: diags
            Sparse matrix of weights
        """
        r = np.sum((moving - fixed) * normals, axis=1)
        if alpha != 0:
            weights = (1 + (r / beta) ** 2) ** (alpha / 2 - 1)
        else:
            weights = beta**2 / (beta**2 + r**2)

        return diags(weights)

    def _apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Applies a 4x4 homogeneous transformation matrix to an array of 3D point
        coordinates.

        Parameters
        ----------
        points: np.array
            Array of 3D points to be transformed
        transform: np.array
            4x4 transformation matrix to apply to points

        Returns
        -------
        transformed_points: np.array
            Array of transformed 3D points
        """
        assert transform.shape == (4, 4), "Transformation matrix is an invalid shape"
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points: np.ndarray = np.transpose(transform @ points.T)[:, 0:3]
        return transformed_points

    def _scaled(
        self, fixed: np.ndarray, normals: np.ndarray, moving: np.ndarray, weights: diags
    ) -> Tuple[np.ndarray, float, float]:
        """
        Solves a scaled rigid-body transformation (7-parameter) that minimizes
        the point to plane distances.

        Parameters
        ----------
        fixed: np.array
            Array of fixed points ordered to correspond to the moving points
        normals: np.array
            Array of normal vectors corresponding to the fixed points
        moving: np.array
            Array of points to be transformed to the fixed point locations
        weights: scipy.sparse.diags
            Sparse matrix of weights for robustness against outliers

        Returns
        -------
        transform: np.array
            4x4 transformation matrix
        euler: float
            The rotation angle in terms of a single rotation axis
        distance: float
            The translation distance
        """
        b = np.sum(fixed * normals, axis=1)
        A1 = np.cross(moving, normals)
        A2 = normals
        A3 = np.expand_dims(np.sum(moving * normals, axis=1), axis=1)
        A = np.hstack((A1, A2, A3))

        if self.config["ICP_ROBUST"]:
            x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ b
        else:
            x = np.linalg.inv(A.T @ A) @ A.T @ b

        x[:3] /= x[6]

        R = np.eye(3)
        T = np.zeros(3)
        R[0, 0] = np.cos(x[2]) * np.cos(x[1])
        R[0, 1] = -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[0, 2] = np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[1, 0] = np.sin(x[2]) * np.cos(x[1])
        R[1, 1] = np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[1, 2] = -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[2, 0] = -np.sin(x[1])
        R[2, 1] = np.cos(x[1]) * np.sin(x[0])
        R[2, 2] = np.cos(x[1]) * np.cos(x[0])
        T[0] = x[3]
        T[1] = x[4]
        T[2] = x[5]
        c = x[6]

        transform = np.eye(4)
        transform[:3, :3] = c * R
        transform[:3, 3] = T

        euler = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        distance = np.sqrt(np.sum(T**2))

        return transform, euler, distance

    def _unscaled(
        self, fixed: np.ndarray, normals: np.ndarray, moving: np.ndarray, weights: diags
    ) -> Tuple[np.ndarray, float, float]:
        """
        Solves a rigid-body transformation (6-parameter) that minimizes
        the point to plane distances.

        Parameters
        ----------
        fixed: np.array
            Array of fixed points ordered to correspond to the moving points
        normals: np.array
            Array of normal vectors corresponding to the fixed points
        moving: np.array
            Array of points to be transformed to the fixed point locations
        weights: scipy.sparse.diags
            Sparse matrix of weights for robustness against outliers

        Returns
        -------
        transform: np.array
            4x4 transformation matrix
        euler: float
            The rotation angle in terms of a single rotation axis
        distance: float
            The translation distance
        """
        b1 = np.sum(fixed * normals, axis=1)
        b2 = np.sum(moving * normals, axis=1)
        b = np.expand_dims(b1 - b2, axis=1)
        A1 = np.cross(moving, normals)
        A2 = normals
        A = np.hstack((A1, A2))

        if self.config["ICP_ROBUST"]:
            x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ b
        else:
            x = np.linalg.inv(A.T @ A) @ A.T @ b

        R = np.eye(3)
        T = np.zeros(3)
        R[0, 0] = np.cos(x[2]) * np.cos(x[1])
        R[0, 1] = -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[0, 2] = np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[1, 0] = np.sin(x[2]) * np.cos(x[1])
        R[1, 1] = np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[1, 2] = -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[2, 0] = -np.sin(x[1])
        R[2, 1] = np.cos(x[1]) * np.sin(x[0])
        R[2, 2] = np.cos(x[1]) * np.cos(x[0])
        T[0] = x[3]
        T[1] = x[4]
        T[2] = x[5]

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = T

        euler = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        distance = np.sqrt(np.sum(T**2))

        return transform, euler, distance

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

        self.registration_parameters: RegistrationParameters = {
            "matrix": X,
            "omega": omega,
            "phi": phi,
            "kappa": kappa,
            "trans_x": tx,
            "trans_y": ty,
            "trans_z": tz,
            "scale": c,
            "n_pairs": self.number_points,
            "rmse_x": self.rmse_xyz[0],
            "rmse_y": self.rmse_xyz[1],
            "rmse_z": self.rmse_xyz[2],
            "rmse_3d": self.rmse_3d,
        }
        output_file = os.path.join(self.config["OUTPUT_DIR"], "registration.txt")

        self.logger.info(f"Saving ICP registration parameters to: {output_file}")
        with open(output_file, "a") as f:
            f.write("ICP REGISTRATION")
            f.write("\n----------------")
            f.write(f"\nTransformation matrix: \n {X}")
            f.write("\nTransformation Parameters:")
            f.write(f"\nOmega = {omega:.3f} degrees")
            f.write(f"\nPhi = {phi:.3f} degrees")
            f.write(f"\nKappa = {kappa:.3f} degrees")
            f.write(f"\nX Translation = {tx:.3f}")
            f.write(f"\nY Translation = {ty:.3f}")
            f.write(f"\nZ Translation = {tz:.3f}")
            f.write(f"\nScale = {c:.6f}")
            f.write(f"\nNumber of pairs = {self.number_points}")
            f.write("\nRMSEs:")
            f.write(
                f"\nX = +/-{self.rmse_xyz[0]:.3f},"
                f"\nY = +/-{self.rmse_xyz[1]:.3f},"
                f"\nZ = +/-{self.rmse_xyz[2]:.3f},"
                f"\n3D = +/-{self.rmse_3d:.3f}"
            )
            f.write("\n\n")

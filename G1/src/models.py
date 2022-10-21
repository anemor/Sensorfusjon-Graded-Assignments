from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import scipy.linalg
from senfuslib import MultiVarGauss, DynamicModel
from states import (ErrorState, ImuMeasurement,
                    CorrectedImuMeasurement, NominalState,
                    GnssMeasurement, EskfState)
from quaternion import RotationQuaterion
from utils.indexing import block_3x3
from utils.cross_matrix import get_cross_matrix
from solution import models as models_solu


@dataclass
class ModelIMU:
    """The IMU is considered a dynamic model instead of a sensar. 
    This works as an IMU measures the change between two states, 
    and not the state itself.."""

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    accm_correction: 'np.ndarray[3, 3]'
    gyro_correction: 'np.ndarray[3, 3]'

    g: 'np.ndarray[3]' = field(default=np.array([0, 0, 9.82]))

    Q_c: 'np.ndarray[12, 12]' = field(init=False, repr=False)

    def __post_init__(self):
        def diag3(x): return np.diag([x]*3)

        accm_corr = self.accm_correction
        gyro_corr = self.gyro_correction

        self.Q_c = scipy.linalg.block_diag(
            accm_corr @ diag3(self.accm_std**2) @ accm_corr.T,
            gyro_corr @ diag3(self.gyro_std**2) @ gyro_corr.T,
            diag3(self.accm_bias_std**2),
            diag3(self.gyro_bias_std**2)
        )

    def correct_z_imu(self,
                      x_est_nom: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_est_nom: previous nominal state
            z_imu: raw IMU measurement

        Returns:
            z_corr: corrected IMU measurement
        """
        S_a = self.accm_correction
        S_g = self.gyro_correction
        acc_est = S_a @ (z_imu.acc - x_est_nom.accm_bias)
        avel_est = S_g @ (z_imu.avel - x_est_nom.gyro_bias)

        z_corr = CorrectedImuMeasurement(acc_est, avel_est)
        return z_corr

    def predict_nom(self,
                    x_est_nom: NominalState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement and a 
        time step, by discretizing (10.58) in the book.

        We assume the change in orientation is negligable when caculating 
        predicted position and velicity, see assignment pdf.

        Hint: You can use: delta_rot = RotationQuaterion.from_avec(something)

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_nom_pred: predicted nominal state
        """
        a = x_est_nom.ori.as_rotmat() @ (z_corr.acc - x_est_nom.accm_bias) + self.g      # z_corr.acc - x_est_nom.accm_bias?
        w = z_corr.avel - x_est_nom.gyro_bias                                            # z_corr.avel - x_est_nom.gyro_bias?
        pos_pred = x_est_nom.pos + dt * x_est_nom.vel + (dt*dt/2) * a
        vel_pred = x_est_nom.vel + dt * a

        delta_rot = RotationQuaterion.from_avec(dt*w)
        ori_pred = x_est_nom.ori.multiply(delta_rot)

        acc_bias_pred = scipy.linalg.expm(-self.accm_bias_p * dt * np.eye(3)) @ a #x_est_nom.accm_bias
        gyro_bias_pred = scipy.linalg.expm(-self.gyro_bias_p * dt * np.eye(3)) @ w #x_est_nom.gyro_bias

        #acc_bias_pred = -self.accm_bias_p * np.eye(3) @ a    #x_est_nom.accm_bias
        #gyro_bias_pred = -self.gyro_bias_p * np.eye(3) @ w    #x_est_nom.gyro_bias

        x_nom_pred = NominalState(pos_pred, vel_pred, ori_pred, acc_bias_pred, gyro_bias_pred)
        # x_nom_pred = models_solu.ModelIMU.predict_nom(self, x_est_nom, z_corr, dt)
        return x_nom_pred

    def A_c(self,
            x_est_nom: NominalState,
            z_corr: CorrectedImuMeasurement,
            ) -> 'np.ndarray[15, 15]':
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        ex: first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev: previous nominal state
            z_corr: corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        A_c = np.zeros((15, 15))
        Rq = x_est_nom.ori.as_rotmat()
        S_acc = get_cross_matrix(z_corr.acc)
        S_omega = get_cross_matrix(z_corr.avel)
        S_a = self.accm_correction
        S_g = self.gyro_correction

        A_c[block_3x3(0,1)] = np.eye(3)
        A_c[block_3x3(1,2)] = -Rq @ S_acc
        A_c[block_3x3(1,3)] = -Rq @ S_a
        A_c[block_3x3(2,2)] = -S_omega
        A_c[block_3x3(2,4)] = -S_g
        A_c[block_3x3(3,3)] = -S_a * self.accm_bias_p
        A_c[block_3x3(4,4)] = -S_g * self.gyro_bias_p

        return A_c

    def get_error_G_c(self,
                      x_est_nom: NominalState,
                      ) -> 'np.ndarray[15, 15]':
        """The continous noise covariance matrix, G, in (10.68)

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_est_nom: previous nominal state
        Returns:
            G_c (ndarray[15, 15]): G in (10.68)
        """
        G_c = np.zeros((15, 12))
        Rq = x_est_nom.ori.as_rotmat()

        G_c[block_3x3(1,0)] = -Rq
        G_c[block_3x3(2,1)] = -np.eye(3)
        G_c[block_3x3(3,2)] = np.eye(3)
        G_c[block_3x3(4,3)] = np.eye(3)
        
        return G_c

    def get_discrete_error_diff(self,
                                x_est_nom: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                dt: float
                                ) -> Tuple['np.ndarray[15, 15]',
                                           'np.ndarray[15, 15]']:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: Use scipy.linalg.expm to get the matrix exponential

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measurement
            dt: time step
        Returns:
            A_d (ndarray[15, 15]): discrede transition matrix
            GQGT_d (ndarray[15, 15]): discrete noise covariance matrix
        """
        A_c = self.A_c(x_est_nom, z_corr)
        G_c = self.get_error_G_c(x_est_nom)
        GQGT_c = G_c @ self.Q_c @ np.transpose(G_c)

        exponent = np.zeros((30,30))
        exponent[0:15, 0:15] = -A_c
        exponent[0:15, 15:30] = GQGT_c
        exponent[15:30, 15:30] = np.transpose(A_c)
        VanLoanMatrix = scipy.linalg.expm(exponent * dt)

        A_d = scipy.linalg.expm(A_c * dt)

        V1 = VanLoanMatrix[15:30, 15:30]
        V2 = VanLoanMatrix[0:15, 15:30]
        GQGT_d = np.transpose(V1) @ V2

        return A_d, GQGT_d

    def predict_err(self,
                    x_est_prev: EskfState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float,
                    ) -> MultiVarGauss[ErrorState]:
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_est_prev: previous estimated eskf state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_err_pred: predicted error state gaussian
        """
        x_est_prev_nom = x_est_prev.nom
        x_est_prev_err = x_est_prev.err
        Ad, GQGTd = self.get_discrete_error_diff(x_est_prev_nom, z_corr, dt)
        P_pred = Ad @ x_est_prev_err.cov @ Ad.T + GQGTd
        x_pred = Ad @ x_est_prev_err.mean

        x_err_pred = MultiVarGauss(x_pred, P_pred)
        return x_err_pred

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from senfuslib import MultiVarGauss
from states import NominalState, GnssMeasurement, EskfState, ErrorState
from quaternion import RotationQuaterion
from utils.cross_matrix import get_cross_matrix
from solution import sensors as sensors_solu


@dataclass
class SensorGNSS:
    gnss_std_ne: float
    gnss_std_d: float
    lever_arm: 'np.ndarray[3]'
    R: 'np.ndarray[3, 3]' = field(init=False)

    def __post_init__(self):
        self.R = np.diag([self.gnss_std_ne**2,
                          self.gnss_std_ne**2,
                          self.gnss_std_d**2])

    def H(self, x_nom: NominalState) -> 'np.ndarray[3, 15]':
        """Get the measurement jacobian, H with respect to the error state.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff. 

        Returns:
            H (ndarray[3, 15]): the measurement matrix
        """
        R = x_nom.ori.R
        S_a = get_cross_matrix(self.lever_arm)

        H = np.zeros((3,15))
        H[0:3, 0:3] = np.eye(3)
        H[0:3, 6:9] = -R @ S_a
   
        return H

    def pred_from_est(self, x_est: EskfState,
                      ) -> MultiVarGauss[GnssMeasurement]:
        """Predict the gnss measurement

        Args:
            x_est: eskf state

        Returns:
            z_gnss_pred_gauss: gnss prediction gaussian
        """
        x_est_nom = x_est.nom
        x_est_err = x_est.err
        H = self.H(x_est_nom)
        R = self.R

        # Should we use x_t = x_nom + x_err ?
        # x_t_pos = x_est_nom.pos + x_est_err.mean.pos
        # x_t_vel = x_est_nom.vel + x_est_err.mean.vel
        # x_t_ori = x_est_nom.ori.multiply(RotationQuaternion.from_avec(1, 0.5*x_est_err.mean.avec))
        # x_t_acc = x_est_nom.accm_bias + x_est_err.mean.accm_bias
        # x_t_gyro = x_est_nom.gyro_bias + x_est_err.mean.gyro_bias
        # x_t = NominalState(x_t_pos, x_t_vel, x_t_ori, x_t_acc, x_t_gyro)

        z_pred = H @ x_est_err.mean
        S = H @ x_est_err.cov @ H.T + R

        z_pred = GnssMeasurement.from_array(z_pred)
        z_gnss_pred_gauss = MultiVarGauss[GnssMeasurement](z_pred, S)

        # Finish task and remove this:
        z_gnss_pred_gauss = sensors_solu.SensorGNSS.pred_from_est(self, x_est)
        return z_gnss_pred_gauss

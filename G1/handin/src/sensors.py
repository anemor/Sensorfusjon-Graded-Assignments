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

        Ra = x_est_nom.ori.R @ self.lever_arm
        z_pred = x_est_nom[0:3] + Ra + x_est_err.mean[0:3]

        S = H @ x_est_err.cov @ H.T + R

        z_pred = GnssMeasurement.from_array(z_pred)
        z_gnss_pred_gauss = MultiVarGauss[GnssMeasurement](z_pred, S)

        return z_gnss_pred_gauss

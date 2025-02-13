import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve


class Calibrator:
    def __init__(self, y_true: np.array, y_pred: np.array):
        self.y_true = y_true
        self.y_pred = y_pred

    def crete_calibration_map(self, n_bins: int):
        self.n_bins = n_bins
        prob_estim, self._prob_pred = calibration_curve(
            self.y_true, self.y_pred, n_bins=n_bins, strategy="quantile"
        )
        # we need to store the bins to calibrate new predictions with it
        quantiles = np.linspace(0, 1, n_bins + 1)
        self.bins = np.percentile(y_pred, quantiles * 100)
        self.df_binid_prob_estim_map = pd.DataFrame({"p_estim": prob_estim})

    def calibrate(self, y_pred_to_calibrate: np.array):
        print(self.bins)
        print(y_pred_to_calibrate)
        y_pred_binids = np.searchsorted(self.bins, y_pred_to_calibrate)
        print(y_pred_binids)
        # len(y_pred_binids) == len(y_pred_to_calibrate)
        df_y_pred_binids = pd.DataFrame({"binid": y_pred_binids})
        # find p estimation by bin id
        y_calibrated = pd.merge(
            df_y_pred_binids,
            self.df_binid_prob_estim_map,
            left_on="binid",
            right_index=True,
        )
        return y_calibrated


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    calibrator = Calibrator(y_true=y_true, y_pred=y_pred)
    calibrator.crete_calibration_map(n_bins=3)
    new_y_pred = np.array([0.2, 0.05, 0.5, 0.3])
    res = calibrator.calibrate(y_pred_to_calibrate=new_y_pred)
    print(calibrator.df_binid_prob_estim_map)
    print(calibrator.bins)
    print(res)

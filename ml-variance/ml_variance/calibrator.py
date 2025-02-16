import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from typing import Protocol


class Predictor(Protocol):
    def predict(self, X): ...


class ProbPredictor:
    def __init__(
        self, predictor: Predictor, df_binid_prob_estim_map: pd.DataFrame, bins
    ):
        self._predictor: Predictor = predictor
        self.df_binid_prob_estim_map = df_binid_prob_estim_map
        self.bins = bins

    def predict(self, X: np.ndarray):
        print(self.bins)
        y_pred_to_calibrate = self._predictor.predict(X)
        print(y_pred_to_calibrate)
        y_pred_binids = np.searchsorted(self.bins, y_pred_to_calibrate)
        print(y_pred_binids)
        # len(y_pred_binids) == len(y_pred_to_calibrate)
        df_y_pred_binids = pd.DataFrame(
            {
                "binid": y_pred_binids,
            }
        )
        # find p estimation by bin id
        y_calibrated = pd.merge(
            df_y_pred_binids,
            self.df_binid_prob_estim_map,
            left_on="binid",
            right_index=True,
        )
        columns = ["binid", "p_estim", "y_pred", "calib_range"]
        return y_calibrated[columns]


class Calibrator:
    def __init__(self, y_true_calib: np.array, X_calib: np.ndarray, n_bins: int):
        self.y_true: np.array = y_true_calib
        self.y_true.flags.writeable = False
        self.X: np.ndarray = X_calib
        self.n_bins = n_bins
        if self.X is not None:
            self.X.flags.writeable = False

    def _make_calibration_map_wiht_y_pred(self, y_pred: np.array):
        prob_estim, prob_pred = calibration_curve(
            self.y_true, y_pred, n_bins=self.n_bins, strategy="quantile"
        )
        # we need to store the bins to calibrate new predictions with it
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        bins = np.percentile(y_pred, quantiles * 100)
        self.first_bin_bound = bins[0]
        bins[0] = 0
        range_strs = [
            f"({round(b[0], 4)}; {round(b[1], 4)})" for b in zip(bins[:-1], bins[1:])
        ]
        df_binid_prob_estim_map = pd.DataFrame(
            {"p_estim": prob_estim, "prob_pred": prob_pred, "calib_range": range_strs}
        )
        return bins[1:], df_binid_prob_estim_map

    def calibrate(self, predictor: Predictor):
        y_pred = predictor.predict(self.X)
        bins, df_binid_prob_estim_map = self._make_calibration_map_wiht_y_pred(
            self.n_bins, y_pred
        )
        return ProbPredictor(
            predictor=predictor,
            df_binid_prob_estim_map=df_binid_prob_estim_map,
            bins=bins,
        )


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    calibrator = Calibrator(y_true_calib=y_true, X_calib=None, n_bins=3)
    print("make calibration")
    calibrator._make_calibration_map_wiht_y_pred(y_pred=y_pred)
    print(f"round binds {np.round(calibrator.bins, decimals=3)}")
    round_binds_true = [0.367, 0.733, 1.0]
    print((round_binds_true == np.round(calibrator.bins, decimals=3)).all())
    new_y_pred = np.array([0.2, 0.05, 0.5, 0.3, 0.8])
    # (bin=interavl):
    #  0=-(0, 0.367), 1=-(0.367, 0.733), 2=(0.733, INF)
    #  0.2 -> 0, 0.05 -> 0
    print(calibrator.df_binid_prob_estim_map)
    # res = calibrator.calibrate(y_pred_to_calibrate=new_y_pred)
    # print(res)

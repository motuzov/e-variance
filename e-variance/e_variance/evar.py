import pandas as pd
import numpy as np
from typing import Protocol, Any


from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupKFold
from sklearn.base import clone


class Predictor(Protocol):
    def predict(self, X: np.ndarray): ...


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.array): ...
    def predict(self, X: np.ndarray): ...


class CalibrationMap:
    def __init__(self, y_true_calib: np.array, X_calib: np.ndarray = None):
        self.y_true: np.array = y_true_calib
        self.y_true.flags.writeable = False
        self.X: np.ndarray = X_calib
        if self.X is not None:
            self.X.flags.writeable = False

    def _make_map_wiht_y_pred(
        self, n_bins: int, y_pred: np.array
    ) -> tuple[Any, pd.DataFrame]:
        self.n_bins = n_bins
        prob_estim, self._prob_pred = calibration_curve(
            self.y_true, y_pred, n_bins=n_bins, strategy="quantile"
        )
        # we need to store the bins to calibrate new predictions with it
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_pred, quantiles * 100)
        self.first_bin_bound = bins[0]
        bins[0] = 0
        range_str = [
            f"({round(b[0], 4)}; {round(b[1], 4)})" for b in zip(bins[:-1], bins[1:])
        ]
        self.df_binid_prob_estim_map = pd.DataFrame(
            {
                "p_estim": prob_estim,
                "calib_range": range_str,
            }
        )
        self.bins = bins[1:]

    def make(self, predictor: Predictor, n_bins: int) -> tuple[Any, pd.DataFrame]:
        y_pred = predictor.predict(self.X)
        self._make_calibration_map_wiht_y_pred(n_bins, y_pred)


class ProbPredictor:
    def __init__(
        self, predictor: Predictor, prob_bins: int, calibration_map: CalibrationMap
    ):
        # binds the calibrator and and the predictor
        if predictor:
            calibration_map.make(predictor, prob_bins)
        self.bins = calibration_map.bins
        self.df_binid_prob_estim_map = calibration_map.df_binid_prob_estim_map
        self.predictor = predictor

    def calibrate(self, y_pred):
        y_pred_binids = np.searchsorted(self.bins, y_pred)
        print(y_pred_binids)
        # len(y_pred_binids) == len(y_pred)
        df_y_pred_binids = pd.DataFrame(
            {
                "binid": y_pred_binids,
                "y_pred": y_pred,
            }
        )
        # find p estimation by bin id
        y_calibrated = pd.merge(
            df_y_pred_binids,
            self.df_binid_prob_estim_map,
            left_on="binid",
            right_index=True,
        )
        return y_calibrated

    def predict(self, X):
        y_pred = self.predictor(X)
        y_calibrated = self.calibrate(y_pred=y_pred)
        return y_calibrated


class EstimatorVar:
    # the estimator states??
    def __init__(self, X: np.ndarray, y: np.array, calibration_map: CalibrationMap):
        self.calibration_map = calibration_map
        self.prob_predictors: list[ProbPredictor] = []
        self.X = X
        self.y = y
        ...

    def fit(
        self,
        estimator: Estimator,
        groups_itr,
        prob_bins,
    ) -> pd.DataFrame:
        # fit predictors acorodng to validation groups and estimate variance
        ...
        self.prob_predictors = []
        for train_index, test_index in groups_itr:
            # estimator = estimator.fit(data)
            estimator = clone(estimator)
            estimator = estimator.fit(self.X[train_index])
            self.calibrator.make_binid_prob_estim_map(
                n_bins=prob_bins, predictor=estimator
            )
            # self.calibrator.make_binid_prob_estim_map(n_bins=3, predictor=estimator)
            prob_pred = ProbPredictor(
                predictor=estimator,
                bins=self.calibrator.bins,
                df_binid_prob_estim_map=self.calibration_map.df_binid_prob_estim_map,
            )
            self.prob_predictors.append(prob_pred)

    def _var(self, df_var_data): ...

    def estimate(self, X: np.ndarray):
        prob_pred_dfs = []
        for i, prob_pred in enumerate(self.prob_predictors):
            df_prob_pred = prob_pred.predict(X)
            df_prob_pred["pred_id"] = i
            prob_pred_dfs.append(df_prob_pred)
        prob_pred_dfs = pd.concat(prob_pred_dfs)
        # self._var(prob_pred_dfs)


def test_calibrator():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    calibration_map = CalibrationMap(y_true_calib=y_true)
    print("make calibration")
    calibration_map._make_map_wiht_y_pred(n_bins=3, y_pred=y_pred)
    print(f"round binds {np.round(calibration_map.bins, decimals=3)}")
    round_binds_true = [0.367, 0.733, 1.0]
    print((round_binds_true == np.round(calibration_map.bins, decimals=3)).all())
    new_y_pred = np.array([0.2, 0.05, 0.5, 0.3, 0.8])
    # (bin=interavl):
    #  0=-(0, 0.367), 1=-(0.367, 0.733), 2=(0.733, INF)
    #  0.2 -> 0, 0.05 -> 0
    print(calibration_map.df_binid_prob_estim_map)
    # res = calibrator.calibrate(y_pred_to_calibrate=new_y_pred)
    # print(res)


def test_groups():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([1, 2, 3, 4, 5, 6])
    groups = np.array([0, 0, 2, 2, 3, 3])
    group_kfold = GroupKFold(n_splits=2)
    group_kfold.get_n_splits(X, y, groups)
    for train_index, test_index in group_kfold.split(X, y, groups):
        print("tarin\\tesst splitting")
        print("X train")
        print(X[train_index])
        print("X test")
        print(X[test_index])
        print("\n")


if __name__ == "__main__":
    test_calibrator()
    # test_groups()


# workwlow cross_val_predict -> y_pred
# model_selection > use estimator = estimator.fit(data, targets) -> to calc -> estmator-var
#  predict_proba
# optuna layer: objective estmator-var to contrl bias var trade-off
# xgb.cb.cv.predict() return the test fold predictions from each CV model.
# y_predicted = estimator(alpha=10).fit(X_train, y_train).predict(X_test)

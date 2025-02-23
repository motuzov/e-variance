import pandas as pd
import numpy as np
from typing import Protocol, Any


from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupKFold
from sklearn.base import clone


class Predictor(Protocol):
    def predict_proba(self, X: np.ndarray): ...


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.array): ...
    def predict_proba(self, X: np.ndarray): ...


class CalibrationMap:
    def __init__(self, y_true_calib: np.array, X_calib: np.ndarray = None):
        self.y_true: np.array = y_true_calib
        self.y_true.flags.writeable = False
        self.X: np.ndarray = X_calib
        if self.X is not None:
            self.X.flags.writeable = False
        # enable_metadata_routing=True

    def _make_map_wiht_y_score(
        self, n_bins: int, y_score: np.array
    ) -> tuple[Any, pd.DataFrame]:
        self.n_bins = n_bins
        prob_estim, self._prob_pred = calibration_curve(
            self.y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
        # we need to store the bins to calibrate new predictions with it
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_score, quantiles * 100)
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

    def make(self, clf: Predictor, n_bins: int) -> tuple[Any, pd.DataFrame]:
        y_score = clf.predict_proba(self.X)
        y_score = y_score[:, 1]
        self._make_map_wiht_y_score(n_bins=n_bins, y_score=y_score)


class CalibratedProbPredictor(Predictor):
    def __init__(self, clf: Predictor, prob_bins: int, calibration_map: CalibrationMap):
        # binds the calibrator and and the predictor
        if clf:
            calibration_map.make(clf, prob_bins)
        self._bins = calibration_map.bins
        self._df_binid_prob_estim_map = calibration_map.df_binid_prob_estim_map
        self._clf = clf

    def calibrate(self, y_score: np.array):
        y_score_binids = np.searchsorted(self._bins, y_score)
        df_y_score_binids = pd.DataFrame(
            {
                "binid": y_score_binids,
                "y_score": y_score,
            }
        )
        y_calib_results = pd.merge(
            df_y_score_binids,
            self._df_binid_prob_estim_map,
            left_on="binid",
            right_index=True,
        )
        return y_calib_results

    def predict_proba(self, X):
        y_score = self._clf.predict_proba(X)
        y_score = y_score[:, 1].ravel()
        y_calib_info = self.calibrate(y_score=y_score)
        return y_calib_info


class EstimatorVar:
    # the estimator states??
    def __init__(
        self, X_train: np.ndarray, y_train: np.array, calibration_map: CalibrationMap
    ):
        self.calibration_map = calibration_map
        self.prob_predictors: list[CalibratedProbPredictor] = []
        self.X_train = X_train
        self.y_train = y_train

    def fit_clfs(
        self,
        estimator: Estimator,
        splits,
        prob_bins,
    ):
        # fit predictors acorodng to groups (D_i - i group data):
        # f_hat(D_i), D_i is i's group data
        self.prob_predictors = []
        for train_index, test_index in splits:
            clf = clone(estimator)
            clf.fit(self.X_train[train_index], self.y_train[train_index])
            prob_pred = CalibratedProbPredictor(
                clf=clf,
                prob_bins=prob_bins,
                calibration_map=self.calibration_map,
            )
            self.prob_predictors.append(prob_pred)

    def _point_wise_var(self, df_var_data):
        self.point_wise_var = df_var_data.groupby(level=0)["p_estim"].var()

    def _bin_wise_var(self, df_var_data: pd.DataFrame):
        fixed_pred_id = 0
        fixed_binid = 0
        df_points_in_fixed_bin = df_var_data[
            (df_var_data["binid"] == fixed_binid)
            & (df_var_data["pred_id"] == fixed_pred_id)
        ]
        df_points_of_fixed_bin = pd.merge(
            left=df_var_data[df_var_data["pred_id"] != fixed_pred_id],
            right=df_points_in_fixed_bin,
            how="inner",
            left_index=True,
            right_index=True,
            suffixes=(f"_{fixed_pred_id}", ""),
        )
        diff_bin_points = df_points_of_fixed_bin[
            df_points_of_fixed_bin[f"binid_{fixed_pred_id}"] != fixed_pred_id
        ]
        (diff_bin_points["p_estim_0"] - diff_bin_points["p_estim"]).var()

    def _var_data_prep(self, X_test: np.ndarray) -> pd.DataFrame:
        prob_pred_dfs = []
        for i, prob_pred in enumerate(self.prob_predictors):
            df_prob_pred = prob_pred.predict_proba(X_test)
            df_prob_pred["pred_id"] = i
            df_prob_pred["point_id"] = df_prob_pred.index
            prob_pred_dfs.append(df_prob_pred)
        return pd.concat(prob_pred_dfs)

    def estimate(self, X_test: np.ndarray):
        self.df_var_data = self._var_data_prep(X_test)
        self._pred_bin_var(self.df_var_data)


def test_calibrator():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    calibration_map = CalibrationMap(y_true_calib=y_true)
    print("make calibration")
    calibration_map._make_map_wiht_y_pred(n_bins=3, y_score=y_score)
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

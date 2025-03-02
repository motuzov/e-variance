import pandas as pd
import numpy as np
from numpy.typing import NDArray, DTypeLike, ArrayLike
from typing import Protocol, Any, Iterator
from evar.data import COLUMN_NAMES


from sklearn.calibration import calibration_curve
from sklearn.base import clone
from collections.abc import Callable


class Predictor(Protocol):
    def predict_proba(self, X: np.ndarray): ...


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.array): ...
    def predict_proba(self, X: np.ndarray): ...


INTERVALS_INFO_FOMAT = "{0:.2f}"


def intervals_info(bins: ArrayLike) -> pd.Series:
    f = INTERVALS_INFO_FOMAT.format
    s_bins = pd.Series(bins)
    return (
        "["
        + s_bins.shift(1).fillna(0).apply(f).str.cat(s_bins.apply(f), sep=", ")
        + "]"
    )


class Calibrator:
    def __init__(
        self, prob_bins: ArrayLike, prob_estim: ArrayLike, mean_score: ArrayLike
    ):
        prob_bins = prob_bins[1:]
        i_info = intervals_info(prob_bins)
        binid_prob_estim_map = pd.DataFrame(
            {
                COLUMN_NAMES.p_estim: prob_estim,
                COLUMN_NAMES.mean_score: mean_score,
                COLUMN_NAMES.bin_interval: i_info,
            }
        )

        self._prob_bins: ArrayLike = prob_bins
        self._binid_prob_estim_map: pd.DataFrame = binid_prob_estim_map
        self._debug_info = {}

    def __call__(self, y_score: ArrayLike) -> pd.DataFrame:
        y_score_binids = np.searchsorted(self._prob_bins, y_score)
        df_y_score_binids = pd.DataFrame(
            {
                COLUMN_NAMES.binid: y_score_binids,
                COLUMN_NAMES.y_score: y_score,
            }
        )
        y_calib_results = pd.merge(
            df_y_score_binids,
            self._binid_prob_estim_map,
            left_on=COLUMN_NAMES.binid,
            right_index=True,
        )
        return y_calib_results

    def save_debug_info(self, k: str, v: Any):
        self._debug_infop[k] = v


class CalibratorFactory:
    def __init__(self, y_true_calib: np.array, X_calib: np.ndarray = None):
        self.y_true: np.array = y_true_calib
        self.y_true.flags.writeable = False
        self.X: np.ndarray = X_calib
        if self.X is not None:
            self.X.flags.writeable = False
        self._bebug = True
        # enable_metadata_routing=True

    def _make_map_wiht_y_score(self, n_bins: int, y_score: np.array) -> Calibrator:
        self.n_bins = n_bins
        prob_estim, mean_score = calibration_curve(
            self.y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
        # We need to store the bins to calibrate predictions with it.
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_score, quantiles * 100)
        calibrator = Calibrator(
            prob_bins=bins, prob_estim=prob_estim, mean_score=mean_score
        )
        return calibrator

    def create(self, clf: Predictor, n_bins: int) -> Calibrator:
        y_score = clf.predict_proba(self.X)
        y_score = y_score[:, 1]
        return self._make_map_wiht_y_score(n_bins=n_bins, y_score=y_score)


class CalibratedProbPredictor(Predictor):
    def __init__(self, clf: Predictor, calibrator: Calibrator):
        self._calibrate = calibrator
        self._clf = clf

    def predict_proba(self, X):
        y_score = self._clf.predict_proba(X)
        y_score = y_score[:, 1].ravel()
        y_calib_info = self._calibrate(y_score=y_score)
        return y_calib_info


class EstimatorVar:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.array,
        estimator: Estimator,
        calibration_factory: CalibratorFactory,
    ):
        self.calibration_factory: CalibratorFactory = calibration_factory
        self.prob_predictors: list[CalibratedProbPredictor] = []
        self.X_train = X_train
        self.y_train = y_train
        self.estimator = estimator

    def fit_prob_predictors(
        self,
        splitter: Callable[[DTypeLike], Iterator[tuple[np.ndarray, np.ndarray]]],
        prob_bins: int,
        n_splits: int,
    ):
        # Fit f_hat calibrated predictions according to groups using the same calibration:
        # f_hat(D_i), D_i is i's group data
        self.prob_predictors = []
        for train_index, test_index in splitter(self.X_train, n_splits):
            clf = clone(self.estimator)
            clf.fit(self.X_train[train_index], self.y_train[train_index])
            calibrator = self.calibration_factory.create(clf, prob_bins)
            prob_pred = CalibratedProbPredictor(clf=clf, calibrator=calibrator)
            self.prob_predictors.append(prob_pred)

    def _bin_wise_var(self, df_var_data: pd.DataFrame):
        fixed_pred_id = 0
        fixed_binid = 0
        df_points_in_fixed_bin = df_var_data[
            (df_var_data[COLUMN_NAMES.binid] == fixed_binid)
            & (df_var_data[COLUMN_NAMES.p_predid] == fixed_pred_id)
        ]
        df_points_of_fixed_bin = pd.merge(
            left=df_var_data[df_var_data[COLUMN_NAMES.p_predid] != fixed_pred_id],
            right=df_points_in_fixed_bin,
            how="inner",
            left_index=True,
            right_index=True,
            suffixes=(f"_{fixed_pred_id}", ""),
        )
        diff_bin_points = df_points_of_fixed_bin[
            df_points_of_fixed_bin[f"binid_{fixed_pred_id}"] != fixed_pred_id
        ]
        (diff_bin_points["p_estim_0"] - diff_bin_points[COLUMN_NAMES.p_predid]).var()

    def _point_wise_var(self, df_var_data: pd.DataFrame):
        self.point_wise_var = df_var_data.groupby(level=0)[COLUMN_NAMES.p_predid].var()

    def _var_data_prep(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        prob_pred_dfs = []
        for i, prob_pred in enumerate(self.prob_predictors):
            df_prob_pred = prob_pred.predict_proba(X_test)
            df_prob_pred[COLUMN_NAMES.p_predid] = i
            prob_pred_dfs.append(df_prob_pred)

        self._var_data = pd.concat(prob_pred_dfs)
        self._var_data[COLUMN_NAMES.y_test] = pd.DataFrame(y_test)
        return self._var_data

    @property
    def var_data(self):
        return self._var_data

    def estimate(self):
        self.df_var_data = self._var_data_prep(self.X_test)
        self._point_wise_var(self.df_var_data)

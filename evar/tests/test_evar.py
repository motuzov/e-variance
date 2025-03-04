import numpy as np
from numpy.typing import ArrayLike, DTypeLike
import pandas as pd
from evar.estimator_var import (
    CalibratedProbPredictor,
    CalibratorFactory,
    Calibrator,
    IntervalInfo,
    Predictor,
)
from evar.data import COLUMN_NAMES


import pytest
from pandas import testing as tm


class FakeProbPredictor(Predictor):
    # The class is a wrapper for y_score
    def __init__(self, y_score: ArrayLike):
        self.y_score = y_score

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
        n_classes = 2, 0 class has a fake score; under the hood, CalibratorFactory uses only class 1 score
        """
        return np.stack((np.zeros(len(self.y_score)), self.y_score), axis=-1)


@pytest.fixture
def y_score_calib():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])


@pytest.fixture
def y_true_calib():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])


@pytest.mark.parametrize(
    "y_true_calib, y_score_calib",
    [
        (
            np.array([1, 1, 0, 0, 1, 1, 1, 0, 0]),
            np.array([1.0, 0.9, 0.3, 0.4, 0.65, 0.7, 0.8, 0.2, 0.1]),
        ),
        (
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0]),
        ),
    ],
)
def test_calibrator_factory(y_true_calib, y_score_calib):
    fake_predictor = FakeProbPredictor(y_score_calib)
    cf = CalibratorFactory(y_true_calib=y_true_calib, X_calib=None, interval_info=None)
    calibrator = cf.create(clf=fake_predictor, n_bins=3)
    tm.assert_series_equal(
        calibrator._binid_prob_estim_map[COLUMN_NAMES.p_estim].round(decimals=3),
        pd.Series([0.0, 0.667, 1.0], name=COLUMN_NAMES.p_estim),
    )


def test_intervals_info():
    bins = np.array([0.1, 0.36666667, 0.73333333, 1.0])
    bin_intervals = IntervalInfo()(bins)
    tm.assert_series_equal(
        left=bin_intervals,
        right=pd.Series(
            [
                "[0.00, 0.10]",
                "[0.10, 0.37]",
                "[0.37, 0.73]",
                "[0.73, 1.00]",
            ]
        ),
        check_names=False,
    )


@pytest.fixture
def calibrator(y_true_calib, y_score_calib) -> Calibrator:
    fake_predictor = FakeProbPredictor(y_score_calib)
    cf = CalibratorFactory(y_true_calib=y_true_calib, X_calib=None, interval_info=None)
    calibrator_ = cf.create(clf=fake_predictor, n_bins=3)
    # calibrator_ = calibrator_factory._make_prob_map_from_y_score(
    #     n_bins=3, y_score=y_score
    # )
    return calibrator_


def test_create_calibrator(calibrator: Calibrator):
    print(calibrator._binid_prob_estim_map)
    tm.assert_series_equal(
        pd.Series(calibrator._prob_bins).round(decimals=4),
        pd.Series([0.3667, 0.7333, 1.0]),
    )


def test_prob_predictor_calibrate(calibrator: Calibrator):
    y_score = np.array([0.2, 0.05, 0.5, 0.3, 0.8])
    y_calib_results = calibrator(y_score=y_score)
    tm.assert_series_equal(
        y_calib_results["p_estim"].round(decimals=2),
        pd.Series([0.0, 0.0, 0.67, 0.0, 1.0]),
        check_names=False,
    )

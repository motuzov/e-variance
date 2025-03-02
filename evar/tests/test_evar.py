import numpy as np
import pandas as pd
from evar.estimator_var import (
    CalibratorFactory,
    Calibrator,
    intervals_info,
)


import pytest
from pandas import testing as tm


@pytest.fixture
def calibrator_factory() -> CalibratorFactory:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    return CalibratorFactory(y_true_calib=y_true)


@pytest.fixture
def y_score() -> np.array:
    return np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])


def test_intervals_info():
    bins = np.array([0.1, 0.36666667, 0.73333333, 1.0])
    bin_intervals = intervals_info(bins)
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
def calibrator(calibrator_factory, y_score) -> CalibratorFactory:
    calibrator_ = calibrator_factory._make_map_wiht_y_score(n_bins=3, y_score=y_score)
    return calibrator_


def test_calibrate(calibrator: Calibrator):
    y_score = np.array([0.2, 0.05, 0.5, 0.3, 0.8])
    y_calib_results = calibrator(y_score=y_score)
    tm.assert_series_equal(
        y_calib_results["p_estim"].round(decimals=2),
        pd.Series([0.0, 0.0, 0.67, 0.0, 1.0]),
        check_names=False,
    )

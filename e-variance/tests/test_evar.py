import numpy as np
from e_variance.evar import CalibrationMap, ProbPredictor
import pandas as pd
from pandas import testing as tm

import pytest


@pytest.fixture
def calibration_map():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    c_map = CalibrationMap(y_true_calib=y_true)
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    c_map._make_map_wiht_y_pred(n_bins=3, y_pred=y_pred)
    return c_map


def test_prob_predictor_calibrate(calibration_map):
    prob_pred = ProbPredictor(
        predictor=None, prob_bins=3, calibration_map=calibration_map
    )
    y_pred = np.array([0.2, 0.05, 0.5, 0.3, 0.8])
    y_calibrated = prob_pred.calibrate(y_pred)
    tm.assert_series_equal(
        y_calibrated["p_estim"].round(decimals=2),
        pd.Series([0.0, 0.0, 0.67, 0.0, 1.0]),
        check_names=False,
    )

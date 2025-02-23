import numpy as np
import pandas as pd
from evar.estimator_var import CalibratedProbPredictor, CalibrationMap


import pytest
from pandas import testing as tm


@pytest.fixture
def calibration_map():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    c_map = CalibrationMap(y_true_calib=y_true)
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    c_map._make_map_wiht_y_score(n_bins=3, y_score=y_score)
    return c_map


def test_prob_predictor_calibrate(calibration_map):
    prob_pred = CalibratedProbPredictor(
        clf=None, prob_bins=3, calibration_map=calibration_map
    )
    y_score = np.array([0.2, 0.05, 0.5, 0.3, 0.8])
    y_calib_results = prob_pred.calibrate(y_score=y_score)
    tm.assert_series_equal(
        y_calib_results["p_estim"].round(decimals=2),
        pd.Series([0.0, 0.0, 0.67, 0.0, 1.0]),
        check_names=False,
    )

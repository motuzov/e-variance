from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationMapColumnNames:
    binid: str = "p_binid"
    y_score: str = "y_score"
    p_estim: str = "p_estim"
    p_predid: str = "p_pred_id"
    bin_interval: str = "bin_interval"
    y_test: str = "y_test"
    mean_score: str = "mean_score"


COLUMN_NAMES = CalibrationMapColumnNames()

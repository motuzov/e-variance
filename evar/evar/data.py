from dataclasses import dataclass
from numpy.typing import ArrayLike
import pandas as pd


@dataclass(frozen=True)
class CalibrationMapColumnNames:
    binid: str = "p_binid"
    y_score: str = "y_score"
    p_estim: str = "p_estim"
    p_predid: str = "p_pred_id"
    bin_interval: str = "bin_interval"
    y_test: str = "y_test"


COLUMN_NAMES = CalibrationMapColumnNames()


class IntervalInfo:
    def __init__(self, format_str: str = "{0:.2f}"):
        self._format_str: str = format_str

    def __call__(self, bins: ArrayLike) -> pd.Series:
        f = self._format_str.format
        s_bins = pd.Series(bins)
        return (
            "["
            + s_bins.shift(1).fillna(0).apply(f).str.cat(s_bins.apply(f), sep=", ")
            + "]"
        )

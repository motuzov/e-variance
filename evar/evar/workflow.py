from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from numpy.typing import DTypeLike
import numpy as np
from typing import Iterator
from sklearn.datasets import load_breast_cancer


import evar.estimator_var as estimator_var
from evar.data import IntervalInfo


def make_kfold_splits(
    X_train: DTypeLike, n_splits
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    group_kfold = KFold(n_splits)
    return group_kfold.split(X_train)


class EvarWorkflow:
    def __init__(self):
        data = load_breast_cancer()
        self.X = data["data"]
        self.y = data["target"]

    def make_estimator_var(self):
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42
        )
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train, y_train, test_size=0.33, random_state=42
        )
        clf = LogisticRegression(random_state=42)
        clf = make_pipeline(StandardScaler(), clf)
        cl_factory = estimator_var.CalibratorFactory(
            y_true_calib=y_calib,
            X_calib=X_calib,
            interval_info=IntervalInfo(format_str="{0:.3f}"),
        )

        self.evar = estimator_var.EstimatorVar(
            X_train=X_train,
            y_train=y_train,
            estimator=clf,
            calibration_factory=cl_factory,
        )

    def comp_var_data(
        self, make_kfold_splits=make_kfold_splits, prob_bins: int = 3, n_splits: int = 4
    ):
        self.evar.fit_prob_predictors(
            splitter=make_kfold_splits, prob_bins=prob_bins, n_splits=n_splits
        )
        var_data = self.evar._var_data_prep(self.X_test, self.y_test)
        return var_data

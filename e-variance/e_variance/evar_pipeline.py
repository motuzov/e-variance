from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import (
    make_circles,
    make_classification,
    make_moons,
    load_breast_cancer,
)
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import evar


def evar_pipeline():
    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = LogisticRegression(random_state=42)
    clf = make_pipeline(StandardScaler(), clf)
    cm = evar.CalibrationMap(X_calib=X_calib, y_true_calib=y_calib)
    var = evar.EstimatorVar(X_train=X_train, y_train=y_train, calibration_map=cm)
    group_kfold = KFold(n_splits=3)
    splits = group_kfold.split(var.X_train)
    var.fit_clfs(estimator=clf, splits=splits, prob_bins=3)
    var_data = var._var_data_prep(X_test)
    print(var_data)


if __name__ == "__main__":
    evar_pipeline()

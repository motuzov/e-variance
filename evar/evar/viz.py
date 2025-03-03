import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evar.data import COLUMN_NAMES


def plot_bin_pred_hist(var_data: pd.DataFrame):
    s = var_data.groupby(level=0)[COLUMN_NAMES.binid].agg(set).astype("str")
    sns.barplot(s.groupby(s).count(), color="lightgreen", edgecolor="red")
    plt.xlabel("Porb bin sets")
    plt.ylabel("Count")
    plt.title(
        """
        A histogram of the number of unique bins the predictors put a single data point in
        """
    )
    plt.show()

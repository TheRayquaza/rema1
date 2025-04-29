import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist_feature(name: str, df: pd.Series, figsize=(5,3), kde=False):
    plt.figure(figsize=figsize)
    sns.histplot(df, kde=kde)
    plt.title("Statistics of {} ({})".format(df.name, name), fontsize=14)
    plt.ylabel("Count")
    plt.xlabel(df.name)
    plt.show()

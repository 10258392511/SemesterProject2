import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


########## pre_train ##########
def plot_curve(axis, csv_filename):
    df = pd.read_csv(csv_filename)
    axis.plot(df["Step"], df["Value"])
    axis.grid(True)


def compute_last_steps_stats(num_steps, csv_filename, keys=("mean", "std", "max", "min")):
    df = pd.read_csv(csv_filename)
    df_extracted = df.iloc[-num_steps:, 2]

    return df_extracted.describe()[list(keys)]
###############################

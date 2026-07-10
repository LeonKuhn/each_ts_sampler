"""
A tool for quick sanity checking of xpcaaba input/output after conversion to parquet files (see: sanity_check.py)
"""

#%% imports
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import pandas as pd

#%% load the data (may take ~ 5 minutes)
input_1 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/joint_samples/joint_0.parquet"
input_2 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/t720_joint/t720_joint_0.parquet"

#input_1 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/joint_samples/joint_0.parquet"
#input_2 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/t60_joint/t60_joint_0.parquet"

#input_1 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/marginal_samples/marginal_0.parquet"
#input_2 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/t720_marginal/t720_marginal_0.parquet"

#input_1 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/marginal_samples/marginal_0.parquet"
#input_2 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/t60_marginal/t60_marginal_0.parquet"

#input_1 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/marginal_samples/marginal_0.parquet"
#input_2 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/t60_joint/t60_joint_0.parquet"

#input_1 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/marginal_samples/marginal_0.parquet"
#input_2 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4_parquet/t60_marginal/t60_marginal_1.parquet"

var = "NO2"

ds_1 = pd.read_parquet(input_1)[var]
ds_2 = pd.read_parquet(input_2)[var]

x = ds_1.to_numpy()
y = ds_2.to_numpy()

lr = linregress(x, y)

fig, (ax1) = plt.subplots(1, 1)
plt.title(f"Scatter plot for variable '{var}'; $R$ = {lr.rvalue:.2f}")
plt.scatter(x, y, rasterized=True)
plt.xlabel("at $t_0$")
plt.ylabel("at $t_0 + ∆t$")
plt.show()

fig, (ax1) = plt.subplots(1, 1)
plt.title(f"Scatter plot for variable '{var}'; $R$ = {lr.rvalue:.2f}")
n_bins=250
bins = np.geomspace(np.min(x), np.max(x), n_bins + 1, endpoint=True)
plt.hist(x, bins=bins)
plt.xscale("log")
plt.show()

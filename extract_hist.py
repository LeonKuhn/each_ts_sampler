"""
Produces histograms of each variable stored in vars (produced by get_var_list.py)
"""

import argparse
import logging
import os
import time
import tomli
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from shared import get_filenames_by_globs, filter_fnames_by_permission, get_var

# parse arguments and toml
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--toml", type=str, help="toml file", default="default.toml")
parser.add_argument("-v", "--var", type=str, help="variable to be processed", required=True)
args = parser.parse_args()

with open(args.toml, "rb") as f:
    toml_cfg = tomli.load(f)
    cfg = {**vars(args), **toml_cfg}


out_format_gases = cfg["out_format_gases"]
input_fname_globs = cfg["input_fname_globs"]
n_bins = cfg["n_bins"]
var = cfg["var"]
ignore_vars = cfg["ignore_vars"]

# fix random seed
np.random.seed(cfg["random_seed"])

out_dir = os.path.join(cfg["work_dir"], "hists", out_format_gases)
os.makedirs(out_dir, exist_ok=True)

if var in ignore_vars:
    print("Exiting extract_hist.py because variable", var, "is in ignore_vars.")
    exit(0)

def estimate_min_max(var, fnames):
    """
    for a given variable named 'var', compute the overall minimum and maximum based on the data from all (fnames) filenames
    """
    all_min = np.inf
    all_max = 0

    for e, fname in enumerate(fnames):
        print(f"Collecting min-max data from {fname} ({e + 1}/{len(fnames)})")

        try:
            with xr.open_dataset(fname) as ds:
                try:
                    da, _ = get_var(ds, var, cfg)
                except KeyError as w:
                    print(str(w))
                    continue

                da = da.where(da > 0)  # because np.geomspace cannot start at 0

                all_min = np.min([all_min, da.min(skipna=True)])
                all_max = np.max([all_max, da.max(skipna=True)])
        except:
            print(f"Error opening file {fname}")

    if all_max == 0 or np.isnan(all_max) or np.isnan(all_min):
        print(f"Only 0s detected. The script will exit without creating a histogram.")
        exit(2)

    return all_min, all_max

def get_full_hist(var, fnames, n_bins):
    """
    for a given variable named 'var', compute a histogram with (n_bin) bins containing data from all (fnames) filenames
    """

    # determine histogram limits and bin edges
    all_min, all_max = estimate_min_max(var, fnames)

    bins = np.geomspace(all_min, all_max, n_bins + 1, endpoint=True)
    da_hist = np.zeros(n_bins)

    for e, fname in enumerate(fnames):
        print(f"Collecting histogram data from {fname} ({e+1}/{len(fnames)})")

        try:
            with xr.open_dataset(fname) as ds:

                try:
                    da, units = get_var(ds, var, cfg)
                except KeyError as w:
                    print(str(w))
                    continue
        except:
            print(f"Error opening file {fname}")

        cts, _ = np.histogram(da, bins=bins)  # standard numpy hist

        da_hist += cts

    return da_hist, bins, units

def plot_results(hist, savename=None):
    """
    plot and save the obtained histograms to disk
    """

    cts, bins, units = hist

    plt.figure()
    plt.title(f"{savename.split('/')[-1]} [{units}]")
    plt.stairs(cts, bins)
    plt.xscale("log")

    if savename is not None:
        full_savename = f"{savename}.png"
        print(f"saving plot to file {full_savename}")
        plt.savefig(full_savename, dpi=72)

    plt.show()

def save_hist(savename, hist):
    """
    saves the obtained histograms to disk
    """

    cts, bins, units = hist
    cts = np.insert(cts, 0, 0)
    out_arr = np.stack([cts, bins]).T
    print(f"saving histogram data to file {savename}")
    np.savetxt(savename, out_arr, delimiter=",", header=f"counts,bin_edges [{units}]")

# load all files matching input_fname_globs
fnames = get_filenames_by_globs(input_fname_globs)

# check which files can be opened (permissions)
fnames = filter_fnames_by_permission(fnames)

# obtain and save histogram data
hist = get_full_hist(var, fnames, n_bins=n_bins)

# save histogram data
save_hist(os.path.join(out_dir, f"{var}.csv"), hist)

# plot histogram
plot_results(hist, f"{os.path.join(out_dir, var)}")

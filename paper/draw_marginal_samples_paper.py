"""
This script uses the histogram data obtained from extract_hist.py to generate an arbitrary amount of marginal samples.
"""

import argparse
import time
import matplotlib.pyplot as plt
import os
from glob import glob
from functools import partial
from pathlib import Path
import re
import tomli
import xarray as xr

import numpy as np
from scipy.interpolate import interp1d

# parse arguments and namelist
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--namelist", type=str, help="namelist file", default="default.nml")
parser.add_argument("-s", "--nr_samples", type=int, help="number of samples to draw", required=True)
parser.add_argument("-d", "--output_dir", type=str, help="output directory (e.g. 100k)", required=True)
parser.add_argument("-o", "--output_name", type=str, help="output name, appended to output_dir", default="")
parser.add_argument("-p", "--draw_plots", help="whether to draw histogram plots", default=False, action="store_true")
parser.add_argument("-t", "--sampling_strategy", type=str, help="which sampling strategy to use", default="reproduce_hist")
parser.add_argument("-g", "--gas", type=str, help="if only a single gas is requested ('all' will draw all gases)", default="all")
args = parser.parse_args()

if args.output_name == "":
    args.output_name = args.output_dir + ".nc"

with open(args.namelist, "rb") as f:
    toml_cfg = tomli.load(f)
    cfg = {**vars(args), **toml_cfg}

work_dir = os.path.join(cfg["work_dir"], "hists", cfg["out_format_gases"])
nr_samples = cfg["nr_samples"]
sampling_strategy = cfg["sampling_strategy"]
gas = cfg["gas"]
draw_plots = cfg["draw_plots"]
output_dir = os.path.join(cfg["work_dir"], "marginal_samples", cfg["output_dir"])
output_name = cfg["output_name"]
remove_suffix = cfg["remove_suffix"]

# fix random seed
np.random.seed(cfg["random_seed"])

os.makedirs(output_dir, exist_ok=True)

def draw_from_hist(hist, size):
    """
    Draws samples from an empirical histogram by sampling a bin
    according to its probability mass, and then sampling uniformly
    within the bin. This avoids artefacts from inverse-CDF interpolation.
    """

    # counts and edges
    cts = hist[1:, 0]
    bins = hist[:, 1]

    # bin widths
    widths = np.diff(bins).astype(float)

    # probability mass of each bin
    p = cts / cts.sum()

    # RNG
    rng = np.random.default_rng()

    # pick bins according to their probability
    idx = rng.choice(len(cts), size=size, p=p)

    # sample uniformly inside each bin
    u = rng.random(size)
    samples = bins[idx] + u * widths[idx]

    return samples

def draw_uniform(hist, low_func, high_func, size):
    """
    Draws samples from a uniform distribution U[a, b), where a, b are determined from histogram data using two functions low_func, high_func.
    Example: low_func = lambda x: np.min(x) and high_func = lambda x: np.max(x); samples are drawn from the range between minimum and maximum of the histogram.
    """

    bins = hist[:, 1]

    # determine lower and upper limit to draw uniformly distributed samples from
    low = low_func(bins)
    high = high_func(bins)

    # draw samples from U[low, high) and apply inverse cdf
    rng = np.random.default_rng()
    uniform_samples = rng.uniform(low, high, size=size)

    return uniform_samples

# here, we define multiple sample strategies, that can be selected in the .nml file
sample_strategies = {
    "reproduce_hist": draw_from_hist,
    "min_max": partial(draw_uniform, low_func=lambda x: np.min(x), high_func=lambda x: np.max(x)),
}


def plot_results(samples, hist, savename=None, variable_units="warning: no units given"):
    """
    plot the obtained samples vs. the original histogram and save to disk
    """

    plt.style.use("paper/paper.mplstyle")

    fig = plt.figure(figsize=(4.72, 3.54))

    ax = plt.gca()

    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    ax.xaxis.get_offset_text().set_visible(False)

    # plot histogram data
    counts_sim = hist[1:, 0]
    bin_edges = hist[:, 1]
    bin_widths = np.diff(bin_edges)
    counts_sim_norm = counts_sim / np.sum(counts_sim * bin_widths)
    l1 = ax.stairs(counts_sim_norm, bin_edges, lw=2, color="brown", label="EMAC simulation")

    # plot samples
    sample_hist, _ = np.histogram(samples, hist[:, 1])
    bin_edges = hist[:, 1]
    bin_widths = np.diff(bin_edges)
    sample_hist_norm = sample_hist / np.sum(sample_hist * bin_widths)
    l2 = ax.stairs(sample_hist_norm, hist[:, 1], lw=2, color="royalblue", label="Marginal samples")

    ax.set_ylabel("Probability density")
    ax.set_xlabel("Concentration [molec cm$^{-3}$]")

    ax.legend(handles=[l1, l2], frameon=False, loc="upper left")

    plt.xlim(10**2, None)
    plt.xscale("log")

    plt.title("Marginal probability density of the NO$_2$ concentration", fontweight="bold")
    plt.tight_layout()

    if savename is not None:
        print(f"Saved plot to {savename}.pdf")
        plt.savefig(f"{savename}.pdf")

    plt.show()

def save_results(savename, arr, variable_names, variable_units):
    """
    save the results to a netCDF4 dataset on disk.
    samples_per_chunk determined into how many files the resulting file will be split.
    """

    ds = xr.Dataset()

    for col_idx in range(arr.shape[1]):
        var_name = variable_names[col_idx]
        var_unit = variable_units[col_idx]
        var_arr = arr[:, col_idx]

        # remove suffixes as specified in namelist.input
        var_name = var_name.split(remove_suffix)[0]

        da = xr.DataArray(var_arr, coords={'sample_nr': range(arr.shape[0])}, dims=['sample_nr'], name=var_name, attrs={"units": var_unit})
        ds[var_name] = da

    # apparently the output file must be deleted, if it already exists under the same name; else there might be Error 13s (permission denied)
    if os.path.exists(savename):
        os.remove(savename)

    print(f"Saving to file {savename}")
    ds.to_netcdf(savename, unlimited_dims={'sample_nr': True})


sample_func = sample_strategies[sampling_strategy]

# results are stored in a numpy array; each column represents one variable; each row represents one sample

if gas == 'all':
    filenames = glob(os.path.join(work_dir, "*.csv"))
else:
    filenames = glob(os.path.join(work_dir, f"{gas}.csv"))
    if filenames == []:
        raise Exception(f"No files found for gas '{gas}'")

result_arr = np.zeros((nr_samples, len(filenames)))
variable_names = []
variable_units_list = []

for col_idx, filename in enumerate(filenames):

    t_start = time.time()

    # grab units
    with open(filename) as f:
        headline = f.readline()
        variable_units = re.match(r"[^[]*\[([^]]*)\]", headline).groups()[0]
        variable_name = filename.split("/")[-1].split(".")[0]

        print(f"Processing ({col_idx + 1}/{len(filenames)}); variable '{variable_name}'; units '{variable_units}'", end="; ")

        variable_names.append(variable_name)
        variable_units_list.append(variable_units)

    hist = np.genfromtxt(filename, delimiter=",")
    samples = sample_func(hist, size=nr_samples)

    result_arr[:, col_idx] = samples

    t_end = time.time()
    print(f"{t_end - t_start:.1f} seconds")

    if draw_plots:
        savename = os.path.join(output_dir, Path(filename).stem)
        plot_results(samples, hist, savename=savename, variable_units=variable_units)

# remove all-zero variables
save_results(os.path.join(output_dir, output_name), result_arr, variable_names, variable_units_list)

#!/usr/bin/env python3
"""
Draws joint samples from a global simulation dataset.
"""

##############################################################################
# import packages:
##############################################################################

import uuid
from pathlib import Path

import argparse
import logging
import os, sys
import re
import time
import random

try:
    import tomllib # python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib # python < 3.11

from glob import glob

import math
import numpy as np
import xarray as xr
import gc

from shared import get_var, filter_fnames_by_permission, get_filenames_by_globs

# import for clustering with hdbscan:
from hdbscan_clusters import cluster_data_3d, ds_info

##############################################################################
# functions:
##############################################################################

# Patch for faster runtimes: Not every process needs to open every data
# file. For example, if we draw 10 × 100.000 samples from 12 files, the
# first process only needs to access the first two files. The logic for
# this is implemented here:

def get_processor_fileindex_samplenumbers(s, P, F):
    """
    Let s := number of samples to be drawn for each processor (e.g. 100000)
        P := total number of processors (e.g. 10)
        F := total number of raw datafiles to access (e.g. 12)

    This function returns a dict, with:
        - keys: the processor indices
        - values: dicts that map each file index to the number of samples to be drawn

    Example:
        d[3] = {4:8000, 5:2000} tells us: processor with index 3 should sample 8000 entries from file with index 4, and 2000 from file with index 5.
    """

    d = {}  # dict that stores the results
    cur_file_idx = 0  # current file index, begin at 0

    spf = math.ceil(s * P / F) # samples per file

    samples_taken_from_files = {}  # track how many samples are already taken from a certain file

    for p in range(P):

        d[p] = {}  # for each processor, begin with an empty dict
        samples_left = s  # samples this processor still needs to draw: begin with s
        while samples_left > 0:

            remaining_in_file = spf - samples_taken_from_files.get(cur_file_idx, 0)
            take = min(samples_left, remaining_in_file)

            if take > 0:
                d[p][cur_file_idx] = take
                samples_left -= take
                samples_taken_from_files[cur_file_idx] = (samples_taken_from_files.get(cur_file_idx, 0) + take)
            if samples_taken_from_files.get(cur_file_idx, 0) >= spf:
                cur_file_idx += 1

    # assert total number of samples is correct
    for v in d.values():
        assert sum(v.values()) == s

    print("processor_fileindex_samplenumbers:", d)
    return d

##############################################################################

def find_matching_files(filename, method, input_fname_globs):
    """
    Simulation results can be stored in multiple files (e.g. in global MECCA simulation: "MOM-ML4GPK.1r*_mecca_gp.nc", "MOM-ML4GPK.1r*_jval_gp.nc").
    These files must be associated with each other correctly.
    """

    valid_methods = ["global_mecca"]

    if method == "global_mecca":
        """
        used for global MECCA simulations
        """
        all_files = get_filenames_by_globs(input_fname_globs)

        m = re.search(r"\d{8}_\d{4}", filename)
        if m is None:
            raise ValueError(f"No timestamp token YYYYMMDD_HHMM found in filename: {filename}")

        identifying_digits = m.group(0)

        matching_files = [f for f in all_files if identifying_digits in f]

    elif method == "global_mecca_r2":
        all_files = get_filenames_by_globs(input_fname_globs)

        m = re.search(r"__\d{6}__", filename)
        if m is None:
            raise ValueError(f"No timestamp token __YYYYMM__ found in filename: {filename}")

        identifying_digits = m.group(0)

        matching_files = [f for f in all_files if identifying_digits in f]

    else:
        raise Exception(f"Invalid parameter 'file_assoc_method'. Must be one of {valid_methods}, user gave '{method}'.")

    return matching_files

##############################################################################

# print some general info:
print(f'python version: {sys.version_info}')
print(f'xarray version: {xr.__version__}')
print(f'numpy  version: {np.__version__}\n')

# parse arguments and namelist
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--namelist", type=str, help="namelist file", default="default.nml")
parser.add_argument("-s", "--nr_samples", type=int, help="number of total samples to draw", required=True)
parser.add_argument("-d", "--output_dir", type=str, help="output directory (e.g. 100k)", required=True)
parser.add_argument("-o", "--output_name", type=str, help="output name, appended to output_dir", default="")
parser.add_argument("-P", "--number_total_processors", type=int, help="number of total processors", required=True)
parser.add_argument("-p", "--processor_index", type=int, help="processor index", required=True)

args = parser.parse_args()

if args.output_name=="":
    args.output_name = args.output_dir + ".nc"

with open(args.namelist, "rb") as f:
    cfg = {**vars(args), **tomllib.load(f)}

cfg["clustering"] = cfg.get("clustering") # set to False if undefined

cfg["output_dir"] = os.path.join(cfg["work_dir"], "joint_samples", cfg["output_dir"])
os.makedirs(cfg["output_dir"], exist_ok=True)

# get all relevant filenames to process
all_fnames = sorted(glob(cfg["input_fname_globs"][0])) # 0 means only mecca, not jval
print('The following files (and the corresponding jval files) will be processed:')
[print(f'  {f}') for f in all_fnames]

processor_fileindex_samplenumbers = get_processor_fileindex_samplenumbers(
    args.nr_samples, args.number_total_processors, len(all_fnames))

# read vars.txt
varsfile = os.path.join(cfg["work_dir"], "vars.txt")
with open(varsfile, "r") as file:
    variables = file.readlines()
    variables = [v.split("\n")[0] for v in variables]  # \n should also not be parsed
print(f'\nOutput variables in {varsfile}:\n', ', '.join(variables), '\n')

# We are possibly iterating over multiple files to take samples from,
# here we store the partial results:
subsets = []

# iterate over the raw files, extract and flatten the requested variables
i_file_unskipped = 1  # track index of unskipped files (just for logging)
total_files_to_be_sampled = len(processor_fileindex_samplenumbers[args.processor_index].values())
for i_file, fname in enumerate(all_fnames):

    # Only process file if this processor is supposed to sample from it in the first place
    if i_file not in processor_fileindex_samplenumbers[args.processor_index]:
        print(f"skipping file {fname}, no samples to be drawn for this processor")
        continue
    
    print(f'cfg["output_dir"]    = {cfg["output_dir"]}')
    print(f'cfg["output_name"]   = {cfg["output_name"]}')
    print(f'i_file               = {i_file}')
    savename = os.path.join(cfg["output_dir"], f'{Path(cfg["output_name"]).stem}.nc')
    print(f'Output nc file will be: {savename}')

    fnames = find_matching_files(fname, cfg["file_assoc_method"], cfg["input_fname_globs"])
    print(f"Opening file '{fname}'")

    # Build the full dataset belonging to one file index (i.e. combine trace gases, J-values, etc.)
    ds_list = []
    for f in fnames:
        with xr.open_dataset(f) as partial_ds:
            ds_list.append(partial_ds)
    ds = xr.merge(ds_list, compat='no_conflicts')
    print('The dataset "ds" now contains the variables from these files:')
    [print(f'  {f}') for f in fnames]
    print(f'  {dict(ds.sizes)}')

    # ------------------------------------------------------------------------
    if (cfg["clustering"]):
        samples_per_file = processor_fileindex_samplenumbers[args.processor_index][i_file]
        ds = cluster_data_3d(ds, variables, samples_per_file, fname)
        # Reduce the number of samples to exactly cfg["nr_samples"]?
        print(f'cfg["nr_samples"] = {cfg["nr_samples"]}; ds.sizes = {ds.sizes}')
        # ds = ds.isel(sample_nr=range(cfg["nr_samples"]))
        # When clustering is switched on, data are written to a file for
        # each fname within the loop over all_fnames:
        ds = ds.assign_coords(sample_nr=range(ds.sizes["sample_nr"]))
        ds["time"].attrs.pop("units", None)
        # Apparently the output file must be deleted, if it already
        # exists under the same name; else there might be Error 13s
        # (permission denied)
        if os.path.exists(savename):
            os.remove(savename)
        print(f"Saving to file {savename}")
        ds.to_netcdf(savename, unlimited_dims={'sample_nr': True})
        # cleanup before creating next xarray dataset:
        del ds
        gc.collect() # gc = garbage collection
    # ------------------------------------------------------------------------
    else:
        # Quick patch to save a lot of memory: We use a variable that is guaranteed in the data (here: O3, is set in config) to determine the shape of the variables
        # Then we already build the sampling indices. This way, the individual variables can be subsampled while iterating over them.
        get_var(ds, cfg["test_var_name"], cfg)
        test_da, _ = get_var(ds, cfg["test_var_name"], cfg)
        test_shape = test_da.shape
        test_size = np.prod(test_shape)
        flat_sampling_indices = np.random.choice(test_size, size=processor_fileindex_samplenumbers[args.processor_index][i_file], replace=False)

        ds_flat = xr.Dataset()

        for i_var, var in enumerate(variables, start=1):

            print(f"[var {i_var}/{len(variables)}] [file {i_file_unskipped}/{total_files_to_be_sampled}] {var}")
            # remove suffix as specified in namelist.input
            var_name_nosuffix = var.split(cfg["remove_suffix"])[0]
            da, units = get_var(ds, var, cfg)
            units = "" if units is None else units

            # flatten
            #da = da.data.flatten()  # todo this can lead to wrong order, can be removed if da.stack below works
            da = da.stack(sample=("time", "eta", "lat", "lon"))
            da = xr.DataArray(da, dims=("sample_nr",), coords={"sample_nr": np.arange(da.shape[0])})
            da = da.isel(sample_nr=flat_sampling_indices)

            ds_flat[var_name_nosuffix] = da
            ds_flat[var_name_nosuffix].attrs["units"] = units

        # now we have a flat dataset with all requested variables → draw random samples
        ds_flat = ds_flat.isel(sample_nr=np.random.choice(
            ds_flat.sample_nr.size, size=processor_fileindex_samplenumbers[args.processor_index][i_file], replace=False))

        # remove all-zero variables
        all_zero_vars = [name for name, da in ds_flat.data_vars.items() if (da.max() == 0 and da.min() == 0).all()]
        ds_flat_cleaned = ds_flat.drop_vars(all_zero_vars)

        # print info:
        print(f"Dropped following variables because they are all-zero (total: {len(all_zero_vars)}):")
        for var in all_zero_vars:
            print(f"{var}")

        subsets.append(ds_flat_cleaned)
        i_file_unskipped += 1
    # ------------------------------------------------------------------------

# When clustering is switched off, data from all files (all_fnames) are
# collected in the subsets list and written to a file after the loop:
if (not cfg["clustering"]):
    ds = xr.concat(subsets, dim="sample_nr")
    ds = ds.isel(sample_nr=range(cfg["nr_samples"]))
    ds = ds.assign_coords(sample_nr=range(ds.sizes["sample_nr"]))
    ds["time"].attrs.pop("units", None)

    savename = os.path.join(cfg["output_dir"], Path(cfg['output_name']).stem + ".nc")

    # apparently the output file must be deleted, if it already exists under the same name; else there might be Error 13s (permission denied)
    if os.path.exists(savename):
        os.remove(savename)

    print(f"Saving to file {savename}")
    ds.to_netcdf(savename, unlimited_dims={'sample_nr': True})

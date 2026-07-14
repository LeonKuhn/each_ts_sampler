#!/usr/bin/env python3
"""
Generate a list of variables contained in the global simulation data.
These variables are later sampled for the training set assembly.
"""

import xarray as xr
import argparse

try:
    import tomllib # python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib # python < 3.11

from glob import glob
import os

# parse toml
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--toml", default="default.toml", type=str, help="toml file")
args = parser.parse_args()

toml = args.toml

print(f"Using toml: {toml}")

with open(args.toml, "rb") as f:
    toml_cfg = tomllib.load(f)
    cfg = {**vars(args), **toml_cfg}


# pick suitable example files (use the first file that matches each input filename glob)
example_files = []
for g in cfg["input_fname_globs"]:
    example_files.append(glob(g)[0])
print("Extracting variables from the following files:")
print("\n".join(example_files))

# vars_found is the set of variables returned at the end; start with the fixed_vars
vars_found = set(cfg["fixed_vars"])

# iterate over the example files and extract variable names based on their units
for fname in example_files:
    with xr.open_dataset(fname) as ds:
        for name, var in ds.data_vars.items():
            units = var.attrs.get("units")
            if units in cfg["accepted_units"] and name not in cfg["blacklist"]:
                vars_found.add(name)

vars_found = sorted(vars_found)

# write the result to a .txt file
print("The following variables were found:")
print("\n".join(vars_found))
os.makedirs(cfg["work_dir"], exist_ok=True)

with open(os.path.join(cfg["work_dir"], "vars.txt"), 'w') as f:
    for v in vars_found:
        f.write(f"{v}\n")

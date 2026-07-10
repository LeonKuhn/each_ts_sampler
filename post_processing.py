"""
A few postprocessing steps:

1. Resolves the issue of sample_nr being shifted by 1 between xpcaaba input and output
2. removes "lev", "lat", "lon" dimensions of the MECCA/CAABA output
3. final selection of variables in the output and saving as .parquet files
"""

from glob import glob
import xarray as xr
from pathlib import Path
import os
import numpy as np
import argparse
import tomli

# parse arguments and namelist
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, help="input directory", required=True)
parser.add_argument("-x", "--feature_dir", type=str, help="sub-directory of the features", required=True)
parser.add_argument("-y", "--target_dir", type=str, help="sub-directory of the targets", required=True)

parser.add_argument("-p", "--processor_idx", type=int, help="processor index (0-based)", required=True)
parser.add_argument("-t", "--total_processors", type=int, help="total processors", required=True)
parser.add_argument("-n", "--namelist", type=str, help="namelist file", default="default.nml")
args = parser.parse_args()

with open(args.namelist, "rb") as f:
    toml_cfg = tomli.load(f)
    cfg = {**vars(args), **toml_cfg}

keep_vars = set(cfg["keep_vars"])
drop_vars = set(cfg["drop_vars"])
base_dir = args.input_dir
x_glob = sorted(glob(os.path.join(base_dir, args.feature_dir, "*.nc")))
y_glob = sorted(glob(os.path.join(base_dir, args.target_dir, "*.nc")))

out_dir = base_dir + "_parquet"
os.makedirs(out_dir, exist_ok=True)

def obtain_x_and_y_column_names(x, y):
    """
    Takes a .nc file of inputs (x) and a .nc file of output (y) and returns the column names to keep
    """

    ds_x = xr.open_dataset(x)
    ds_y = xr.open_dataset(y)

    # drop unused dimensions from targets
    ds_y = ds_y.squeeze(drop=True)

    # convert both to pandas
    df_x = ds_x.to_dataframe().reset_index()
    df_y = ds_y.to_dataframe().reset_index()

    ## select variables
    intersected_vars = set(df_x.columns) & set(df_y.columns)
    x_vars_not_in_intersect = {v for v in df_x.columns if v not in intersected_vars}
    y_vars_not_in_intersect = {v for v in df_y.columns if v not in intersected_vars}

    """
    x contains the J_* variables that y does not contain. These should be kept.
    blacklist contains all variable names of y that are not in the intersect (unless they are in keep_vars) + the variables in drop_vars
    """
    blacklist = set.union(y_vars_not_in_intersect - keep_vars, drop_vars)
    keep_x_vars = [v for v in df_x.columns if v not in blacklist]
    keep_y_vars = [v for v in df_y.columns if v not in blacklist and v != "time"]  # y has a time variable that should be dropped
    df_x = df_x[keep_x_vars]
    df_y = df_y[keep_y_vars]

    x_not_in_y = {v for v in df_x.columns if v not in df_y.columns}
    y_not_in_x = {v for v in df_y.columns if v not in df_x.columns}
    print("variables in X but not in Y:")
    print(x_not_in_y)
    print("variables in Y but not in X:")
    print(y_not_in_x)

    # check if there are any all-0 columns
    is_zero_x = (df_x==0).all(axis=0)
    is_zero_y = (df_y==0).all(axis=0)

    # find zero-columns
    zero_cols_x = df_x.columns[is_zero_x]
    zero_cols_y = df_y.columns[is_zero_y]

    print("X columns that are all zero:")
    print(list(zero_cols_x))

    print("Y columns that are all zero:")
    print(list(zero_cols_y))

    # drop them
    df_x = df_x.drop(columns=zero_cols_x)
    df_y = df_y.drop(columns=zero_cols_y)

    return df_x.columns, df_y.columns

def x_and_y_to_parquet(x_fn, y_fn, x_cols, y_cols):
    """
    Takes a pair of .nc files, representing features (x) and targets (y) and converts them to parquet.
    """

    x = xr.open_dataset(x_fn).squeeze(drop=True)
    x = x.to_pandas().reset_index()
    x = x[x_cols]

    y = xr.open_dataset(y_fn).squeeze(drop=True)
    y = y.to_pandas().reset_index()
    print("Reducing sample_nr in features by 1!"); y["sample_nr"] = y["sample_nr"] - 1
    x["cair"] = y["cair"]
    y = y[y_cols]

    return x, y

x_cols, y_cols = obtain_x_and_y_column_names(x_glob[-1], y_glob[-1])

# Distribute over processors
p, t = args.processor_idx, args.total_processors
x_files = np.array_split(x_glob, t)[p]
y_files = np.array_split(y_glob, t)[p]

for xf, yf in zip(x_files, y_files):
    print(f"Processing X={xf} and Y={yf}")
    x_ds, y_ds = x_and_y_to_parquet(xf, yf, x_cols, y_cols)

    x_out_fn = os.path.join(out_dir, Path(xf).parent.stem, Path(xf).stem + ".pq")
    y_out_fn = os.path.join(out_dir, Path(yf).parent.stem, Path(yf).stem + ".pq")

    os.makedirs(Path(x_out_fn).parent, exist_ok=True)
    os.makedirs(Path(y_out_fn).parent, exist_ok=True)

    print(f"Writing to file '{x_out_fn}'")
    x_ds.to_parquet(x_out_fn)

    print(f"Writing to file '{y_out_fn}'")
    y_ds.to_parquet(y_out_fn)

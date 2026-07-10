"""
A collection of shared functions
"""

import os
from glob import glob

def filter_fnames_by_permission(fnames):
    """
    Takes a list of filenames (fnames) and returns only those for which there exist read permissions.
    """

    fnames_with_permissions = []

    for f in fnames:
        read_permissions = os.access(f, os.R_OK)
        if read_permissions:
            fnames_with_permissions.append(f)
        else:
            print(f"Warning: File {f} has no read permissions.")

    return fnames_with_permissions

def get_var(ds, var, cfg):
    """
    read a variable from an xarray dataset.
    - Some variables (like 'p' in the MECCA/CAABA simulations) are only implicitly contained through hybrid pressure coordinates. These can be reconstructed using lambda functions.
    - Conversion between mole fractions (mol/mol) and concentrations (µg/m^3) is also included here
    """

    broadcast_example_var = "NO2_start" # for broadcasting of variables like latitude or time

    if var == "p":
        da = ds["hyam"] + ds["aps"] * ds["hybm"]
        units = "Pa"

    # the following variables must be broadcasted
    elif var == "lat":
        da = ds.lat
        units = "degrees_north"
        da = da.broadcast_like(ds[broadcast_example_var])

    elif var == "lon":
        da = ds.lon
        units = "degrees_east"
        da = da.broadcast_like(ds[broadcast_example_var])

    elif var == "time":
        da = ds.time
        units = ""
        da = da.broadcast_like(ds[broadcast_example_var])

    elif var in ds.variables:  # case 1: var exists in the dataset

        da, units = ds[var], None

        if var[:2] == "J_":
            units = "1/s"

        elif var == "tm1":
            units = "K"

        elif "units" in da.attrs:

            if da.attrs["units"] == "molec/cm3":
                # assume a trace gas
                if cfg["out_format_gases"] == "conc":
                    units = "molec/cm3"
                elif cfg["out_format_gases"] == "vmr":  # if chosen, convert to vmr
                    da = da / ds[cfg["cair_variable_name"]]
                    units = "mol/mol"
                else:
                    raise Exception("Invalid parameter 'out_format_gases'")

            elif da.attrs["units"] == "mol/mol":
                # assume a trace gas
                if cfg["out_format_gases"] == "vmr":
                    units = "mol/mol"
                elif cfg["out_format_gases"] == "conc":
                    da = da * ds[cfg["cair_variable_name"]]
                    units = "molec/cm3"
                else:
                    raise Exception("Invalid parameter 'out_format_gases'")

    else:  # case 2: var cannot be returned
        raise KeyError(f"Variable '{var}' is not contained in '{ds.encoding["source"]}' and has no recipe for reconstruction from other variables")

    return da, units

def get_filenames_by_globs(fname_globs):
    # given a valid glob, return all matching filenames.

    fnames = []
    for g in fname_globs:
        for fname in glob(g):
            fnames.append(fname)

    return list(set(fnames))

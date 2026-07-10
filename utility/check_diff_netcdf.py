"""
A tool to check the difference between xpcaaba input files (e.g. from extract_hist.py + draw_marginal_samples.py vs. draw_joint_samples_v3.py)
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

input_1 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4/marginal/10m_0.nc"
input_2 = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4/joint/10m_0.nc"

ds_1 = xr.open_dataset(input_1)
ds_2 = xr.open_dataset(input_2)

# compare dataset dimensions
print("ds_1 dimensions:", ds_1.dims)
print("ds_2 dimensions:", ds_2.dims)

dimensions_ok = True
for dim in ds_1.dims:
    if dim not in ds_2.dims:
        print(f"Dimension '{dim}' exists in ds_1, but not in ds_2!")
        dimensions_ok = False

if dimensions_ok:
    print("Dimensions of the two datasets are identical")
else:
    print("Dimensions of the two datasets are NOT identical")

# ensure both datasets have the same variables
same_vars_exist = True
missing_vars = []
for var in ds_1.variables:
    if var not in ds_2.variables:
        print(f"Variable '{var}' exists in ds_1 but not in ds_2")
        same_vars_exist = False
        missing_vars.append(var)

if same_vars_exist:
    print("Variables of the two datasets are identical")
else:
    print("Variables of the two datasets are NOT identical. The following vars exist in ds_1 but not in ds_2:")
    print(missing_vars)

# ensure variables have the same coords
coords_ok = True
faulty_coord_vars = []
for var in ds_1.variables:
    coords_1 = ds_1[var].coords
    coords_2 = ds_2[var].coords
    if not coords_1.equals(coords_2):
        print(f"Variable '{var}' has different coords in ds_1 and ds_2")
        coords_ok = False
        faulty_coord_vars.append(var)

if coords_ok:
    print("Coordinates of the two datasets are identical")
else:
    print("Coordinates of the two datasets are NOT identical. The following vars are affected:")
    print(faulty_coord_vars)

# ensure variables have the same units
units_ok = True
faulty_units_vars = []
for var in ds_1.variables:
    try:
        units_1 = ds_1[var].units
    except:
        print(f"variable '{var}' has no units in ds_1, skipping")
        continue

    units_2 = ds_2[var].units
    if not units_1 == units_2:
        print(f"Variable '{var}' has different units in ds_1 ({units_1}) and ds_2 ({units_2})")
        units_ok = False
        faulty_units_vars.append(var)

if units_ok:
    print("Units of the two datasets are identical")
else:
    print("Units of the two datasets are NOT identical. The following vars are affected:")
    print(faulty_units_vars)

# check mean values of variables
warning_threshold = 0.50
means_ok = True
faulty_means_vars = []

for var in ds_1.variables:
    if var in ds_2.variables:
        da_1, da_2 = ds_1[var], ds_2[var]
        m1, m2 = da_1.mean(), da_2.mean()
        rel_diff = np.abs((m1 - m2) / m1)

        if rel_diff > warning_threshold:
            print(f"Variable '{var}' has a mean relative difference of {rel_diff:.2%} between ds_1 ({float(m1):.1E}) and ds_2 ({float(m2):.1E})")
            means_ok = False
            faulty_means_vars.append(var)

if means_ok:
    print("Means of the two datasets are identical")
else:
    print("Means of the two datasets are NOT identical. The following vars are affected:")
    print(faulty_means_vars)

# draw a test histogram
var = "SO2"
n_bins = 50
x_scale, y_scale = "log", "linear"

v1, v2 = ds_1[var], ds_2[var]
all_min = min(v1.min(), v2.min())
all_max = min(v1.max(), v2.max())

bins = np.geomspace(all_min, all_max, n_bins + 1, endpoint=True)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.hist(ds_1[var].data.flatten(), bins=bins)
ax2.hist(ds_2[var].data.flatten(), bins=bins)

for ax in (ax1, ax2):
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlabel(f"{var} [{v1.units}]")

plt.show()
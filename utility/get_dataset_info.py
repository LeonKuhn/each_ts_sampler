"""
A small utility that obtains the number of species and other information about xpcaaba input files
"""
import xarray as xr

input_file = "/work/mm0062/b302092/datasets/MOM-ML4GPK.1r_v4/marginal/10m_0.nc"
ds = xr.open_dataset(input_file)

# count tracers, based on units
ctr = 0
for var in ds.variables:
    if getattr(ds[var], "units", None) in ["mol/mol", "molec/cm3"]:
        ctr += 1

print(f"The dataset contains {ctr} chemical species")

# count photolysis constants
ctr = 0
for var in ds.variables:
    if var[:2] == "J_":
        ctr += 1

print(f"The dataset contains {ctr} photolysis constants")

"""
Counts the number of entries in the global simulation
"""

from glob import glob
import xarray as xr

input_glob = glob("/work/mm0062/b302052/MOM/EXP/MOM-ML4GPK.1r/MOM-ML4GPK.*_mecca_gp.nc")

entries = 0
for fn in input_glob:

    ds = xr.open_dataset(fn)
    entries += ds.time.size * ds.lon.size * ds.lat.size * ds.eta.size

print("Total entries:", entries)
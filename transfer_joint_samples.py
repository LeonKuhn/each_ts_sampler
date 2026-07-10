"""
A small tool to copy chunks of the joint initial states to the "datasets" folder.
Should not be done by hand because draw_joint_samples.sh occasionally times out due to the Levante job time limits,
i.e. the filename numbering may have some missing numbers etc.
"""

import os
from glob import glob

fname_glob = "/work/mm0062/b302092/each_ts_sampler/MOM-ML4GPK.1r/joint_samples/10m_v4/*.nc"
target_dir = "/work/mm0062/b302092/each_ts_sampler/MOM-ML4GPK.1r/joint_selected/10m_4/"
max_fnames = 100

os.makedirs(target_dir, exist_ok=True)

fnames = sorted(glob(fname_glob))

for i, fname in enumerate(fnames):
    if i == max_fnames:
        break
    new_fname = fname.split("/")[-1].split("_")[0] + f"_{i}.nc"
    print(f"Copying {fname} -> {new_fname}")
    os.system(f"cp {fname} {os.path.join(target_dir, new_fname)}")

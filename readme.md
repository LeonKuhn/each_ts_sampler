## About `each_ts_sampler`
This is a the repository for the sampling routine from MOM simulation data used within the EACH project.
For context, please refer to our ESSD publication: <Todo: add link to paper>

## Requirements
A list of dependencies is given in `env.yml`, which can be used to install a suitable environment via `conda` through
```
conda env create -f env.yml
```

## Usage

Check the pre-supplied toml file (`MOM-ML4GPK.2r_conc.toml`) and see if it matches your needs. The meaning of the configuration
keys is given below.

`each_ts_sampler` expects a default toml (`default.toml`), which can be created e.g. through a soft link:
```
cd config
ln -sf MOM-ML4GPK.2r_conc.toml default.toml
```

Relevant config keys:
- `input_fname_globs`: list of glob patterns for the input files
- `work_dir`: the working directory, where `each_ts_sampler` will write its output

Before running any code, make sure to load a conda environment, ideally built using the supplied ``env.yml`` file.

#### 1) Create list of variables

In this first step, test files from the global simulation (see config key `input_fname_globs`) are inspected
to determine which variables are present and will later be sampled.

Run the python file `get_var_list.py`.

Relevant config keys:
- `fixed_vars`: list of variables that are always extracted
- `accepted_units`: all variables whose units are in this list are extracted
- `blacklist`: list of variables that are ignored, even if their units are in `accepted_units`

#### 2) Draw joint samples

This is the canonical sampling method. Samples are drawn by accessing the global simulation data at
random indices.

Joint sampling is executed via slurm job arrays:

```
sbatch --array=0-11 draw_joint_samples.sh -s 1000000 -d 12m_output
```

with the arguments:
- `-s`: number of samples to draw *per task*
- `-d`: output subdirectory (within the working directory `work_dir` defined in the toml file)
- `-t`: (optional) toml file, defaults to `config/default.toml`"


#### 3) Draw marginal samples

##### 3.1) Producing histograms

First, the histogram of each variable is determined from all files of the global simulation: Open the slurm submission script `extract_hist.sh` and change the variable `NWORKERS` to the number of job array elements you wish to run (e.g. 20). The number must
match the requested job array size (e.g. 0-19). Then, run the slurm submission script:

```
sbatch extract_hist.sh
```

##### 3.2) Drawing samples from the histograms

Afterwards, run
```
sbatch --array=0-11 draw_marginal_samples.sh -s 1000000 -d 12m_output
```

#### 4) Reproducing paper plot of marginal histograms

The plot is reproduced by running
```
python paper/draw_marginal_samples_paper.py -s 100000 -d paper_hist -p -g NO2_start 
```

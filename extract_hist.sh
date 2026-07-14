#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=mm0062
#SBATCH --job-name=extract_hists
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/extract_hists.%A_%a.out
#SBATCH --error=logs/extract_hists.%A_%a.err

# number of workers (must match array size)
NWORKERS=20

# default toml
toml="default.toml"

while getopts ":t:" opt; do
  case $opt in
    t) toml="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

# parse config
work_dir=$(sed -n 's/^work_dir *= *"\([^"]*\)".*/\1/p' "$toml")
var_file="${work_dir}/vars.txt"

TASK_ID=$SLURM_ARRAY_TASK_ID

echo "Worker $TASK_ID processing every $NWORKERS-th line"

# total number of lines
NLINES=$(wc -l < "$var_file")

for ((i=TASK_ID+1; i<=NLINES; i+=NWORKERS)); do
    var=$(sed -n "${i}p" "$var_file")
    echo "Worker $TASK_ID processing: $var"
    python -u extract_hist.py -t "$toml" -v "$var"
done

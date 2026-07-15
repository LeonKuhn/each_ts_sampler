#!/bin/bash -l
#SBATCH -J draw_joint_samples
#SBATCH -o ./logs/draw_joint_samples.%A_%a.out
#SBATCH -e ./logs/draw_joint_samples.%A_%a.err

#SBATCH --partition=compute
#SBATCH --account=mm0062

#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --cpus-per-task=1

# parse arguments:
while getopts ":t:s:d:o:" opt; do
  case $opt in
    t) TOML="$OPTARG" ;;
    s) N_SAMPLES="$OPTARG" ;;
    d) OUTPUT_DIR="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

# construct command
CMD=(python -u draw_joint_samples.py
     -s "$N_SAMPLES"
     -d "$OUTPUT_DIR"
     -p "$SLURM_ARRAY_TASK_ID"
     -P "$SLURM_ARRAY_TASK_COUNT")

[ -t "$TOML" ] && CMD+=(-t "$TOML")

echo "Running command:" "${CMD[@]}"
"${CMD[@]}"

#!/bin/bash -l
#SBATCH -J draw_marginal_samples
#SBATCH -o ./logs/draw_marginal_samples.%A_%a.out
#SBATCH -e ./logs/draw_marginal_samples.%A_%a.err

#SBATCH --partition=compute
#SBATCH --account=mm0062

#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1

# parse arguments:
while getopts ":t:s:d:" opt; do
  case $opt in
    t) TOML="$OPTARG" ;;
    s) N_SAMPLES="$OPTARG" ;;
    d) OUTPUT_DIR="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

# strip the ".nc" extension and append array ID
BASENAME="${OUTPUT_DIR%.nc}"
OUTPUT_FILENAME_FINAL="${BASENAME}_${SLURM_ARRAY_TASK_ID}.nc"

conda activate each_ts_sampler  # ← change to environment name of your choice

# construct command
CMD=(python -u draw_marginal_samples.py
     -s "$N_SAMPLES"
     -d "$OUTPUT_DIR"
     -o "$OUTPUT_FILENAME_FINAL")

[ -n "$TOML" ] && CMD+=(-t "$TOML")

echo "Running command:" "${CMD[@]}"
"${CMD[@]}"

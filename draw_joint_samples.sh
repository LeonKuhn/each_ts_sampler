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
while getopts ":n:s:d:o:" opt; do
  case $opt in
    n) NAMELIST="$OPTARG" ;;
    s) N_SAMPLES="$OPTARG" ;;
    d) OUTPUT_DIR="$OPTARG" ;;
    o) OUTPUT_FILENAME="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

P=${SLURM_ARRAY_TASK_COUNT}
p=${SLURM_ARRAY_TASK_ID}
# p2d is p with 2 digits:
p2d=$(printf "%02d" $((p+1)))

# strip the ".nc" extension and append array ID
BASENAME="${OUTPUT_FILENAME%.nc}"
OUTPUT_FILENAME_FINAL="${BASENAME}_${p2d}.nc"

# construct command
CMD=(python -u draw_joint_samples.py
     -s "$N_SAMPLES"
     -d "$OUTPUT_DIR"
     -o "$OUTPUT_FILENAME_FINAL"
     -p "$p"
     -P "$P")

[ -n "$NAMELIST" ] && CMD+=(-n "$NAMELIST")

echo "Running command:" "${CMD[@]}"
"${CMD[@]}"

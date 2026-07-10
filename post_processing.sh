#!/bin/bash -l
#SBATCH --job-name=post_processing
#SBATCH --output=logs/post_processing_%A_%a.out
#SBATCH --error=logs/post_processing_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --array=0-19
#SBATCH --partition=compute
#SBATCH --account=mm0062

# parse arguments:
while getopts ":n:i:x:y:" opt; do
  case $opt in
    n) NAMELIST="$OPTARG" ;;
    i) INPUT_DIR="$OPTARG" ;;
    x) X_SUBDIR="$OPTARG" ;;
    y) Y_SUBDIR="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

# processor index and total processors
P=$SLURM_ARRAY_TASK_ID
T=$SLURM_ARRAY_TASK_COUNT

# construct command
CMD=(python -u post_processing.py
     -i "$INPUT_DIR"
     -x "$X_SUBDIR"
     -y "$Y_SUBDIR"
     -y "$Y_SUBDIR"
     -p "$P"
     -t "$T")

[ -n "$NAMELIST" ] && CMD+=(-n "$NAMELIST")

echo "Running command:" "${CMD[@]}"
"${CMD[@]}"
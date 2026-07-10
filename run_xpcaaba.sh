#!/bin/bash

# arguments passed from sbatch
NAME=$1         # e.g., "m32_t720"
B_VAL=$2        # value for -b
T_VAL=$3        # value for -t
START=$4        # start index
END=$5          # end index
MAX_CONCURRENT=$6   # number of concurrent runs
INPUT_PATTERN=$7    # input file name pattern, e.g.

# safety checks for input variables
if [[ -z "$NAME" || -z "$B_VAL" || -z "$T_VAL" || -z "$START" || -z "$END" || -z "$MAX_CONCURRENT" || -z "$INPUT_PATTERN" ]]; then
    echo "Usage: $0 <NAME> <B_VAL> <T_VAL> <START> <END> <MAX_CONCURRENT> <INPUT_PATTERN>"
    echo "Example: $0 m32_t720 4 720 0 9 5 \"initial_conditions_%d.nc\""
    exit 1
fi

if (( START > END )); then
    echo "Error: start index greater than end index"
    exit 1
fi

if (( MAX_CONCURRENT < 1 )); then
    echo "Error: MAX_CONCURRENT must be at least 1"
    exit 1
fi

# loop through desired indices
for i in $(seq "$START" "$END"); do
    IDENTIFIER="${NAME}_${i}"
    INPUT_FILE=$(printf "$INPUT_PATTERN" "$i")

    # respect job submission limits
    while (( $(squeue -u "$USER" -n xpcaaba-EACH | tail -n +2 | wc -l) >= MAX_CONCURRENT )); do
        echo "Concurrency limit reached ($MAX_CONCURRENT). Waiting..."
        sleep 5
    done

    echo "Submitting: ./xpcaaba-EACH -i ${IDENTIFIER} -b ${B_VAL} -t ${T_VAL} ${INPUT_FILE}"
    sbatch ./xpcaaba-EACH -i "$IDENTIFIER" -b "$B_VAL" -t "$T_VAL" "$INPUT_FILE"
    sleep 0.2   # optional: avoid spamming the scheduler
done

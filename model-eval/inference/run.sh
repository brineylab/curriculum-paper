#!/bin/bash

# error handling
error_handling() {
    echo "Error on line $1"
    exit 1
}
trap 'error_handling $LINENO' ERR

# allow ctrl c to kill all background jobs
trap 'echo "Caught SIGINT, stopping all background jobs"; kill 0' SIGINT

# models to run inference on
declare -A models
models=(  
    ["curriculum"]="./models/mxd-curr-max0.7-k15_50M-ESM_100k-stp_2024-12-11/"
)

# output file
output_csv="./test_CEL-accuracy.csv"
echo "model_name,unpaired_CEL,unpaired_accuracy,paired_CEL,paired_accuracy" > "$output_csv"

# inference
for model_name in "${!models[@]}"; do
    echo "Running inference for $model_name"
    model_path="${models[$model_name]}"
    
    python3 1-inference-loss_no-sep.py --model "$model_path" --model_name "$model_name" --output_file "$output_csv" & # runs for all models at the same time, remove & to run one at a time
    
done

wait # waits for all processes to terminate

echo "All inference is complete"
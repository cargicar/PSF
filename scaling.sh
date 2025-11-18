#!/bin/bash

# Script to run train_flow_g4.py with --npoints varying from 500 to 10000 
# in increments of 500.

# Define the starting point, step size, and end point for --npoints
START=500
STEP=500
END=10000
MODEL_NAME="pvcnn2"
NITER=1

# Use the 'seq' command to generate the sequence of numbers (500, 1000, 1500, ..., 10000)
# and iterate over them using a for loop.
for NPOINTS_VALUE in $(seq $START $STEP $END); do
    
    # Record the current date and time before starting the run
    START_TIME=$(date +%Y-%m-%d\ %H:%M:%S)
    
    # Inform the user which training run is starting
    echo "=================================================="
    echo "STARTING RUN: npoints=${NPOINTS_VALUE}"
    echo "START TIME: ${START_TIME}" # Display the recorded start time
    echo "=================================================="

    # Construct and execute the full command
    # We use double quotes around the variables in the command just in case,
    # though here they hold numbers.
    python scaling.py \
        --niter ${NITER} \
        --enable_profiling \
        --model_name ${MODEL_NAME} \
        --npoints ${NPOINTS_VALUE}
        
    # Check the exit status of the previous command (the python script)
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Training run for npoints=${NPOINTS_VALUE} completed successfully."
    else
        echo "FAILURE: Training run for npoints=${NPOINTS_VALUE} encountered an error."
        # Optionally, uncomment the line below to stop the script after the first failure
        # exit 1
    fi

    echo "--------------------------------------------------"
    echo "" # Add an empty line for readability between runs

done

echo "All training runs complete!"
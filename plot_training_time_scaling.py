import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

def parse_log_file(file_path, model_name):
    """
    Reads a log file, splits it into runs, and extracts npoints, start time,
    and end time for each run.
    """
    if not os.path.exists(file_path):
        print(f"Error: Log file not found at {file_path}")
        return []

    print(f"Parsing log file: {file_path} for model: {model_name}")
    
    with open(file_path, 'r') as f:
        npoints = []
        times = []
        intervals = []
        for line in f:
            if "STARTING RUN:" in line:
                line = line.split('=')
                npoints.append(int(line[1]))
            elif "START TIME:" in line:
                line = line.split('TIME: ')
                times.append(line[1].strip())
                
        times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in times]

        # Iterate from the second element (index 1) to the end
        for i in range(1, len(times)):
            # Calculate the difference: Current time - Previous time
            interval = times[i] - times[i-1]
            interval = interval.total_seconds()
            intervals.append(interval)

        results= {
                    'npoints': npoints,
                    'intervals': intervals,
                    'model': model_name
                }

    return results

def plot_comparison(df):
    """
    Generates a scatter and line plot comparing 'npoints' vs 'duration_seconds'.
    """
    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plotting using model name as the hue
    for name, group in df.groupby('model'):
        # Scatter plot for individual data points
        plt.scatter(group['npoints'], group['intervals'], label=f'{name} (Data)', marker='o', s=100)
        # Line plot connecting the data points
        #plt.plot(group['npoints'], group['intervals'], linestyle='--', alpha=0.6, label=f'{name} (Trend)')

    plt.title('Training Time vs. Number of Points (npoints) Scaling', fontsize=16)
    plt.xlabel('Number of Points (npoints)', fontsize=14)
    plt.ylabel('Total Training Duration (Seconds)', fontsize=14)
    plt.legend(title='Model', loc='upper left')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.xticks(sorted(df['npoints'].unique())) # Ensure only relevant ticks are shown
    plt.tight_layout()
    plt.savefig('training_time_scaling_comparison.png')

if __name__ == '__main__':
    # --- Configuration ---
    LOG_FILE_1 = 'output_scaling_calopodit.txt'
    MODEL_NAME_1 = 'calopodit'
    
    LOG_FILE_2 = 'output_scaling_pvcnn2.txt'
    MODEL_NAME_2 = 'pvcnn2'
    # ---------------------
    
    # 1. Parse both log files
    data_calopodit = parse_log_file(LOG_FILE_1, MODEL_NAME_1)
    data_pvcnn2 = parse_log_file(LOG_FILE_2, MODEL_NAME_2)
    plt.scatter(data_calopodit['npoints'][:6], data_calopodit['intervals'][:6], label=f'{data_calopodit['model']}', marker='o', s=10)
    plt.scatter(data_pvcnn2['npoints'][:len(data_pvcnn2['intervals'])], data_pvcnn2['intervals'], label=f'{data_pvcnn2['model']}', marker='x', s=10)
        # Line plot connecting the data points
        #plt.plot(group['npoints'], group['intervals'], linestyle='--', alpha=0.6, label=f'{name} (Trend)')

    plt.title('Training Time vs. Number of Points (npoints) Scaling', fontsize=10)
    plt.xlabel('Number of Points (npoints)', fontsize=8)
    plt.ylabel('Total Training Duration (Seconds)', fontsize=8)
    plt.legend(title='Model', loc='upper right')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('training_time_scaling_comparison.png')
 
    # 2. Combine results and create a DataFrame
    # all_data = data_calopodit + data_pvcnn2
    # df = pd.DataFrame(all_data)
    
    # # 3. Sort by npoints for correct plotting order
    # df = df.sort_values(by='npoints')
    
    # # 4. Generate the plot
    # plot_comparison(df)

    # print("\nDataFrame containing all extracted data:")
    # print(df)
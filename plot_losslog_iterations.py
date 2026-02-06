import re
import matplotlib.pyplot as plt
import sys
import numpy as np  # Added for efficient average calculation

def calculate_moving_average(data, window_size):
    """
    Calculates the simple moving average of a list or array.
    
    Args:
        data (list or np.array): The numerical data to smooth.
        window_size (int): The number of points to include in the average.
        
    Returns:
        np.array: The smoothed data.
    """
    if len(data) < window_size:
        return np.array(data)
        
    # Create a simple kernel (window) where every item has equal weight
    kernel = np.ones(window_size) / window_size
    # Use convolution to slide the window across the data
    # mode='valid' ensures we only compute averages where we have a full window
    return np.convolve(data, kernel, mode='valid')

def plot_loss_from_log(file_path):
    iterations = []
    losses = []
    
    # Regex to parse the log line
    pattern = re.compile(r'\[\s*(\d+)[^\]]*\]\[\s*(\d+)/(\d+)\]\s+loss:\s+([\d\.]+)')

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    curr_iter = int(match.group(2))
                    total_iter_per_epoch = int(match.group(3))
                    loss = float(match.group(4))
                    
                    global_step = (epoch * total_iter_per_epoch) + curr_iter
                    iterations.append(global_step)
                    losses.append(loss)
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    if not iterations:
        print("No loss data found. Check your log file format.")
        return

    # --- SMOOTHING LOGIC ---
    # Define how many points to average over (adjust this to make it smoother/sharper)
    window_size = 200
    smoothed_losses = calculate_moving_average(losses, window_size)
    
    # Adjust iterations to match the length of smoothed data 
    # (Convolution with mode='valid' shortens the array by window_size - 1)
    smoothed_iterations = iterations[window_size - 1:]

    # --- PLOTTING ---
    plt.figure(figsize=(10, 5))
    
    # 1. Plot the raw, jittery data (faded out)
    plt.plot(iterations, losses, linewidth=1, alpha=0.3, color='gray', label='Raw Training Loss')
    
    # 2. Plot the soothing average (solid line)
    plt.plot(smoothed_iterations, smoothed_losses, linewidth=2, color='tab:blue', label=f'Moving Average (n={window_size})')
    
    plt.xlabel('Global Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iterations (Smoothed)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save logic (kept from your original script)
    output_path = f"{file_path[:-11]}/syn/loss_plot.png"
    # Ensure directory exists or handle path carefully in your real env
    try:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    except Exception as e:
        print(f"Could not save file (check directory existence): {e}")
        plt.show() # Fallback to showing plot if save fails

if __name__ == "__main__":
    log_file = "/global/homes/c/ccardona/PSF2/PSF/output/train_calopodit/2026-02-06-07-02-00/output.log"
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        
    plot_loss_from_log(log_file)
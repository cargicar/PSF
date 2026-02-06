import re
import matplotlib.pyplot as plt
import sys

def plot_loss_from_log(file_path):
    iterations = []
    losses = []
    # Regex to parse the log line
    # Matches: [Epoch/TotalEpochs][Iter/TotalIter] loss: Value
    # Example: [  0/20000][ 32/967]    loss:      1.5571,
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
                    # Calculate global iteration (cumulative steps)
                    global_step = (epoch * total_iter_per_epoch) + curr_iter
                    
                    iterations.append(global_step)
                    losses.append(loss)
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    if not iterations:
        print("No loss data found. Check your log file format.")
        return

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, losses, linewidth=1, label='Training Loss')
    plt.xlabel('Global Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iterations')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path[:-11]}/syn/loss_plot.png")

if __name__ == "__main__":
    # Change this to your actual log file path
    log_file = "/global/homes/c/ccardona/PSF2/PSF/output/train_calopodit/2026-02-05-13-11-29/output.log"
    
    # If passing file path as a command line argument
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        
    plot_loss_from_log(log_file)
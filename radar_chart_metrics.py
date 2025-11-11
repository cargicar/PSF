import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes

# --- Data Preparation ---
# Metrics selected for the plot (8 axes for a nice octagon)
#mmd: lower is better
#cov: higher is better
#1-NN-acc_t: lower is better but about 0.5 is better
metrics_to_plot = [
    'lgan_mmd-CD',  'lgan_mmd_smp-CD', 'lgan_mmd-EMD','lgan_mmd_smp-EMD',
    'lgan_cov-CD', 'lgan_cov-EMD',  
    '1-NN-CD-acc_t', '1-NN-CD-acc_f', '1-NN-CD-acc',  
    'JSD']

# Direction where the metric is 'better' ('lower' or 'higher')
better_directions = [
    'lower', 'lower', 'lower', 'lower', 
    'higher', 'higher',
    'lower', 'lower', 'lower', 
    'higher']

# Notice that I am scaling the large values so the are all less than 1
# to fit them in the radar chart properly.
pvcnn_values = {'lgan_mmd-CD': 474.1660461425781/1000,'lgan_mmd_smp-EMD': 2732.073974609375/10000,'lgan_mmd_smp-CD': 335.8780517578125/1000,  'lgan_mmd-EMD': 3005.349365234375/10000, 
                'lgan_cov-EMD': 0.15625,  'lgan_cov-CD': 0.3125, 
              '1-NN-CD-acc_t': 0.375, '1-NN-CD-acc_f': 0.96875, '1-NN-CD-acc': 0.671875,   
                'JSD': 0.026629936536614274}

calopodit_values = {'lgan_mmd-CD': 363.5998229980469/1000,   'lgan_mmd_smp-EMD': 2680.424560546875/10000,'lgan_mmd_smp-CD': 326.1539611816406/1000,'lgan_mmd-EMD': 2740.2236328125/10000,
                    'lgan_cov-EMD': 0.1875, 'lgan_cov-CD': 0.34375, 
                    '1-NN-CD-acc_t': 0.59375, '1-NN-CD-acc_f': 0.875, '1-NN-CD-acc': 0.734375, 
                    'JSD': 0.10879843043929771}


# Combine into a DataFrame
df = pd.DataFrame([pvcnn_values, calopodit_values], index=['PVCNN', 'CaloPoDit'])
model_names = df.index.tolist()

def plot_standard_radar(df, metrics, better_directions, model_names, filename="radar_chart.png"):
    """Generates a standard polar plot (radar chart) for comparison."""
    
    for i, metric in enumerate(metrics):
        # Invert the scale if "lower is better" so that 1 is always best.
        if better_directions[i] == 'lower':
            df[metric] = 1 - df[metric]
    # 2. Setup the plot
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the plot
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2) # Rotate so the first axis is up
    ax.set_theta_direction(-1) # Clockwise direction
    
    # Set the y-limits to 0 to 1 for the normalized data
    ax.set_ylim(0, 0.5)
    # Set the labels for the axes (metric names)
    ax.set_xticks(angles[:-1])
    # Use newline for cleaner labels
    ax.set_xticklabels([m.replace('-', '\n-') for m in metrics], fontsize=10)
    
    # Set the grid lines
    ax.set_yticks(np.arange(0.2, 1.0, 0.2))
    ax.set_yticklabels([r"20%", r"40%", r"60%", r"80%"], color="grey", size=8, verticalalignment='center')
    ax.grid(True)

    # 3. Plot each model
    for model_name, color in zip(model_names, ['blue', 'red']):
        values = df.loc[model_name].values.flatten().tolist()
        values += values[:1] # Close the circle
        
        ax.plot(angles, values, label=model_name, linewidth=2, linestyle='solid', color=color, marker='o', markersize=4)
        ax.fill(angles, values, color, alpha=0.25)

    # Add a title and legend
    ax.set_title('Point Cloud Model Performance Comparison (Normalized)', size=14, y=1.1)
    ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.9), frameon=False)

    # Save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Radar chart saved as {filename}")

# Execute the revised plotting function
plot_standard_radar(df, metrics_to_plot, better_directions, model_names)

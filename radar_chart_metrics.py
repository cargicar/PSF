import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes

def create_radar_chart(df, metrics, better_directions, model_names, filename="radar_chart.png"):
    """
    Generates a radar chart for comparing model metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metric values.
        metrics (list): List of metric names to plot.
        better_directions (list): List of strings ('lower' or 'higher') indicating 
                                  the optimal direction for each metric.
        model_names (list): List of model names to plot.
        filename (str): The name of the file to save the chart to.
    """
    
    # 1. Normalize the data (Min-Max Scaling)
    df_normalized = pd.DataFrame(index=df.index)
    for i, metric in enumerate(metrics):
        min_val = df[metric].min()
        max_val = df[metric].max()
        
        # Min-Max Scaling: (X - Min) / (Max - Min)
        normalized = (df[metric] - min_val) / (max_val - min_val)
        
        # Invert the scale if "lower is better" so that 1 is always best.
        if better_directions[i] == 'lower':
            df_normalized[metric] = 1 - normalized
        else:
            df_normalized[metric] = normalized

    # 2. Setup the plot
    num_vars = len(metrics)
    # The angle of each axis (equal spacing)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the plot by duplicating the first score and angle
    angles += angles[:1] 
    
    # Function to define the projection for the radar chart
    def radar_factory(num_vars, frame='circle'):
        # calculate the vertices of the polygon
        x = np.cos(np.deg2rad(np.linspace(0, 360, num_vars, endpoint=False)))
        y = np.sin(np.deg2rad(np.linspace(0, 360, num_vars, endpoint=False)))
        
        # define the vertices for the Axes - 0, 0 is the center of the plot
        v = np.vstack((x, y))
        path = Path(v.T)
        
        class RadarAxes(PolarAxes):
            # name the axes
            name = 'radar'
            
            # Use the Path for the boundary of the grid
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # rotate plot such that first axis is at the top
                self.set_theta_zero_location('N')
            
            def _gen_axes_patch(self):
                # The Axes patch must be centered at (0.5, 0.5) and have a radius of 0.5
                return path.get_patch_factory()(self.center, radius=0.5, transform=self.transData)

            def fill(self, *args, closed=True, **kwargs):
                """Override fill so that closed=True by default"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot so that closed=True by default"""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    line.set_path_effects([path.get_path_factory()(self.center, radius=0.5, transform=self.transData)])
                return lines
                
            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(angles[:-1]), labels)
            
            def _gen_axes_spines(self):
                spine = Spine(axes=self, spine_type='circle', path=path)
                spine.set_transform(self.transAxes)
                return {'polar': spine}

        return new_projection(RadarAxes, num_vars=num_vars)


    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2) # Rotate so the first axis is up
    ax.set_theta_direction(-1) # Clockwise direction
    
    # Set the y-limits to 0 to 1 for the normalized data
    ax.set_ylim(0, 1)
    # Set the labels for the axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('-', '\n-') for m in metrics])
    # Set the grid lines (0.2, 0.4, 0.6, 0.8)
    ax.set_yticks(np.arange(0.2, 1.0, 0.2))
    ax.set_yticklabels(["", "", "", ""], color="grey", size=8)

    # Plot each model
    for model_name in model_names:
        values = df_normalized.loc[model_name].values.flatten().tolist()
        values += values[:1] # Close the circle
        
        if model_name == 'Model 1':
            line, = ax.plot(angles, values, label=model_name, linewidth=2, linestyle='solid', color='blue')
            ax.fill(angles, values, 'blue', alpha=0.25)
        else: # Model 2
            line, = ax.plot(angles, values, label=model_name, linewidth=2, linestyle='solid', color='red')
            ax.fill(angles, values, 'red', alpha=0.25)

    # Add a title and legend
    ax.set_title('Point Cloud Model Performance Comparison (Normalized)', size=16, y=1.1)
    ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.9))

    # Save the figure
    plt.savefig(filename)
    plt.close(fig)
    print(f"Radar chart saved as {filename}")

# --- Data Preparation ---

# Metrics selected for the plot (8 axes for a nice octagon)
#mmd: lower is better
#cov: higher is better
#1-NN-acc_t: lower is better but about 0.5 is better
metrics_to_plot = [
    'lgan_mmd-CD',  'lgan_mmd_smp-CD', 'lgan_mmd-EMD','lgan_mmd_smp-EMD',
    'lgan_cov-CD', 'lgan_cov-EMD',  
    '1-NN-CD-acc_t', '1-NN-CD-acc_f', '1-NN-CD-acc', '1-NN-EMD-acc_t', '1-NN-EMD-acc', 
    'JSD']

# Direction where the metric is 'better' ('lower' or 'higher')
better_directions = [
    'lower', 'lower', 'lower', 'lower', 
    'higher', 'higher',
    'lower', 'lower', 'lower', 'lower', 'lower', 
    'higher']

# Actual Model 1 Data (from your log)
pvcnn_values = {'lgan_mmd-CD': 474.1660461425781,'lgan_mmd_smp-EMD': 2732.073974609375,'lgan_mmd_smp-CD': 335.8780517578125,  'lgan_mmd-EMD': 3005.349365234375, 
                'lgan_cov-EMD': 0.15625,  'lgan_cov-CD': 0.3125, 
              '1-NN-CD-acc_t': 0.375, '1-NN-CD-acc_f': 0.96875, '1-NN-CD-acc': 0.671875, '1-NN-EMD-acc_t': 0.96875,  
                '1-NN-EMD-acc': 0.984375,'JSD': 0.026629936536614274}

calopodit_values = {'lgan_mmd-CD': 363.5998229980469,   'lgan_mmd_smp-EMD': 2680.424560546875,'lgan_mmd_smp-CD': 326.1539611816406,'lgan_mmd-EMD': 2740.2236328125,
                    'lgan_cov-EMD': 0.1875, 'lgan_cov-CD': 0.34375, 
                    '1-NN-CD-acc_t': 0.59375, '1-NN-CD-acc_f': 0.875, '1-NN-CD-acc': 0.734375, '1-NN-EMD-acc_t': 0.96875,
                    '1-NN-EMD-acc': 0.984375, 'JSD': 0.10879843043929771}


# Combine into a DataFrame
df = pd.DataFrame([pvcnn_values, calopodit_values], index=['PVCNN', 'CaloPoDit'])
model_names = df.index.tolist()

# Generate the chart
# Note: The custom radar_factory is required for a cleaner polar plot, 
# but it's complex to define within a code execution block. 
# For simplicity and reliability in the execution environment, I will use the standard matplotlib 
# polar plot, which achieves the same goal but with a circular frame instead of an octagonal one.

# --- Revised Plotting Script using standard Polar Plot ---
def plot_standard_radar(df, metrics, better_directions, model_names, filename="radar_chart.png"):
    """Generates a standard polar plot (radar chart) for comparison."""
    
    # 1. Normalize the data (Min-Max Scaling)
    df_normalized = pd.DataFrame(index=df.index)
    for i, metric in enumerate(metrics):
        min_val = df[metric].min()
        max_val = df[metric].max()
        
        # Avoid division by zero if all values are the same
        if max_val == min_val:
            df_normalized[metric] = 0.5 # Neutral score if no difference
        else:
            normalized = (df[metric] - min_val) / (max_val - min_val)
            # Invert the scale if "lower is better" so that 1 is always best.
            if better_directions[i] == 'lower':
                df_normalized[metric] = 1 - normalized
            else:
                df_normalized[metric] = normalized
    
    # 2. Setup the plot
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the plot
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2) # Rotate so the first axis is up
    ax.set_theta_direction(-1) # Clockwise direction
    
    # Set the y-limits to 0 to 1 for the normalized data
    ax.set_ylim(0, 1)
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
        values = df_normalized.loc[model_name].values.flatten().tolist()
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
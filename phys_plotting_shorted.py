# From omni_jet_alpha_c
import json
import math
import h5py
import os
from datetime import datetime

import torch
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
#import vector
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance

#vector.register_awkward()

from metrics.physics.metrics import quantiled_kl_divergence
from metrics.physics.plotting_utils import (
    get_good_linestyles,
    KL,
    find_max_energy_z,
    get_COG_ak,
    sum_energy_per_layer,
    sum_energy_per_radial_distance,
    write_distances_to_json,
    plot_ratios
)

def plot_paper_plots(feature_sets: list, labels: list = None, colors: list = None, material: str = None, **kwargs):
    """Plots the features of multiple constituent or shower sets.

    Args:
        feature_sets: A list of dictionaries, each containing awkward arrays for "x", "y", "z", and "energy" features.
        labels: (Optional) A list of labels for the feature sets (defaults to 'Set 1', 'Set 2', etc.).
        colors: (Optional) A list of colors for the feature sets (defaults to a matplotlib colormap).
        kwargs: Additional keyword arguments to pass to the plotting functions.
    """
    num_sets = len(feature_sets)
    #new_feature_sets = feature_sets.transpose(1,0,2)

    if labels is None:
        labels = [f"Set {i + 1}" for i in range(num_sets)]
    if colors is None:
        colors = plt.cm.get_cmap("tab10").colors  # Use matplotlib's colormap

    # Preprocessing & feature extraction'
    features_list = []
    for features in feature_sets:
        # Filter voxels with energy > 0.1
        mask = features["energy"] > 0.1
        #mask = features[3] > 0.1

        filtered_features = {
            "x": features["x"][mask],
            "y": features["y"][mask],
            "z": features["z"][mask],
            "energy": features["energy"][mask],
        }
        #NOTE. This file uses akward array. So we will transform to:
        # filtered_features = {
        #     "x": ak.from_numpy(features[0][mask][None, :]),
        #     "y": ak.from_numpy(features[1][mask][None, :]),
        #     "z": ak.from_numpy(features[2][mask][None, :]),
        #     "energy": ak.from_numpy(features[3][mask][None, :]),
        # }
        features_list.append(
            {
                "voxel": ak.to_numpy(ak.num(filtered_features["x"])),
                "energy": ak.flatten(features["energy"]).to_numpy(),  # Keep all energies here
                #"energy": ak.flatten(features[3][None,:]),  # Keep all energies here
                "shower_energy": ak.to_numpy(ak.sum(filtered_features["energy"], axis=1)),
                # "max_z": find_max_energy_z(filtered_features["energy"], filtered_features["z"]),
                "x_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["x"], filtered_features["energy"])
                ),
                "y_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["y"], filtered_features["energy"])
                ),
                "z_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["z"], filtered_features["energy"])
                ),
                "x": ak.flatten(filtered_features["x"]).to_numpy(),
                "y": ak.flatten(filtered_features["y"]).to_numpy(),
                "z": ak.flatten(filtered_features["z"]).to_numpy(),
                "distance": np.mean(
                    sum_energy_per_radial_distance(
                        filtered_features["x"], filtered_features["y"], filtered_features["energy"]
                    ),
                    axis=0,
                ),
                "energy_filtered": ak.flatten(filtered_features["energy"]).to_numpy(),
                "energy_per_layer": np.mean(
                    sum_energy_per_layer(filtered_features["z"], filtered_features["energy"]),
                    axis=0,
                ),
                "pixel": np.arange(0, 21) + 0.5,
                "hits": np.arange(0, 29) + 0.5,
            }
        )
    # Plotting (two figures)
    mpl.rcParams["xtick.labelsize"] = 15
    mpl.rcParams["ytick.labelsize"] = 15
    # mpl.rcParams['font.size'] = 28
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(18, 12), facecolor="white")

    """Plots the distributions of energy, energy sum, number of hits, and z start layer."""
    gs = fig.add_gridspec(
        5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8]
    )  # 3 rows for the different distributions
    # print("Plotting distributions:max(features_list[z])",  max(features_list["z"]))

    # Binning setup (adjust ranges and bins as needed for your data)
    fontsize_labels = 18

    energy_sum = 2000
    energy = 70
    n_hits = 1700

    energy_bins = np.logspace(np.log10(0.01), np.log10(energy), 50)  # Logarithmic bins for energy
    energy_sum_bins = np.arange(0, energy_sum, 75)
    voxel_bins = np.arange(0, n_hits, 50)  # The number of hits
    dist_e_bins = np.arange(0, 21, 1)  # The distance
    bins_cog = np.arange(8, 22, 0.5)
    bins_z = np.arange(0, 31.5, 1)

    # Energy Distribution
    ax0 = fig.add_subplot(gs[0, 0])  # vis cell energy x/y log
    ax1 = fig.add_subplot(gs[0, 1])  # energy sum
    ax2 = fig.add_subplot(gs[0, 2])  # number of hits
    ax3 = fig.add_subplot(gs[3, 0])  # center of gravity Z
    ax4 = fig.add_subplot(gs[3, 1])  # spatial distribution Z
    ax5 = fig.add_subplot(gs[3, 2])  # energy over distance

    # looping through all input data to be plottet on the different distributions
    for features, label, color in zip(features_list, labels, colors):
        histtype = "stepfilled" if features is features_list[0] else "step"
        edgecolor = "gray" if histtype == "stepfilled" else color
        linestyle = (
            "--"
            if len(features_list) > 2
            and (
                features is features_list[2]
                or len(features_list) > 3
                and (features is features_list[3])
            )
            else "-"
        )
        alpha = 0.95
        ax0.hist(
            features["energy"],
            bins=energy_bins,
            linestyle=linestyle,
            histtype=histtype,
            edgecolor=edgecolor,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax1.hist(
            features["shower_energy"],
            bins=energy_sum_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            density=True,
            color=color,
        )
        ax2.hist(
            features["voxel"],
            bins=voxel_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            density=True,
            color=color,
        )
        ax3.hist(
            features["z_zero"],
            bins=bins_cog,
            histtype=histtype,
            lw=2,
            alpha=alpha,
            linestyle=linestyle,
            label=label,
            edgecolor=edgecolor,
            density=True,
            color=color,
        )
        ax4.hist(
            features["hits"],
            bins=bins_z,
            histtype=histtype,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
            linestyle=linestyle,
            weights=features["energy_per_layer"],
        )
        ax5.hist(
            features["pixel"],
            bins=dist_e_bins,
            weights=features["distance"],
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
    # ax0.set_xlabel("Energy (MeV)")
    ax0.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(left=0.01)
    ax0.axvspan(0.01, 0.1, ymin=0, ymax=0.73, facecolor="lightgray", alpha=0.2, hatch="/")
    ax0.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax0.get_ylim()
    new_ymax = ymax + 1620 * ymax
    ax0.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot

    mask = [0.7, 1.3]
    ax0_twin = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax0_twin.set_xlim(left=0.01)
    plot_ratios(ax0_twin, features_list, energy_bins, "energy", labels, colors, mask=mask)
    # Add horizontal line at y=1
    ax0_twin.axhline(y=1, color="gray", linestyle="--")
    ax0_twin.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")
    # Set y-axis limits
    ax0_twin.set_ylim(mask)
    ax0_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax0_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax0_twin.tick_params(axis="y", labelcolor="black")

    # Energy Sum Distribution
    ax1.set_ylabel("normalized", fontsize=fontsize_labels)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax1.get_ylim()
    new_ymax = ymax + 0.45 * ymax
    ax1.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax1_twin = fig.add_subplot(gs[1, 1], sharex=ax1)
    plot_ratios(
        ax1_twin, features_list, energy_sum_bins, "shower_energy", labels, colors, mask=mask
    )
    ax1_twin.axhline(y=1, color="gray", linestyle="--")
    # Set y-axis limits
    ax1_twin.set_ylim(mask)
    ax1_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax1_twin.set_xlabel("energy sum [MeV]", fontsize=fontsize_labels)
    ax1_twin.tick_params(axis="y", labelcolor="black")

    # Number of Hits (Voxel) Distribution
    mask = [0.6, 1.4]
    ax2.set_ylabel("normalized", fontsize=fontsize_labels)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax2.get_ylim()
    new_ymax = ymax + 0.44 * ymax
    ax2.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax2_twin = fig.add_subplot(gs[1, 2], sharex=ax2)
    plot_ratios(ax2_twin, features_list, voxel_bins, "voxel", labels, colors, mask)

    ax2_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax2_twin.set_ylim(mask)
    ax2_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax2_twin.set_xlabel("number of hits", fontsize=fontsize_labels)
    ax2_twin.tick_params(axis="y", labelcolor="black")

    # Center of Gravity Z Distribution
    ax3.set_ylabel("normalized", fontsize=fontsize_labels)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax3.get_ylim()
    new_ymax = ymax + 0.48 * ymax
    ax3.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax3_twin = fig.add_subplot(gs[4, 0], sharex=ax3)
    mask = (0.4, 1.6)
    plot_ratios(ax3_twin, features_list, bins_cog, "z_zero", labels, colors, mask=mask)

    ax3_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits

    ax3_twin.set_ylim(mask)
    ax3_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax3_twin.set_xlabel("center of gravity Z [layer]", fontsize=fontsize_labels)
    ax3_twin.tick_params(axis="y", labelcolor="black")

    # Z Distribution
    ax4.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax4.tick_params(axis="x", labelbottom=False)
    ax4.set_yscale("log")
    ax4.set_xlim(0, 30)
    ymin, ymax = ax4.get_ylim()
    new_ymax = ymax + 40 * ymax
    ax4.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax4_twin = fig.add_subplot(gs[4, 1], sharex=ax4)
    mask = [0.7, 1.3]
    plot_ratios(
        ax4_twin, features_list, bins_z, "hits", labels, colors, mask, weights="energy_per_layer"
    )

    ax4_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits

    ax4_twin.set_ylim(mask)
    ax4_twin.set_xlim(0, 30)
    ax4_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax4_twin.set_xlabel("layer", fontsize=fontsize_labels)
    ax4_twin.tick_params(axis="y", labelcolor="black")

    # Energy Distribution per Layer
    ax5.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax5.set_yscale("log")
    ax5.set_xlim(0, 21)
    ax5.tick_params(axis="x", labelbottom=False, labelsize=fontsize_labels)
    ymin, ymax = ax5.get_ylim()
    new_ymax = ymax + 40 * ymax
    ax5.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax5_twin = fig.add_subplot(gs[4, 2], sharex=ax5)
    mask = [0.7, 1.3]
    plot_ratios(
        ax5_twin, features_list, dist_e_bins, "pixel", labels, colors, mask, weights="distance"
    )

    ax5_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax5_twin.set_ylim(mask)
    ax5_twin.set_xlim(0, 21)
    ax5_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax5_twin.set_xlabel("radius [pixels]", fontsize=fontsize_labels)
    ax5_twin.tick_params(axis="y", labelcolor="black")

    # Add legend to the first subplot (energy)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            label=label,
            linestyle="--"
            if len(features_list) > 2
            and (
                features is features_list[2]
                or len(features_list) > 3
                and (features is features_list[3])
            )
            else "-",
        )
        for color, label, features in zip(colors, labels, features_list)
    ]
    # Create the figure
    ax5.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax0.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax4.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax1.set_title(f"Material: {material}", fontsize=22)

    return fig        

# This function take a list: file_path= [(original hdf5 file dataset), (generated samples file: ak, pth, or another)]

def read_generated(file_path, 
                   material_list=["G4_Ta","G4_W",], 
                   num_showers=-1, 
                   material="G4_W", 
                   use_mask=False, 
                   energy_threshold=0.0,
                   enforce_conservation = True):
    material_dict = {"G4_W" : 0, "G4_Ta": 1, "G4_Pb" : 2 }
    #if pth gen_tensor = [x.cpu(), pts.cpu(), mask.cpu(), int_energy.cpu(), gap_pid.cpu()] 
    gen_dict = {
        "x": [],
        "y": [],
        "z": [],
        "energy": []
    }
    data_dict = {
        "x": [],
        "y": [],
        "z": [],
        "energy": [],
    }

    # Read it back later
    if "parquet" in file_path[1]:
        loaded_showers = ak.from_parquet(file_path[1])
    elif "pth" in file_path[1]:
        data = torch.load(file_path[1], map_location='cpu')
        if isinstance(data, list):
            _ , gen_tensor, mask, int_energy, gap_pid = data
    else:
        print(f"Array or Tensor format not implemented yet")
        return None, None # Safety return

    with h5py.File(file_path[0], "r") as h5file:
        showers_idx = list(h5file.keys()) # Convert to list for safer indexing
        if num_showers == -1:
            num_showers = len(showers_idx)
        
        # Determine actual loop limit based on available tensor size if applicable
        # (Optional safety check if gen_tensor is smaller than num_showers)
        if "pth" in file_path[1] and hasattr(gen_tensor, 'shape'):
             num_showers = min(num_showers, gen_tensor.shape[0])
        n=0
        for i, idx in enumerate(showers_idx):
            if i >= num_showers:
                break
            
            shower = h5file[idx]
            # E = shower.attrs['initial_energy']
            x, y, z = shower['indices'][()].T
            energy = shower['values'][()]
            mat = shower['material'][()]
            
            if i % 5000 == 0 or i == num_showers:
                print(f"Shower #: {i}/{num_showers}, Material: {mat}")

            # Decode material if needed
            mat_decoded = mat.decode('utf-8') if hasattr(mat, 'decode') else str(mat)

            if mat_decoded != material:
                continue
            
            if "pth" in file_path[1]:
                # Apply global mask if requested
                if use_mask: 
                    # Note: Ensure shapes align for this multiplication
                    gen_tensor_i = gen_tensor[i] * mask.unsqueeze(-1)
                else:
                    gen_tensor_i = gen_tensor[i]
                if gap_pid[i] != material_dict[material]:
                    n+=1
                    #print(f"Material mistmatch: gap :{gap_pid[i]}, material dict {material_dict[material]}")
                    continue
                # Unpack based on shape [Batch, Channels, Points] vs [Batch, Points, Channels]
                if gen_tensor_i.shape[0] == 4 and len(gen_tensor_i.shape) == 2: 
                    # Case: [4, npoints]
                    xg, yg, zg, eg = gen_tensor_i
                elif gen_tensor_i.shape[-1] == 4:
                     # Case: [npoints, 4] -> Transpose to unpack
                    xg, yg, zg, eg = gen_tensor_i.T
                else:
                    # Fallback or specific user case logic
                    xg, yg, zg, eg = gen_tensor_i.T
                
                # --- FILTER LOGIC STARTS HERE ---
                # Create a boolean mask for points above the threshold
                valid_mask = eg > energy_threshold
                
                # Apply the mask to all components
                xg = xg[valid_mask]
                yg = yg[valid_mask]
                zg = zg[valid_mask]
                eg = eg[valid_mask]
                # --- FILTER LOGIC ENDS HERE ---

                # Enforce energy conservation
                if enforce_conservation:
                    # 1. Calculate the current sum of the generated shower
                    current_sum = torch.sum(eg)
                    
                    # 2. Get the target sum from the conditional input (int_energy)
                    # Ensure we handle shape (1,) or scalar
                    target_sum = np.sum(energy)
                    
                    # 3. Scale if the shower is not empty
                    if current_sum > 1e-9:
                        scale_factor = target_sum / current_sum
                        eg = eg * scale_factor
                
                gen_dict["x"].append(zg)
                gen_dict["z"].append(xg)
                gen_dict["y"].append(yg)
                gen_dict["energy"].append(eg)

            data_dict["x"].append(z)
            data_dict["z"].append(x)
            data_dict["y"].append(y)
            data_dict["energy"].append(energy)
        print(f"Total Material mistmatch: {n}")

    if "pth" in file_path[1]:
        # Convert list of tensors to awkward array
        # ak.Array can handle variable length lists created by the filtering
        ak_array = ak.Array(gen_dict)
    elif "parquet" in file_path[1]:
        ak_array = loaded_showers
    else:
        ak_array = None
        
    ak_array_truth = ak.Array(data_dict)
    
    return ak_array, ak_array_truth


# read_generated with hit_prob
# This function reads a pth files that contain a list [x_tensor, gen_tensor, mask], i.e, a file where original and generated samples have been storage along side
def read_generated_pth(file_path, num_showers=-1, prob_threshold=0.0):
    material_list=["G4_Ta","G4_W",]
    material="G4_W"
    # Now expecting 5 channels: x, y, z, E, hit_prob
    x_tensor, gen_tensor, mask = torch.load(file_path, map_location= 'cpu')
    x_tensor = x_tensor.detach()
    gen_tensor= gen_tensor.detach()
    gen_dict = {"x": [], "y": [], "z": [], "energy": []}
    data_dict = {"x": [], "y": [], "z": [], "energy": []}
    if num_showers == -1:
        num_showers = x_tensor.shape[0]-1
    with torch.no_grad():
        for i in range(num_showers):

            # Target data (Ground Truth)
            if x_tensor.shape[1] == 4:
                x, y, z, e = x_tensor[i] # [4, N]    
                # Generated data (Model Output)
                # Assuming shape [5, N] where indices are:
                # 0:x, 1:y, 2:z, 3:E, 4:hit_prob
                xg, yg, zg, eg = gen_tensor[i] 
            else:
                x, y, z, e = x_tensor[i].T # [4, N]
                # Generated data (Model Output)
                # Assuming shape [5, N] where indices are:
                # 0:x, 1:y, 2:z, 3:E, 4:hit_prob
                xg, yg, zg, eg = gen_tensor[i].T 
            
            
            # --- THE FILTERING STEP ---
            # Only keep points where the model is confident a hit exists
            #mask = pg > prob_threshold
            #redefine mask
            # filtered_xg = xg[mask]
            # filtered_yg = yg[mask]
            # filtered_zg = zg[mask]
            # filtered_eg = eg[mask]

            # Append Ground Truth
            data_dict["z"].append(x)
            data_dict["y"].append(y)
            data_dict["x"].append(z)
            data_dict["energy"].append(e)

            # # Append Filtered Generated Data
            # gen_dict["z"].append(filtered_xg)
            # gen_dict["y"].append(filtered_yg)
            # gen_dict["x"].append(filtered_zg)
            # gen_dict["energy"].append(filtered_eg)

            # Append Filtered Generated Data
            gen_dict["z"].append(xg)
            gen_dict["y"].append(yg)
            gen_dict["x"].append(zg)
            gen_dict["energy"].append(eg)
            
        ak_array_truth = ak.Array(data_dict)
        ak_array = ak.Array(gen_dict)
    return ak_array, ak_array_truth

def make_plots(file_paths: list[str], #list containig file paths for simulation and generated data
                material_list=["G4_Ta","G4_W","G4_Pb"],
                material = "G4_W",
                num_showers=-1,
                title= None):
    #filepath[0] : simulation data
    #filepath[1] : generated data
    #os.makedirs("Plots",exist_ok=True)
    #filename = file_path.split("/")[-1][:-3]

    #for material in material_list:
    if isinstance(file_paths, list):
        generated_features, ground_truth_features = read_generated(file_paths, material_list, num_showers, material, enforce_conservation = True)
    else:
        generated_features, ground_truth_features = read_generated_pth(file_paths, num_showers)
    fig = plot_paper_plots(
        [ground_truth_features, generated_features],
        labels=["Ground Truth", "Generated"],
        colors=["lightgrey", "cornflowerblue"], material=material
    )

    #fig.savefig(f"Plots/{filename}_{material}.pdf", dpi=300)
    #fig.savefig(f"file_paths[0][-4]{title}", dpi=300)
    fig.savefig(f"{title}", dpi=300)

if __name__ == "__main__":
    material = "G4_Ta"
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    parser = argparse.ArgumentParser(description="Plot generated showers vs ground truth")
    #parser.add_argument("--file_path", type=str, required=True, help="Path to the HDF5 file containing generated showers")
    if material == "G4_W":
        parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/train_dbg/w_sim/photon-shower-1_corrected_compressed.hdf5') #For the case when we want to read the original data from the original dataset hdf5 files
    if material == "G4_Ta":
        parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/train_dbg/ta_sim/photon-shower-1_corrected_compressed.hdf5') #For the case when we want to read the original data from the original dataset hdf5 files
        #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/pth/combined_batches_calopodit_gen_Jan_17.pth') # For the case when we can to read original and generated from the same pth file
    parser.add_argument('--title', default=f'phys_metrics_calopodit_{date_str}_{material}')
    #parser.add_argument('--genroot', default='/pscratch/sd/c/ccardona/logs/example_backbone/runs/2026-02-02_pretrained_from_paper/gen_samples/showers.parquet')
    parser.add_argument('--genroot', default='/pscratch/sd/c/ccardona/datasets/pth/combined_batches_calopodit_gen_Feb_5.pth')
    #parser.add_argument('--genroot', default='/global/homes/c/ccardona/PSF/output/test_flow_g4/2026-01-06_clopodit_idl_mask/syn/combined_photon_samples.pth')
    parser.add_argument("--num_showers", type=int, default=-1, help="Number of showers to process (-1 for all)")
    args = parser.parse_args()

    if 'pth' in args.dataroot:
        filepaths = args.dataroot
    else:
        filepaths = [args.dataroot, args.genroot]
    make_plots(filepaths, num_showers=args.num_showers, title = args.title, material= material)
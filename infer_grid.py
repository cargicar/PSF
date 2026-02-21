import torch
import numpy as np
from tqdm import tqdm
from datasets.idl_dataset import LazyIDLDataset

def infer_grid_from_dataset(dataset, num_samples=1000):
    """
    Analyzes the coordinates in the dataset to find unique voxel centers.
    """
    print(f"Analyzing {num_samples} samples to infer grid structure...")
    
    all_x = []
    all_y = []
    all_z = []

    # 1. Collect unique coordinates from the dataset
    # Standard mode returns: (shower, energy, y_pid, gap_pid, idx)
    # Reflow mode returns: (x, x0, mask, init_e, y_pid, gap_pid, idx)
    for i in tqdm(range(min(len(dataset), num_samples))):
        data = dataset[i]
        # Coordinates are in the first element of the tuple
        shower = data[0].numpy() 
        
        # If the shower is padded, we only want the non-zero (masked) points
        # Assuming the energy value (index 3) is > 0 for real hits
        mask = shower[:, 3] > 0
        real_hits = shower[mask]
        
        all_x.extend(real_hits[:, 0].tolist())
        all_y.extend(real_hits[:, 1].tolist())
        all_z.extend(real_hits[:, 2].tolist())

    # 2. Extract Unique Coordinate Values (The Grid Centers)
    # We use a small tolerance for floating point precision issues
    def get_grid_info(values, name):
        unique_vals = np.sort(np.unique(np.round(values, decimals=4)))
        diffs = np.diff(unique_vals)
        
        if len(unique_vals) < 2:
            return f"{name}: Single value detected ({unique_vals[0]})"
        
        avg_spacing = np.mean(diffs)
        return {
            "name": name,
            "count": len(unique_vals),
            "spacing": avg_spacing,
            "min": unique_vals.min(),
            "max": unique_vals.max(),
            "unique_centers": unique_vals.tolist()
        }

    grid_x = get_grid_info(all_x, "X")
    grid_y = get_grid_info(all_y, "Y")
    grid_z = get_grid_info(all_z, "Z")

    # 3. Print Results
    for res in [grid_x, grid_y, grid_z]:
        print(f"\n--- Dimension {res['name']} ---")
        print(f"Voxel Count:   {res['count']}")
        print(f"Voxel Spacing: {res['spacing']:.4f}")
        print(f"Range:         [{res['min']}, {res['max']}]")

    breakpoint()
    return grid_x, grid_y, grid_z


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Infer grid strcuture")
    #parser.add_argument("--file_path", type=str, required=True, help="Path to the HDF5 file containing generated showers")
    parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/train_dbg/') #For the case when we want to read the original data from the original dataset hdf5 files
    
    #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/pth/combined_batches_calopodit_gen_Jan_17.pth') # For the case when we can to read original and generated from the same pth file
    args = parser.parse_args()

    # data_dir = "/path/to/your/h5/files"
    dataset = LazyIDLDataset(data_dir=args.dataroot)
    grid_info = infer_grid_from_dataset(dataset)
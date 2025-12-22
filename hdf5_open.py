import h5py
import numpy as np
import os

def load_hdf5_data(filepath):
    """
    Opens an HDF5 file and loads the 'indices' and 'values' datasets 
    from all top-level groups (e.g., /0, /1, etc.).
    
    Args:
        filepath (str): The path to the HDF5 file.
        
    Returns:
        tuple: A tuple containing two lists: 
               - loaded_indices (list of numpy arrays)
               - loaded_values (list of numpy arrays)
               Returns ([], []) if the file cannot be read.
    """
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at '{filepath}'")
        return [], []

    print(f"Loading data from **{filepath}**...")
    
    loaded_indices = []
    loaded_values = []
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Iterate over all top-level keys (which are your groups: '0', '1', etc.)
            for group_name in f.keys():
                
                # Check if the key corresponds to a group (like /0)
                if isinstance(f[group_name], h5py.Group):
                    
                    group = f[group_name]
                    
                    # Check for the required datasets within the group
                    if 'indices' in group and 'values' in group:
                        print(f"✅ Loading datasets from group: **/{group_name}**")
                        
                        # Load the full array data into memory
                        indices = group['indices'][:]
                        values = group['values'][:]
                        
                        loaded_indices.append(indices)
                        loaded_values.append(values)
                    else:
                        print(f"⚠️ Skipping group /{group_name}: 'indices' or 'values' dataset not found.")
                else:
                    print(f"ℹ️ Skipping key /{group_name}: Not a group.")
                        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred while loading data: {e}")
        return [], []
        
    print(f"\nSuccessfully loaded data for **{len(loaded_indices)}** events.")
    return loaded_indices, loaded_values

# --- USAGE EXAMPLE ---
# 1. **REPLACE THIS with the path to your HDF5 file**
HDF5_FILE_PATH = "/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/Pb_Simulation/photon-shower-0_corrected_compressed.hdf5" 

# 2. **Load the data**
all_indices, all_values = load_hdf5_data(HDF5_FILE_PATH)
breakpoint()
# 3. **Example of how to access and inspect the loaded data**
# if all_indices:
#     print("\n--- Inspection ---")
#     print(f"Total events loaded: {len(all_indices)}")
    
#     # Accessing the data for the first event (e.g., the data from group /0)
#     first_event_indices = all_indices[0]
#     first_event_values = all_values[0]
    
#     print(f"\nFirst Event (Group /0) Indices Shape: {first_event_indices.shape}")
#     print(f"First Event (Group /0) Values Shape: {first_event_values.shape}")
#     print(f"Sample of first indices: {first_event_indices[:3]}")
#     print(f"Sample of first values: {first_event_values[:3]}")
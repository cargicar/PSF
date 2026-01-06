import h5py
import numpy as np
import os

def explore_hdf5_file(filepath):
    """
    Opens an HDF5 file and recursively prints its group structure, 
    datasets, and key metadata.
    
    Args:
        filepath (str): The path to the HDF5 file.
    """
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at '{filepath}'")
        return

    print(f"🔍 Exploring HDF5 file: **{filepath}**\n")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Recursive function to traverse the structure
            def walk_file(name, obj):
                indent = '  ' * (name.count('/') + 1)
                
                if isinstance(obj, h5py.Group):
                    # It's a Group (like a directory)
                    print(f"{indent}📁 **GROUP**: /{name}")
                    
                    # Check and print group attributes (metadata attached to the group)
                    if obj.attrs:
                        print(f"{indent}  ATTRS: {dict(obj.attrs)}")

                elif isinstance(obj, h5py.Dataset):
                    # It's a Dataset (contains the actual data)
                    shape_str = str(obj.shape)
                    dtype_str = str(obj.dtype)
                    
                    print(f"{indent}📊 **DATASET**: /{name}")
                    print(f"{indent}  - Shape: **{shape_str}**")
                    print(f"{indent}  - Dtype: **{dtype_str}**")
                    
                    # Check and print dataset attributes
                    if obj.attrs:
                        print(f"{indent}  - ATTRS: {dict(obj.attrs)}")
                        
                    # Optionally, print a small slice of the data for inspection
                    try:
                        if obj.shape and obj.size > 0:
                            # Try to get the first few elements for small datasets
                            if obj.ndim == 1 and obj.shape[0] < 6:
                                sample = obj[:]
                            # Try to get a slice for larger or multi-dimensional datasets
                            elif obj.ndim == 1:
                                sample = obj[:3]
                            elif obj.ndim > 1:
                                sample = obj[tuple([0] * (obj.ndim - 1) + [slice(None, min(obj.shape[-1], 3))])]
                            else:
                                sample = "..."
                                
                            print(f"{indent}  - Sample Data (first few): {sample}")
                        elif obj.size == 0:
                             print(f"{indent}  - Sample Data: **(Empty Dataset)**")
                    except Exception as e:
                        # Catch potential errors when trying to read the data
                        print(f"{indent}  - Sample Data: **(Error reading data: {e})**")

            # Start the traversal from the root group '/'
            f.visititems(walk_file)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

# --- USAGE EXAMPLE ---
# 1. **REPLACE THIS with the path to your HDF5 file**
HDF5_FILE_PATH = "/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/Pb_Simulation/photon-shower-0_corrected_compressed.hdf5" 

# 2. **Run the function**
explore_hdf5_file(HDF5_FILE_PATH)
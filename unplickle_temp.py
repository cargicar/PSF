import pickle
import numpy as np
import os

data_dir = '/data/ccardona/datasets/G4_individual_sims_pkl_test/'
data_save = '/data/ccardona/datasets/G4_individual_sims_npy_e_liquidArgon/'
file_paths = [os.path.join(data_dir, f) 
                      for f in os.listdir(data_dir) if f.endswith('.pkl')]
        
# We need a robust way to get the count without loading the heavy arrays.
for file_path in file_paths:
    try:
        # 1. Load the file structure (often a small dictionary). 
        # This should be memory-efficient if the structure is not the full data.
        if ("e-" and "liquidArgon") in file_path:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            pc = data['showers'][0]
            energies = data['energies'][0]
            gap = data['gap_pid'][0]
            new_filename = file_path[52:].replace('.pkl', '_pc.npy')
            npy_file = os.path.join(data_save, new_filename) 
            np.save(npy_file, pc)
            # Or for multiple arrays:
            # np.savez('converted_data.npz', array1=data['array1'], array2=data['array2'])
            
            # Ensure data is released immediately after reading its length
            del data 
        
    except (pickle.UnpicklingError, FileNotFoundError, KeyError, IndexError, TypeError) as e:
        print(f"Error indexing file structure '{file_path}': {e}. Skipping file.")


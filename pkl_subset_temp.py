import pickle
# Assuming the data arrays are compatible with NumPy
import numpy as np 

input_filename = '/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_e_liquidArgon_50/all_sims_genAi_e-_brass_liquidArgon.pkl'
output_filename = '/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_e_liquidArgon_50_subset/G4_e_liquidArgon.pkl'
subset_size = 500

# 1. Load the original data
with open(input_filename, 'rb') as f:
    data = pickle.load(f)

# Extract the tensors (assuming they are wrapped in a list at index 0)
all_showers = data['showers'][0]
all_energies = data['energies'][0]
pid = data['pid']
gap = data['gap_pid']
                
# 2. Slice the tensors
subset_showers = all_showers[:subset_size]
subset_energies = all_energies[:subset_size]

print(f"Subset Showers Shape: {subset_showers.shape}")

# 3. Create the new dictionary and save
new_data = {
    'showers': [subset_showers],
    'energies': [subset_energies],
    'pid': pid,
    'gap_pid': gap
}

with open(output_filename, 'wb') as f:
    pickle.dump(new_data, f)

print(f"\nSuccessfully saved subset data to: {output_filename}")
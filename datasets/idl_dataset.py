import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import numpy as np
import pickle
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import pickle
from pathlib import Path        

material_dict = {"G4_W" : 0, "G4_Ta": 1, "G4_Pb" : 2 }
particle_dict = {"Photon" : 0, "electron" : 1}

class LazyIDLDataset(Dataset):
    def __init__(self, data_dir, transform=None, reflow=False):
        self.data_dir = data_dir
        self.transform = transform
        self.reflow = reflow
        
        # The core of lazy loading: a map from global index to (file_path, local_index)
        self.global_index_map = [] 
        # Cache for open file handles (HDF5) or loaded tensors (PTH)
        self.files = {} 
        
        # Container for global dataset statistics
        self.stats = None 
        self.max_particles = None
        
        self._create_global_index_map()

    def _create_global_index_map(self):
        cache_path = Path(self.data_dir) / f"dataset_cache.pkl"
        
        # --- 1. Try Loading from Cache ---
        if cache_path.exists():
            print(f"Loading dataset index from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            
            # Check if cache is the new dict format or the old list format
            if isinstance(cached_data, dict) and 'index_map' in cached_data:
                self.global_index_map = cached_data['index_map']
                self.stats = cached_data.get('stats', None)
                self.max_particles = cached_data.get('max_particles', None)
                if self.stats:
                    print(f"Loaded Stats -> Mean: {self.stats['mean']}, Std: {self.stats['std']}")
                if self.max_particles:
                    print(f"Loaded max-particles -> {self.max_particles}")
                  
            else:
                # Fallback for old cache (just a list)
                self.global_index_map = cached_data
                self.stats = None
                print("Warning: Old cache format found (no statistics). Delete .pkl to recompute.")
            return

        # --- 2. Compute Index & Stats if not cached ---
        base_path = Path(self.data_dir)
        
        # Stats Accumulators (x, y, z, log_energy)
        # Using double precision for accumulation to avoid overflow/precision loss
        running_sum = torch.zeros(4, dtype=torch.float64)
        running_sum_sq = torch.zeros(4, dtype=torch.float64)
        total_points = 0
        
        if self.reflow:
            # --- Reflow (PTH) Case ---
            #FIXME (Stats ignored so far)
            file_paths = list(base_path.rglob("*.pth"))
            if len(file_paths)>0:
                print(f"Found {len(file_paths)} .pth files for reflow.")
            else: #single file
                file_paths = [base_path]
            
            for file_path in file_paths:
                try:
                    data = torch.load(file_path, map_location='cpu')
                    # ... [Existing Reflow Logic] ...
                    if isinstance(data, dict):
                        key = 'x' if 'x' in data else list(data.keys())[0]
                        num_particles = len(data[key])
                    elif isinstance(data, (list, tuple)):
                        num_particles = len(data[0])
                    else:
                        continue
                        
                    for i in range(num_particles):
                        self.global_index_map.append((str(file_path), i))
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        else:
            # --- Standard (H5) Case ---
            file_paths = list(base_path.rglob("*.h5")) + list(base_path.rglob("*.hdf5"))
            lengths = [] # Temporary list to find the max particles
            print(f"Indexing {len(file_paths)} files and computing statistics...")
            
            for file_path in file_paths:
                if "Pb_Simulation" in str(file_path):
                    #FIXME I am hardcoding ignoring Pb_here, NOT NEEED!
                    continue
                try:
                    with h5py.File(file_path, "r") as f:
                        for key in f.keys():
                            self.global_index_map.append((str(file_path), key))
                            
                            # --- Compute Statistics ---
                            # Read raw data to accumulate mean/std
                            group = f[key]
                            indices = group["indices"][:] # (N, 3)
                            values = group["values"][:]   # (N, 1) or (N,)
                            num_particles = indices.shape[0]
                            lengths.append(num_particles)
                            
                            # Handle shape consistency
                            if values.ndim == 1:
                                values = values[:, None]
                            
                            # --- MODIFICATION: Log Transform Energy for Stats ---
                            # We want stats on log(E) so the scaler works correctly.
                            # Add epsilon 1e-6 to avoid log(0)
                            values_log = np.log(values + 1e-6)
                            
                            # Stack: (N, 4) -> [x, y, z, log_energy]
                            data_chunk = np.column_stack((indices, values_log)).astype(np.float64)
                            data_tensor = torch.from_numpy(data_chunk)
                            
                            # Accumulate sums
                            running_sum += torch.sum(data_tensor, dim=0)
                            running_sum_sq += torch.sum(data_tensor ** 2, dim=0)
                            total_points += data_tensor.shape[0]
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            # --- Finalize Statistics ---
            if total_points > 0 and not self.reflow:
                self.max_particles = max(lengths) if lengths else 0
                # E[X]
                mean = (running_sum / total_points).float()
                # Std = sqrt( E[X^2] - (E[X])^2 )
                mean_sq = running_sum_sq / total_points
                std = torch.sqrt(mean_sq - mean.double()**2).float()
                
                # Replace tiny stds with 1.0 to avoid division by zero later
                std = torch.where(std < 1e-6, torch.tensor(1.0), std)
                
                self.stats = {
                    'mean': mean, # [x_mean, y_mean, z_mean, log_e_mean]
                    'std': std,
                    'total_points': total_points
                }
                print(f"Computed Stats (log-energy):\nMean: {mean}\nStd:  {std}")

            # --- 3. Save to Cache ---
            cache_payload = {
                'index_map': self.global_index_map,
                'stats': self.stats,
                'max_particles': self.max_particles
            }
            
            with open(cache_path, "wb") as f:
                pickle.dump(cache_payload, f)
            
            print(f"Dataset indexed. Total events: {len(self.global_index_map)}")

    def __len__(self):
        return len(self.global_index_map)

    def __getitem__(self, idx):
        file_path, local_key = self.global_index_map[idx]
        
        # ... [Existing Reflow Logic] ...
        if self.reflow:
            if file_path not in self.files:
                self.files[file_path] = torch.load(file_path, map_location='cpu')
            data = self.files[file_path]
            sample_idx = int(local_key)
            if isinstance(data, dict):
                x0 = data.get('x', data.get('x0'))[sample_idx]
                x1 = data.get('recons', data.get('x1'))[sample_idx]
                mask = data.get('mask', data.get('mask'))[sample_idx]
                init_e = data.get('init_e', data.get('init_e'))[sample_idx]
            else:
                x0 = data[0][sample_idx]
                x1 = data[1][sample_idx]
                mask = data[2][sample_idx]
                init_e = data[3][sample_idx]
            return (x0, x1, mask, init_e, 0.0, 0.0, idx)

        # --- STANDARD / HDF5 CASE ---
        else:                
            if file_path not in self.files:
                self.files[file_path] = h5py.File(file_path, "r", libver='latest', swmr=True)
            
            f = self.files[file_path]
            
            try:
                group = f[local_key]
                indices = group["indices"][:]
                values = group["values"][:]
                material = group['material'][()].decode('utf-8')
                gap_pid = material_dict[material]
                
                # Note: We return RAW values here. 
                # The Scaler will apply log transform + normalization during training.
                shower = np.column_stack((indices, values))
                
                initial_energy = group.attrs.get("initial_energy", 0.0)
                initial_energy = ((initial_energy - 10.) / 45.) - 1.0
                
                if self.transform:
                    shower = self.transform(shower)

                return (
                    torch.from_numpy(shower).float(),
                    torch.tensor(initial_energy).float(),
                    torch.tensor(0).long(), 
                    torch.tensor(gap_pid).long(),
                    idx
                )
            except Exception as e:
                print(f"Error accessing {local_key} in {file_path}: {e}")
                raise e

# Old version before adding reflow
# class LazyIDLDataset(Dataset):
#     def __init__(self, data_dir, transform=None, reflow= False):
#         self.data_dir = data_dir
#         self.transform = transform
#         # The core of lazy loading: a map from global index to (file_path, local_index)
#         self.global_index_map: List[Tuple[str, int]] = []
#         self.reflow = reflow
#         self.files = {}
#         self._create_global_index_map()
#         # if self.reflow:
#         #     reflow_data = torch.load('data_dir', map_location='cpu')
#         #     self.x0 = reflow_data[0]
#         #     self.x1 = reflow_data[1]
#         #     self.mask = reflow_data[2]

#     def _create_global_index_map(self):
#         base_path = Path(self.data_dir)
#         file_paths = list(base_path.rglob("*.h5")) + list(base_path.rglob("*.hdf5"))
#         cache_path = Path(self.data_dir) / "dataset_cache.pkl"
#         self.max_particles = 2200 #FIXME hardcoded max particles
        
#         lengths = [] # Temporary list to find the min
#         if cache_path.exists():
#             print(f"Loading dataset index from cache: {cache_path}")
#             with open(cache_path, "rb") as f:
#                 self.global_index_map = pickle.load(f)
#             return

#         for file_path in file_paths:
#             if "Pb_Simulation" in str(file_path):
#                 continue
#             try:
#                 with h5py.File(file_path, "r") as f:
#                     for key in f.keys():
#                         group = f[key]
#                         # Read the shape of the dataset without loading the data
#                         #num_particles = group["indices"].shape[0]
#                         self.global_index_map.append((str(file_path), key))
#                         #lengths.append(num_particles)
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
                
#         # Calculate the global minimum points across the entire dataset
#         #self.max_particles = max(lengths) if lengths else 0
#         # Cache the index map for future use
#         with open(cache_path, "wb") as f:
#             pickle.dump(self.global_index_map, f)
        
#         print(f"Dataset indexed. Total events: {len(self.global_index_map)}")
#         print(f"Max particles found in dataset: {self.max_particles}")

#     def __len__(self):
#         """Returns the total number of individual events (showers) in the dataset."""
#         return len(self.global_index_map)

#     def __getitem__(self, idx):
#         """Retrieves a single data sample by loading the necessary file on demand."""
        
#         file_path, group_key = self.global_index_map[idx]
        
#         if self.reflow:
#             # Fix: Use the actual idx provided by the DataLoader
#             sample_idx = idx % self.x0.shape[0]
#             return (self.x0[sample_idx], self.x1[sample_idx], 0.0, 0.0, idx)
#         # if self.reflow:
#         #     #idx = random.randint(0, 1001) #FIXME num samples 1001 hardcoded
#         #     x0 = self.x0[idx % len(self.x0)]
#         #     x0 = self.x0[idx, :, :]
#         #     x1 = self.x1[idx, :, :]
#         #     self.pkl = False
#         #     self.npy = False
#         #     return (x0, x1, 0.0, 0.0, idx)
#         else:                
#             #Load the entire file (the I/O-heavy step, done only when needed)
#             # LAZY OPENING: Check if this file is already open in this worker
#             #Because we are keeping file handles open in self.files, they will stay open until the DataLoader workers are destroyed at the end of the epoch. This is usually fine, but if you have thousands of separate HDF5 files, you might hit the OS "Maximum Open Files" limit.
#             if file_path not in self.files:
#                 # 'swmr=True' allows multiple workers to read the same file safely
#                 self.files[file_path] = h5py.File(file_path, "r", libver='latest', swmr=True)
            
#             f = self.files[file_path]
            
#             try:
#                 group = f[group_key]
#                 # Use [:] for faster slicing than [()]
#                 indices = group["indices"][:]
#                 values = group["values"][:]
#                 material = group['material'][()].decode('utf-8')
#                 gap_pid = material_dict[material]
#                 # Efficiently stack instead of concatenate + newaxis
#                 shower = np.column_stack((indices, values))
                
#                 initial_energy = group.attrs.get("initial_energy", 0.0)
#                 initial_energy = ((initial_energy - 10.) / 45.) - 1.0
                
#                 if self.transform:
#                     shower = self.transform(shower)

#                 return (
#                     torch.from_numpy(shower).float(),
#                     torch.tensor(initial_energy).float(),
#                     torch.tensor(0).long(), # particle_index#TODO hardcoded photon
#                     torch.tensor(gap_pid).long(), # gap_pid
#                     idx
#                 )
#             except Exception as e:
#                 print(f"Error accessing {group_key} in {file_path}: {e}")
#                 # Return a dummy sample or re-raise
#                 raise e
            

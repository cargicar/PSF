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
        # Reflow usually implies reading samples generated in a subfolder, 
        # but we need the stats from the parent folder's original training data.
        self.data_dir = data_dir if not reflow else Path(data_dir).parent
        self.transform = transform
        self.reflow = reflow
        
        self.global_index_map = [] 
        self.files = {} 
        
        self.stats = None 
        self.max_particles = None
        
        self._create_global_index_map()

    def _create_global_index_map(self):
        cache_path = Path(self.data_dir) / "dataset_cache.pkl"
        
        # --- 1. Try Loading Stats from Cache ---
        if cache_path.exists():
            print(f"Loading dataset info from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            
            if isinstance(cached_data, dict):
                self.stats = cached_data.get('stats', None)
                self.max_particles = cached_data.get('max_particles', None)
                
                if self.stats:
                    print(f"Loaded Stats -> Mean: {self.stats['mean']}, Std: {self.stats['std']}")
                
                # CRITICAL CHANGE: 
                # If we are in reflow mode, we DO NOT want the cached index_map 
                # (which points to the original H5 files). We only want the stats.
                if not self.reflow and 'index_map' in cached_data:
                    self.global_index_map = cached_data['index_map']
                    return 
            else:
                # Old cache format fallback
                if not self.reflow:
                    self.global_index_map = cached_data
                    return

        # --- 2. Compute Index (if reflow or cache missing) ---
        base_path = Path(self.data_dir)
        
        if self.reflow:
            # --- REFLOW CASE: Index .pth files ---
            # Search recursively for .pth files (assumed to be in subdirs of base_path)
            file_paths = list(base_path.rglob("*.pth"))
            
            if not file_paths:
                print(f"Warning: No .pth files found in {base_path} for reflow.")
            else:
                print(f"Indexing {len(file_paths)} .pth files for reflow...")

            for file_path in file_paths:
                try:
                    # Load file to get batch size. 
                    # weights_only=False needed for arbitrary list/tuple structures
                    data = torch.load(file_path, map_location='cpu') #, weights_only=False)
                    
                    num_samples = 0
                    
                    # Handle the list structure: [x, pts, mask, int_energy, gap_pid]
                    if isinstance(data, (list, tuple)):
                        # data[0] is 'x' with shape (Batch, Points, Feats)
                        num_samples = len(data[0])
                        
                    # Handle dictionary structure (fallback)
                    elif isinstance(data, dict):
                        # Try finding a known key to determine length
                        possible_keys = ['x', 'x0', 'data', 'recons']
                        for k in possible_keys:
                            if k in data:
                                num_samples = len(data[k])
                                break
                        # Fallback: take length of first key
                        if num_samples == 0 and len(data) > 0:
                            num_samples = len(data[list(data.keys())[0]])

                    # Add entries to map
                    for i in range(num_samples):
                        self.global_index_map.append((str(file_path), i))
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            print(f"Reflow index created. Total samples: {len(self.global_index_map)}")

        else:
            # --- STANDARD CASE: Index H5 files and compute stats ---
            file_paths = list(base_path.rglob("*.h5")) + list(base_path.rglob("*.hdf5"))
            lengths = [] 
            
            # ... [Stats computation logic remains the same] ...
            # (Initializing accumulators)
            running_sum = torch.zeros(4, dtype=torch.float64)
            running_sum_sq = torch.zeros(4, dtype=torch.float64)
            total_points = 0

            print(f"Indexing {len(file_paths)} H5 files and computing statistics...")
            #FIXME I am hardcoding ignoring Pb_here, NOT NEEED!
            for file_path in file_paths:
                try:
                    with h5py.File(file_path, "r") as f:
                        for key in f.keys():
                            self.global_index_map.append((str(file_path), key))
                            
                            # Stats accumulation
                            group = f[key]
                            indices = group["indices"][:] 
                            values = group["values"][:]   
                            num_particles = indices.shape[0]
                            lengths.append(num_particles)
                            
                            if values.ndim == 1: values = values[:, None]
                            
                            # Log transform for stats
                            values_log = np.log(values + 1e-6)
                            data_chunk = np.column_stack((indices, values_log)).astype(np.float64)
                            data_tensor = torch.from_numpy(data_chunk)
                            
                            running_sum += torch.sum(data_tensor, dim=0)
                            running_sum_sq += torch.sum(data_tensor ** 2, dim=0)
                            total_points += data_tensor.shape[0]
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            # Finalize Stats
            if total_points > 0:
                self.max_particles = max(lengths) if lengths else 0
                mean = (running_sum / total_points).float()
                mean_sq = running_sum_sq / total_points
                std = torch.sqrt(mean_sq - mean.double()**2).float()
                std = torch.where(std < 1e-6, torch.tensor(1.0), std)
                
                self.stats = {'mean': mean, 'std': std, 'total_points': total_points}
                print(f"Computed Stats (log-energy):\nMean: {mean}\nStd:  {std}")

            # Save Cache
            cache_payload = {
                'index_map': self.global_index_map,
                'stats': self.stats,
                'max_particles': self.max_particles
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_payload, f)

    def __len__(self):
        return len(self.global_index_map)

    def __getitem__(self, idx):
        file_path, local_key = self.global_index_map[idx]
        
        # --- REFLOW LOADING ---
        if self.reflow:
            if file_path not in self.files:
                # Load the full batch file once and cache it
                self.files[file_path] = torch.load(file_path, map_location='cpu') #, weights_only=False)
            
            data = self.files[file_path]
            sample_idx = int(local_key)
            
            if isinstance(data, (list, tuple)):
                # Expected List format: [x, pts, mask, int_energy, gap_pid]
                x = data[0][sample_idx]
                x0 = data[1][sample_idx]
                mask = data[2][sample_idx]
                init_e = data[3][sample_idx]
                gap_pid = data[4][sample_idx]
                y_pid = torch.tensor(0).long() #FIXME hardcoded photon=0 for now
            
            elif isinstance(data, dict):
                # Dictionary fallback
                x0 = data.get('x', data.get('x0'))[sample_idx]
                x = data.get('recons', data.get('x1'))[sample_idx]
                mask = data.get('mask')[sample_idx]
                init_e = data.get('init_e')[sample_idx]
                gap_pid = torch.tensor(0).long() # Fallback
                y_pid = torch.tensor(0).long()

            return (x, x0, mask, init_e, y_pid, gap_pid, idx)

        # --- STANDARD / HDF5 LOADING ---
        else:                
            if file_path not in self.files:
                self.files[file_path] = h5py.File(file_path, "r", libver='latest', swmr=True)
            
            f = self.files[file_path]
            try:
                group = f[local_key]
                indices = group["indices"][:]
                values = group["values"][:]
                material = group['material'][()].decode('utf-8')
                # Assuming material_dict is available in scope
                gap_pid = material_dict[material] 
                
                shower = np.column_stack((indices, values))
                
                initial_energy = group.attrs.get("initial_energy", 0.0)
                # Normalize initial energy to [-1, 1] range (approx)
                initial_energy = ((initial_energy - 10.) / 45.) - 1.0
                
                if self.transform:
                    shower = self.transform(shower)
                
                return (
                    torch.from_numpy(shower).float(),
                    torch.tensor(initial_energy).float(),
                    torch.tensor(0).long(), #FIXME hardcoded photon=0 for now
                    torch.tensor(gap_pid).long(),
                    idx
                )
            except Exception as e:
                print(f"Error accessing {local_key} in {file_path}: {e}")
                raise e
# class LazyIDLDataset(Dataset):
#     def __init__(self, data_dir, transform=None, reflow=False):
#         self.data_dir = data_dir if not reflow else Path(data_dir).parent
#         self.transform = transform
#         self.reflow = reflow
        
#         # The core of lazy loading: a map from global index to (file_path, local_index)
#         self.global_index_map = [] 
#         # Cache for open file handles (HDF5) or loaded tensors (PTH)
#         self.files = {} 
        
#         # Container for global dataset statistics
#         self.stats = None 
#         self.max_particles = None
        
#         self._create_global_index_map()

#     def _create_global_index_map(self):
#         cache_path = Path(self.data_dir) / f"dataset_cache.pkl"
        
#         # --- 1. Try Loading from Cache ---
#         if cache_path.exists():
#             print(f"Loading dataset index from cache: {cache_path}")
#             with open(cache_path, "rb") as f:
#                 cached_data = pickle.load(f)
            
#             # Check if cache is the new dict format or the old list format
#             if isinstance(cached_data, dict) and 'index_map' in cached_data:
#                 self.global_index_map = cached_data['index_map']
#                 breakpoint() #DEBUG
#                 self.stats = cached_data.get('stats', None)
#                 self.max_particles = cached_data.get('max_particles', None)
#                 if self.stats:
#                     print(f"Loaded Stats -> Mean: {self.stats['mean']}, Std: {self.stats['std']}")
#                 if self.max_particles:
#                     print(f"Loaded max-particles -> {self.max_particles}")
                  
#             else:
#                 # Fallback for old cache (just a list)
#                 self.global_index_map = cached_data
#                 self.stats = None
#                 print("Warning: Old cache format found (no statistics). Delete .pkl to recompute.")
#             return

#         # --- 2. Compute Index & Stats if not cached ---
#         base_path = Path(self.data_dir)
#         # Stats Accumulators (x, y, z, log_energy)
#         # Using double precision for accumulation to avoid overflow/precision loss
#         running_sum = torch.zeros(4, dtype=torch.float64)
#         running_sum_sq = torch.zeros(4, dtype=torch.float64)
#         total_points = 0
        
#         if self.reflow:
#             # --- Reflow (PTH) Case ---
#             file_paths = list(base_path.rglob("*.pth"))
#             if len(file_paths)>0:
#                 print(f"Found {len(file_paths)} .pth files for reflow.")
#             else: #single file
#                 file_paths = [base_path]
            
#             for file_path in file_paths:
#                 try:
#                     data = torch.load(file_path, map_location='cpu')
#                     # ... [Existing Reflow Logic] ...
#                     if isinstance(data, dict):
#                         key = 'x' if 'x' in data else list(data.keys())[0]
#                         num_particles = len(data[key])
#                     elif isinstance(data, (list, tuple)):
#                         num_particles = len(data[0])
#                     else:
#                         continue
                        
#                     for i in range(num_particles):
#                         self.global_index_map.append((str(file_path), i))
                        
#                 except Exception as e:
#                     print(f"Error reading {file_path}: {e}")

#         else:
#             # --- Standard (H5) Case ---
#             file_paths = list(base_path.rglob("*.h5")) + list(base_path.rglob("*.hdf5"))
#             lengths = [] # Temporary list to find the max particles
#             print(f"Indexing {len(file_paths)} files and computing statistics...")
            
#             for file_path in file_paths:
#                 if "Pb_Simulation" in str(file_path):
#                     #FIXME I am hardcoding ignoring Pb_here, NOT NEEED!
#                     continue
#                 try:
#                     with h5py.File(file_path, "r") as f:
#                         for key in f.keys():
#                             self.global_index_map.append((str(file_path), key))
                            
#                             # --- Compute Statistics ---
#                             # Read raw data to accumulate mean/std
#                             group = f[key]
#                             indices = group["indices"][:] # (N, 3)
#                             values = group["values"][:]   # (N, 1) or (N,)
#                             num_particles = indices.shape[0]
#                             lengths.append(num_particles)
                            
#                             # Handle shape consistency
#                             if values.ndim == 1:
#                                 values = values[:, None]
                            
#                             # --- MODIFICATION: Log Transform Energy for Stats ---
#                             # We want stats on log(E) so the scaler works correctly.
#                             # Add epsilon 1e-6 to avoid log(0)
#                             values_log = np.log(values + 1e-6)
                            
#                             # Stack: (N, 4) -> [x, y, z, log_energy]
#                             data_chunk = np.column_stack((indices, values_log)).astype(np.float64)
#                             data_tensor = torch.from_numpy(data_chunk)
                            
#                             # Accumulate sums
#                             running_sum += torch.sum(data_tensor, dim=0)
#                             running_sum_sq += torch.sum(data_tensor ** 2, dim=0)
#                             total_points += data_tensor.shape[0]
                            
#                 except Exception as e:
#                     print(f"Error reading {file_path}: {e}")
            
#             # --- Finalize Statistics ---
#             if total_points > 0 and not self.reflow:
#                 self.max_particles = max(lengths) if lengths else 0
#                 # E[X]
#                 mean = (running_sum / total_points).float()
#                 # Std = sqrt( E[X^2] - (E[X])^2 )
#                 mean_sq = running_sum_sq / total_points
#                 std = torch.sqrt(mean_sq - mean.double()**2).float()
                
#                 # Replace tiny stds with 1.0 to avoid division by zero later
#                 std = torch.where(std < 1e-6, torch.tensor(1.0), std)
                
#                 self.stats = {
#                     'mean': mean, # [x_mean, y_mean, z_mean, log_e_mean]
#                     'std': std,
#                     'total_points': total_points
#                 }
#                 print(f"Computed Stats (log-energy):\nMean: {mean}\nStd:  {std}")

#             # --- 3. Save to Cache ---
#             cache_payload = {
#                 'index_map': self.global_index_map,
#                 'stats': self.stats,
#                 'max_particles': self.max_particles
#             }
            
#             with open(cache_path, "wb") as f:
#                 pickle.dump(cache_payload, f)
            
#             print(f"Dataset indexed. Total events: {len(self.global_index_map)}")

#     def __len__(self):
#         return len(self.global_index_map)

#     def __getitem__(self, idx):
#         file_path, local_key = self.global_index_map[idx]
        
#         # ... [Existing Reflow Logic] ...
#         if self.reflow:
#             if file_path not in self.files:
#                 self.files[file_path] = torch.load(file_path, map_location='cpu', weights_only = False)
#             data = self.files[file_path]
#             sample_idx = int(local_key)
#             # From test: torch.save([x.cpu(), pts.cpu(), mask.cpu(), int_energy.cpu(), gap_pid.cpu()], save_path)  
#             if isinstance(data, dict):
#                 x0 = data.get('x', data.get('x0'))[sample_idx]
#                 x1 = data.get('recons', data.get('x1'))[sample_idx]
#                 mask = data.get('mask', data.get('mask'))[sample_idx]
#                 init_e = data.get('init_e', data.get('init_e'))[sample_idx]
#             else:
#                 x = data[0][sample_idx]
#                 x0 = data[1][sample_idx]
#                 mask = data[2][sample_idx]
#                 init_e = data[3][sample_idx]
#                 gap_pid = data[4][sample_idx]
#                 y_pid = torch.tensor(0).long() #FIXME hardcoded photon=
#             return (x, x0, mask, init_e, y_pid, gap_pid, idx)

#         # --- STANDARD / HDF5 CASE ---
#         else:                
#             if file_path not in self.files:
#                 self.files[file_path] = h5py.File(file_path, "r", libver='latest', swmr=True)
            
#             f = self.files[file_path]
            
#             try:
#                 group = f[local_key]
#                 indices = group["indices"][:]
#                 values = group["values"][:]
#                 material = group['material'][()].decode('utf-8')
#                 gap_pid = material_dict[material]
                
#                 # Note: We return RAW values here. 
#                 # The Scaler will apply log transform + normalization during training.
#                 shower = np.column_stack((indices, values))
                
#                 initial_energy = group.attrs.get("initial_energy", 0.0)
#                 initial_energy = ((initial_energy - 10.) / 45.) - 1.0
                
#                 if self.transform:
#                     shower = self.transform(shower)
#                 #FIXME hardcoded photon
#                 return (
#                     torch.from_numpy(shower).float(),
#                     torch.tensor(initial_energy).float(),
#                     torch.tensor(0).long(), 
#                     torch.tensor(gap_pid).long(),
#                     idx
#                 )
#             except Exception as e:
#                 print(f"Error accessing {local_key} in {file_path}: {e}")
#                 raise e


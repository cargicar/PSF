import os
from pathlib import Path
import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
from typing import List, Tuple, Literal
import pickle
        
class LazyIDLDataset(Dataset):
    def __init__(self, data_dir, transform=None, reflow= False):
        self.data_dir = data_dir
        self.transform = transform
        # The core of lazy loading: a map from global index to (file_path, local_index)
        self.global_index_map: List[Tuple[str, int]] = []
        self.reflow = reflow
        self.files = {}
        self._create_global_index_map()
        if self.reflow:
            reflow_data = torch.load('G4_reflow_DATASET_100.pth', map_location='cpu')
            self.x0 = reflow_data[0]
            self.x1 = reflow_data[1]

    def _create_global_index_map(self):
        base_path = Path(self.data_dir)
        file_paths = list(base_path.rglob("*.h5")) + list(base_path.rglob("*.hdf5"))
        cache_path = Path(self.data_dir) / "dataset_cache.pkl"
        self.max_particles = 1700 #FIXME hardcoded max particles
        
        lengths = [] # Temporary list to find the min
        if cache_path.exists():
            print(f"Loading dataset index from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                self.global_index_map = pickle.load(f)
            return

        for file_path in file_paths:
            if "Pb_Simulation" in str(file_path):
                continue
            try:
                with h5py.File(file_path, "r") as f:
                    for key in f.keys():
                        group = f[key]
                        # Read the shape of the dataset without loading the data
                        #num_particles = group["indices"].shape[0]
                        self.global_index_map.append((str(file_path), key))
                        #lengths.append(num_particles)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
        # Calculate the global minimum points across the entire dataset
        #self.max_particles = max(lengths) if lengths else 0
        # Cache the index map for future use
        with open(cache_path, "wb") as f:
            pickle.dump(self.global_index_map, f)
        
        print(f"Dataset indexed. Total events: {len(self.global_index_map)}")
        print(f"Max particles found in dataset: {self.max_particles}")

    def __len__(self):
        """Returns the total number of individual events (showers) in the dataset."""
        return len(self.global_index_map)

    def __getitem__(self, idx):
        """Retrieves a single data sample by loading the necessary file on demand."""
        
        file_path, group_key = self.global_index_map[idx]
        
        if self.reflow:
            # Fix: Use the actual idx provided by the DataLoader
            sample_idx = idx % self.x0.shape[0]
            return (self.x0[sample_idx], self.x1[sample_idx], 0.0, 0.0, idx)
        # if self.reflow:
        #     #idx = random.randint(0, 1001) #FIXME num samples 1001 hardcoded
        #     x0 = self.x0[idx % len(self.x0)]
        #     x0 = self.x0[idx, :, :]
        #     x1 = self.x1[idx, :, :]
        #     self.pkl = False
        #     self.npy = False
        #     return (x0, x1, 0.0, 0.0, idx)
        else:                
            #Load the entire file (the I/O-heavy step, done only when needed)
            # LAZY OPENING: Check if this file is already open in this worker
            #Because we are keeping file handles open in self.files, they will stay open until the DataLoader workers are destroyed at the end of the epoch. This is usually fine, but if you have thousands of separate HDF5 files, you might hit the OS "Maximum Open Files" limit.
            if file_path not in self.files:
                # 'swmr=True' allows multiple workers to read the same file safely
                self.files[file_path] = h5py.File(file_path, "r", libver='latest', swmr=True)
            
            f = self.files[file_path]
            
            try:
                group = f[group_key]
                # Use [:] for faster slicing than [()]
                indices = group["indices"][:]
                values = group["values"][:]
                
                # Efficiently stack instead of concatenate + newaxis
                shower = np.column_stack((indices, values))
                
                initial_energy = group.attrs.get("initial_energy", 0.0)
                initial_energy = ((initial_energy - 10.) / 45.) - 1.0
                
                if self.transform:
                    shower = self.transform(shower)

                return (
                    torch.from_numpy(shower).float(),
                    torch.tensor(initial_energy).float(),
                    torch.tensor(0).long(), # material_index
                    torch.tensor(0).long(), # gap_pid
                    idx
                )
            except Exception as e:
                print(f"Error accessing {group_key} in {file_path}: {e}")
                # Return a dummy sample or re-raise
                raise e
            
#load latents from pcvae model
class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, latent_file):
        data = torch.load(latent_file)
        self.mu = data['mu']
        self.logvar = data['logvar']

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        # Sample from the distribution (Reparameterization trick)
        mu = self.mu[idx]
        std = torch.exp(0.5 * self.logvar[idx])
        z = mu + std * torch.randn_like(std)
        return z
    

class ECAL_Chunked_Dataset(Dataset):
    def __init__(self, file_path: str,
                 max_seq_length=1700,
                 #energy_digitizer=None,
                 verbose=False,
                 ordering: Literal['energy', 'spatial'] = 'spatial',
                 material_list: list = ["G4_W", "G4_Ta", "G4_Pb"],
                 inference_mode: bool = False
                ):

        # Constant shape per shot
        self.shape = (30, 30, 30)
        self.max_seq_length = max_seq_length
        #self.energy_digitizer = energy_digitizer
        self.verbose = verbose
        self.ordering = ordering
        self.material_list = material_list
        self.SOS_token = 0
        # Positional Tokens 1-27000
        self.EOS_token = 27000 + 1  # 27001
        self.pad_token = self.EOS_token + 1  # 27002
        self.inference_mode = inference_mode

        # Energy tokens
        self.energy_EOS_token = 24938 + 1
        self.energy_pad_token = 24938 + 2

        self.file_list = [os.path.join(file_path, f) 
                      for f in os.listdir(file_path)]# if f.endswith('.pkl')]
        if self.inference_mode:
            self.memory_cache, self.ground_truth_cache = self._load_all_into_memory()
        else:
            self.memory_cache = self._load_all_into_memory()

    def _encode_sample(self, indices, values):
        if self.ordering == 'energy':
            tokens = self.energy_digitizer.tokenize((indices, values))

            if tokens.size > self.max_seq_length:
                topk_idx = np.argpartition(tokens, -self.max_seq_length)[-self.max_seq_length:]
                sorted_positions = topk_idx[np.argsort(-tokens[topk_idx])]
            else:
                sorted_positions = np.argsort(-tokens)

            sorted_energies = tokens[sorted_positions]

            cut_index = np.argmax(sorted_energies == 1)
            if sorted_energies[cut_index] != 1:
                cut_index = len(sorted_energies)

            sorted_positions = sorted_positions[:cut_index]
            sorted_energies = sorted_energies[:cut_index]

        elif self.ordering == 'spatial':
            flat = np.ravel_multi_index(indices.T, self.shape)
            order = np.argsort(flat)
            flat_sorted = flat[order]
            vals_sorted = values[order]

            sorted_energies = np.digitize(vals_sorted, self.energy_digitizer.e_bins)
            if flat_sorted.size > self.max_seq_length:
                flat_sorted = flat_sorted[:self.max_seq_length]
                sorted_energies = sorted_energies[:self.max_seq_length]
            sorted_positions = flat_sorted

        else:
            raise ValueError(f"Unknown ordering: {self.ordering}")

        return sorted_positions, sorted_energies

    def _load_all_into_memory(self):
        # if self.energy_digitizer is None:
        #     raise ValueError("Energy digitizer must be provided for tokenization.")

        cache = []
        file_to_groupkeys = {}
        ground_truth_cache = []

        if self.verbose:
            print('Loading Files Into Memory...')

        for file_path in self.file_list:
            if self.verbose:
                print(f'Processing file: {file_path}')
            with h5py.File(file_path, "r") as f:
                for key in f.keys():
                    group = f[key]
                    indices = group["indices"][()]
                    values = group["values"][()]
                    initial_energy = group.attrs["initial_energy"].item()
                    material = group['material'][()].decode('utf-8')

                    if self.inference_mode:
                        ground_truth_cache.append(initial_energy)
                    material_index = self.material_list.index(material)
                    #initial_energy = ((initial_energy - 10.) / 45.) - 1.0 # Scale to roughly -1 to 1
                    #print("Material:", material, "Index:", material_index, "Initial Energy:", initial_energy)

                    # Tokenize + order
                    # sorted_positions, sorted_energies = self._encode_sample(indices, values)
                    # sorted_positions, sorted_energies = self._encode_sample(indices, values)
                    # # Add SOS/EOS
                    # sorted_positions = np.insert(sorted_positions, 0, self.SOS_token)
                    # sorted_positions = np.append(sorted_positions, self.EOS_token)

                    # sorted_energies = np.insert(sorted_energies, 0, self.SOS_token)
                    # sorted_energies = np.append(sorted_energies, self.energy_EOS_token)

                    #cache.append((sorted_positions, sorted_energies, initial_energy, material_index))
                    cache.append((indices, values, initial_energy, material_index))


        if not self.inference_mode:
            random.shuffle(cache)
            return cache

        else:
            return cache,ground_truth_cache

    def __len__(self):
        return len(self.memory_cache)

    def __getitem__(self, idx):
        pos, ene, initial_energy, material_index = self.memory_cache[idx]
        initial_energy_t = self.ground_truth_cache[idx]
        assert len(pos) == len(ene)
        # Pad to max_seq_length
        if len(pos) < self.max_seq_length:
            pos = np.pad(pos, (0, self.max_seq_length - len(pos)), constant_values=self.pad_token)
            ene = np.pad(ene, (0, self.max_seq_length - len(ene)), constant_values=self.energy_pad_token)
        elif len(pos) > self.max_seq_length:
            pos = pos[:self.max_seq_length]
            ene = ene[:self.max_seq_length]
        else:
            pass

        if not self.inference_mode:
            return pos, ene, initial_energy, material_index

        else:
            return pos, ene, initial_energy, material_index, initial_energy_t

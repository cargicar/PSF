import os
import json
import torch
from copy import copy
import open3d as o3d
from torch.utils.data import Dataset
import numpy as np
import pickle
from typing import List, Tuple

class LazyPklDataset(Dataset):
    """
    A PyTorch Dataset that loads data from pickle files lazily (on demand) 
    to avoid loading the entire dataset into memory at initialization.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # The core of lazy loading: a map from global index to (file_path, local_index)
        self.global_index_map: List[Tuple[str, int]] = []
        #FIXME Temporary
        self.pkl = None
        self.npy = None
        self._create_global_index_map()

    def _create_global_index_map(self):
        """
        Scans all files to determine the total number of events and creates 
        the global index map. This only reads file structure/metadata, NOT the heavy data.
        """
        file_paths_pkl = [os.path.join(self.data_dir, f) 
                      for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        file_paths_npy = [os.path.join(self.data_dir, f) 
                      for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        if len(file_paths_pkl)>0: self.pkl = True
        if len(file_paths_npy)>0: self.npy = True
        # We need a robust way to get the count without loading the heavy arrays.
        if self.pkl:
            for file_path in file_paths_pkl:
                try:
                    # 1. Load the file structure (often a small dictionary). 
                    # This should be memory-efficient if the structure is not the full data.
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    # Assuming 'showers' contains a list with a single array: [np.array(N_events, ...)]
                    showers_array = data['showers'][0]
                    
                    # Check the actual number of events in this file
                    num_showers = len(showers_array)
                    
                    # Ensure data is released immediately after reading its length
                    del data 

                    # 2. Map all events in this file to their file path and local index
                    for local_idx in range(num_showers):
                        self.global_index_map.append((file_path, local_idx))
                        
                except (pickle.UnpicklingError, FileNotFoundError, KeyError, IndexError, TypeError) as e:
                    print(f"Error indexing file structure '{file_path}': {e}. Skipping file.")
        elif self.npy:
            for file_path in file_paths_npy:
                try:
                    # 1. Load the file structure (often a small dictionary). 
                    # This should be memory-efficient if the structure is not the full data.
                    data = np.load(file_path)
                    # Assuming 'showers' contains a list with a single array: [np.array(N_events, ...)]
                    #showers_array = data['showers'][0]
                    #FIXME Temporary
                    # Check the actual number of events in this file
                    num_showers = len(data)
                    
                    # Ensure data is released immediately after reading its length
                    del data 

                    # 2. Map all events in this file to their file path and local index
                    for local_idx in range(num_showers):
                        self.global_index_map.append((file_path, local_idx))
                        
                except (FileNotFoundError, KeyError, IndexError, TypeError) as e:
                    print(f"Error indexing file structure '{file_path}': {e}. Skipping file.")        

        print(f"Dataset indexed. Total events found: {len(self.global_index_map)}")

    def __len__(self):
        """Returns the total number of individual events (showers) in the dataset."""
        return len(self.global_index_map)

    def __getitem__(self, idx):
        """Retrieves a single data sample by loading the necessary file on demand."""
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # 1. Look up the file path and local index for the requested global index
        file_path, local_idx = self.global_index_map[idx]
        
        if self.pkl:
                
            #Load the entire file (the I/O-heavy step, done only when needed)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # Extract the necessary data arrays (assuming they are single-element lists)
            all_showers_in_file = data['showers'][0]
            all_energies_in_file = data['energies'][0]
            pid_in_file = data['pid'][0]
            gap_pid_in_file = data['gap_pid'][0]
            
            # 3. Retrieve the specific sample (the "lazy" selection)
            shower = all_showers_in_file[local_idx]
            energy = all_energies_in_file[local_idx]
            
            # Since pid/gap_pid were replicated in the original dataset, 
            # we assume the single scalar value applies to all showers in the file.
            pid = pid_in_file 
            gap_pid = gap_pid_in_file

            # NOTE: Using hardcoded max/min
            max_e = 1000000
            min_e = 1000
            
            # Normalization
            energy = (energy - min_e) / (max_e - min_e)
        elif self.npy:
            #Load the entire file (the I/O-heavy step, done only when needed)
            data = np.load(file_path)
                
            # Extract the necessary data arrays (assuming they are single-element lists)
            all_showers_in_file = data
            
            # 3. Retrieve the specific sample (the "lazy" selection)
            shower = all_showers_in_file[local_idx]
            energy = 0.0
            pid = 0
            gap_pid = 0

        if self.transform:
            shower = self.transform(shower)

        # The showers data has shape (N_particles, 4)
        shower = torch.from_numpy(shower).float()
        
        # The energy, pid, and gap_pid are scalars
        energy = torch.tensor(energy).float()
        pid = torch.tensor(pid).long()
        gap_pid = torch.tensor(gap_pid).long()

        # Return the tensors
        return (shower, energy, pid, gap_pid)

class PointCloudMasks(object):
    '''
    render a view then save mask
    '''
    def __init__(self, radius : float=10, elev: float =45, azim:float=315, ):

        self.radius = radius
        self.elev = elev
        self.azim = azim


    def __call__(self, points):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        camera = [self.radius * np.sin(90-self.elev) * np.cos(self.azim),
                  self.radius * np.cos(90 - self.elev),
                  self.radius * np.sin(90 - self.elev) * np.sin(self.azim),
                  ]
        # camera = [0,self.radius,0]
        _, pt_map = pcd.hidden_point_removal(camera, self.radius)

        mask = torch.zeros_like(points)
        mask[pt_map] = 1

        return mask #points[pt_map]


####################################################################################

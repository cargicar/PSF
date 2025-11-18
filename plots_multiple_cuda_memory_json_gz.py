import os
import glob
import gzip
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
_CATEGORY_TO_COLORS = {
                "PARAMETER": "darkgreen",
                "OPTIMIZER_STATE": "goldenrod",
                "INPUT": "black",
                "TEMPORARY": "mediumpurple",
                "ACTIVATION": "red",
                "GRADIENT": "mediumblue",
                "AUTOGRAD_DETAIL": "royalblue",
                None: "grey",}                

_ACTION= {
    "PREEXISTING": 1,
    "CREATE" : 2,
    "INCREMENT_VERSION" : 3,
    "DESTROY" : 4,
}

_ACTION_TO_INDEX = {c: i for i, c in enumerate(_ACTION)}

_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}

def _coalesce_timeline(timeline):
        """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
        times: list[int] = []
        sizes: list[list[int]] = []


        # def update(key, version, delta):
        #     category = (
        #         self.categories.get(key, version)
        #         if isinstance(key, TensorKey)
        #         else None
        #     )
        #     index = _CATEGORY_TO_INDEX[category] + 1
        #     sizes[-1][index] += int(delta)
        def update(category, delta):
            #index = _CATEGORY_TO_INDEX[category] + 1
            index = category + 1
            sizes[-1][index] += int(delta)
        
        t_min = -1
        for t, action,  numbytes, category in timeline:
            
            # Convert timestamps from ns to us, to match trace events.
            if t != -1:
                t = int(t / 1000)

            # Save the smallest timestamp to populate pre-existing allocs.
            if t_min == -1 or (t < t_min and t > 0):
                t_min = t

            # Handle timestep
            if len(times) == 0:
                times.append(t)
                sizes.append([0] + [0 for _ in _CATEGORY_TO_INDEX])

            elif t != times[-1]:
                times.append(t)
                sizes.append(sizes[-1].copy())
            # Handle memory and categories
            update(category, numbytes)

        times = [t_min if t < 0 else t for t in times]
        
        return times, sizes


def open_json_gz(base_folder, device_str='cuda:0'):
    
    patterns = ["calopodit_*", "pvcnn2_*"]
    npoints_calopodit = []
    npoints_pvcnn2 = []
    max_mem_calopodit= []
    max_mem_pvcnn2= []
    for pattern in patterns:
        search_pattern = os.path.join(base_folder, pattern)
            
        subfolders = glob.glob(search_pattern)

        # loop through each found subfolder
        
        for folder_path in subfolders:
            # Check if the path is actually a directory (important for robustness)
            if os.path.isdir(folder_path):
                file_name = "profiling/memory_timeline.raw.json.gz"
                file_path = os.path.join(folder_path, file_name)
                # 5. Check if the file exists
                if os.path.exists(file_path):
                    try:    
                    
                        #with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            '''data is a list. Each item in the list is a raw memory event consisting of (timestamp, action, numbytes, category)'''
                            data = json.load(f)

                        mt = _coalesce_timeline(data)
                        times, sizes = np.array(mt[0]), np.array(mt[1])
                        # For this timeline, start at 0 to match Chrome traces.
                        t_min = min(times)
                        times -= t_min
                        stacked = np.cumsum(sizes, axis=1) / 1024**3
                        max_mem = stacked.max()
                        if "calopodit" in pattern:
                            max_mem_calopodit.append(max_mem)
                            npoints_calopodit.append(int(folder_path.split('_')[-1]))
                        elif "pvcnn2" in pattern:
                            max_mem_pvcnn2.append(max_mem)
                            npoints_pvcnn2.append(int(folder_path.split('_')[-1]))
                    except Exception as e:
                        print(f"  - 🚨 Error processing file {file_path}: {e}")
                    
                else:
                    print(f"\n⚠️ File not found in {folder_path}: {file_name}")
            else:
                # This case is unlikely if using glob.glob(search_pattern) but good practice
                print(f"⚠️ Path is not a directory: {folder_path}")
        
    plt.scatter(npoints_calopodit, max_mem_calopodit, label=patterns[0])
    plt.scatter(npoints_pvcnn2, max_mem_pvcnn2, label=patterns[1])
    plt.xlabel("Number of Points (K)")
    plt.ylabel("Max CUDA Memory (GB)")
    plt.title("Max CUDA Memory vs Number of Points")
    plt.legend()
    plt.savefig(os.path.join(base_folder, "max_cuda_memory_plot.png"))

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="output/scaling/" )

    args = parser.parse_args()

    return args

 


if __name__ == "__main__":
    args = parse_args()
    file_path =args.file_path
    open_json_gz(file_path)


    import os
    import glob
    import gzip
    import json

    def process_memory_files(base_folder="scaling"):
        """
        Loops through all 'calopodit_xxx' subfolders in the base_folder
        and processes the 'memory.raw.json.gz' file found inside each one.
        """
        print(f"--- Starting to process folders in: {base_folder} ---")

        # 1. Define the search pattern
        # glob.glob finds all pathnames matching a specified pattern.
        # The pattern matches any directory named 'calopodit_' followed by anything ('*')
        search_pattern = os.path.join(base_folder, "calopodit_*")
        
        # 2. Get a list of all matching subfolders
        subfolders = glob.glob(search_pattern)
        
        if not subfolders:
            print(f"❌ No subfolders matching 'calopodit_*' found in '{base_folder}'.")
            return

        # 3. Loop through each found subfolder
        for folder_path in subfolders:
            # Check if the path is actually a directory (important for robustness)
            if os.path.isdir(folder_path):
                # 4. Define the target file path
                file_name = "memory.raw.json.gz"
                file_path = os.path.join(folder_path, file_name)
                
                # 5. Check if the file exists
                if os.path.exists(file_path):
                    print(f"\n✅ Found file: {file_path}")
                    
                    try:
                        # --- START: File Processing Logic ---
                        
                        # Open the gzipped file for reading (mode 'rt' for text)
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            # Load the JSON data
                            data = json.load(f)
                            
                            # **TODO: Your processing logic goes here**
                            # Example: Print the ID and the number of keys in the loaded data
                            folder_id = os.path.basename(folder_path).split('_')[-1]
                            print(f"  - Folder ID: **{folder_id}**")
                            print(f"  - Data keys loaded: **{len(data.keys())}**")
                            
                            # Example: Perform an aggregation or write to a new file
                            # if 'some_key' in data:
                            #     print(f"  - Value of 'some_key': {data['some_key']}")
                            
                        # --- END: File Processing Logic ---

                    except Exception as e:
                        print(f"  - 🚨 Error processing file {file_path}: {e}")
                else:
                    print(f"\n⚠️ File not found in {folder_path}: {file_name}")
            else:
                # This case is unlikely if using glob.glob(search_pattern) but good practice
                print(f"⚠️ Path is not a directory: {folder_path}")

        print("\n--- Processing complete ---")

    # --- Execute the function ---
    # Make sure your current working directory contains the 'scaling' folder
    process_memory_files()
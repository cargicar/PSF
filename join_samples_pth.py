import torch
import os
from pathlib import Path

def combine_pth_files(input_dir, output_filename, pattern="*photon_samples_*.pth"):
    input_path = Path(input_dir)
    
    # 1. Filter and sort files (sorting ensures samples_1 comes before samples_2)
    files = sorted(list(input_path.glob(pattern)), 
                   key=lambda x: int(x.stem.split('_')[-1]))
    
    if not files:
        print(f"No files matching {pattern} found in {input_dir}")
        return
    
    print(f"Found {len(files)} files. Loading...")

    # 2. Load all files into a list
    loaded_data = []
    for f in files:
        data = torch.load(f)
        if isinstance(data, list):
            # If it's a list, extend our main list with its elements
            loaded_data.extend(data)
        else:
            # If it's already a tensor, just add it
            loaded_data.append(data)
    
    # 3. Concatenate data
    # Note: dim=0 assumes you want to stack them along the first dimension
    combined_data = torch.cat(loaded_data, dim=0)
    
    # 4. Save the result
    torch.save(combined_data, output_filename)
    print(f"Successfully saved concatenated tensor to {output_filename}")
    print(f"Final shape: {combined_data.shape}")

# Usage
if __name__ == "__main__":
    # Update these paths to match your setup
    target_directory = "./" 
    output_file = "combined_photon_samples.pth"
    
    combine_pth_files(target_directory, output_file)
import torch
import os
from pathlib import Path

def combine_pth_files(input_dir, output_filename, pattern="*_calopodit_gen_Jan_17_batch*.pth"):
    input_path = Path(input_dir)
    
    # 1. Filter and sort files (sorting ensures samples_1 comes before samples_2)
    files = sorted(list(input_path.glob(pattern)), 
                   key=lambda x: int(x.stem.split('_')[-1]))
    
    if not files:
        print(f"No files matching {pattern} found in {input_dir}")
        return
    
    print(f"Found {len(files)} files. Loading...")

    # 2. Load all files into a list
    xs = []
    gens = []
    masks = []
    for f in files:
        #data = torch.load(f)
        x, gen, mask = torch.load(f, map_location='cpu')
        xs.append(x)
        gens.append(gen)
        masks.append(mask)
    
    # Concatenate data
    # Note: dim=0 assumes you want to stack them along the first dimension
    xs = torch.cat(xs, dim=0)
    gens = torch.cat(gens, dim=0)
    masks = torch.cat(masks, dim=0)
    
    # 4. Save the result
    torch.save([xs, gens, masks], output_filename)
    print(f"Successfully saved concatenated tensor to {output_filename}")
    print(f"Final shape: {xs.shape}")

# Usage
if __name__ == "__main__":
    # Update these paths to match your setup
    target_directory = "/pscratch/sd/c/ccardona/datasets/pth" 
    output_file = f"/pscratch/sd/c/ccardona/datasets/pth/combined_batches_calopodit_gen_Jan_17.pth"
    
    combine_pth_files(target_directory, output_file)
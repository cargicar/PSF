import torch
import os
from pathlib import Path

def combine_pth_files(input_dir, output_filename, pattern="_calopodit_gen_gapclasses_sample_w_Feb_1_rank_*.pth"):
    input_path = Path(input_dir)
    
    # 1. Filter and sort files (sorting ensures samples_1 comes before samples_2)
    files = sorted(list(input_path.glob(pattern)), 
                   key=lambda x: int(x.stem.split('_')[-1]))
    
    if not files:
        print(f"No files matching {pattern} found in {input_dir}")
        return
    
    print(f"Found {len(files)} files. Loading...")

    # 2. Load all files into a list
    #torch.save([x.cpu(), pts.cpu(), mask.cpu(), int_energy.cpu, gap_pid.cpu()], save_path)  
    xs = []
    gens = []
    masks = []
    init_es = []
    gaps = []

    for f in files:
        #data = torch.load(f)
        x, gen, mask, init_e, gap = torch.load(f, map_location='cpu', weights_only = False)
        xs.append(x)
        gens.append(gen)
        masks.append(mask)
        init_es.append(init_e)
        gaps.append(gap)
    
    # Concatenate data
    # Note: dim=0 assumes you want to stack them along the first dimension
    xs = torch.cat(xs, dim=0)
    gens = torch.cat(gens, dim=0)
    masks = torch.cat(masks, dim=0)
    init_es = torch.cat(init_es, dim=0)
    gaps = torch.cat(gaps, dim=0)
    
    # 4. Save the result
    torch.save([xs, gens, masks, init_es, gaps], output_filename)
    print(f"Successfully saved concatenated tensor to {output_filename}")
    print(f"Final shape: {xs.shape}")

# Usage
if __name__ == "__main__":
    # Update these paths to match your setup
    target_directory = "/pscratch/sd/c/ccardona/datasets/pth" 
    output_file = f"/pscratch/sd/c/ccardona/datasets/pth/combined_batches_calopodit_gen_Feb_3_sample_w.pth"
    
    combine_pth_files(target_directory, output_file)
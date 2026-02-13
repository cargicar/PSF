import torch
import os
from pathlib import Path

def combine_pth_files(input_dir, output_filename, pattern="*.pth"):
    input_path = Path(input_dir)
    
    # 1. Filter and sort files
    # Note: Your lambda assumes the filename ends exactly in '_rank_X.pth'. 
    # If filenames vary, consider a more robust sort or try-except block.
    try:
        files = sorted(list(input_path.glob(pattern)), 
                       key=lambda x: int(x.stem.split('_')[-1]))
    except ValueError:
        print("Warning: Could not sort files numerically. Using default sort.")
        files = sorted(list(input_path.glob(pattern)))
    
    if not files:
        print(f"No files matching {pattern} found in {input_dir}")
        return
    
    print(f"Found {len(files)} files. Loading...")

    xs = []
    gens = []
    masks = []
    init_es = []
    gaps = []

    files_skipped = 0

    for f in files:
        try:
            # Load the data
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            # --- NEW: Safety Check ---
            if len(data) < 5:
                print(f"Warning: Skipping {f.name} - Found only {len(data)} components, expected 5.")
                files_skipped += 1
                continue
            
            # Unpack safely
            x, gen, mask, init_e, gap = data
            
            xs.append(x)
            gens.append(gen)
            masks.append(mask)
            init_es.append(init_e)
            gaps.append(gap)
            
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
            files_skipped += 1

    if not xs:
        print("Error: No valid data found to combine!")
        return

    # 3. Concatenate data
    print(f"Concatenating {len(xs)} valid files (Skipped {files_skipped})...")
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
    target_directory = "/pscratch/sd/c/ccardona/datasets/pth/" 
    pattern = "_calopodit_samples_Reflow_unNormalized_Feb_11_2_steps_rank_*.pth"
    output_file = f"{target_directory}/combined_batches_calopodit_UnNormalized_Feb_13_500_steps.pth"
    
    combine_pth_files(target_directory, output_file, pattern= pattern)
import torch
import time
import pandas as pd
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from model.calopodit import DiT, DiTConfig
from rectified_flow.rectified_flow import RectifiedFlow
from utils.train_utils import *


def measure_generation_speed(model, 
                             euler_sampler, 
                             num_steps, 
                             name="Model", 
                             BATCH_SIZE= 36, 
                             NUM_POINTS=1700, 
                             DEVICE=None):
    print(f"\n--- Benchmarking {name} ({num_steps} steps) ---")
    
    # 1. Create Synthetic Inputs (Bypassing Dataset)
    # Standard Gaussian Noise (Latent space)
    x_0 = torch.randn(BATCH_SIZE, NUM_POINTS, 4, device=DEVICE)
    
    # Synthetic Conditions (e.g., all class 0, energy 100 GeV)
    y = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    gap = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    energy = torch.ones(BATCH_SIZE, 1, device=DEVICE) #* 100.0
    #mask = torch.zeros((len(showers_list), max_particles), dtype=torch.bool, device= device)
    mask = torch.zeros((BATCH_SIZE, NUM_POINTS), dtype= torch.bool, device=DEVICE) # No padding for benchmark
    
    # Warm-up pass (to initialize CUDA kernels, ignore this in timing)
    _ = euler_sampler.sample_loop(
        x_0=x_0, y=y, gap=gap, energy=energy, mask=mask, 
        num_samples=BATCH_SIZE, num_steps=2 # Fast warm-up
    )
    torch.cuda.synchronize()

    # 2. Timing the full generation loop
    start_time = time.perf_counter()
    
    traj = euler_sampler.sample_loop(
        x_0=x_0, 
        y=y, 
        gap=gap, 
        energy=energy, 
        mask=None, 
        num_samples=BATCH_SIZE, 
        num_steps=num_steps,
        cfg_scale=1.0 # Keep CFG=1.0 for fair speed comparison
    )
    
    torch.cuda.synchronize() # Crucial: ensure GPU is finished before stopping clock
    end_time = time.perf_counter()
    
    # 3. Calculate Results
    total_time = end_time - start_time
    time_per_sample = (total_time / BATCH_SIZE) * 1000 # in milliseconds
    samples_per_sec = BATCH_SIZE / total_time
    
    print(f"Total Time for {BATCH_SIZE} samples: {total_time:.4f}s")
    print(f"Latency: {time_per_sample:.2f} ms/sample")
    print(f"Throughput: {samples_per_sec:.2f} samples/sec")
    
    return total_time

def load_model_on_gpu(ckpt_path, config):
    """
    Loads a pretrained CaloPoDiT model onto a single GPU, 
    cleaning DDP prefixes if they exist.
    """
    print(f"Initializing model and loading weights from: {ckpt_path}")
    
    # 1. Initialize the raw model on the target device
    model = DiT(config).to(DEVICE)
    
    # 2. Load checkpoint to CPU first to prevent GPU memory spikes
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['model_state']
    
    # 3. Strip 'module.' prefix if it exists (common if trained with DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # 4. Load weights into the model
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    print(f"Model successfully loaded on {DEVICE}")
    return model

def benchmark_sweep(ckpt_path, name, config):

    # --- CONFIG ---
    BATCH_SIZE = 128
    NUM_POINTS = 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    STEPS_TO_TEST = [1, 2, 5, 10, 25, 50, 100]

    results = []
    
    # 1. Load Model
    model = DiT(config).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    rf = RectifiedFlow(velocity_field=model, data_shape=(NUM_POINTS, 4))
    sampler = MyEulerSampler(rectified_flow=rf)

    # 2. Synthetic Inputs
    x_0 = torch.randn(BATCH_SIZE, NUM_POINTS, 4, device=DEVICE)
    y = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    gap = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    energy = torch.ones(BATCH_SIZE, 1, device=DEVICE) * 50.0 # Mid-range energy
    mask = torch.ones(BATCH_SIZE, NUM_POINTS, device=DEVICE)

    print(f"\n--- Sweeping {name} ---")

    for steps in STEPS_TO_TEST:
        # Warm-up
        _ = sampler.sample_loop(x_0=x_0, y=y, gap=gap, energy=energy, mask=mask, num_steps=1)
        torch.cuda.synchronize()

        # Timing
        start = time.perf_counter()
        with torch.no_grad():
            traj = sampler.sample_loop(
                x_0=x_0, y=y, gap=gap, energy=energy, mask=mask, 
                num_samples=BATCH_SIZE, num_steps=steps, cfg_scale=1.0
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Metrics
        total_time = end - start
        throughput = BATCH_SIZE / total_time
        
        # Simple Physics Check: Mean Energy of generated points
        # If the flow is "broken", this usually drifts far from the input '50.0'
        gen_mean_energy = traj.x_t[..., 3].mean().item()

        results.append({
            "Steps": steps,
            "Latency_ms": (total_time / BATCH_SIZE) * 1000,
            "Throughput": throughput,
            "Energy_Consistency": gen_mean_energy
        })
        print(f"Steps: {steps:>3} | Latency: {results[-1]['Latency_ms']:>6.2f}ms | Energy Check: {gen_mean_energy:>6.2f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Define your model configuration

    # --- BENCHMARK CONFIG ---
    BATCH_SIZE = 24
    NUM_POINTS = 1000
    NUM_STEPS = 1   # Reflow Model target steps
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Note: Ensure these match the weights you are loading!
    # Path to your weights
    path_base = "output/train_calopodit/2026-02-06-07_train_w_ta/epoch_18.pth"
    path_reflow = "output/train_calopodit/2026-02-11-05_reflow_dropout/epoch_19.pth"
    
    DiT_config = DiTConfig(
            #Point transformer config
            k = 16,
            nblocks =  4,
            name= "calopodit",
            num_points = NUM_POINTS,
            energy_cond = True,#opt.energy_cond,
            in_features=4,
            transformer_features = 128, #512 = hidden_size in current implementation
            #DiT config
            num_classes = 0,
            gap_classes = 2,
            out_channels=4, #opt.out_channels,
            hidden_size=128,
            depth=13,
            num_heads=8,
            mlp_ratio=4,
            use_long_skip=True,
        )
        
    use_model = "reflow" #base
    name = use_model
    data_shape = (NUM_POINTS, 4)  # (N, 4) 4 for (x,y,z,energy)

    if use_model == "reflow":
        model = load_model_on_gpu(path_reflow, DiT_config)
        model.eval()
        rectified_flow = RectifiedFlow(
        data_shape=data_shape,
        interp="straight",
        source_distribution="normal",
        is_independent_coupling=True,
        train_time_distribution="uniform",
        train_time_weight="uniform",
        criterion="mse", #opt.criterion,
        velocity_field=model,
        #device=accelerator.device,
        dtype=torch.float32,
        )

        rectified_flow.device = DEVICE
        euler_sampler = MyEulerSampler(
                            rectified_flow=rectified_flow,
                        )
            

    elif use_model == "base":
        model = load_model_on_gpu(path_base, DiT_config)
        model.eval()
        rectified_flow = RectifiedFlow(
            data_shape=data_shape,
            interp="straight",
            source_distribution="normal",
            is_independent_coupling=True,
            train_time_distribution="uniform",
            train_time_weight="uniform",
            criterion="mse", #opt.criterion,
            velocity_field=model,
            #device=accelerator.device,
            dtype=torch.float32,
        )
        
        rectified_flow.device = DEVICE
        euler_sampler = MyEulerSampler(
                                rectified_flow=rectified_flow,
                            )
    else:
        model = None
        euler_sampler = None
        print(f"pick a set of weights.")
    

    
    time_step = measure_generation_speed(model,
                                        euler_sampler, 
                                        NUM_STEPS, 
                                        name, 
                                        DEVICE=DEVICE, 
                                        BATCH_SIZE=BATCH_SIZE)

    
    # Comparison
    #improvement = time_base / time_reflow
    print(f"\nSPEED: model {use_model} is {time_step}.") #{time_step:.1f}
    #Benachmark sweep spot
    # df_base = benchmark_sweep("path/to/base_model.pt", "Base Model")
    # df_reflow = benchmark_sweep("path/to/reflow_model.pt", "Reflow Model")

    # # Final summary can be printed or saved to CSV
    # print("\nSummary (Reflow Model):")
    # print(df_reflow)


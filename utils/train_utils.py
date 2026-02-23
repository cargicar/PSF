import torch.nn as nn
import torch.multiprocessing as mp
import torch.utils.data
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.g4_pc_dataset import LazyPklDataset
#from datasets.idl_dataset import ECAL_Chunked_Dataset as IDLDataset
from datasets.idl_dataset import LazyIDLDataset as IDLDataset
from datasets.transforms import MinMaxNormalize, CentroidNormalize, Compose
from rectified_flow.samplers.base_sampler import Sampler
from rectified_flow.samplers import EulerSampler
from rectified_flow.flow_components.loss_function import RectifiedFlowLossFunction
import numpy as np
import torch.nn.functional as F
import awkward as ak
from OmniJetAlphaC_phys_plotting import plot_paper_plots
import matplotlib.pyplot as plt



def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def get_dataset(dataroot, npoints,category, name='shapenet', reflow = False):
    if name == 'shapenet':
        train_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
            categories=[category], split='train',
            tr_sample_size=npoints,
            te_sample_size=npoints,
            scale=1.,
            reflow = False,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            random_subsample=True)
        test_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
            categories=[category], split='val',
            tr_sample_size=npoints,
            te_sample_size=npoints,
            scale=1.,
            reflow = False,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            all_points_mean=tr_dataset.all_points_mean,
            all_points_std=tr_dataset.all_points_std,
        )
    elif name == 'g4':
        
        centroid_transform = CentroidNormalize()
        #minmax_transform = MinMaxNormalize(min_vals, max_vals)

        composed_transform = Compose([
                            centroid_transform,
        #                    minmax_transform,
                            ])

        #dataset.transform = minmax_transform
        dataset = LazyPklDataset(os.path.join(dataroot), transform=None)
        #NOTE in case we want to do the splits in this form. Is cleaner to do it "in-house'"
        # total_size = len(dataset)
        # num_train = int(total_size * 0.8)
        # num_val = total_size - num_train
        # lengths = [num_train, num_val]

        # RNG = torch.Generator().manual_seed(42)

        # # 2. Pass the generator to random_split
        # train_dataset, test_dataset = torch.utils.data.random_split(
        #     dataset, 
        #     lengths, 
        #     generator=RNG  # This line makes the split reproducible
        # )
        train_dataset = dataset
        test_dataset = None
        #te_dataset = LazyPklDataset(os.path.join(dataroot, 'val'), transform
    elif name == 'idl':
        if reflow:
            dataset = IDLDataset(dataroot, reflow = True)
        else:
            dataset = IDLDataset(dataroot)#, max_seq_length=npoints, ordering='spatial', material_list=["G4_W", "G4_Ta", "G4_Pb"], inference_mode=False)
        train_dataset = dataset
        test_dataset = None
    return train_dataset, test_dataset

def pad_collate_fn(batch, max_particles=1000):
    """
    Custom collate function to handle batches of showers with varying numbers of particles.
    It pads or truncates each shower to a fixed size and then stacks them.

    Args:
        batch (list): A list of data samples from the dataset.
        max_particles (int): The maximum number of particles to keep per shower.
    Returns:
        A tuple of batched PyTorch tensors.
    """
    showers_list, energies_list, pids_list, gap_pids_list, idx = zip(*batch)
    nfeatures, dtype, device = showers_list[0].shape[1], showers_list[0].dtype, showers_list[0].device
    
    # Initialize tensors for padded data and masks
    # (Batch_Size, Max_N, 3)
    padded_batch = torch.zeros((len(showers_list), max_particles, nfeatures), dtype= dtype, device= device)
    mask = torch.zeros((len(showers_list), max_particles), dtype=torch.bool, device= device)
    
    for i, shower in enumerate(showers_list):
        num_particles = shower.shape[0]
        padded_batch[i, :num_particles, :] = shower
        mask[i, :num_particles] = True
        

    # Stack all tensors to create the batch
    energies_batch = torch.stack(energies_list, dim=0)
    pids_batch = torch.stack(pids_list, dim=0)
    gap_pids_batch = torch.stack(gap_pids_list, dim=0)
    
    return padded_batch, mask, energies_batch, pids_batch, gap_pids_batch, idx

def get_dataloader(opt, train_dataset, test_dataset=None, collate_fn=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler, pin_memory = True,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True, collate_fn=collate_fn )

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler, pin_memory = True,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False,  collate_fn=collate_fn)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler

def multi_gpu_wrapper(model, f):
        return f(model)


class MyEulerSamplerPVCNN(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, **model_kwargs):
        # Extract the current time, next time point, and current state
        t, t_next, x_t = self.t, self.t_next, self.x_t
        x_t = x_t.transpose(1,2)
        # Compute the velocity field at the current state and time
        v_t = self.rectified_flow.get_velocity(x_t=x_t, t=t, **model_kwargs)
        v_t = v_t.transpose(1,2)
        # Update the state using the Euler formula
        self.x_t = self.x_t + (t_next - t) * v_t


# class MyEulerSampler(Sampler):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def step(self, **model_kwargs):
#         # Extract the current time, next time point, and current state
#         t, t_next, x_t = self.t, self.t_next, self.x_t
#         # Compute the velocity field at the current state and time
#         v_t = self.rectified_flow.get_velocity(x_t=x_t, t=t, **model_kwargs)
#         # Update the state using the Euler formula
#         self.x_t = self.x_t + (t_next - t) * v_t     

#         # in self.x_t to stay at 0 so they don't drift during sampling
#         if "mask" in model_kwargs and model_kwargs["mask"] is not None:
#             mask = model_kwargs["mask"].unsqueeze(-1).to(self.x_t.device)
#             self.x_t = self.x_t * mask

class MyEulerSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, **model_kwargs):
            t, t_next, x_t = self.t, self.t_next, self.x_t
            cfg_scale = model_kwargs.get("cfg_scale", 1.0)
            
            if cfg_scale > 1.0:
                batch_size = x_t.shape[0]
                t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.float32)
                
                if hasattr(self.rectified_flow.velocity_field, 'module'):
                    self.rectified_flow.velocity_field = self.rectified_flow.velocity_field.module
                v_t = self.rectified_flow.velocity_field.forward_with_cfg(
                    x=x_t, t=t_tensor, **model_kwargs 
                )
            else:
                v_t = self.rectified_flow.get_velocity(x_t=x_t, t=t, **model_kwargs)
                
            # 1. Update the state using the Euler formula
            self.x_t = self.x_t + (t_next - t) * v_t     

            # REMOVED THE SPATIAL CLAMPING FROM HERE!
            # Just update the mask
            if "mask" in model_kwargs and model_kwargs["mask"] is not None:
                mask = model_kwargs["mask"].unsqueeze(-1).to(self.x_t.device)
                self.x_t = self.x_t * mask

@torch.no_grad()
def apply_voxel_collision_filter(pts_snapped, mask, grid_dims=(30, 30, 30)):
    """Merges overlapping points in the same voxel by summing linear energies."""
    B, N, _ = pts_snapped.shape
    Dx, Dy, Dz = grid_dims
    device = pts_snapped.device

    # Convert Log-Energy to Linear Energy for addition
    log_energy = pts_snapped[..., 3]
    linear_energy = torch.exp(log_energy) - 1e-6
    linear_energy = torch.clamp(linear_energy, min=0.0) 

    if mask is not None:
        linear_energy = linear_energy * mask.float()

    # Map 3D -> 1D index
    x = pts_snapped[..., 0].long().clamp(0, Dx - 1)
    y = pts_snapped[..., 1].long().clamp(0, Dy - 1)
    z = pts_snapped[..., 2].long().clamp(0, Dz - 1)
    flat_idx = x + (y * Dx) + (z * Dx * Dy)

    filtered_pts = torch.zeros_like(pts_snapped)
    new_mask = torch.zeros_like(mask) if mask is not None else torch.zeros(B, N, device=device)

    for b in range(B):
        grid_energies = torch.bincount(flat_idx[b], weights=linear_energy[b], minlength=Dx*Dy*Dz)
        
        active_mask = grid_energies > 0
        active_flat_idx = torch.nonzero(active_mask).squeeze(-1)
        active_energies = grid_energies[active_mask]
        
        # Reconstruct 3D Coordinates
        active_z = active_flat_idx // (Dx * Dy)
        rem = active_flat_idx % (Dx * Dy)
        active_y = rem // Dx
        active_x = rem % Dx
        
        # Back to Log-Energy
        active_logE = torch.log(active_energies + 1e-6)
        
        num_keep = min(len(active_flat_idx), N) 
        filtered_pts[b, :num_keep, 0] = active_x[:num_keep].float()
        filtered_pts[b, :num_keep, 1] = active_y[:num_keep].float()
        filtered_pts[b, :num_keep, 2] = active_z[:num_keep].float()
        filtered_pts[b, :num_keep, 3] = active_logE[:num_keep]
        
        if mask is not None:
            new_mask[b, :num_keep] = 1

    return filtered_pts, new_mask

@torch.no_grad()
def sample_with_voxel_snapping(euler_sampler, centers, **kwargs):
    # 1. Run ODE integration
    traj = euler_sampler.sample_loop(**kwargs)
    pts = traj.x_t 
    
    spatial = torch.clamp(pts[..., :3], 0.0, 29.0)
    energy = pts[..., 3:4]
    
    # 2. Snapping
    dists = torch.abs(spatial.unsqueeze(-1) - centers.view(1, 1, 1, -1))
    nearest_idx = torch.argmin(dists, dim=-1)
    snapped_spatial = centers[nearest_idx]
    
    pts_snapped = torch.cat([snapped_spatial, energy], dim=-1)
    
    mask = kwargs.get('mask', None)
    if mask is not None:
        pts_snapped = pts_snapped * mask.unsqueeze(-1)

    # 3. Collision Filter
    filtered_pts, new_mask = apply_voxel_collision_filter(pts_snapped, mask)
        
    return filtered_pts, new_mask

class MaskedPhysicalRectifiedFlowLoss(RectifiedFlowLossFunction):
    def __init__(self, centers, loss_type="mse", energy_weight=10.0, grid_weight=10.0, e_channel_weight=20.0):
        super().__init__(loss_type=loss_type)
        self.energy_weight = energy_weight
        self.grid_weight = grid_weight
        self.e_channel_weight = e_channel_weight # NEW: Boosts energy gradient
        self.centers = centers

    def __call__(self, v_t, dot_x_t, x_t, t, time_weights, mask=None):
        if self.loss_type != "mse":
            return super().__call__(v_t, dot_x_t, x_t, t, time_weights)
        if self.centers.device != v_t.device:
            self.centers = self.centers.to(v_t.device)
        #FIXME to be able to capture the mask here, I have edited the rectified_flow.get_loss to pass the mask as an argument. This is a bit hacky, but it allows us to keep the loss function logic here while still having access to the mask. We can consider refactoring this later for cleaner design.
        # to find path python -c "import rectified_flow; print(rectified_flow.__file__)"x
    
        # 1. SPLIT SQUARED ERROR
        # Spatial error (X, Y, Z)
        spatial_sq_diff = (v_t[..., :3] - dot_x_t[..., :3]) ** 2
        
        # Energy error (LogE) - Multiplied by the balancing weight
        energy_sq_diff = ((v_t[..., 3:4] - dot_x_t[..., 3:4]) ** 2) * self.e_channel_weight
        
        # Recombine
        sq_diff = torch.cat([spatial_sq_diff, energy_sq_diff], dim=-1)

        # 2. Standard Masking and Mean
        if mask is not None:
            masked_sq_diff = sq_diff * mask.unsqueeze(-1).float()
            points_per_instance = mask.sum(dim=1).clamp(min=1)
            per_instance_loss = masked_sq_diff.sum(dim=(1, 2)) / (points_per_instance * v_t.shape[-1])
        else:
            per_instance_loss = torch.mean(sq_diff, dim=list(range(1, v_t.dim())))

        loss_mse = torch.mean(time_weights * per_instance_loss)
        # 3. Physics & Grid Constraints
        if mask is not None:
            
            # Energy Sum Constraint
            pred_lin_energy = torch.exp(v_t[..., 3]) * mask.float()
            target_lin_energy = torch.exp(dot_x_t[..., 3]) * mask.float()
            
            #  Sum the actual physical energy deposits
            pred_sum = pred_lin_energy.sum(dim=1)
            target_sum = target_lin_energy.sum(dim=1)
            
            # Calculate Normalized MSE
            # Dividing by target_sum + epsilon makes this a 'fractional' error.
            # This keeps the value small (usually between 0 and 1).
            rel_diff = (pred_sum - target_sum) / (target_sum + 1e-6)
            loss_sumE = self.energy_weight * torch.mean(rel_diff ** 2)

            # 3D Grid Quantization Constraint
            pred_x1 = x_t + (1.0 - t.view(-1, 1, 1)) * v_t
            spatial_pred = pred_x1[..., :3]
            dists = torch.abs(spatial_pred.unsqueeze(-1) - self.centers.view(1, 1, 1, -1))
            min_dists = torch.min(dists, dim=-1).values
            quant_loss = self.grid_weight *(min_dists.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
        loss = loss_mse + loss_sumE +  quant_loss
        return loss, loss_mse, loss_sumE, quant_loss
            
@torch.no_grad()
def validate_reflow_straightness(model, rectified_flow, val_batch, scaler):
    model.eval()
    # val_batch: (x_1_raw, x_0_raw, mask, energy, y, gap)
    x_1_raw, x_0_raw, mask, energy, y, gap, _ = val_batch
    
    # 1. Prepare Inputs
    # IMPORTANT: We use the noise x_0 that the model was trained on!
    # No scaler on x_0!
    x_0 = x_0_raw.cuda() 
    
    # 2. Run deterministic Euler with very few steps (e.g., 5 or 10)
    # If paths are straight, 5 steps is plenty!
    num_steps = 5 
    dt = 1.0 / num_steps
    x_t = x_0.clone()
    
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((x_0.shape[0],), t, device=x_0.device)
        
        # Get velocity (No CFG for this diagnostic)
        v_t = model(x=x_t, t=t_batch, y=y.cuda(), gap=gap.cuda(), energy=energy.cuda(), mask=mask.cuda())
        
        x_t = x_t + v_t * dt
        x_t = x_t * mask.cuda().unsqueeze(-1) # Keep padding clean

    # 3. Convert back to raw physics units
    x_gen_raw = scaler.inverse_transform(x_t, mask=mask.cuda())
    
    # 4. Metric: MSE between Generated x_1 and Ground Truth x_1
    reflow_error = F.mse_loss(x_gen_raw, x_1_raw.cuda())
    return reflow_error, x_gen_raw

def enforce_energy_conservation(samples, target, energy_idx=-1, eps=1e-6):
    """
    Scales the energy channel of generated points so their sum matches the target energy.
    
    Args:
        samples (torch.Tensor): (Batch, Points, Channels).
        target_energy (torch.Tensor): (Batch,). The conditioning energy (incident or sum).
        energy_idx (int): Index of the energy channel (usually -1 or 3).
        eps (float): Small value to prevent division by zero.
        
    Returns:
        torch.Tensor: The corrected samples.
    """
    # 1. Clone to avoid in-place modification errors
    samples = samples.clone()
    target = target.clone()
    # 2. Extract generated energy channel
    # relu ensures we don't sum negative noise, though your model should predict > 0
    gen_energies = torch.nn.functional.relu(samples[..., energy_idx]) 
    target_energies = torch.nn.functional.relu(target[..., energy_idx]) 
    
    # 3. Calculate the sum per shower
    gen_sums = torch.sum(gen_energies, dim=1) # Shape: (Batch,)
    target_sums = torch.sum(target_energies, dim=1) # Shape: (Batch,)
    # 4. Calculate Scaling Factor (Target / Generated_Sum)
    # Ensure we don't divide by zero
    gen_sums = torch.clamp(gen_sums, min=eps)
    
    # target_energy might need to be reshaped to match gen_sums
    # if target_energy.dim() > 1:
    #     target_energy = target_energy.view(-1)
        
    scale_factors = target_sums / gen_sums # Shape: (Batch,)
    
    # 5. Apply Scaling
    # Reshape for broadcasting: (Batch, 1) to multiply across (Batch, Points)
    scale_factors = scale_factors.unsqueeze(1)
    
    # Apply ONLY to the energy channel
    samples[..., energy_idx] = samples[..., energy_idx] * scale_factors
    
    return samples

def plot_4d_reconstruction(original, reconstructed, savepath=".reconstructed.png", index=0):
    # original/reconstructed: [B, 4, N]
    orig = original[index].cpu().numpy() # [4, N]
    recon = reconstructed[index].cpu().numpy() # [4, N]
    assert orig.shape[0] == 4
    fig = plt.figure(figsize=(12, 6))
    
    # Plot Original
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(orig[0], orig[1], orig[2], c=orig[3], cmap='viridis', s=2)
    ax1.set_title("Original (Color=Energy)")
    
    # Plot Reconstruction
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(recon[0], recon[1], recon[2], c=recon[3], cmap='viridis', s=2)
    ax2.set_title("Reconstructed")
    
    plt.colorbar(sc2, ax=ax2, label='Energy Value')
    fig.savefig(f"{savepath}", dpi=300)
    plt.close(fig)

#NOTE Temporary, move to phys_metrics

import torch.nn as nn
import torch.multiprocessing as mp
import torch.utils.data
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.g4_pc_dataset import LazyPklDataset
#from datasets.idl_dataset import ECAL_Chunked_Dataset as IDLDataset
from datasets.idl_dataset import LazyIDLDataset as IDLDataset
from datasets.transforms import MinMaxNormalize, CentroidNormalize, Compose, NormalizePC4D
from rectified_flow.samplers.base_sampler import Sampler
from rectified_flow.samplers import EulerSampler
from rectified_flow.flow_components.loss_function import RectifiedFlowLossFunction
import numpy as np
import torch.nn.functional as F
import awkward as ak
import matplotlib.pyplot as plt
from phys_plotting import plot_paper_plots


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


def get_dataset(dataroot, npoints,category, name='shapenet'):
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
        #transform = NormalizePC4D()
        transform = None
        print(f"warning: using hardcoded E_MIN and E_MAX for energy normalization")
        dataset = IDLDataset(dataroot, transform=transform)#, max_seq_length=npoints, ordering='spatial', material_list=["G4_W", "G4_Ta", "G4_Pb"], inference_mode=False)
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


class MyEulerSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, **model_kwargs):
        # Extract the current time, next time point, and current state
        t, t_next, x_t = self.t, self.t_next, self.x_t
        # Compute the velocity field at the current state and time
        v_t = self.rectified_flow.get_velocity(x_t=x_t, t=t, **model_kwargs)
        # Update the state using the Euler formula
        self.x_t = self.x_t + (t_next - t) * v_t     

        # in self.x_t to stay at 0 so they don't drift during sampling
        if "mask" in model_kwargs and model_kwargs["mask"] is not None:
            mask = model_kwargs["mask"].unsqueeze(-1).to(self.x_t.device)
            self.x_t = self.x_t * mask


class DDIMSampler:
    def __init__(self, model, n_steps=1000, ddim_discretize="uniform", ddim_eta=0.0):
        self.model = model  # Your AdaLN DiT
        self.n_steps = n_steps # Total training steps (usually 1000)
        self.eta = ddim_eta # 0.0 for deterministic DDIM
        
        # Define the beta schedule (should match your training)
        betas = torch.linspace(0.0001, 0.02, n_steps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

    def get_sampling_timesteps(self, n_sample_steps):
        """Creates the subset of timesteps to jump through."""
        times = np.linspace(0, self.n_steps - 1, n_sample_steps, dtype=int)
        return list(reversed(times))

    @torch.no_grad()
    def sample(self, shape, e_init, n_sample_steps=50, cfg_scale=3.0):
        device = next(self.model.parameters()).device
        z = torch.randn(shape, device=device)
        timesteps = self.get_sampling_timesteps(n_sample_steps)
        
        # Pre-calculate alpha values for the subset
        for i, step in enumerate(timesteps):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            prev_step = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            # 1. Get CFG Noise Prediction
            # Conditioned
            eps_cond = self.model(z, t, e_init, mask_condition=torch.ones(1, device=device))
            # Unconditioned
            eps_uncond = self.model(z, t, e_init, mask_condition=torch.zeros(1, device=device))
            # Final Guided Noise
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            # 2. DDIM Math
            alpha_t = self.alphas_cumprod[step]
            alpha_prev = self.alphas_cumprod[prev_step]
            
            # Predict z_0
            pred_z0 = (z - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            
            # Direction pointing to z_t
            dir_zt = torch.sqrt(1 - alpha_prev) * eps
            
            # Update z
            z = torch.sqrt(alpha_prev) * pred_z0 + dir_zt
            
        return z

class MaskedPhysicalRectifiedFlowLoss(RectifiedFlowLossFunction):
    def __init__(self, loss_type: str = "mse", energy_weight: float = 0.1):
        super().__init__(loss_type=loss_type)
        self.energy_weight = energy_weight

    def __call__(
        self,
        v_t: torch.Tensor,
        dot_x_t: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        time_weights: torch.Tensor,
        mask: torch.Tensor = None,           # Added mask
        target_energy: torch.Tensor = None,   # Added for physics constraint
    ) -> torch.Tensor:
        """
        Calculates a masked MSE loss + Energy Conservation constraint.
        """
        if self.loss_type != "mse":
            return super().__call__(v_t, dot_x_t, x_t, t, time_weights)
    
        # Compute Element-wise Squared Error
        # Shape: (Batch, Max_Particles, Features)
        sq_diff = (v_t - dot_x_t) ** 2
        if mask is not None:
            # Apply Mask (B, P) -> (B, P, 1)
            # This zeroes out the error for padded points
            masked_sq_diff = sq_diff * mask.unsqueeze(-1).float()
            #  Sum over dimensions (Particles and Features)
            # per_instance_loss shape: (Batch,)
            # We divide by the number of active points in each instance for stability
            points_per_instance = mask.sum(dim=1).clamp(min=1)
            per_instance_loss = masked_sq_diff.sum(dim=(1, 2)) / (points_per_instance * v_t.shape[-1])
        else:
            # Fallback to standard mean if no mask provided
            per_instance_loss = torch.mean(sq_diff, dim=list(range(1, v_t.dim())))

        # Standard RF weighting
        loss = torch.mean(time_weights * per_instance_loss)

        #  Physics Constraint: Energy Conservation
        # Constrain the sum of velocities in the energy dimension (index 3)

        if target_energy is not None and mask is not None:
            # Velocity toward the target sum: sum(v_t_energy) should match sum(dot_x_t_energy)
            pred_energy_sum = (v_t[:, :, 3] * mask).sum(dim=1)
            target_energy_sum = (dot_x_t[:, :, 3] * mask).sum(dim=1)
            
            physics_loss = F.mse_loss(pred_energy_sum, target_energy_sum)
            loss = loss + (self.energy_weight * physics_loss)

        return loss
    
def masked_chamfer_4d(pc_pred, pc_gt, mask):
    """
    pc_pred: [B, 4, N] (Reconstructed)
    pc_gt:   [B, 4, N] (Ground Truth padded)
    mask:    [B, N]    (1 for real points, 0 for padded)
    """
    # Transpose to [B, N, 4] for distance calculation
    pc_pred = pc_pred.transpose(1, 2)
    pc_gt = pc_gt.transpose(1, 2)
    
    # Calculate pairwise 4D squared distance: [B, N, N]
    # dist[b, i, j] is the distance between pred_point_i and gt_point_j
    dist_matrix = torch.cdist(pc_pred, pc_gt, p=2)**2
    
    # Masking: We only care about distances to REAL ground truth points
    # For pred -> gt: ignore columns where mask is 0
    # For gt -> pred: ignore rows where mask is 0
    
    # 1. Prediction to Ground Truth (forward)
    # Set distances to padded GT points to infinity so they aren't picked as 'min'
    dist_to_gt = dist_matrix.masked_fill(mask.unsqueeze(1) == 0, float('inf'))
    min_dist_to_gt, _ = torch.min(dist_to_gt, dim=2) # [B, N]
    # Ignore the min_dist of padded predicted points
    loss_forward = (min_dist_to_gt * mask).sum(dim=1) / mask.sum(dim=1)

    # 2. Ground Truth to Prediction (backward)
    # We only care about the distance from REAL GT points to the reconstruction
    min_dist_from_gt, _ = torch.min(dist_matrix, dim=1) # [B, N]
    loss_backward = (min_dist_from_gt * mask).sum(dim=1) / mask.sum(dim=1)
    
    return (loss_forward + loss_backward).mean()

def vae_loss_function(x, recon_x, mu, logvar, e_init, mask):
    """
    recon_x: [B, 5, N] (x, y, z, E, hit_prob)
    x: [B, 4, N] (target x, y, z, E)
    mask: [B, N] (1 for real hits, 0 for padding)
    """
    #pred_hit_prob = recon_x[:, 4, :] # [B, N]
    # This forces hit_prob to 0 where the ground truth is padding
    #loss_hit = F.binary_cross_entropy(pred_hit_prob, mask.float())
    
    # Split coordinates and energy
    pred_xyz = recon_x[:, :3, :]
    target_xyz = x[:, :3, :]
    pred_e = recon_x[:, 3, :]
    target_e = x[:, 3, :]
    #e_init = e_init.squeeze(1)
    m = mask.unsqueeze(1) # [B, 1, N]
    # 1. Coordinate Loss: MSE + Chamfer for Volumetric Shape
    # Chamfer helps fix the "stringy" geometry by encouraging clusters
    loss_xyz = F.mse_loss(pred_xyz * m, target_xyz * m)
    loss_chamfer = masked_chamfer_4d(pred_xyz, target_xyz, mask)

    #  Energy Distribution Loss
    # Since sum(pred_e) == e_init, we treat this as a distribution problem.
    # Kullback-Leibler Divergence or simply MSE on the normalized energies.
    nparticles = x.shape[2]
    loss_energy = F.mse_loss(pred_e * m, target_e * m)
    pred_esum= pred_e.sum(dim=1)/nparticles
    target_esum = target_e.sum(dim=1)/nparticles
    #energy_weights = torch.softmax(energy_logits, dim=1)
    loss_sum_e = F.mse_loss(pred_esum, target_esum)
    #  KLD Loss (Standard VAE)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss with weighting (adjust these based on training behavior)
    #TODO here scales for each loss has been hard coded. Pass it from somewhere else
    total_loss = 0.1*loss_xyz + 0.1 * loss_chamfer+ loss_energy + loss_sum_e + 0.001 * kld_loss
    return total_loss, 0.1*loss_xyz, 0.1 * loss_chamfer, loss_energy, loss_sum_e, 0.01 * kld_loss

def masked_chamfer_distance(pc_a, pc_b, mask_a, mask_b):
    """
    pc_a, pc_b: [B, 4, N] (4th dim is Energy)
    mask_a, mask_b: [B, N] (1 for real, 0 for padded)
    """
    # Reshape for broadcasting: [B, N, 1, 4] and [B, 1, N, 4]
    pc_a = pc_a.transpose(1, 2).unsqueeze(2) 
    pc_b = pc_b.transpose(1, 2).unsqueeze(1)
    
    # Compute squared Euclidean distance in 4D space
    # dist shape: [B, N, N]
    dist = torch.sum((pc_a - pc_b) ** 2, dim=-1)
    
    # MASKING: We want to ignore distances involving padded points.
    # For the min() operation, set padded distances to infinity.
    mask_2d = mask_a.unsqueeze(2) * mask_b.unsqueeze(1) # [B, N, N]
    dist_masked = dist.masked_fill(mask_2d == 0, float('inf'))
    
    # Distance from A to B: min distance for each point in A to any point in B
    dist_a_to_b, _ = torch.min(dist_masked, dim=2) # [B, N]
    # Distance from B to A: min distance for each point in B to any point in A
    dist_b_to_a, _ = torch.min(dist_masked, dim=1) # [B, N]
    
    # Apply 1D mask to ignore the 'inf' values of padded points in the final sum
    dist_a_to_b = dist_a_to_b.masked_fill(mask_a == 0, 0.0)
    dist_b_to_a = dist_b_to_a.masked_fill(mask_b == 0, 0.0)
    
    # Average over real points only
    loss = (dist_a_to_b.sum(dim=1) / mask_a.sum(dim=1) + 
            dist_b_to_a.sum(dim=1) / mask_b.sum(dim=1))
    
    return loss.mean()



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

def make_phys_plots(real, gen, material_list=["G4_W"], savepath="./Phys_plots/"):
    #TODO read the gap_pid and pass it to the plotting function
    material = material_list[0]
    gen_dict = {
        "x": [],
        "y": [],
        "z": [],
        "energy": []
    }
    data_dict = {
        "x": [],
        "y": [],
        "z": [],
        "energy": [],
    }
    for i in range(gen.shape[0]):
        if real[i].shape[0]==4:
            x, y, z, e = real[i]
            xg, yg, zg, eg = gen[i]
        else:
            x, y, z, e = real[i].T
            xg, yg, zg, eg = gen[i].T


        gen_dict["x"].append(zg)
        gen_dict["z"].append(xg)
        gen_dict["y"].append(yg)
        gen_dict["energy"].append(eg)

        data_dict["x"].append(z)
        data_dict["z"].append(x)
        data_dict["y"].append(y)
        data_dict["energy"].append(e)
                
        ak_array_truth = ak.Array(data_dict)
        ak_array_gen = ak.Array(gen_dict)

    fig = plot_paper_plots(
            [ak_array_truth, ak_array_gen],
            labels=["Ground Truth", "Generated"],
            colors=["lightgrey", "cornflowerblue"], material=material
        )
        #fig.savefig(f"Plots/{filename}_{material}.pdf", dpi=300)
    fig.savefig(f"{savepath}/phys_metrics.png", dpi=300)
    plt.close(fig)

##### temporal utils #######
import torch
import matplotlib.pyplot as plt


def plot_histogram(data_tensor, bins=30, title="Histogram", save_path="histogram.png"):
    """
    Takes a tensor of size [npoints] and saves a histogram plot.
    """
    # Convert to numpy for matplotlib compatibility
    data_np = data_tensor.detach().cpu().numpy()
    
    # Create the plot
    plt.hist(data_np, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved to {save_path}")

# Example usage:
# n_points = 1000
# my_tensor = torch.randn(n_points)
# plot_histogram(my_tensor, title=f"Distribution of {n_points} points")
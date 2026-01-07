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
import numpy as np

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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True, collate_fn=collate_fn )

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
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
        #NOTE: If using a mask, force the padded points 
        # in self.x_t to stay at 0 so they don't drift during sampling
        
        if "mask" in model_kwargs and model_kwargs["mask"] is not None:
            mask = model_kwargs["mask"].unsqueeze(-1).to(self.x_t.device)
            self.x_t = self.x_t * mask


class MaskedFlowMatchingCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        #     return self.criterion(
        #     v_t=v_t,
        #     dot_x_t=dot_x_t,
        #     x_t=x_t,
        #     t=t,
        #     time_weights=time_weights,
        # )

    def forward(self, v_t, dot_x_t, mask, time_weights=None, **kwargs):
        # 1. Compute Raw Squared Error: (B, N, C)
        sq_error = (v_t - dot_x_t) ** 2
        
        # 2. Apply Mask: (B, N) -> (B, N, 1)
        # This ignores the 'velocity' of the padding
        masked_error = sq_error * mask.unsqueeze(-1).float()
        
        # 3. Apply Time Weights (if used)
        if time_weights is not None:
            # time_weights is usually (B,)
            masked_error = masked_error * time_weights.view(-1, 1, 1)
            
        # 4. Reduction
        if self.reduction == 'mean':
            # Normalize by the number of actual points in the entire batch
            return masked_error.sum() / mask.sum().clamp(min=1)
        else:
            return masked_error.sum()

import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.calopodit import DiT, DiTConfig
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.g4_pc_dataset import LazyPklDataset
#from datasets.idl_dataset import ECAL_Chunked_Dataset as IDLDataset
from datasets.idl_dataset import LazyIDLDataset as IDLDataset

from datasets.transforms import MinMaxNormalize, CentroidNormalize, Compose
#from rectified_flow.models.dit import DiT, DiTConfig

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


def train(gpu, opt, output_dir, noises_init):
    
    set_seed(opt)
    logger = setup_logging(output_dir)
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')
    if opt.enable_profiling:
        out_prof, = setup_output_subdirs(output_dir, 'profiling')
     
    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveIter =  int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)

    ''' data '''
    #NOTE passing here to read max_particles from the args. Look for a way to do it from dataset
    def pad_collate_fn(batch, max_particles=opt.npoints):
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

        padded_showers = []
        for shower in showers_list:
            num_particles = shower.shape[0]
            if num_particles > max_particles:
                # Truncate if the number of particles exceeds the max
                padded_shower = shower[:max_particles]
            else:
                # Pad with zeros if the number of particles is less than the max
                padding = torch.zeros(max_particles - num_particles, shower.shape[1], dtype=shower.dtype, device=shower.device)
                padded_shower = torch.cat([shower, padding], dim=0)
            padded_showers.append(padded_shower)

        # Stack all tensors to create the batch
        showers_batch = torch.stack(padded_showers, dim=0)
        energies_batch = torch.stack(energies_list, dim=0)
        pids_batch = torch.stack(pids_list, dim=0)
        gap_pids_batch = torch.stack(gap_pids_list, dim=0)

        return showers_batch, energies_batch, pids_batch, gap_pids_batch, idx

    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category, name = opt.dataname)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, test_dataset = None, collate_fn=pad_collate_fn)

    if should_diag:
        logger.info(opt)


    for i, data in enumerate(dataloader):
        if opt.dataname == 'g4' or opt.dataname == 'idl':
            x, int_energy, y, gap_pid, idx = data
            # x_pc = x[:,:,:3]
            # outf_syn = f"/global/homes/c/ccardona/PSF"
            # visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
            #                        x_pc, None, None,
            #                        None)
            if opt.model_name == "pvcnn2":
                x = x.transpose(1,2)
            #noises_batch = noises_init[list(idx)].transpose(1,2)
        elif opt.dataname == 'shapenet':
            x = data['train_points']
            if opt.model_name == "pvcnn2":      
                x = x.transpose(1,2)
            #noises_batch = noises_init[data['idx']].transpose(1,2)
        
        if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
            x = x.cuda(gpu)
            #noises_batch = noises_batch.cuda(gpu)
        elif opt.distribution_type == 'single':
            x = x.cuda()
            #noises_batch = noises_batch.cuda()
        
        visualize_pointcloud_batch(f'dataset_samples.png', x.transpose(1,2), None,
                                    None,
                                    None)
        breakpoint()
    dist.destroy_process_group()

def main():
    opt = parse_args()
    if 1:
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)

    ''' workaround '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category, name =opt.dataname)
    noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
    
    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        train(opt.gpu, opt, output_dir, noises_init)



def parse_args():

    parser = argparse.ArgumentParser()
    ''' Data '''
    #parser.add_argument('--dataroot', default='/data/ccardona/datasets/ShapeNetCore.v2.PC15k/')
    #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_e_liquidArgon_50/')
    parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/')
    parser.add_argument('--category', default='car')
    parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--bs', type=int, default=5, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--npoints',  type=int, default=2048)
    
    '''model'''
    parser.add_argument("--model_name", type=str, default="pvcnn2", help="Name of the velovity field model. Choose between ['pvcnn2', 'calopodit', 'graphcnn'].")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)
    
    '''Flow'''
    parser.add_argument("--interp", type=str, default="straight", help="Interpolation method for the rectified flow. Choose between ['straight', 'slerp', 'ddim'].")
    parser.add_argument("--source_distribution", type=str, default="normal", help="Distribution of the source samples. Choose between ['normal'].")
    parser.add_argument("--is_independent_coupling", type=bool, default=True,help="Whether training 1-Rectified Flow")
    parser.add_argument("--train_time_distribution", type=str, default="uniform", help="Distribution of the training time samples. Choose between ['uniform', 'lognormal', 'u_shaped'].")
    parser.add_argument("--train_time_weight", type=str, default="uniform", help="Weighting of the training time samples. Choose between ['uniform'].")
    parser.add_argument("--criterion", type=str, default="mse", help="Criterion for the rectified flow. Choose between ['mse', 'l1', 'lpips'].")
    parser.add_argument("--num_steps", type=int, default=100, help=(
            "Number of steps for generation. Used in training Reflow and/or evaluation"),)
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images.",)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='multi', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=8, help='unit: epoch')
    parser.add_argument('--diagIter', default=8, help='unit: epoch')
    parser.add_argument('--vizIter', default=8, help='unit: epoch')
    parser.add_argument('--print_freq', default=8, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    '''profiling'''
    parser.add_argument('--enable_profiling', action='store_true', help='Enable profiling during training.')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()

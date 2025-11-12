import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
from model.calopodit import DiT, DiTConfig
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.g4_pc_dataset import LazyPklDataset
from datasets.transforms import MinMaxNormalize, CentroidNormalize, Compose
#from rectified_flow.models.dit import DiT, DiTConfig
from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers import EulerSampler
from rectified_flow.samplers.base_sampler import Sampler

class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


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

def train(gpu, opt, output_dir, noises_init):
    
    set_seed(opt)
    logger = setup_logging(output_dir)
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

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


    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    #model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.model_name == 'pvcnn2':
        model = PVCNN2(num_classes=opt.nc, 
                    embed_dim=opt.embed_dim, 
                    use_att=opt.attention,
                    dropout=opt.dropout, 
                    extra_feature_channels=1) #<--- energy. #NOTE maybe we can add the remaining features as extra channels?? 
    elif opt.model_name == 'calopodit':
        #TODO clean up this config. Delet unused params and add new useful ones.
        DiT_config = DiTConfig(
            #Point transformer config
            k = 16,
            nblocks =  4,
            name= "calopodit",
            num_points = opt.npoints,
            energy_cond = False,#opt.energy_cond,
            in_features=opt.nc,
            transformer_features = 128, #512 = hidden_size in current implementation
            #DiT config
            num_classes = 0, #opt.num_classes,
            gap_classes = opt.gap_classes if hasattr(opt, 'gap_classes') else 0,
            out_channels=4, #opt.out_channels,
            hidden_size=128,
            depth=13,
            num_heads=8,
            mlp_ratio=4,
            use_long_skip=True,
            final_conv=False,
        )
        model = DiT(DiT_config)
    
    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        #model.multi_gpu_wrapper(_transform_)
        model = multi_gpu_wrapper(model, _transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        #model.multi_gpu_wrapper(_transform_)
        model = multi_gpu_wrapper(model, _transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    optimizer= optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        #optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    # def new_x_chain(x, num_chain):
    #     return torch.randn(num_chain, *x.shape[1:], device=x.device)
    #Rectified_Flow
    data_shape = (opt.npoints ,opt.nc)  # (N, 4) 4 for (x,y,z,energy)
    rectified_flow = RectifiedFlow(
        data_shape=data_shape,
        interp=opt.interp,
        source_distribution=opt.source_distribution,
        is_independent_coupling=opt.is_independent_coupling,
        train_time_distribution=opt.train_time_distribution,
        train_time_weight=opt.train_time_weight,
        criterion=opt.criterion,
        velocity_field=model,
        #device=accelerator.device,
        dtype=torch.float32,
    )

    for epoch in range(start_epoch, opt.niter):
        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)

        for i, data in enumerate(dataloader):  
            if opt.dataname == 'g4':
                x, energy, y, gap_pid, idx = data
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
            
            rectified_flow.device = x.device      
            
            x_0 = rectified_flow.sample_source_distribution(x.shape[0])
            if opt.model_name == "pvcnn2":
                x_0 = x_0.transpose(1,2)
            t = rectified_flow.sample_train_time(x.shape[0])
            t= t.squeeze()
            #FIXME for now we will not use conditionals
            loss = rectified_flow.get_loss(
                        x_0=x_0,
                        x_1=x,
                        t=t,
                    )
    
            # loss = rectified_flow.get_loss(
            #             x_0=x_0,
            #             x_1=x,
            #             y= y,
            #             gap= gap_pid,
            #             energy=energy,
            #             t=t,
            #         )

            #loss = model.get_loss_iter(x, noises_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            #netpNorm, netgradNorm = getGradNorm(model)
            #if opt.grad_clip is not None:
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()


            if i % opt.print_freq == 0 and should_diag:

                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             .format(
                        epoch, opt.niter, i, len(dataloader),loss.item()
                        ))

        
        
        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            logger.info('Generation: eval')

            model.eval()
            #x = x
            with torch.no_grad():
                if opt.model_name == "pvcnn2":
                    euler_sampler = MyEulerSamplerPVCNN(
                        rectified_flow=rectified_flow,
                        num_steps=opt.num_steps,
                        num_samples=opt.sample_batch_size,
                    )
                else:
                    euler_sampler = MyEulerSampler(
                            rectified_flow=rectified_flow,
                            num_steps=opt.num_steps,
                            num_samples=opt.sample_batch_size,
                        )
                        
                # Sample method
                #FIXME for now we will not use conditionals
                traj1 = euler_sampler.sample_loop(seed=233)
                # traj1 = euler_sampler.sample_loop(
                #     seed=233,
                #     y=y,
                #     gap= gap_pid,
                #     energy=energy,
                #     )
                pts= traj1.x_t
                trajectory = traj1.trajectories
                #Ehistogram(X,pts, y, gap_pid, energy, title=f"Ehist_calopodit_del")
                #plot_batch_3d(pts, y, gaps=gap_pid, energies= energy, title = "Model sampler_G4")

                #cf = chamfer_distance(X, pts, squared=True)
                #print(f"Chamfer Batch AVG Distance calopodit sampler: {cf.item():.6f}")
        

                # x_gen_eval = model.gen_samples(new_x_chain(x, 25).shape, x.device, clip_denoised=False)
                # x_gen_list = model.gen_sample_traj(new_x_chain(x, 1).shape, x.device, freq=40, clip_denoised=False)
                # x_gen_all = torch.cat(x_gen_list, dim=0)

                # gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                # gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                # logger.info('      [{:>3d}/{:>3d}]  '
                #              'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                #              'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                #     .format(
                #     epoch, opt.niter,
                #     *gen_eval_range, *gen_stats,
                # ))

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                                       trajectory, None, None,
                                       None)

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                                       pts, None,
                                       None,
                                       None)

            # visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x.transpose(1,2), None,
            #                            None,
            #                            None)

            logger.info('Generation: train')
            model.train()

        if (epoch + 1) % opt.saveIter == 0:

            if should_diag:


                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }

                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))


            if opt.distribution_type == 'multi':
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                model.load_state_dict(
                    torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])

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
    parser.add_argument('--dataroot', default='/home/carlos/Rnet_local/datasets/G4_individual_sims_pkl_test')
    parser.add_argument('--category', default='car')
    parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--bs', type=int, default=128, help='input batch size')
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
    parser.add_argument('--saveIter', default=100, help='unit: epoch')
    parser.add_argument('--diagIter', default=100, help='unit: epoch')
    parser.add_argument('--vizIter', default=100, help='unit: epoch')
    parser.add_argument('--print_freq', default=10, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()

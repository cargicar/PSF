import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast


import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from utils.train_utils import *
from model.pvcnn_generation import PVCNN2Base
from model.calopodit import DiT, DiTConfig
import torch.distributed as dist


#from rectified_flow.models.dit import DiT, DiTConfig
from rectified_flow.rectified_flow import RectifiedFlow
#from rectified_flow.flow_components.loss_function import RectifiedFlowLossFunction
from contextlib import contextmanager
import torch.profiler
from functools import partial


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



@contextmanager
def profile(enable_profiling, record_shapes=True, tensor_board=True, output_dir="profiling"):
    activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
    if enable_profiling:
        tb_exec_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if tensor_board:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=record_shapes,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/trace-{tb_exec_id}"),
                profile_memory=True,
                with_stack=True
                ) as prof:
                yield prof
        else:
            with profile(activities=activities, record_shapes=record_shapes) as prof:
                yield prof
    else:
        yield None

def profiler_table_output(prof, output_filename="profiling/cuda_memory_profile.txt"):
    profiler_table_output = prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    )

    with open(output_filename, "w") as f:
        f.write(profiler_table_output)

    print(f"Profiler table saved to {output_filename}")

def gather_all_gpu_tensors(local_tensor):
    # Ensure tensor is contiguous before gathering (crucial after transpose operations)
    print(f"gathering tensors from gpus to save samples...")
    local_tensor = local_tensor.contiguous()
    
    # Create a list to hold the gathered tensors (one for each GPU)
    world_size = dist.get_world_size()
    gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    
    # Collect data from all ranks
    dist.all_gather(gathered_list, local_tensor)
    print(f"tensors gathered.")
    # Concatenate along the batch dimension (dim 0) to get size 256
    return torch.cat(gathered_list, dim=0)

def train(gpu, opt, output_dir, noises_init):
    debug = False
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

    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category, name = opt.dataname)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, test_dataset = None, collate_fn=partial(pad_collate_fn, max_particles= train_dataset.max_particles))


    '''
    create networks
    '''
    #betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
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
            energy_cond = True,#opt.energy_cond,
            in_features=opt.nc,
            transformer_features = 128, #512 = hidden_size in current implementation
            #DiT config
            num_classes = opt.num_classes if hasattr(opt, 'num_classes') else 0,
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

    scaler = GradScaler(enabled=True)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        #optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scaler_state' in ckpt: 
            scaler.load_state_dict(ckpt['scaler_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    # def new_x_chain(x, num_chain):
    #     return torch.randn(num_chain, *x.shape[1:], device=x.device)
    #Rectified_Flow
    #rf_criterion = RectifiedFlowLossFunction(loss_type = "mse")
    rf_criterion = MaskedPhysicalRectifiedFlowLoss(loss_type= "mse", energy_weight= 0.1)
    
    data_shape = (train_dataset.max_particles, opt.nc)  # (N, 4) 4 for (x,y,z,energy)
    rectified_flow = RectifiedFlow(
        data_shape=data_shape,
        interp=opt.interp,
        source_distribution=opt.source_distribution,
        is_independent_coupling=opt.is_independent_coupling,
        train_time_distribution=opt.train_time_distribution,
        train_time_weight=opt.train_time_weight,
        criterion=rf_criterion, #opt.criterion,
        velocity_field=model,
        #device=accelerator.device,
        dtype=torch.float32,
    )
    ##################################################################################
    ''' training '''
    ##################################################################################
    profiling = opt.enable_profiling
    out_prof = None
    with profile(profiling, output_dir=out_prof) as prof:
        with torch.profiler.record_function("train_trace"):   
            for epoch in range(start_epoch, opt.niter):
                if opt.distribution_type == 'multi':
                    train_sampler.set_epoch(epoch)
                lr_scheduler.step(epoch)
                
                for i, data in enumerate(dataloader):
                    if opt.dataname == 'g4' or opt.dataname == 'idl':
                        x, mask, int_energy, y, gap_pid, idx = data
                        
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
                        x = x.cuda(gpu,  non_blocking=True)
                        mask = mask.cuda(gpu,  non_blocking=True)
                        y = y.cuda(gpu,  non_blocking=True)
                        gap_pid = gap_pid.cuda(gpu,  non_blocking=True)
                        int_energy = int_energy.cuda(gpu,  non_blocking=True)
                        #noises_batch = noises_batch.cuda(gpu)
                    elif opt.distribution_type == 'single':
                        x = x.cuda()
                        mask = mask.cuda()
                        y = y.cuda()
                        gap_pid = gap_pid.cuda()
                        int_energy = int_energy.cuda()
                        #noises_batch = noises_batch.cuda()
                    
                    rectified_flow.device = x.device      
                    optimizer.zero_grad()
                    with autocast(enabled=True):
                        x_0 = rectified_flow.sample_source_distribution(x.shape[0])
                        if opt.model_name == "pvcnn2":
                            x_0 = x_0.transpose(1,2)
                        t = rectified_flow.sample_train_time(x.shape[0])
                        t= t.squeeze()
                        #NOTE to pass the mask to the loss function, we have edited rectified_flow.get_loss.criterion(mask=kwargs.get(mask))
                        loss = rectified_flow.get_loss(
                                    x_0=x_0,
                                    x_1=x,
                                    y= y,
                                    gap= gap_pid,
                                    energy=int_energy,
                                    t=t,
                                    mask = mask,
                                )
                    scaler.scale.loss.backward()
                    if hasattr(opt, 'grad_clip') and opt.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    #netpNorm, netgradNorm = getGradNorm(model)
                    #if opt.grad_clip is not None:
                    #    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

                    scaler.step(optimizer)
                    scaler.update()
                    if prof is not None:
                        prof.step()

                    if i % opt.print_freq == 0 and should_diag:

                        logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                                    .format(
                                epoch, opt.niter, i, len(dataloader),loss.item()
                                ))
                    #TODO temporary. Instead of eval, save the generation and tested with physics metrics outside this script
                    
                if (epoch + 1) % opt.vizIter == 0 and should_diag:
                    logger.info('Generation: eval')

                    model.eval()
                    #x = x
                    #TODO CFG has to be done here
                    num_samples=opt.bs
                    num_steps = opt.num_steps
                    with torch.no_grad():
                        if opt.model_name == "pvcnn2":
                            euler_sampler = MyEulerSamplerPVCNN(
                                rectified_flow=rectified_flow,
                            )
                        else:
                            euler_sampler = MyEulerSampler(
                                    rectified_flow=rectified_flow,
                                )
                                
                        # Sample method
                        #FIXME we should be using a validatioon small dataset instead
                        with autocast(enabled=True):
                            traj1 = euler_sampler.sample_loop(
                                seed=233,
                                y=y,
                                gap= gap_pid,
                                energy=int_energy,
                                mask=mask,
                                num_samples=num_samples,
                                num_steps=num_steps,
                                )
                            pts= traj1.x_t
                            trajectory = traj1.trajectories
                        if opt.distribution_type == 'multi':
                            full_x = gather_all_gpu_tensors(x)
                            full_pts = gather_all_gpu_tensors(pts)
                            full_mask = gather_all_gpu_tensors(mask)
                        else:
                            # Fallback for single GPU training
                            full_x, full_pts, full_mask = x, pts, mask
                        if gpu ==0:
                            torch.save([full_x, full_pts, full_mask], f'{opt.pthsave}calopodit_train_Jan_17_epoch_{epoch}_m.pth')  
                            print(f"Samples fir testing save to {opt.pthsave}")
                        
                        with torch.no_grad():
                            plot_4d_reconstruction(x.transpose(1,2), pts.transpose(1,2), savepath=f"{outf_syn}/reconstruction_ep_{epoch}.png", index=0)
                    # if debug:
                    #     visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                    #                             trajectory, None, None,
                    #                             None)

                    #     visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                    #                             pts, None,
                    #                             None,
                    #                             None)

                    #     visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x, None,
                    #                                 None,
                    #                                 None)
                    #     #make_phys_plots(x, pts, savepath = outf_syn)
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
    if gpu==0:
        prof.export_memory_timeline(f"{out_prof}/memory_timeline.raw.json.gz", device=f"cuda:{gpu}")
        prof.export_memory_timeline(f"{out_prof}/memory_timeline.html", device=f"cuda:{gpu}")
    profiler_table_output(prof, output_filename=f"{out_prof}/cuda_memory_profile_rank{opt.rank}.txt")
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
    #parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/')
    parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/')
    parser.add_argument('--category', default='all', help='category of dataset')
    parser.add_argument('--pthsave', default='/pscratch/sd/c/ccardona/datasets/pth/')
    #parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--dataname',  default='idl', help='dataset name: shapenet | g4')
    parser.add_argument('--bs', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--npoints',  type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=0, help=("Number of primary particles used in simulated data"),)
    parser.add_argument("--gap_classes", type=int, default=0, help=("Number of calorimeter materials used in simulated data"),)
    
    '''model'''
    parser.add_argument("--model_name", type=str, default="calopodit", help="Name of the velovity field model. Choose between ['pvcnn2', 'calopodit', 'graphcnn'].")
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
    parser.add_argument("--num_steps", type=int, default=1000, help=(
            "Number of steps for generation. Used in training Reflow and/or evaluation"),)
    parser.add_argument("--sample_batch_size", type=int, default=100, help="Batch size (per device) for sampling images.",)

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
    parser.add_argument('--grad_clip', type=float, default=1.0, help='weight decay for EBM')
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
    parser.add_argument('--saveIter', type=int, default=8, help='unit: epoch')
    parser.add_argument('--diagIter', type=int, default=8, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=8, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=8, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    '''profiling'''
    parser.add_argument('--enable_profiling', action='store_true', help='Enable profiling during training.')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()

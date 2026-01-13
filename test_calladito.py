import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data


import argparse
from torch.distributions import Normal
from collections import OrderedDict

from utils.file_utils import *
from utils.visualize import *
from utils.train_utils import *

from models.calladito import DiTConfig, LatentDiT
import torch.distributed as dist


#from rectified_flow.models.dit import DiT, DiTConfig
from rectified_flow.rectified_flow import RectifiedFlow
#from rectified_flow.flow_components.loss_function import RectifiedFlowLossFunction
from contextlib import contextmanager
import torch.profiler
from functools import partial
from models.PCVAE import PointCloud4DVAE

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


def test(gpu, opt, output_dir, noises_init):
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
    if opt.model_name == 'calladito':
        #TODO clean up this config. Delet unused params and add new useful ones.
        DiT_config = DiTConfig(
            name= "calladito",
            input_dim= 512,
            hidden_size= 1024,
            depth= 13,
            num_heads= 16,
            num_particle_classes= 0,# 2,
            gap_classes= 0,#4,
            in_features= 4, #4 channels: x,y,z,E 
        )
        model = LatentDiT(DiT_config)
        latent_dim=DiT_config.input_dim
        pcvae = PointCloud4DVAE(latent_dim=latent_dim, max_points=train_dataset.max_particles)
        checkpoint = torch.load(opt.pcvae_model)
        for param in pcvae.parameters():
            param.requires_grad = False
        try:
            pcvae.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state'].items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' (7 characters)
                new_state_dict[name] = v
            pcvae.load_state_dict(new_state_dict)
        #pcvae.to(device)
    #  Setup DataLoader (No shuffling needed for extraction)

    else:
        raise NotImplementedError(f"Model {opt.model_name} not implemented yet.")
    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)#TODO check if find_unused_parameters is needed

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

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        #optimizer.load_state_dict(ckpt['optimizer_state'])

    # def new_x_chain(x, num_chain):
    #     return torch.randn(num_chain, *x.shape[1:], device=x.device)
    #Rectified_Flow
    #rf_criterion = RectifiedFlowLossFunction(loss_type = "mse")
    #rf_criterion = MaskedPhysicalRectifiedFlowLoss(loss_type= "mse", energy_weight= 0.1)
    rf_criterion = "mse"
    
    #data_shape = (train_dataset.max_particles, opt.nc)  # (N, 4) 4 for (x,y,z,energy)
    data_shape = (latent_dim,) #(B, latent_dim=512)
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
    ''' sample generation '''
    ##################################################################################
    profiling = opt.enable_profiling
    out_prof = None
    xs = []
    recons =[]
    with profile(profiling, output_dir=out_prof) as prof:
        with torch.profiler.record_function("train_trace"):   
            for i, data in enumerate(dataloader):
                if i > 21:
                    break
                if opt.dataname == 'g4' or opt.dataname == 'idl':
                    x, mask, e_init, y, gap_pid, idx = data
                    x = x.transpose(1,2)
                if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                    x = x.cuda(gpu,  non_blocking=True)
                    mask = mask.cuda(gpu,  non_blocking=True)
                    y = y.cuda(gpu,  non_blocking=True)
                    gap_pid = gap_pid.cuda(gpu,  non_blocking=True)
                    e_init = e_init.cuda(gpu,  non_blocking=True)
                    pcvae = pcvae.cuda(gpu)
                    #noises_batch = noises_batch.cuda(gpu)
                elif opt.distribution_type == 'single':
                    x = x.cuda()
                    mask = mask.cuda()
                    y = y.cuda()
                    gap_pid = gap_pid.cuda()
                    e_init = e_init.cuda()
                    pcvae = pcvae.cuda()
                    #noises_batch = noises_batch.cuda()
                
                rectified_flow.device = x.device      
                
                if prof is not None:
                    prof.step()
            
                logger.info('Generation: eval')

                model.eval()
                #x = x
                #TODO CFG has to be done here
                num_samples=opt.bs
                num_steps = opt.num_steps
                with torch.no_grad():
                    euler_sampler = MyEulerSampler(
                            rectified_flow=rectified_flow,
                        )
                    # Sample method
                    #FIXME we should be using a validatioon small dataset instead
                    #Sampling with RF API
                    #CFG
                    cfg_mask = (torch.rand(e_init.shape[0], device=x.device) > 0.1).float()
                    cfg_scale = 7.0
                    traj_cond = euler_sampler.sample_loop(
                        seed=233,
                        #y=y,
                        #gap= gap_pid,
                        e_init=e_init,
                        mask_condition=cfg_mask,
                        num_samples=num_samples,
                        num_steps=num_steps,
                        )
                    traj_uncond = euler_sampler.sample_loop(
                        seed=233,
                        #y=y,
                        #gap= gap_pid,
                        e_init=e_init,
                        mask_condition=torch.zeros(1),
                        num_samples=num_samples,
                        num_steps=num_steps,
                        )
                    #pts= traj_uncond.x_t 
                    pts= traj_uncond.x_t + cfg_scale * (traj_cond.x_t - traj_uncond.x_t) 
                    recon = pcvae.decoder(pts, e_init, x.shape[2])
                    xs.append(x)
                    recons.append(recon)
                    print(f"Latents generated")

                    print(f"Point clouds reconstructed batch {i}")
                    
                if debug:
                    
                    # visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                    #                         trajectory, None, None,
                    #                         None)

                    # visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                    #                         pts, None,
                    #                         None,
                    #                         None)

                    # visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x, None,
                    #                             None,
                    #                             None)
                    outf_syn = f"output/test_calladito/"
                    make_phys_plots(x, recon, savepath = outf_syn)
                    print(f"phys metrics plotted and saved to {outf_syn}")    
            xs = torch.cat(xs, 0)
            recons = torch.cat(recons, 0)
            torch.save([xs, recons], f'{opt.pthsave}_calladito_gen_Jan_13.pth')               
    dist.destroy_process_group()

# @torch.no_grad()
# def sample_rectified_flow(vae, dit, steps=50, num_points=2048, device="cuda"):
#     vae.eval()
#     dit.eval()
    
#     # 1. Start with Gaussian Noise
#     zt = torch.randn(1, 512).to(device)
#     dt = 1.0 / steps
    
#     # 2. Euler Integration (Solving the ODE)
#     for i in range(steps):
#         t = torch.full((1,), i / steps, device=device)
        
#         # Get velocity at current position and time
#         vt = dit(zt, t)
        
#         # Step forward along the straight line
#         zt = zt + vt * dt
        
#     # 3. Final Latent to Point Cloud
#     # zt is now roughly z1 (the data distribution)
#     generated_pc = vae.decoder(zt, num_points=num_points)
#     return generated_pc
#########################################

def mse_norm(pred, target):
    return torch.mean((pred - target) ** 2)

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
        mp.spawn(test, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        test(opt.gpu, opt, output_dir, noises_init)



def parse_args():

    parser = argparse.ArgumentParser()
    ''' Data '''
    #parser.add_argument('--dataroot', default='/data/ccardona/datasets/ShapeNetCore.v2.PC15k/')
    #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_e_liquidArgon_50/')
    #parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/')
    parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/')
    parser.add_argument('--pthsave', default='/pscratch/sd/c/ccardona/datasets/pth/')
    parser.add_argument('--category', default='all', help='category of dataset')
    #parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--dataname',  default='idl', help='dataset name: shapenet | g4')
    parser.add_argument('--bs', type=int, default=256, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--npoints',  type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=0, help=("Number of primary particles used in simulated data"),)
    parser.add_argument("--gap_classes", type=int, default=0, help=("Number of calorimeter materials used in simulated data"),)
    
    '''model'''
    parser.add_argument("--model_name", type=str, default="calladito", help="Name of the velovity field model. Choose between ['pvcnn2', 'calopodit', 'graphcnn'].")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    parser.add_argument('--model', default='', help="path to model (to continue training)")
    parser.add_argument('--pcvae_model', default= 'output/train_pcvae/2026-01-10-pcvae-transf-dec/epoch_69.pth', help="path to latent pretrained model")

    
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
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

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
    parser.add_argument('--saveIter', type=int, default=16, help='unit: epoch')
    parser.add_argument('--diagIter', type=int, default=16, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=16, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=8, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    '''profiling'''
    parser.add_argument('--enable_profiling', action='store_true', help='Enable profiling during training.')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()

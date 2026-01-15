import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Subset

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from utils.train_utils import *

"""import models"""
#from model.pvcnn_generation import PVCNN2Base
#from model.calopodit import DiT, DiTConfig
from models.PCVAE import PointCloud4DVAE, PointCloudVAELoss, KLAnnealer
import torch.distributed as dist


#from rectified_flow.models.dit import DiT, DiTConfig
#from rectified_flow.rectified_flow import RectifiedFlow
#from rectified_flow.flow_components.loss_function import RectifiedFlowLossFunction
from contextlib import contextmanager
import torch.profiler
from functools import partial


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
    

@torch.no_grad()
def validate(gpu, opt, model, val_loader, save_samples = False):
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    
    # To store samples for visualization
    sample_pcs = []
    sample_recons = []
    for i, data in enumerate(val_loader):
        if opt.dataname == 'g4' or opt.dataname == 'idl':
            x, mask, int_energy, y, gap_pid, idx = data
        x = x.transpose(1,2)
        
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
        pcs_recon, mu, logvar = model(x, mask)
        
        # Calculate losses using the masked Chamfer function from before
        #recon_loss = masked_chamfer_distance(x, pcs_recon, mask, mask)
        recon_loss = masked_chamfer_4d(x, pcs_recon, mask, mask)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        # Save the first batch for visual inspection
        if save_samples and i == 0:
            sample_pcs.append(x[:4].cpu())       # Original
            sample_recons.append(pcs_recon[:4].cpu()) # Reconstructed

            return sample_pcs, sample_recons

    avg_recon = total_recon_loss / len(val_loader)
    avg_kl = total_kl_loss / len(val_loader)
    
    print(f"Validation - Recon (Chamfer): {avg_recon:.6f}, KL: {avg_kl:.6f}")
    
    return None, None

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

    ## TODO create validation dataset. Using a portion of the training data for now
    subset_size = 10
    # indices = list(range(subset_size)) # First 1000
    indices = np.random.choice(len(train_dataset), subset_size, replace=False) # Random 
    train_subset = Subset(train_dataset, indices)
    # 4. Pass the SUBSET to your existing dataloader function
    val_loader, _, train_sampler, _ = get_dataloader(opt, train_subset,test_dataset=None, 
        collate_fn=partial(pad_collate_fn, max_particles=train_dataset.max_particles))

    '''
    create networks
    '''
    #betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    #model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    if opt.model_name == 'pc4dvae':
        model = PointCloud4DVAE(latent_dim=opt.latent_dim, max_points=train_dataset.max_particles)
    else:
        print(f"Model name {opt.model_name} not implemented.")
    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)#, find_unused_parameters=True)#TODO find_unused_parameters set True because the query_embed in the transformer encoder set num_point to 2048 and if the pc has less points, then the embedding get unsued parameter. Remeber to set Flase if dynamic queries with pos encoding are used later

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
    criterion = PointCloudVAELoss(
            lambda_e_sum=0.00001, 
            lambda_hit=0.000001,
            lambda_emd=0.5
    )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)
    # This will reach 0.005 by epoch 30
    annealer = KLAnnealer(target_kl=0.005, start_epoch=0, end_epoch=30)


    if opt.model_path != '':
        ckpt = torch.load(opt.model_path)
        model.load_state_dict(ckpt['model_state'])
        #optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model_path != '':
        start_epoch = torch.load(opt.model_path)['epoch'] + 1
    else:
        start_epoch = 0

    ##################################################################################
    ''' training '''
    ##################################################################################
    profiling = opt.enable_profiling
    out_prof = None
    beta = opt.kl_beta

    with profile(profiling, output_dir=out_prof) as prof:
        with torch.profiler.record_function("train_trace"):   
            for epoch in range(start_epoch, opt.niter):
                if opt.distribution_type == 'multi':
                    train_sampler.set_epoch(epoch)
                lr_scheduler.step(epoch)
                xs = []
                recons = []
                masks = []
                current_kl_weight = annealer.get_weight(epoch)
                for i, data in enumerate(dataloader):
                    if opt.dataname == 'g4' or opt.dataname == 'idl':
                        x, mask, e_init, y, gap_pid, idx = data
                        x = x.transpose(1,2)
                    elif opt.dataname == 'shapenet':
                        x = data['train_points']
                        mask = None
                        #noises_batch = noises_init[data['idx']].transpose(1,2)
                    
                    if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                        x = x.cuda(gpu,  non_blocking=True)
                        mask = mask.cuda(gpu,  non_blocking=True)
                        y = y.cuda(gpu,  non_blocking=True)
                        gap_pid = gap_pid.cuda(gpu,  non_blocking=True)
                        e_init = e_init.cuda(gpu,  non_blocking=True)
                        criterion = criterion.cuda(gpu)
                        #noises_batch = noises_batch.cuda(gpu)
                    elif opt.distribution_type == 'single':
                        x = x.cuda()
                        mask = mask.cuda()
                        y = y.cuda()
                        gap_pid = gap_pid.cuda()
                        e_init = e_init.cuda()
                        criterion = criterion.cuda()
                        #noises_batch = noises_batch.cuda()
                    pcs_recon, mu, logvar = model(x, mask, e_init)
                    #NOTE to pass the mask to the loss function, we have edited rectified_flow.get_loss.criterion(mask=kwargs.get(mask))
                    #loss = masked_chamfer_distance(x, pcs_recon, mask, mask)
                    #loss, loss_xyz, loss_chamfer, loss_energy, loss_sum_e, kld_loss = vae_loss_function(x, pcs_recon, mu, logvar, init_energy, mask)
                    loss_dict = criterion(
                            preds=pcs_recon, 
                            target=x, 
                            target_mask=mask, 
                            mu=mu, 
                            logvar=logvar, 
                            e_init=e_init,
                            kl_weight = current_kl_weight,
                        )
                    loss = loss_dict['loss']
                                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if prof is not None:
                        prof.step()

                    if i % opt.print_freq == 0 and should_diag:

                        logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},  loss_emd {:>10.4f}, loss_energy {:>10.4f}, loss_global_e {:>10.4f},  kld_loss{:>10.4f}, hit_count{:>10.4f} '
                                    .format(
                                epoch, opt.niter, i, len(dataloader),loss.item(),  loss_dict["emd"], loss_dict["local_E"], loss_dict["global_E"], loss_dict["kld"], loss_dict["hit_count"]
                                ))
                    #TODO temporary. Instead of eval, save the generation and tested with physics metrics outside this script
                    if i < 21:
                        xs.append(x)
                        recons.append(pcs_recon)
                        masks.append(mask)
                xs = torch.cat(xs, 0)
                recons = torch.cat(recons, 0)
                masks = torch.cat(masks, 0)
                
                torch.save([xs, recons, masks], f'{opt.pthsave}_pcvae_train_Jan_14_epoch_{epoch}_m.pth')  
                print(f"Samples fir testing save to {opt.pthsave}")
                
                if (epoch + 1) % opt.vizIter == 0 and should_diag:
                    logger.info('eval')
                    #TODO add validation loader
                    #val_loader = dataloader
                    #pcs, gen = validate(gpu, opt, model, val_loader)
                    #model.eval()
                    with torch.no_grad():
                        plot_4d_reconstruction(x, pcs_recon, savepath=f"{outf_syn}/reconstruction_ep_{epoch}.png", index=0)
                    if debug and x is not None and pcs_recon is not None:
                        visualize_pointcloud_batch('%s/epoch_%03d_samples_gen.png' % (outf_syn, epoch),
                                                gen, None,
                                                None,
                                                None)

                        visualize_pointcloud_batch('%s/epoch_%03d_samples_sim.png' % (outf_syn, epoch), pcs, None,
                                                    None,
                                                    None)
                        #make_phys_plots(pcs, gen, savepath = outf_syn)
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
    parser.add_argument('--pthsave', default='/pscratch/sd/c/ccardona/datasets/pth/')
    parser.add_argument('--category', default='all', help='category of dataset')
    #parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--dataname',  default='idl', help='dataset name: shapenet | g4')
    parser.add_argument('--bs', type=int, default=200, help='input batch size') #lower bs if using repulsion loss
    parser.add_argument('--workers', type=int, default=32, help='workers')
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--npoints',  type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=0, help=("Number of primary particles used in simulated data"),)
    parser.add_argument("--gap_classes", type=int, default=0, help=("Number of calorimeter materials used in simulated data"),)
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train for')
    '''model'''
    parser.add_argument("--model_name", type=str, default="pc4dvae", help="Name of the velovity field model. Choose between ['pvcnn2', 'calopodit', 'graphcnn'].")
    parser.add_argument('--kl_beta', default=0.0005)
    parser.add_argument('--schedule_type', default='linear')
    '''encoder decoder'''
    parser.add_argument('--latent_dim',  type=int, default=512)
    
    #params
    
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model_path', default='', help="path to model (to continue training)")


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
    parser.add_argument('--print_freq', type=int, default=32, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    '''profiling'''
    parser.add_argument('--enable_profiling', action='store_true', help='Enable profiling during training.')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()

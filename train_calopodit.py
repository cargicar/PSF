import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
#from torch.cuda.amp import GradScaler, autocast


import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from utils.train_utils import *
from datasets.transforms import PointCloudPhysicsScaler #PointCloudStandardScaler
from model.calopodit import DiT, DiTConfig
import torch.distributed as dist


#from rectified_flow.models.dit import DiT, DiTConfig
from rectified_flow.rectified_flow import RectifiedFlow
#from rectified_flow.flow_components.loss_function import RectifiedFlowLossFunction
from contextlib import contextmanager
import torch.profiler
from functools import partial

#torch.autograd.set_detect_anomaly(True)
#TODO list to improve fidelity:
#### Add train_utils.enforce_energy_conservation to the loss

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

def load_checkpoint_with_gap_surgery(model, checkpoint_path, new_gap_classes, new_particle_classes=0): #NOTE new_gap_clasess = total number
    # Load the existing state dict
    ckpt = torch.load(checkpoint_path, map_location="cpu") # map_location avoids GPU memory spikes
    state_dict = ckpt['model_state']
    
    # --- DDP Handling Steps ---
    # 1. Check if the IN-MEMORY model is wrapped in DDP (has "module." prefix)
    is_model_ddp = hasattr(model, "module")
    
    # 2. Check if the CHECKPOINT has "module." prefix
    # We look at the first key to guess
    first_key = next(iter(state_dict.keys()))
    is_ckpt_ddp = first_key.startswith("module.")

    # 3. Create a helper to map logical names to actual keys in this specific checkpoint
    def get_ckpt_key(logical_name):
        if is_ckpt_ddp:
            return f"module.{logical_name}"
        return logical_name
    
    # Define the logical name of the layer we want to edit
    target_layer_name = 'gap_embedder.embedding_table.weight'
    actual_key = get_ckpt_key(target_layer_name)

    # --- Surgery Logic ---
    if actual_key in state_dict:
        old_weight = state_dict[actual_key] # Shape: (Old_N + 1, Dim)
        current_rows, dim = old_weight.shape
        target_rows = new_gap_classes + 1
        if target_rows > current_rows:
            print(f"Expanding gap embedder from {current_rows} to {target_rows} rows...")
            
            # Keep the existing classes (indices 0 to N-1)
            # Keep the NULL token (usually the last index)
            existing_classes = old_weight[:-1] 
            null_token = old_weight[-1:]
            
            # Initialize new class embeddings
            mean_emb = existing_classes.mean(dim=0, keepdim=True)
            num_new = target_rows - current_rows
            new_embeddings = mean_emb.repeat(num_new, 1) + torch.randn(num_new, dim) * 0.02
            
            # Concatenate: [Existing, New, Null]
            new_weight = torch.cat([existing_classes, new_embeddings, null_token], dim=0)
            
            # Update state dict using the actual key found in the file
            state_dict[actual_key] = new_weight
    else:
        print(f"Warning: Key '{actual_key}' not found in checkpoint. Skipping surgery.")

    # --- Loading Logic ---
    # We must ensure the state_dict keys match the model's expectation.
    # If the checkpoint has 'module.' but model doesn't (or vice versa), we must fix it.
    
    final_state_dict = {}
    for k, v in state_dict.items():
        # Normalize key: strip 'module.' if it exists
        clean_key = k[7:] if k.startswith("module.") else k
        
        # Add 'module.' back ONLY if the target model needs it
        if is_model_ddp:
            final_key = f"module.{clean_key}"
        else:
            final_key = clean_key
            
        final_state_dict[final_key] = v

    # Load into model
    # strict=True ensures we didn't mess up the keys
    model.load_state_dict(final_state_dict, strict=True)
    
    return model

def fine_tuning_modulate(model, lr=0.001, decay = 0.01):
    # Call after model = DiT(config)... and loading weights...
    # Returns Optimizer!
    
    # --- DDP ADAPTATION ---
    # If wrapped in DDP, we need to access attributes via 'model.module'
    # If not wrapped, we use 'model' directly.
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
    # ----------------------

    # Freeze everything first
    # Note: iterating model.parameters() works for both DDP and standard models
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the Gap Embedder
    # Use 'actual_model' to access specific sub-modules safely
    if hasattr(actual_model, "gap_embedder"):
        for param in actual_model.gap_embedder.parameters():
            param.requires_grad = True
    else:
        print("Warning: 'gap_embedder' not found in model.")

    # Unfreeze the Condition Fuser
    if hasattr(actual_model, "condition_fuser"):
        for param in actual_model.condition_fuser.parameters():
            param.requires_grad = True
    else:
        print("Warning: 'condition_fuser' not found in model.")

    # Unfreeze adaLN Modulation
    # We use actual_model.named_parameters() to avoid "module." prefixes in names,
    # though strict string matching "adaLN_modulation" would usually work either way.
    for name, param in actual_model.named_parameters():
        if "adaLN_modulation" in name:
            param.requires_grad = True

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Fine-tuning {trainable_params} / {total_params} parameters.")

    # Create optimizer ONLY for trainable parameters
    # The filter function works perfectly on the DDP wrapper or the inner model
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        weight_decay=decay
    )
    
    return optimizer

def train(gpu, opt, output_dir):
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

    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category, name = opt.dataname, reflow= not opt.is_independent_coupling)
    if opt.is_independent_coupling:
        dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, test_dataset = None, collate_fn=partial(pad_collate_fn, max_particles= train_dataset.max_particles))
    else:
        dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, test_dataset = None)

    # Transforms
    scaler = PointCloudPhysicsScaler(train_dataset.stats)    

    '''
    create networks
    '''
    #betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    #model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    if opt.model_name == 'calopodit':
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
            hidden_size=256,
            depth=11,
            num_heads=8,
            mlp_ratio=4,
            use_long_skip=True,
        )
        model = DiT(DiT_config)
    else:
        print(f"Model {opt.model_name} not implemented or not included in this script")
    
    
    # Load model checkpoint and define optimizer
    if opt.model_ckpt != '':
        if opt.fine_tuning:
            # Your existing fine-tuning logic
            model = load_checkpoint_with_gap_surgery(model, opt.model_ckpt, new_gap_classes=opt.gap_classes) 
            optimizer = fine_tuning_modulate(model, lr=opt.lr*0.1, decay=opt.decay)
        else:
            print(f"Loading checkpoint from {opt.model_ckpt}...")
            ckpt = torch.load(opt.model_ckpt, map_location='cpu') # Always load to CPU first to avoid GPU OOM
            state_dict = ckpt['model_state']

            # Check if the checkpoint was saved from DDP
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:] # remove 'module.'
                else:
                    name = k
                new_state_dict[name] = v
            # -----------------------------------

            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("Model loaded successfully.")
            except RuntimeError as e:
                # If strict loading fails, it might be due to architecture changes 
                # (e.g., adding CrossAttn or PointEmbedder to an old checkpoint).
                print(f"Strict loading failed: {e}")
                print("Attempting non-strict loading (ignoring missing/unexpected keys)...")
                model.load_state_dict(new_state_dict, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))
    if opt.model_ckpt != '':
        start_epoch = torch.load(opt.model_ckpt)['epoch'] + 1
    else:
        start_epoch = 0
    print(f"Starting training from epoch {start_epoch}...")

    #Schedulers
    # 1. Warmup for first 5 epochs (or ~1000 steps)
    # warmup_epochs = 3
    # warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)

    # # 2. Cosine Decay (Smoothly drops to 0 by the last epoch)
    # # T_max should be (total_epochs - warmup_epochs)
    # # Calculate total steps
    # total_steps = (opt.niter - warmup_epochs) * len(dataloader)

    # main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # # Combine
    # lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # One CicleLR for short runs
    # 1. Calculate Total Training Steps
    # (Make sure to account for gradient accumulation if you use it, though your code doesn't seem to)
    
    #TODO Add flag to change between OneCicleLR for short runs and CosineAnnealingLR for longer runs
    steps_per_epoch = len(dataloader)
    total_steps = (opt.niter -start_epoch)* steps_per_epoch

    # 2. Define the Scheduler
    # max_lr: The peak learning rate. Since 1e-3 was unstable, let's try 1e-4 or 3e-4.
    # pct_start: The percentage of training spent warming up (2 epochs / 10 epochs = 0.2)
    # div_factor: Initial LR = max_lr / div_factor. Default is 25.
    # final_div_factor: Final LR = Initial LR / final_div_factor. Default is 1e4.

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,  # Peak LR (Try 1e-4 if 3e-4 is still unstable)
        #max_lr= 1e-5, 
        total_steps=total_steps, # Exact number of batch updates
        pct_start=0.2,           # Warmup for first 20% (2 epochs)
        anneal_strategy='cos',   # Cosine shape
        div_factor=25.0,         # Starts at 3e-4 / 25 = 1.2e-5
        final_div_factor=100.0,  # Ends at 1.2e-5 / 100 = 1.2e-7
    )

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

    #scaler = GradScaler(enabled=True)

    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    # def new_x_chain(x, num_chain):
    #     return torch.randn(num_chain, *x.shape[1:], device=x.device)
    #Rectified_Flow
    #rf_criterion = RectifiedFlowLossFunction(loss_type = "mse")
    rf_criterion = MaskedPhysicalRectifiedFlowLoss(loss_type= "mse", energy_weight= 1.0)
    p_drop = opt.dropout
    
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
                #lr_scheduler.step(epoch) #seem to be deprecated?
                #lr_scheduler.step()
                for i, data in enumerate(dataloader):
                    if opt.dataname == 'g4' or opt.dataname == 'idl':
                        if opt.is_independent_coupling:
                            x, mask, int_energy, y, gap_pid, idx = data
                            #TODO I am just gonna used Ta to reflow, correct this later
                            gap_pid = gap_pid.long() # safe guard, force cast to long just in case, Critical for nn.Embedding 
                            y = y.long() # safe guard, force cast to long just in case, Critical for nn.Embedding 
                        else:
                            #NOTE reflow pairs have been saved normalized
                            #idl_dataset.reflow: [x_0.cpu(), pts_norm.cpu(), mask.cpu(), int_energy.cpu(), gap_pid.cpu()]
                            x_0, x_1, mask, int_energy, y, gap_pid, idx = data
                            gap_pid = gap_pid.long() # safe guard, force cast to long just in case, Critical for nn.Embedding 
                            #y = y.long()

                    if opt.model_name == "pvcnn2":
                        x = x.transpose(1,2)
                        #noises_batch = noises_init[list(idx)].transpose(1,2)
                    elif opt.dataname == 'shapenet':
                        x = data['train_points']
                        if opt.model_name == "pvcnn2":      
                            x = x.transpose(1,2)
                        #noises_batch = noises_init[data['idx']].transpose(1,2)
                    
                    if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                        if opt.is_independent_coupling:
                            x = x.cuda(gpu,  non_blocking=True)
                        else:
                            x_0 = x_0.cuda(gpu,  non_blocking=True)
                        x_1 = x_1.cuda(gpu,  non_blocking=True) if not opt.is_independent_coupling else None
                        mask = mask.cuda(gpu,  non_blocking=True)
                        y = y.cuda(gpu,  non_blocking=True)
                        gap_pid = gap_pid.cuda(gpu,  non_blocking=True)
                        int_energy = int_energy.cuda(gpu,  non_blocking=True)
                        scaler = scaler.cuda(gpu)
                        #noises_batch = noises_batch.cuda(gpu)
                    elif opt.distribution_type == 'single':
                        if opt.is_independent_coupling:
                            x = x.cuda()
                        else:
                            x_0 = x_0.cuda()
                        x_1 = x_1.cuda() if not opt.is_independent_coupling else None
                        mask = mask.cuda()
                        y = y.cuda()
                        gap_pid = gap_pid.cuda()
                        int_energy = int_energy.cuda()
                        scaler = scaler.cuda()
                        #noises_batch = noises_batch.cuda()
                    
                    optimizer.zero_grad()
                    
                    #Transform
                    if opt.is_independent_coupling:  #rectified flow dataset saved normalized, so we need to transform here.
                        x_1 = scaler.transform(x, mask=mask)
                        rectified_flow.device = x.device      
                        x_0 = rectified_flow.sample_source_distribution(x_1.shape[0])
                    else:
                        rectified_flow.device = x_1.device      

                    if debug:
                        means = x_1.mean(dim=0)
                        stds= x_1.std(dim=0)
                        print(f"Channel Means (Target ~0): {means.cpu().numpy()}")
                        print(f"Channel Stds  (Target ~1): {stds.cpu().numpy()}")
                        
                        # Energy Channel Check (Index 3)
                        if abs(means[3]) > 0.5 or abs(stds[3] - 1.0) > 0.5:
                            print("WARNING: Energy channel normalization is OFF. Check dataset stats.")
                        else:
                            print("Normalization looks healthy.")

                    #with autocast(enabled=True):
                    if opt.model_name == "pvcnn2":
                        x_0 = x_0.transpose(1,2)
                    t = rectified_flow.sample_train_time(x_1.shape[0])
                    t= t.squeeze()

                    #CFG
                    batch_size = x_1.shape[0]
                    if p_drop > 0:
                        force_drop_ids = torch.bernoulli(torch.full((batch_size,), p_drop, device=x_1.device)).bool()
                    else:
                        force_drop_ids = None
                    #NOTE to pass the mask to the loss function, we have edited rectified_flow.get_loss.criterion(mask=kwargs.get(mask))
                    loss = rectified_flow.get_loss(
                                x_0=x_0,
                                x_1=x_1,
                                y= y,
                                gap= gap_pid,
                                energy=int_energy,
                                t=t,
                                mask = mask,
                                force_drop_ids = force_drop_ids
                            )
                    #scaler.scale(loss).backward()
                    loss.backward()
                    if hasattr(opt, 'grad_clip') and opt.grad_clip is not None:
                        #scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    #netpNorm, netgradNorm = getGradNorm(model)
                    #if opt.grad_clip is not None:
                    #    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

                    #scaler.step(optimizer)
                    lr_scheduler.step() #FIXME or #TODO doing lr updates per batch
                    optimizer.step()
                    #scaler.update()
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
                    num_samples=opt.num_samples
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
                        if gpu == 0:
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
                            #     trajectory = traj1.trajectories
                            # if opt.distribution_type == 'multi':
                            #     full_x = gather_all_gpu_tensors(x)
                            #     full_pts = gather_all_gpu_tensors(pts)
                            #     full_mask = gather_all_gpu_tensors(mask)
                            # else:
                            #     # Fallback for single GPU training
                            #     full_x, full_pts, full_mask = x, pts, mask
                            # if gpu ==0:
                            #     torch.save([full_x, full_pts, full_mask], f'{opt.pthsave}calopodit_train_Jan_17_epoch_{epoch}_m.pth')  
                            torch.save([x_1, pts, mask], f'{opt.pthsave}calopodit_train_Jan_28_epoch_{epoch}.pth')  
                            print(f"Samples for testing save to {opt.pthsave}")
                            
                        with torch.no_grad():
                            plot_4d_reconstruction(x_1.transpose(1,2), pts.transpose(1,2), savepath=f"{outf_syn}/reconstruction_ep_{epoch}.png", index=0)
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
                        # In your train loop, inside the saving block:
                        model_to_save = model.module if hasattr(model, "module") else model

                        save_dict = {
                            'epoch': epoch,
                            'model_state': model_to_save.state_dict(), # Saves raw keys 'layer_name...'
                            'optimizer_state': optimizer.state_dict()
                        }
                        torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))


                    # if opt.distribution_type == 'multi':
                    #     dist.barrier()
                    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                    #     model.load_state_dict(
                    #         torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])
                    if opt.distribution_type == 'multi':
                        dist.barrier()
                        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                        
                        model.module.load_state_dict(
                            torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state']
                        )
                        # -------------------
    # if gpu==0:
    #     prof.export_memory_timeline(f"{out_prof}/memory_timeline.raw.json.gz", device=f"cuda:{gpu}")
    #     prof.export_memory_timeline(f"{out_prof}/memory_timeline.html", device=f"cuda:{gpu}")
    # profiler_table_output(prof, output_filename=f"{out_prof}/cuda_memory_profile_rank{opt.rank}.txt")
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

    # ''' workaround '''
    # if opt.is_independent_coupling:
    #     train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category, name =opt.dataname)
    #     noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
        
    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        train(opt.gpu, opt, output_dir)



def parse_args():
   #NOTE regular python train_calopodit.py --model_ckpt output/train_calopodit/2026-02-06-07_train_w_ta/epoch_18.pth --niter 10 --num_steps 500 
   #NOTE REFLOW python train_calopodit.py --model_ckpt output/train_calopodit/2026-02-09-17-27-32/epoch_19.pth --niter 80 --no_independent_coupling --num_steps 300 --dropout 0.0 
    parser = argparse.ArgumentParser()
    ''' Data '''
    #parser.add_argument('--dataroot', default='/data/ccardona/datasets/ShapeNetCore.v2.PC15k/')
    #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_e_liquidArgon_50/')
    #parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/')
    parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/train/')
    #parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/train_dbg/')# Training two class
    #parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/finetune/') #Finetuning
    #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/pth/reflow/combined_batches_reflow_calopodit_Normalized_Feb_10_500_steps.pth') #Reflow
    parser.add_argument('--category', default='all', help='category of dataset')
    parser.add_argument('--pthsave', default='/pscratch/sd/c/ccardona/datasets/pth/reflow/')
    #parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--dataname',  default='idl', help='dataset name: shapenet | g4')
    parser.add_argument('--bs', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--npoints',  type=int, default=1700)
    parser.add_argument("--num_classes", type=int, default=0, help=("Number of primary particles used in simulated data"),)
    parser.add_argument("--gap_classes", type=int, default=2, help=("Number of calorimeter materials used in simulated data"),) 
    
    '''model'''
    parser.add_argument("--model_name", type=str, default="calopodit", help="Name of the velovity field model. Choose between ['pvcnn2', 'calopodit', 'graphcnn'].")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)
    
    '''Flow'''
    parser.add_argument("--interp", type=str, default="straight", help="Interpolation method for the rectified flow. Choose between ['straight', 'slerp', 'ddim'].")
    parser.add_argument("--source_distribution", type=str, default="normal", help="Distribution of the source samples. Choose between ['normal'].")
    #parser.add_argument("--is_independent_coupling", type=bool, default= True, help="Whether training 1-Rectified Flow")
    parser.add_argument("--no_independent_coupling", #NOTE Reflow
                    dest="is_independent_coupling", 
                    action="store_false", 
                    help="Disable independent coupling")
    parser.set_defaults(is_independent_coupling=True)

    parser.add_argument("--train_time_distribution", type=str, default="uniform", help="Distribution of the training time samples. Choose between ['uniform', 'lognormal', 'u_shaped'].")
    parser.add_argument("--train_time_weight", type=str, default="uniform", help="Weighting of the training time samples. Choose between ['uniform'].")
    parser.add_argument("--criterion", type=str, default="mse", help="Criterion for the rectified flow. Choose between ['mse', 'l1', 'lpips'].")
    parser.add_argument("--num_steps", type=int, default=1000, help=(
            "Number of steps for generation. Used in training Reflow and/or evaluation"),)
    #parser.add_argument("--sample_batch_size", type=int, default=100, help="Batch size (per device) for sampling images.",)
    parser.add_argument("--num_samples", type=int, default=64, help="Batch size (per device) for sampling images.",)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', type = float,  default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0.01, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='lr decay for EBM')

    parser.add_argument('--model_ckpt', default='', help="path to model checkpoint (to continue training)")


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
    parser.add_argument('--saveIter', type=int, default=4, help='unit: epoch')
    parser.add_argument('--diagIter', type=int, default=4, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=8000, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=32, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    '''profiling'''
    parser.add_argument('--enable_profiling', action='store_true', help='Enable profiling during training.')
    '''fine tuning'''
    parser.add_argument('--fine_tuning', action='store_true', help='Enable fine tuning training.')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()

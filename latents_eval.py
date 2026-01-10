import argparse
from functools import partial
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Subset
from models.PCVAE import PointCloud4DVAE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.train_utils import *
from datasets.idl_dataset import LazyIDLDataset as IDLDataset

def check_latent_distribution(mus, savepath):
    # mus: [Total_Samples, 512]
    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=30)
    z_2d = tsne.fit_transform(mus.numpy())
    
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=5)
    plt.title("t-SNE of Point Cloud Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(savepath)

def extract_pcvae_latents_and_reconstruction(args, device="cuda:0"):
    parent_dir = str(Path(args.model_path).parent)
    saveplot_parent = f"{parent_dir}/syn/"
    saveplot = f"{saveplot_parent}reconstruction_eval.png"
    saveplot_latent = f"{saveplot_parent}/latent_space_eval.png"
    dataset = IDLDataset(args.dataroot, transform=None)
    # Setup Model
    # Ensure the latent_dim matches what you used during training
    #model = PointCloud4DVAE(latent_dim=512).to(device)
    model = PointCloud4DVAE(latent_dim=args.latent_dim, max_points=dataset.max_particles)
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    for param in model.parameters():
        param.requires_grad = False
    #unwrap ddp model if needed
    #Create a new state_dict without the 'module.' prefix
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state'].items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' (7 characters)
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.to(device)
    #  Setup DataLoader (No shuffling needed for extraction)

    
    #dataloader, _, train_sampler, _ = get_dataloader(args, train_dataset, test_dataset = None, collate_fn=partial(pad_collate_fn, max_particles= train_dataset.max_particles))
    ## TODO create validation dataset. Using a portion of the training data for now
    subset_size = args.bs
    indices = list(range(subset_size)) # First 1000
    #indices = np.random.choice(len(dataset), subset_size, replace=False) # Random 
    train_subset = Subset(dataset, indices)        
    val_loader = DataLoader(train_subset, batch_size=args.bs, pin_memory = True,
                                num_workers=args.workers, drop_last=True, 
                                collate_fn=partial(pad_collate_fn, max_particles=dataset.max_particles) )

    all_mus = []
    all_logvars = []
    #loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    #model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x, mask, init_energy, y, gap_pid, idx = data
            x = x.transpose(1,2)
            x, mask, init_energy= x.to(device), mask.to(device), init_energy.to(device)
            pcs_recon, mu, logvar = model(x, mask, init_energy)
            # collect lantents
            all_mus.append(mu.cpu())
            all_logvars.append(logvar.cpu())
        #plot one example
        plot_4d_reconstruction(x, pcs_recon, saveplot, index=0)
        print(f"Plot save to {saveplot}")
    # Concatenate all batches
    all_mus = torch.cat(all_mus, dim=0)
    all_logvars = torch.cat(all_logvars, dim=0)
    #latent space visualization

    check_latent_distribution(all_mus, saveplot_latent)
    #save in dataser folders
    savedata = "/pscratch/sd/c/ccardona/datasets/"
    torch.save({'mu': all_mus, 'logvar': all_logvars}, f"{savedata}pcvae_encoded_w_small_dataset_BN.pt")
    print(f"Latents extracted and saved to {savedata}pcvae_encoded_w_small_dataset_BN.pt")    
    


def main():
    #TODO yhis could be easily wrapped on ddp
    args = parse_args()
    extract_pcvae_latents_and_reconstruction(args, device="cuda:0")
    print("Latent space visualization saved.")
    # If you see distinct clusters or a smooth manifold, your VAE is ready.
    # If you see a single exploded cloud, check your KL Divergence weight.

def parse_args():

    parser = argparse.ArgumentParser()
    ''' Data '''
    parser.add_argument('--model_path', default='', help="path to model (to continue training)")
    #parser.add_argument('--dataroot', default='/data/ccardona/datasets/ShapeNetCore.v2.PC15k/')
    #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_e_liquidArgon_50/')
    #parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/')
    parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/')
    parser.add_argument('--category', default='all', help='category of dataset')
    #parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--dataname',  default='idl', help='dataset name: shapenet | g4')
    parser.add_argument('--latent_dim',  type=int, default=512)
    parser.add_argument('--bs', type=int, default=256, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--npoints',  type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=0, help=("Number of primary particles used in simulated data"),)
    parser.add_argument("--gap_classes", type=int, default=0, help=("Number of calorimeter materials used in simulated data"),)
    
    '''model'''
    parser.add_argument("--model_name", type=str, default="calopodit", help="Name of the velovity field model. Choose between ['pvcnn2', 'calopodit', 'graphcnn'].")
    
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
    
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    '''profiling'''
    parser.add_argument('--enable_profiling', action='store_true', help='Enable profiling during training.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()

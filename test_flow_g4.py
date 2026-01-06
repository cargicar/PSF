import torch
import torch.multiprocessing as mp
from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD
import torch.nn as nn
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
from model.calopodit import DiT, DiTConfig

from tqdm import tqdm

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.g4_pc_dataset import LazyPklDataset
from datasets.idl_dataset import LazyIDLDataset as IDLDataset
from datasets.transforms import MinMaxNormalize, CentroidNormalize, Compose
from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers import EulerSampler
from rectified_flow.samplers.base_sampler import Sampler
from functools import partial


'''
models
'''

from scipy import integrate
#import sde_lib
def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.
    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      rtol=atol=1e-5
      method='RK45'
      eps=1e-3
      shape = z.shape
      device = z.device
      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()

        print(x.shape, x.min(), x.max())
      else:
        x = z

      model_fn = model#mutils.get_model_fn(model, train=False)


      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)


        return to_flattened_numpy(drift)


      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (eps, 1), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)


      #x = inverse_scaler(x)
      print(x.min(), x.max(), x.shape, nfe)

      #import torchvision
      #torchvision.utils.save_image(x.clamp_(0.0, 1.0), 'figs/samples.png', nrow=10, normalize=True)
      #assert False

      return x, nfe



def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type, step = 1000):
        self.loss_type = loss_type
        self.step = step
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)
        if self.step == 1:
            self.jump = 1000
        return data + model_output * 1. / 1000. * self.jump
        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False, use_var=True):
        """
        Sample from the model
        """
        model_mean = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        return model_mean
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean
        if use_var:
            print('?')
        else:
            print('!')
        '''
        if use_var:
            sample = sample + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        '''
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, constrain_fn=lambda x, t:x,
                      clip_denoised=True, max_timestep=None, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """
        if max_timestep is None:
            final_time = self.num_timesteps
        else:
            final_time = max_timestep
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        x0 = img_t

        #img_t, nfe = ode_sampler(denoise_fn, z = x0)
        #print(nfe)
        #return img_t
        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        self.jump = float(1000 // self.step)
        for t in range(0, self.step):
            #if t % 50 == 0:
            #    print(t)
            t_ = t * self.jump
            if self.step == 1:
                t_ = 0
            img_t = constrain_fn(img_t, t_)
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t_)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False).detach()
            if self.step == 1:
                print('break')
                break

        assert img_t.shape == shape
        return img_t

    def reconstruct(self, x0, t, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t


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



class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type, step = args.step)

        self.model = PVCNN2(num_classes=args.nc, 
                            embed_dim=args.embed_dim, 
                            use_att=args.attention,
                            dropout=args.dropout, 
                            extra_feature_channels=1) #<--- energy. #NOTE maybe we can add the remaining features as extra channels?? 

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t):
        B, D,N= data.shape
        assert data.dtype == torch.float
        #assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, noise_fn=torch.randn, constrain_fn=lambda x, t:x,
                    clip_denoised=False, max_timestep=None,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            constrain_fn=constrain_fn,
                                            clip_denoised=clip_denoised, max_timestep=max_timestep,
                                            keep_running=keep_running)

    def reconstruct(self, x0, t, constrain_fn=lambda x, t:x):

        return self.diffusion.reconstruct(x0, t, self._denoise, constrain_fn=constrain_fn)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


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

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


#############################################################################
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
        #FIXME
        test_dataset = train_dataset
    return train_dataset, test_dataset




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


def evaluate_gen(opt, ref_pcs, logger):
    #NOTE passing here to read max_particles from the args. Look for a way to do it from dataset
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


    if ref_pcs is None:
        _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category, name = opt.dataname)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False,  collate_fn=pad_collate_fn)
        #NOTE for debugging purposses 
        subset_size = 32
        subset_indices = torch.randperm(len(test_dataset))[:subset_size]
        subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
        subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False,  collate_fn=partial(pad_collate_fn, max_particles= train_dataset.max_particles))
                                                      

        ref = []
        #for data in tqdm(test_dataloader, total=len(test_dataloader), desc='Generating Samples'):
        for data in tqdm(subset_dataloader, total=len(subset_dataloader), desc='Generating Samples'):
            if opt.dataname == 'g4':
                x, mask, energy, y, gap_pid, idx = data
                # x_pc = x[:,:,:3]
                # outf_syn = f"/global/homes/c/ccardona/PSF"
                # visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                #                        x_pc, None, None,
                #                        None)
                if opt.model_name == 'pvcnn2':
                    x = x.transpose(1,2)
                #FIXME m and s hardcoded for debbuging
                m=0
                s=1
            elif opt.dataname == 'shapenet':      
                x = data['test_points'].transpose(1,2)
                m, s = data['mean'].float(), data['std'].float()
            ref.append(x*s + m)

        
        ref_pcs = torch.cat(ref, dim=0).contiguous()

    logger.info("Loading sample path: %s"
      % (opt.eval_path))
    sample_pcs = torch.load(opt.eval_path).contiguous()

    logger.info("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))

    if opt.model_name == 'pvcnn2':
        ref_pcs = ref_pcs.transpose(1,2)
        sample_pcs = sample_pcs.transpose(1,2)
    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, opt.bs)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}

    pprint(results)
    logger.info(results)
    #FIXME JSD will be computed on spatial dimension for now. Due to implementation constrainst
    if opt.dataname == 'g4':
        sample_pcs = sample_pcs[:,:,:3]
        ref_pcs = ref_pcs[:,:,:3]
    jsd = JSD(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    pprint('JSD: {}'.format(jsd))
    logger.info('JSD: {}'.format(jsd))



def generate(model, opt):

    ''' data '''
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
    
    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category, name = opt.dataname)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False,  collate_fn=partial(pad_collate_fn, max_particles= test_dataset.max_particles))
    
    #Rectified_Flow
    #data_shape = (opt.npoints ,opt.nc)  # (N, 4) 4 for (x,y,z,energy)
    data_shape = (test_dataset.max_particles ,opt.nc)  # (N, 4) 4 for (x,y,z,energy)
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

    with torch.no_grad():
        if opt.model_name == "pvcnn2":
            euler_sampler = MyEulerSamplerPVCNN(
                rectified_flow=rectified_flow,
                num_steps=opt.num_steps,
                #num_samples=opt.sample_batch_size,
                num_samples=opt.bs,
            )
        else:
            euler_sampler = MyEulerSampler(
                    rectified_flow=rectified_flow,
                    num_steps=opt.num_steps,
                    #num_samples=opt.sample_batch_size,
                    num_samples=opt.bs,
                )
        
        #NOTE for debugging purposses 
        subset_size = opt.bs*50
        subset_indices = torch.randperm(len(test_dataset))[:subset_size]
        subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
        subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=opt.bs,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False,  collate_fn=partial(pad_collate_fn, max_particles= test_dataset.max_particles))
        # subset_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,
        #                                               shuffle=False, num_workers=int(opt.workers), drop_last=False,  collate_fn=partial(pad_collate_fn, max_particles= test_dataset.max_particles))
                                                      

        ref = []
        samples = []
        #for data in tqdm(test_dataloader, total=len(test_dataloader), desc='Generating Samples'):
    
        #for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):
        i = 0
        for data in tqdm(subset_dataloader, total=len(subset_dataloader), desc='Generating Samples'): 
            if opt.dataname == 'g4' or opt.dataname == 'idl':
                i+=1
                x, mask, int_energy, y, gap_pid, idx = data
                
                # x_pc = x[:,:,:3]
                # outf_syn = f"/global/homes/c/ccardona/PSF"
                # visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                #                        x_pc, None, None,
                #                        None)
                if opt.model_name == 'pvcnn2':
                    x = x.transpose(1,2)
                #FIXME m and s hardcoded for debbuging
                m=0
                s=1
            elif opt.dataname == 'shapenet':      
                x = data['test_points']
                if opt.model_name == 'pvcnn2':
                    x = x.transpose(1,2)
                m, s = data['mean'].float(), data['std'].float()
            x= x.cuda()
            rectified_flow.device = x.device      
            # Sample method
            #FIXME we should be using a validatioon small dataset instead
            # num_samples = opt.sample_batch_size
            # y =y[:num_samples]
            # gap_pid = gap_pid[:num_samples]
            # int_energy = int_energy[:num_samples]
            # mask = mask[:num_samples]
            #FIXME choosing only one material temporarly:
            if gap_pid[0]!=0:
                continue
            traj1 = euler_sampler.sample_loop(
                seed=233,
                y=y,
                gap= gap_pid,
                energy=int_energy,
                mask=mask,
                )
            gen = traj1.x_t.detach().cpu()
            trajectory = traj1.trajectories            
            # gen = model.gen_samples(x.shape,
            #                            'cuda', clip_denoised=False).detach().cpu()
            if opt.model_name == 'pvcnn2':
                gen = gen.transpose(1,2).contiguous()
            # gen = gen.transpose(1,2)
            # x = x.transpose(1,2).contiguous()

            gen = gen * s + m
            x = x * s + m
            samples.append(gen)
            ref.append(x)
            torch.save(samples, f'Jan_02_photon_samples_{i}.pth')
            #exit(0)
            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'x.png'), gen[:64], None,
                                       None, None)

        samples = torch.cat(samples, dim=0)
        torch.save(samples, 'appendix_distill_samples_{}_{}.pth'.format(opt.category, opt.num_steps))
        ref = torch.cat(ref, dim=0)

        torch.save(samples, opt.eval_path)



    return ref


def main(opt):

    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)
    logger = setup_logging(output_dir)

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

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
    

    if opt.cuda:
        model.cuda()

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    model = model.cuda()
    model = multi_gpu_wrapper(model, _transform_)
    #model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():

        logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        state_dict = resumed_param['model_state']
        model.load_state_dict(state_dict)


        ref = None
        if opt.generate:
            opt.eval_path = os.path.join(outf_syn, opt.category + 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            ref=generate(model, opt)

        if opt.eval_gen:
            # Evaluate generation
            evaluate_gen(opt, ref, logger)


def parse_args():

    parser = argparse.ArgumentParser()
    ''' Data '''
    #parser.add_argument('--dataroot', default='/data/ccardona/datasets/ShapeNetCore.v2.PC15k/')
    #parser.add_argument('--dataroot', default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_e_liquidArgon_50/')
    #parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_1mill/Pb_Simulation/')
    parser.add_argument('--dataroot', default='/global/cfs/cdirs/m3246/hep_ai/ILD_debug/')
    parser.add_argument('--category', default='car')
    parser.add_argument('--dataname',  default='g4', help='dataset name: shapenet | g4')
    parser.add_argument('--bs', type=int, default=128, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--npoints',  type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=0, help=("Number of primary particles used in simulated data"),)
    parser.add_argument("--gap_classes", type=int, default=2, help=("Number of calorimeter materials used in simulated data"),)
    
    
    '''model'''
    parser.add_argument("--model_name", type=str, default="pvcnn2", help="Name of the velovity field model. Choose between ['pvcnn2', 'calopodit', 'graphcnn'].")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=100)
    
    '''Flow'''
    parser.add_argument("--interp", type=str, default="straight", help="Interpolation method for the rectified flow. Choose between ['straight', 'slerp', 'ddim'].")
    parser.add_argument("--source_distribution", type=str, default="normal", help="Distribution of the source samples. Choose between ['normal'].")
    parser.add_argument("--is_independent_coupling", type=bool, default=True,help="Whether training 1-Rectified Flow")
    parser.add_argument("--train_time_distribution", type=str, default="uniform", help="Distribution of the training time samples. Choose between ['uniform', 'lognormal', 'u_shaped'].")
    parser.add_argument("--train_time_weight", type=str, default="uniform", help="Weighting of the training time samples. Choose between ['uniform'].")
    parser.add_argument("--criterion", type=str, default="mse", help="Criterion for the rectified flow. Choose between ['mse', 'l1', 'lpips'].")
    parser.add_argument("--num_steps", type=int, default=1000, help=(
            "Number of steps for generation. Used in training Reflow and/or evaluation"),)
    parser.add_argument("--sample_batch_size", type=int, default=32, help="Batch size (per device) for sampling images.",)

    parser.add_argument('--generate',default=True)
    parser.add_argument('--eval_gen', default=False)

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
    parser.add_argument('--saveIter', default=80, help='unit: epoch')
    parser.add_argument('--diagIter', default=80, help='unit: epoch')
    parser.add_argument('--vizIter', default=80, help='unit: epoch')
    parser.add_argument('--print_freq', default=10, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt

if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)

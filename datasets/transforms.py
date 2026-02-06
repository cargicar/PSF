import torch
import torch.nn as nn

import numpy as np
import math
import random
import numbers
import random
from itertools import repeat

class Compose:
    """
    Composes several transforms together.
    Args:
        transforms (list of callables): List of transform objects to be applied.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
import torch
import torch.nn as nn

class PointCloudPhysicsScaler(nn.Module):
    """
    Hybrid Scaler: 
    - X, Y, Log(Energy): Z-score (Mean/Std)
    - Z (Layer): Min-Max to [-1, 1]
    """
    def __init__(self, stats_dict, z_min=0.0, z_max=30.0, device="cpu"):
        super().__init__()
        
        # 1. Standard Stats (Mean/Std)
        mean = stats_dict['mean'].to(device) # [mu_x, mu_y, mu_z, mu_logE]
        std = stats_dict['std'].to(device)   # [sig_x, sig_y, sig_z, sig_logE]
        
        self.register_buffer("mean", mean.view(1, 1, 4))
        self.register_buffer("std", std.view(1, 1, 4))
        
        # 2. Z-Axis Specifics
        # Pre-calculating Min-Max constants for the Z channel (index 2)
        self.z_min = z_min
        self.z_max = z_max
        # We map [z_min, z_max] -> [-1, 1]
        self.z_scale = 2.0 / (z_max - z_min + 1e-6)

    def transform(self, x, mask=None):
        """
        Input x: (B, N, 4) -> [x, y, z, raw_energy]
        """
        # 1. Log-Transform Energy
        coords_raw = x[..., :3]
        energy_raw = x[..., 3:4]
        log_energy = torch.log(energy_raw + 1e-6)
        x_log = torch.cat([coords_raw, log_energy], dim=-1) # [x, y, z, logE]

        # 2. Apply Z-score to X, Y, and LogE
        # We skip the Z-index (2) in the standard mu/sigma application
        x_norm = (x_log - self.mean) / (self.std + 1e-6)
        
        # 3. OVERWRITE Z-channel with Min-Max
        # Map original Z to [-1, 1]
        z_raw = x[..., 2:3]
        z_minmax = (z_raw - self.z_min) * self.z_scale - 1.0
        x_norm[..., 2:3] = z_minmax

        # 4. Masking
        if mask is not None:
            x_norm = x_norm * mask.unsqueeze(-1)
            
        return x_norm

    def inverse_transform(self, x_norm, mask=None):
        """
        Input x_norm: (B, N, 4) -> Normalized [x, y, z, log_energy]
        """
        # 1. Un-standardize X, Y, LogE
        x_log = x_norm * self.std + self.mean
        
        # 2. Un-MinMax Z-channel
        z_norm = x_norm[..., 2:3]
        z_raw = (z_norm + 1.0) / self.z_scale + self.z_min
        
        # 3. Inverse Log for Energy
        coords = torch.cat([x_log[..., :2], z_raw], dim=-1) # [x, y, z_raw]
        log_energy = x_log[..., 3:4]
        energy = torch.exp(log_energy)
        
        x_raw = torch.cat([coords, energy], dim=-1)
        
        if mask is not None:
            x_raw = x_raw * mask.unsqueeze(-1)
            
        return x_raw

# class PointCloudStandardScaler(nn.Module):
#     """
#     Standardizes point cloud data (x, y, z, energy) using pre-computed statistics.
#     Includes an automatic log-transform for the energy channel to handle large dynamic ranges.
    
#     The input stats_dict is expected to contain:
#     - mean: [mean_x, mean_y, mean_z, mean_log_energy]
#     - std:  [std_x,  std_y,  std_z,  std_log_energy]
#     """
#     def __init__(self, stats_dict, device="cpu"):
#         """
#         Args:
#             stats_dict (dict): Dictionary containing 'mean' and 'std' tensors from LazyIDLDataset.
#             device (str or torch.device): Device to store the statistics on.
#         """
#         super().__init__()
        
#         # Extract stats
#         # shape: (4,) -> [x, y, z, log_energy]
#         mean = stats_dict['mean'].to(device)
#         std = stats_dict['std'].to(device)
        
#         # Register as buffers (non-trainable parameters that move with the model)
#         # Reshape to (1, 1, 4) for easy broadcasting against (Batch, Points, 4)
#         self.register_buffer("mean", mean.view(1, 1, 4))
#         self.register_buffer("std", std.view(1, 1, 4))

#     def transform(self, x, mask=None):
#         """
#         Input: (Batch, Points, 4) [x, y, z, raw_energy]
#         Output: (Batch, Points, 4) Normalized [x, y, z, log_energy]
#         """
#         # 1. Log-Transform Energy Channel
#         # We must apply this BEFORE normalization because our stats are for log(E)
#         coords = x[..., :3]
#         energy = x[..., 3:4]
        
#         # Add epsilon to prevent log(0)
#         log_energy = torch.log(energy + 1e-6)
        
#         # Recombine into a single tensor [x, y, z, log_energy]
#         x_log = torch.cat([coords, log_energy], dim=-1)
        
#         # 2. Standardize (Z-Score Normalization)
#         # (x - mu) / sigma
#         x_norm = (x_log - self.mean) / (self.std + 1e-6)
        
#         # 3. Apply Mask (Keep 0s as 0s)
#         # Normalization usually shifts 0 to something else (e.g., -1.5). 
#         # We must zero out the padding explicitly.
#         if mask is not None:
#             x_norm = x_norm * mask.unsqueeze(-1)
            
#         return x_norm

#     def inverse_transform(self, x, mask=None):
#         """
#         Input: (Batch, Points, 4) Normalized [x, y, z, log_energy]
#         Output: (Batch, Points, 4) Raw [x, y, z, raw_energy]
#         """
#         # 1. Un-Standardize
#         # x * sigma + mu
#         x_log = x * self.std + self.mean
        
#         # 2. Inverse Log-Transform Energy Channel
#         coords = x_log[..., :3]
#         log_energy = x_log[..., 3:4]
        
#         # Exp to get raw energy back
#         energy = torch.exp(log_energy)
        
#         # 3. Recombine
#         x_raw = torch.cat([coords, energy], dim=-1)
        
#         # 4. Apply Mask
#         if mask is not None:
#             x_raw = x_raw * mask.unsqueeze(-1)
            
#         return x_raw

class MinMaxNormalize:
    """
    Min-Max normalization transform for PyTorch datasets.
    It normalizes data to a [0, 1] range based on pre-calculated min and max values.
    """
    def __init__(self, min_vals: np.ndarray, max_vals: np.ndarray):
        """
        Initializes the normalizer with pre-calculated min and max values.

        Args:
            min_vals (np.ndarray): A NumPy array containing the minimum value for each feature.
            max_vals (np.ndarray): A NumPy array containing the maximum value for each feature.
        """
        if min_vals.shape != max_vals.shape:
            raise ValueError("min_vals and max_vals must have the same shape.")
        
        self.min_vals = min_vals
        self.max_vals = max_vals

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the min-max normalization to the input data.

        Args:
            x (np.ndarray): The input data (e.g., a shower array) to be normalized.
                            Expected shape is (N_particles, N_features).

        Returns:
            np.ndarray: The normalized data with values in the range [0, 1].
        """
        # Ensure that the dimensions match
        if x.shape[-1] != self.min_vals.shape[-1]:
            raise ValueError(f"Input data has {x.shape[-1]} features, but normalizer was initialized with {self.min_vals.shape[-1]} features.")
        
        # Apply the Min-Max normalization formula: (x - min) / (max - min)
        # Using broadcasting for efficient computation
        normalized_x = (x - self.min_vals) / (self.max_vals - self.min_vals)
        return normalized_x

class CentroidNormalize:
    """
    A transform that centers the data around the origin by subtracting the centroid.
    The centroid is calculated as the mean of each feature dimension, excluding the third feature (index 3).
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the centroid normalization to the input data.

        Args:
            x (np.ndarray): The input data (e.g., a shower array) to be centered.
                            Expected shape is (N_particles, 4).

        Returns:
            np.ndarray: The centered data. The third feature remains unchanged.
        """
        # Create a copy to avoid modifying the original array
        centered_x = x.copy()
        
        # Calculate the mean of the first three features (indices 0, 1, 2)
        # and ignore the fourth one (index 3).
        centroid = np.mean(x[:, :3], axis=0)
        
        # Subtract the centroid from the first three features only.
        # This uses NumPy's broadcasting.
        centered_x[:, :3] = x[:, :3] - centroid
        
        return centered_x
class Center(object):
    r"""Centers node positions around the origin."""

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, data):
        for key in self.attr:
            data[key] = data[key] - data[key].mean(dim=-2, keepdim=True)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class FixedPoints(object):
    r"""Samples a fixed number of :obj:`num` points and features from a point
    cloud.
    Args:
        num (int): The number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples fixed
            points without replacement. In case :obj:`num` is greater than
            the number of points, duplicated points are kept to a
            minimum. (default: :obj:`True`)
    """

    def __init__(self, num, replace=True):
        self.num = num
        self.replace = replace
        # warnings.warn('FixedPoints is not deterministic')

    def __call__(self, data):
        num_nodes = data['pos'].size(0)
        data['dense'] = data['pos']

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        for key, item in data.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes and key != 'dense':
                data[key] = item[choice]

        return data

    def __repr__(self):
        return '{}({}, replace={})'.format(self.__class__.__name__, self.num,
                                           self.replace)


class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.
    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix, attr):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix
        self.attr = attr

    def __call__(self, data):
        for key in self.attr:
            pos = data[key].view(-1, 1) if data[key].dim() == 1 else data[key]

            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible '
                'shape.')

            data[key] = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())


class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, attr, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.attr = attr

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix), attr=self.attr)(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)


class AddNoise(object):

    def __init__(self, std=0.01, noiseless_item_key='clean'):
        self.std = std
        self.key = noiseless_item_key

    def __call__(self, data):
        data[self.key] = data['pos']
        data['pos'] = data['pos'] + torch.normal(mean=0, std=self.std, size=data['pos'].size())
        return data


class AddRandomNoise(object):

    def __init__(self, std_range=[0, 0.10], noiseless_item_key='clean'):
        self.std_range = std_range
        self.key = noiseless_item_key

    def __call__(self, data):
        noise_std = random.uniform(*self.std_range)
        data[self.key] = data['pos']
        data['pos'] = data['pos'] + torch.normal(mean=0, std=noise_std, size=data['pos'].size())
        return data


class AddNoiseForEval(object):

    def __init__(self, stds=[0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15]):
        self.stds = stds
        self.keys = ['noisy_%.2f' % s for s in stds]

    def __call__(self, data):
        data['clean'] = data['pos']
        for noise_std in self.stds:
            data['noisy_%.2f' % noise_std] = data['pos'] + torch.normal(mean=0, std=noise_std, size=data['pos'].size())
        return data


class IdentityTransform(object):
    
    def __call__(self, data):
        return data


class RandomScale(object):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}
    for three-dimensional positions.
    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """

    def __init__(self, scales, attr):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.attr = attr

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        for key in self.attr:
            data[key] = data[key] * scale
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.scales)


class RandomTranslate(object):
    r"""Translates node positions by randomly sampled translation values
    within a given interval. In contrast to other random transformations,
    translation is applied separately at each position.
    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """

    def __init__(self, translate, attr):
        self.translate = translate
        self.attr = attr

    def __call__(self, data):
        (n, dim), t = data['pos'].size(), self.translate
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=dim))
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data['pos'].new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

        for key in self.attr:
            data[key] = data[key] + torch.stack(ts, dim=-1)

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.translate)


class Rotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degree, attr, axis=0):
        self.degree = degree
        self.axis = axis
        self.attr = attr

    def __call__(self, data):
        degree = math.pi * self.degree / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix), attr=self.attr)(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)

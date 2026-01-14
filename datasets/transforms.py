import torch
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

class NormalizePC4D:
    """
    pc: [4, N] (x, y, z, E)
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        coords = x[:3, :] # [3, N]
        energy = x[3:, :] # [1, N]
        
        # TODO: read max e and min energy from somewhere else
        #NOTE Z-score is probably more robust normalization
        E_MIN = 0.0
        E_MAX = 40.0
        C_MIN = 0.0
        C_MAX = 30.0

        # Center and scale spatial coordinates
        # axis=1 is equivalent to dim=1; keepdims=True is equivalent to keepdim=True
        #centroid = np.mean(coords, axis=1, keepdims=True)
        #coords = coords - centroid
        
        # np.linalg.norm is a cleaner way to calculate sqrt(sum(coords**2))
        #m_dist = np.max(np.sqrt(np.sum(coords**2, axis=0)))
        #coords = coords / (m_dist + 1e-6)
        #Lets do just some MIn-Max
        #Ncoords = (coords-C_MIN)/ (C_MAX - C_MIN + 1e-6)
        # Log-scale energy and Min-Max normalize
        energy = np.log1p(energy) 
        energy = (energy - E_MIN) / (E_MAX - E_MIN + 1e-6)
        
        #NOTE Global Z-score
        #energy = (energy - e_mean) / (e_std + 1e-6)
        #energy = energy.unsqueeze(0)
        # np.concatenate with axis=0 is equivalent to torch.cat with dim=0
        return np.concatenate([coords, energy], axis=0)    
    
def invert_normalize_pc4d(x_norm: torch.Tensor, m_dist = None, centroid = None) -> np.ndarray:
    """
    Inverts the NormalizePC4D transformation.
    
    Args:
        x_norm: [4, N] Normalized point cloud (x, y, z, E)
    """
    coords = x_norm[..., :3, :] # [3, N]
    energy = x_norm[..., 3:, :] # [1, N]
    
    # 1. Invert Energy Normalization
    E_MIN = 0.0
    E_MAX = 40.0
    C_MIN = 0.0
    C_MAX = 30.0
    # Reverse Min-Max
    energy = energy * (E_MAX - E_MIN + 1e-6) + E_MIN
    coords = coords * (C_MAX - C_MIN + 1e-6) + C_MIN
    
    # Reverse log1p (e^x - 1)
    energy = torch.expm1(energy)
    
    # 2. Invert Spatial Normalization
    # Reverse Scaling
    if m_dist is not None:
        coords = coords * (m_dist + 1e-6)
    
    # Reverse Centering
    if centroid is not None:
        coords = coords + centroid
        
    return torch.cat([coords, energy], dim=-2)

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


class NormalizeScale(object):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self, attr):
        self.center = Center(attr=attr)
        self.attr = attr

    def __call__(self, data):
        data = self.center(data)

        for key in self.attr:
            scale = (1 / data[key].abs().max()) * 0.999999
            data[key] = data[key] * scale

        return data


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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
from model.pointnet_util import index_points, square_distance

#NOTE Still have not used this
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: point cloud data, [B, N, C] (B=batch size, N=number of points, C=coordinates dimension)
        npoint: number of samples to select
    Return:
        centroids: selected point indices, [B, npoint]
    """
    # Get dimensions
    device = xyz.device
    B, N, C = xyz.shape
    
    # Initialize the tensor to hold the indices of the sampled points
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    
    # Initialize the tensor to hold the distance from each point to the nearest sampled point
    # Start with a large distance (infinity)
    distance = torch.ones(B, N).to(device) * 1e10
    
    # Randomly select the first point (index 0 for simplicity/consistency)
    # The `farthest` index will hold the index of the point currently selected as the farthest
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # B indices

    # Create a batch index array for easier indexing later
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        # Record the index of the farthest point as the i-th centroid
        centroids[:, i] = farthest
        
        # Get the coordinates of the farthest point
        # xyz[B, N, C] -> use batch_indices and farthest indices to get [B, 1, C]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        
        # Calculate the squared distance from all points to the new centroid
        # square_distance(xyz: [B, N, C], centroid: [B, 1, C]) -> [B, N, 1] (or [B, N])
        # We need to reshape the output to [B, N] for element-wise comparison
        dist = square_distance(xyz, centroid).view(B, N)
        
        # Update the distance array: a point's distance is the *minimum* # of its previous distance and its new distance to the latest centroid
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Find the index of the point that is *farthest* from the *set* of sampled points
        # The farthest point will have the maximum value in the updated distance tensor
        farthest = torch.max(distance, -1)[1] # [1] gets the indices
        
    return centroids

# Assuming TransformerBlock is already defined as above
#NOTE Still have not used this
class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # Simple MLP for feature aggregation (replace with your TransformerBlock if needed)
        # Note: A proper aggregation step would often be more complex (e.g., PointNet-style)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3 # 3 for coordinates
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            features: input points features data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, npoint, 3]
            new_features: sampled points features data, [B, npoint, C']
        """
        
        # 1. Farthest Point Sampling (FPS)
        # fps_idx: [B, npoint] indices of the selected points
        fps_idx = farthest_point_sample(xyz, self.npoint)
        
        # new_xyz: [B, npoint, 3] coordinates of the sampled points
        new_xyz = index_points(xyz, fps_idx)
        
        # 2. Grouping (Ball Query not shown for brevity, assuming you have it or use a simple index_points)
        # For simplicity, we'll just index points directly based on the FPS indices
        # In a real PointNet++ setup, you'd use Ball Query or KNN around new_xyz
        
        if features is not None:
            # new_features: [B, npoint, C] features of the sampled points
            new_features = index_points(features, fps_idx)
        else:
            new_features = None
            
        # 3. Aggregation (Simplified: just using the sampled points' features)
        # A true SA layer would perform feature aggregation from a local neighborhood (grouping)
        
        return new_xyz, new_features

class TransformerBlock(nn.Module):
    def __init__(self, in_features, transformer_features, d_model, k) -> None: 
        super().__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(in_features, transformer_features),
            nn.ReLU(),
            nn.Linear(transformer_features, transformer_features)
        )
        self.fc1 = nn.Linear(transformer_features, d_model)
        self.fc2 = nn.Linear(d_model, transformer_features)
        self.fc_delta = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x in_feat = b x n x 4
    def forward(self, xyz, mask = None):
        
        #dist: bxnxn = bx(n_{i,j}= square distance n_i to n_j) (why not square-root?)
        dists = square_distance(xyz, xyz)
        # Masking Distances
        if mask is not None:
            # If a point is a pad (mask=0), make it infinitely far away 
            # so it is never a neighbor of a real point.
            fill_value = 1e9
            # mask shape (B, P) -> (B, 1, P) for broadcasting across rows
            dists = dists.masked_fill(~mask.unsqueeze(1), fill_value)
        # ------------------------------
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k = bx(firs_k(distance n_i to k n_j ))
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x in_features = bx(4 dim coordinates of k closest points to n_i)
        # in simplest words, each point n_i has attached to it is closest k neightbors
        
        features = self.fc0(xyz) # features: b x n x hidden_size =(b, n, 128)
        pre = features # Save for residual connection, (b, n, 128)

        #projection to attention dim x: bxnx 128 *linear( 128 xd_model:128)=bxnx d_model
        x = self.fc1(features) 
        # q : (x:bxnx512)*(linear(d_model:d_model,d_model)) = bxnxd_model
        # k : index_points(w_ks(x):bxnx512, knn_idx) = bxnx(k indexes of closes k points to poin n_i)xd_model
        # v isomorphic k
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        #self.dc_delta iso linear(3,d_model:512)
        # xyzk :=(xyz[:,:,None]:bxnx1x3-knn_xyz):bxnxkx3= bxnxkx3 (Per each point position, substracts the position of k closer points (?))
        #self.dc_delta(xyzk): bxnxkx3* linear(3,512)=bxnxkxd_model. Projection of xyzk into attention dim
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d_model
        # pos_enc is pretty much the projection of knn_xyz into attention dim. 
        # self.fc_gamma iso linear( d_model, d_model)
        # q[:, :, None]:bxnx1xk - k:bxnx[k]xd_model + pos_enc:b x n x k x d_model = b x n x k x d_model
        # attn = bxnxkxd_model*linear(d_model, d_model): bxnxkxd_model
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        # einstein sumation over k dim: bxnxkxd_model*bxnxkxd_model-> bxnxd_model
        # res attn*(v+pos_enc)
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        # res:bxnxd_model*linear(d_model,d_points) = bxnx(d_points=32)
        res = self.fc2(res) + pre
        #Zero out padded features before returning ---
        if mask is not None:
            res = res * mask.unsqueeze(-1)
        return res, attn
    
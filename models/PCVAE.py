import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#TODO find culprit for wrogn statistics at eval mode. Likely, change BN to LN
class PointCloud4DVAE(nn.Module):
    def __init__(self, latent_dim=512, cond_dim=128, max_points=1700):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # --- Energy Conditioning MLP ---
        # Projects the scalar E_init into a higher-dimensional embedding
        self.cond_net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # --- Encoder ---
        # We append the cond_dim to the input features (4 coordinates + cond)
        # self.enc_conv = nn.Sequential(
        #     nn.Conv1d(4 + cond_dim, 64, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 128, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 256, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, 1024, 1)
        # )
        self.enc_conv = nn.Sequential(
            nn.Conv1d(4 + cond_dim, 64, 1),
            nn.GroupNorm(1, 64), # LayerNorm equivalent
            nn.ReLU(),
            
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(1, 128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, 1),
            nn.GroupNorm(1, 256),
            nn.ReLU(),
            
            nn.Conv1d(256, 1024, 1),
            nn.GroupNorm(1, 1024),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # --- Decoder ---
        self.decoder = ConditionalTransformerSpatialDecoder(
            latent_dim=latent_dim, 
            cond_dim=cond_dim, 
            out_channels=4
        )

    def encode(self, x, mask, e_init):
        # x: [B, 4, N], e_init: [B, 1]
        B, _, N = x.shape
        if e_init.dim() == 1:
            e_init = e_init.unsqueeze(1)
        # Process Condition
        e_emb = self.cond_net(e_init) # [B, cond_dim]
        e_feat = e_emb.unsqueeze(2).repeat(1, 1, N) # [B, cond_dim, N]
        
        # 2. Concat features and encode
        x_in = torch.cat([x, e_feat], dim=1) # [B, 4 + cond_dim, N]
        feat = self.enc_conv(x_in)
        
        # 3. Masked Max Pooling
        m = mask.unsqueeze(1)
        feat = feat.masked_fill(m == 0, -1e9)
        global_feat = torch.max(feat, dim=2)[0] # [B, 1024]
        
        return self.fc_mu(global_feat), self.fc_logvar(global_feat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask, e_init):
        # x: [B, 4, N], mask: [B, N], e_init: [B, 1]
        if e_init.dim() == 1:
            e_init = e_init.unsqueeze(1)

        mu, logvar = self.encode(x, mask, e_init)
        z = self.reparameterize(mu, logvar)
        
        # Decode conditioned on BOTH z and e_init
        recon = self.decoder(z, e_init, x.shape[2])
        return recon, mu, logvar

class ConditionalTransformerSpatialDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, out_channels=4, max_points=1700):
        super().__init__()
        self.d_model = latent_dim + cond_dim
        self.max_points = max_points
        
        # 1. Learnable Queries (Replaces the fixed 2D grid)
        # This allows the model to learn 1700 unique "prototypes" 
        # that can move anywhere in 3D space.
        self.query_embed = nn.Embedding(max_points, self.d_model)

        # Condition projection
        self.cond_net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=1024, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # --- Output Heads ---
        # 3 coords (x, y, z)
        self.fc_coords = nn.Linear(self.d_model, 3) 
        
        # Energy: 1 scalar
        self.fc_energy = nn.Linear(self.d_model, 1)
        
        # Probability of hit: 1 scalar (The Occupancy Head)
        self.fc_hit = nn.Linear(self.d_model, 1)

    def forward(self, z, e_init, num_points=None):
        # z: [B, latent_dim]
        # e_init: [B] or [B, 1]
        
        B = z.shape[0]
        N = self.max_points if num_points is None else num_points
        
        # --- 1. Prepare Memory (Key/Value) ---
        if e_init.dim() == 1:
            e_init = e_init.unsqueeze(1)
            
        e_emb = self.cond_net(e_init) # [B, cond_dim]
        
        # Combine z and conditioning
        # Memory shape: [B, 1, d_model]
        memory = torch.cat([z, e_emb], dim=1).unsqueeze(1) 

        # --- 2. Prepare Queries ---
        # Expand learnable queries for the batch
        # shape: [B, N, d_model]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # --- 3. Transformer Decoding ---
        # Out: [B, N, d_model]
        latent_points = self.transformer(queries, memory)

        # --- 4. Prediction Heads ---
        
        # A. Coordinates
        coords = self.fc_coords(latent_points)
        
        # B. Energy
        # Use Softplus to enforce positivity. 
        # Adding a small epsilon prevents log(0) issues downstream
        energy = F.softplus(self.fc_energy(latent_points)) + 1e-6
        
        # C. Hit Probability (logits)
        hit_logits = self.fc_hit(latent_points)
        hit_prob = torch.sigmoid(hit_logits)
        
        # Concatenate outputs: [B, N, 5] -> (x, y, z, E, prob)
        # Transpose to [B, 5, N] to match expected loss format if needed
        full_recon = torch.cat([coords, energy, hit_prob], dim=-1)
        
        return full_recon.transpose(1, 2)    

class KLAnnealer:
    def __init__(self, target_kl=0.005, start_epoch=0, end_epoch=50):
        self.target_kl = target_kl
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def get_weight(self, epoch):
        if epoch < self.start_epoch:
            return 0.0
        if epoch >= self.end_epoch:
            return self.target_kl
        
        # Linear interpolation
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return progress * self.target_kl

class PointCloudVAELoss(nn.Module):
    def __init__(self, lambda_e_sum=1.0, lambda_hit=1.0, 
                 lambda_chamfer=1.0, lambda_repulsion=0.5, lambda_emd=0.0):
        super().__init__()        
        self.lambda_e_sum = lambda_e_sum    # Global Energy conservation
        self.lambda_hit = lambda_hit        # Occupancy (BCE/Count)
        self.lambda_chamfer = lambda_chamfer # Scale for Chamfer
        self.lambda_repulsion = lambda_repulsion # Weight for Repulsion (Fixes clustering)
        self.lambda_emd = lambda_emd        # Weight for EMD (Alternative to Chamfer)
        
    def get_repulsion_loss(self, pred_xyz, h=0.05):
        """
        Penalizes predicted points that are too close to each other.
        pred_xyz: [B, N, 3]
        h: Threshold distance. Points closer than this are penalized.
        """
        # Pairwise distance between predicted points
        # dist_sq: [B, N, N]
        dist_sq = torch.cdist(pred_xyz, pred_xyz, p=2) ** 2
        dist = torch.sqrt(dist_sq + 1e-8) # Avoid NaN gradients at 0
        
        # We want to ignore self-distance (diagonal is always 0)
        B, N, _ = pred_xyz.shape
        identity = torch.eye(N, device=pred_xyz.device).unsqueeze(0).expand(B, -1, -1)
        
        # Add a large number to diagonal so it's greater than h
        dist = dist + (identity * 1e9)
        
        # Calculate repulsion: max(0, h - dist)
        # Only penalize if distance < h
        repulsion = torch.clamp(h - dist, min=0.0)
        
        # Mean over the batch and pairs
        return repulsion.mean()

    def get_emd_loss(self, x, y, eps=0.005, iters=50):
        """
        Approximate Earth Mover's Distance using Sinkhorn iterations.
        x: [B, N, 3] (Predictions)
        y: [B, N, 3] (Targets)
        Note: This assumes N_pred == N_target. If padding exists, 
        it usually requires more complex masking, but this is a robust approximation.
        """
        B, N, _ = x.shape
        # Cost matrix: Squared Euclidean distance
        # [B, N, N]
        C = torch.cdist(x, y, p=2) ** 2
        
        # Optimal Transport kernel
        K = torch.exp(-C / eps)
        
        # Vectors u and v (initialization)
        u = torch.ones_like(K[:, :, 0]) / N
        v = torch.ones_like(K[:, 0, :]) / N
        
        # Sinkhorn Iterations
        for _ in range(iters):
            u = 1.0 / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            v = 1.0 / (torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            
        # Transport Plan T = diag(u) * K * diag(v)
        # Cost = Sum(T * C)
        # Efficient calculation:
        transport_cost = (u.unsqueeze(2) * (K * C) * v.unsqueeze(1)).sum(dim=(1, 2))
        return transport_cost.mean()

    def forward(self, preds, target, target_mask, mu, logvar, e_init, kl_weight):
        """
        preds:       [B, 5, N_pred] -> (x, y, z, E, hit_prob)
        target:      [B, 4, N_gt]   -> (x, y, z, E)
        target_mask: [B, N_gt]      -> 1.0 for real points, 0.0 for padding
        mu, logvar:  [B, latent_dim]
        e_init:      [B] or [B, 1]  -> Total energy condition
        """
        # Force inputs to Float
        target_mask = target_mask.float() 
        e_init = e_init.float()
        batch_size = preds.shape[0]

        # --- 1. Unpack Predictions ---
        preds = preds.transpose(1, 2)   # [B, N, 5]
        target = target.transpose(1, 2) # [B, N, 4]
        
        pred_xyz = preds[..., :3]       # [B, N, 3]
        pred_E   = preds[..., 3]        # [B, N]
        pred_hit = preds[..., 4]        # [B, N] 

        target_xyz = target[..., :3]    # [B, N, 3]
        target_E   = target[..., 3]     # [B, N]

        # --- 2. Repulsion Loss (NEW) ---
        # Fixes the clustering by pushing points apart
        loss_repulsion = self.get_repulsion_loss(pred_xyz, h=0.2) * self.lambda_repulsion

        # --- 3. EMD Loss (NEW/Optional) ---
        # If enabled, calculate EMD. Note: EMD is expensive.
        # We only apply it if lambda_emd > 0 to save compute.
        loss_emd = torch.tensor(0.0, device=preds.device)
        if self.lambda_emd > 0:
            loss_emd = self.get_emd_loss(pred_xyz, target_xyz) * self.lambda_emd

        # --- 4. Masked Chamfer Distance ---
        # Pairwise squared distances: [B, N_pred, N_gt]
        # (x-y)^2 = x^2 + y^2 - 2xy
        # Using built-in cdist for efficiency
        dist_sq = torch.cdist(pred_xyz, target_xyz, p=2) ** 2 
        
        # A. Pred -> Nearest Target (Precision)
        # We mask columns corresponding to padding in target so preds don't match with "ghosts"
        # Expand mask to [B, 1, N_gt] and make invalid locations infinity
        inf_mask = (target_mask.unsqueeze(1) == 0) # True where padding
        # Ensure float('inf') matches the tensor dtype
        dist_sq_masked = dist_sq.masked_fill(inf_mask, 1e18)
        
        # min_dist_pred: Distance from each pred point to nearest VALID target
        # idx_pred: Which target point did it match?
        min_dist_pred, idx_pred = torch.min(dist_sq_masked, dim=2) # [B, N_pred]
        
        # B. Target -> Nearest Pred (Recall)
        # We want every REAL target point to be matched by some prediction.
        min_dist_target, idx_target = torch.min(dist_sq, dim=1)    # [B, N_gt]
        
        # Loss terms
        #loss_chamfer_target = (min_dist_target * target_mask).sum() / (target_mask.sum() + 1e-6)
        #loss_chamfer_pred   = min_dist_pred.mean()
        
        # Apply lambda_chamfer here
        #loss_chamfer = (loss_chamfer_target + loss_chamfer_pred) * self.lambda_chamfer

        # --- 5. Local Energy Match ---
        batch_indices = torch.arange(batch_size, device=idx_pred.device).unsqueeze(1).expand(-1, idx_pred.shape[1])
        matched_target_E = target_E[batch_indices, idx_pred] 
        loss_local_E = F.l1_loss(pred_E, matched_target_E)

        # --- 6. Occupancy / Hit Loss ---
        # Count match
        n_hits_pred = pred_hit.sum(dim=1)
        n_hits_target = target_mask.sum(dim=1).float()
        loss_hit_count = F.mse_loss(n_hits_pred, n_hits_target)
        
        # Entropy (Binary regularization)
        loss_hit_entropy = -(pred_hit * torch.log(pred_hit + 1e-6) + \
                             (1-pred_hit) * torch.log(1-pred_hit + 1e-6)).mean()

        # --- 7. Global Energy Conservation ---
        total_pred_E = (pred_E * pred_hit).sum(dim=1)
        total_target_E = (target_E * target_mask).sum(dim=1)
         #if e_init.dim() > 1: e_init = e_init.squeeze()
        
        #loss_global_E_sum = F.mse_loss(total_pred_E, e_init)
        #instead of comparing pred_e_sum with e_inint, I want to compare it with total_target_e
        loss_global_E_sum = F.mse_loss(total_pred_E, total_target_E)

        # --- 8. KLD Loss ---
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kld = (loss_kld / batch_size) * kl_weight

        # Weighted Sums
        loss_global_e = self.lambda_e_sum * loss_global_E_sum
        loss_hit_weighted = self.lambda_hit * loss_hit_count
        loss_hit_entr = 0.1 * loss_hit_entropy
        
        #total_loss = (loss_chamfer) + \
        total_loss  = (loss_repulsion) + \
                     (loss_emd) + \
                     (loss_local_E) + \
                     (loss_kld) + \
                     (loss_global_e) + \
                     (loss_hit_weighted) + \
                     (loss_hit_entr)
        return {
            "loss": total_loss,
            #"chamfer": loss_chamfer.item(),
            "repulsion": loss_repulsion.item(), # New log
            "emd": loss_emd.item(),             # New log
            "local_E": loss_local_E.item(),
            "global_E": loss_global_e.item(),
            "hit_count": loss_hit_weighted.item(),
            "kld": loss_kld.item()
        }
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
        self.enc_conv = nn.Sequential(
            nn.Conv1d(4 + cond_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 1024, 1)
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
 # MODEL with LN 

# class PointCloud4DVAE(nn.Module):
#     def __init__(self, latent_dim=512, cond_dim=128, max_points=2048):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.cond_dim = cond_dim

#         # --- Energy Conditioning MLP ---
#         self.cond_net = nn.Sequential(
#             nn.Linear(1, cond_dim),
#             nn.SiLU(),
#             nn.Linear(cond_dim, cond_dim)
#         )

#         # --- Encoder ---
#         # Using nn.GroupNorm(1, channels) as it acts as LayerNorm for 1D convolutions
#         # without needing constant tensor reshaping.
#         self.enc_conv = nn.Sequential(
#             nn.Conv1d(4 + cond_dim, 64, 1),
#             nn.GroupNorm(1, 64), # LayerNorm equivalent
#             nn.ReLU(),
            
#             nn.Conv1d(64, 128, 1),
#             nn.GroupNorm(1, 128),
#             nn.ReLU(),
            
#             nn.Conv1d(128, 256, 1),
#             nn.GroupNorm(1, 256),
#             nn.ReLU(),
            
#             nn.Conv1d(256, 1024, 1),
#             nn.GroupNorm(1, 1024),
#             nn.ReLU()
#         )
        
#         self.fc_mu = nn.Linear(1024, latent_dim)
#         self.fc_logvar = nn.Linear(1024, latent_dim)

#         # --- Decoder ---
#         self.decoder = ConditionalTransformerSpatialDecoder(
#             latent_dim=latent_dim, 
#             cond_dim=cond_dim, 
#             out_channels=4
#         )

#     def encode(self, x, mask, e_init):
#         B, _, N = x.shape
#         e_emb = self.cond_net(e_init) 
#         e_feat = e_emb.unsqueeze(2).repeat(1, 1, N) 
        
#         x_in = torch.cat([x, e_feat], dim=1) 
#         feat = self.enc_conv(x_in)
        
#         # Masked Max Pooling
#         m = mask.unsqueeze(1)
#         feat = feat.masked_fill(m == 0, -1e9)
#         global_feat = torch.max(feat, dim=2)[0] 
        
#         return self.fc_mu(global_feat), self.fc_logvar(global_feat)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x, mask, e_init):
#         if e_init.dim() == 1:
#             e_init = e_init.unsqueeze(1)

#         mu, logvar = self.encode(x, mask, e_init)
#         z = self.reparameterize(mu, logvar)
        
#         recon = self.decoder(z, e_init, x.shape[2])
#         return recon, mu, logvar

# class ConditionalTransformerSpatialDecoder(nn.Module):
#     def __init__(self, latent_dim, cond_dim, out_channels=4):
#         super().__init__()
#         self.d_model = latent_dim + cond_dim 
        
#         self.cond_net = nn.Sequential(
#             nn.Linear(1, cond_dim),
#             nn.SiLU(),
#             nn.Linear(cond_dim, cond_dim)
#         )
        
#         self.pe_projection = nn.Linear(128, self.d_model)
        
#         # Transformer LayerNorm is built-in to the TransformerDecoderLayer
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=self.d_model, 
#             nhead=8, 
#             dim_feedforward=1024, 
#             batch_first=True,
#             norm_first=True # Better stability for VAEs
#         )
#         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
#         # Final output layer
#         self.fc_out = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.LayerNorm(self.d_model), # Final refinement
#             nn.SiLU(),
#             nn.Linear(self.d_model, out_channels)
#         )

#     def generate_2d_grid(self, n, device):
#         side = int(math.sqrt(n))
#         x = torch.linspace(-1, 1, side, device=device)
#         y = torch.linspace(-1, 1, side, device=device)
#         grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
#         if grid.shape[0] < n:
#             pad = torch.zeros((n - grid.shape[0], 2), device=device)
#             grid = torch.cat([grid, pad], 0)
#         return grid[:n]

#     def get_pe(self, grid):
#         freqs = torch.exp(torch.arange(0, 64, 2, device=grid.device) * -(math.log(10000.0) / 64))
#         u_enc = grid[:, 0:1] * freqs
#         v_enc = grid[:, 1:2] * freqs
#         return torch.cat([torch.sin(u_enc), torch.cos(u_enc), torch.sin(v_enc), torch.cos(v_enc)], -1)

#     def forward(self, z, e_init, num_points):
#         B = z.shape[0]
#         e_emb = self.cond_net(e_init) 
#         memory = torch.cat([z, e_emb], dim=1).unsqueeze(1) 
        
#         grid = self.generate_2d_grid(num_points, z.device)
#         grid = grid + torch.randn_like(grid) * 0.05 
#         queries = self.pe_projection(self.get_pe(grid)).unsqueeze(0).repeat(B, 1, 1)
        
#         out = self.transformer(queries, memory)
#         return self.fc_out(out).transpose(1, 2)   


class PointCloudVAELoss(nn.Module):
    def __init__(self, lambda_e_sum=10.0, lambda_hit=20.0):
        super().__init__()        
        self.lambda_e_sum = lambda_e_sum    # Weight for Global Energy conservation
        self.lambda_hit = lambda_hit        # Weight for Occupancy (BCE)
        
    def forward(self, preds, target, target_mask, mu, logvar, e_init, kl_weight):
        """
        preds:       [B, 5, N_pred] -> (x, y, z, E, hit_prob)
        target:      [B, 4, N_gt]   -> (x, y, z, E)
        target_mask: [B, N_gt]      -> 1.0 for real points, 0.0 for padding
        mu, logvar:  [B, latent_dim]
        e_init:      [B] or [B, 1]  -> Total energy condition
        """
        # Force inputs to Float right at the start (avoid error message in backwards pass)
        target_mask = target_mask.float() 
        e_init = e_init.float()
        # --- 1. Unpack Predictions ---
        # preds is [B, 5, N]. Transpose to [B, N, 5] for easier handling
        preds = preds.transpose(1, 2)   # [B, N, 5]
        target = target.transpose(1, 2) # [B, N, 4]
        
        pred_xyz = preds[..., :3]       # [B, N, 3]
        pred_E   = preds[..., 3]        # [B, N]
        pred_hit = preds[..., 4]        # [B, N] (Probabilities 0-1)

        target_xyz = target[..., :3]    # [B, N, 3]
        target_E   = target[..., 3]     # [B, N]

        # --- 2. Masked Chamfer Distance (Spatial Structure) ---
        # We need to find the best matches, but we must IGNORE padded points in the target.
        
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
        
        # Apply mask to the target loss: don't penalize for failing to match padding
        loss_chamfer_target = (min_dist_target * target_mask).sum() / target_mask.sum()
        loss_chamfer_pred   = min_dist_pred.mean() # Average over all generated points
        
        loss_chamfer = loss_chamfer_target + loss_chamfer_pred

        # --- 3. Local Energy Match (Energy per point) ---
        # We want the energy of a point to match the energy of its nearest spatial neighbor.
        
        # Gather the energies of the matched target points
        # idx_pred contains indices of nearest target for each predicted point
        # We gather valid target energies
        batch_size, n_pred = idx_pred.shape
        batch_indices = torch.arange(batch_size, device=idx_pred.device).unsqueeze(1).expand(-1, n_pred)
        
        matched_target_E = target_E[batch_indices, idx_pred] # [B, N_pred]
        
        # Calculate Energy L1 Loss on these matches
        # Note: We might want to weight this by the hit probability in the future, 
        # but for now, we force the 'potential' energy to be correct.
        loss_local_E = F.l1_loss(pred_E, matched_target_E)

        # --- 4. Occupancy Loss (BCE) ---
        # Problem: The target mask size (N_gt) might differ from N_pred if not careful, 
        # but usually in this VAE N_pred == N_gt (max size).
        # We want pred_hit to approximate target_mask.
        # However, because of the permutation invariance of point clouds, we can't just 
        # do BCE(pred_hit, target_mask) directly because index i doesn't correspond to index i.
        
        # *Strategy*: We used Chamfer to align them.
        # BUT, the simplified approach for this VAE is:
        # The model should output N points. We want the model to learn to output `k` high-prob points
        # where `k` is the number of real hits.
        # We sort the predicted hit probabilities and match them to the sorted mask? 
        # No, that breaks gradients.
        
        # *Robust Strategy*: Hungarian Matching is best, but expensive.
        # *Efficient Strategy*: If we rely on the decoder to handle permutation, 
        # we treat `fc_hit` as a "Keep/Discard" switch.
        # Since we are using Learnable Queries, the queries might specialize. 
        # However, a simpler proxy is to rely on the Global Energy sum and 
        # a "Sparsity Penalty" or match the number of hits.
        
        # Alternative: We skip direct BCE on mask and rely on this:
        # We want sum(pred_hit) approx sum(target_mask)
        n_hits_pred = pred_hit.sum(dim=1)
        n_hits_target = target_mask.sum(dim=1).float()
        loss_hit_count = F.mse_loss(n_hits_pred, n_hits_target)
        
        # To force binaries (0 or 1) rather than all 0.5s:
        # Regularize pred_hit towards 0 or 1.
        loss_hit_entropy = -(pred_hit * torch.log(pred_hit + 1e-6) + \
                            (1-pred_hit) * torch.log(1-pred_hit + 1e-6)).mean()

        # --- 5. Global Energy Conservation ---
        # The sum of (Energy * Probability) should equal the condition E_init
        total_pred_E = (pred_E * pred_hit).sum(dim=1) # [B]
        if e_init.dim() > 1: e_init = e_init.squeeze()
        
        loss_global_E_sum = F.mse_loss(total_pred_E, e_init)

        # --- 6. KLD Loss ---
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kld = loss_kld / batch_size
        lambda_chamfer = 0.001
        # --- Total Loss ---
        loss_chamf = loss_chamfer*lambda_chamfer
        losskld= kl_weight * loss_kld
        loss_global_e =self.lambda_e_sum * loss_global_E_sum
        loss_hit = self.lambda_hit * loss_hit_count
        loss_hit_entr= 0.1 * loss_hit_entropy

        total_loss = (loss_chamf) + \
                     (loss_local_E) + \
                     (losskld) + \
                     (loss_global_e) + \
                     (loss_hit) + \
                     (loss_hit_entr)
        return {
            "loss": total_loss,
            "chamfer": loss_chamf.item(),
            "local_E": loss_local_E.item(),
            "global_E": loss_global_e.item(),
            "hit_count": loss_hit.item(),
            "kld": losskld.item()
        }

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
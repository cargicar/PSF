import torch
import torch.nn as nn
import torch.nn.functional as F

"""This handles the discrete codebook lookup and the straight-through gradient estimation."""
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # The Codebook: [K, D]
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B, Latent_Dim] (Flattened)
        # Convert to [B, 1, D] for broadcasting
        inputs = inputs.unsqueeze(1) 
        
        # Calculate distances between inputs and all codebook vectors
        # (x-y)^2 = x^2 + y^2 - 2xy
        distances = (torch.sum(inputs**2, dim=2, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
            
        # Find nearest codebook vector
        encoding_indices = torch.argmin(distances, dim=2).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], encoding_indices.shape[1], self.num_embeddings, device=inputs.device)
        encodings.scatter_(2, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight) # [B, 1, D]
        quantized = quantized.squeeze(1) # [B, D]
        inputs = inputs.squeeze(1) # [B, D]

        # Loss
        # 1. Dictionary learning loss (move codes towards encoder outputs)
        # 2. Commitment loss (move encoder outputs towards codes)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator:
        # Pass gradients from decoder directly to encoder, skipping the non-differentiable argmin
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices
    
class PointCloud4DVQVAE(nn.Module):
    def __init__(self, latent_dim=512, cond_dim=128, max_points=1700, num_embeddings=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.max_points = max_points

        # --- Conditioning ---
        self.cond_net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # --- Encoder (PointNet-style with GroupNorm) ---
        self.enc_conv = nn.Sequential(
            nn.Conv1d(4 + cond_dim, 64, 1),
            nn.GroupNorm(8, 64), # GN is safer than BN for small batches
            nn.SiLU(), # SiLU is generally better for Deep Generative Models
            
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            
            nn.Conv1d(128, 256, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            
            nn.Conv1d(256, latent_dim, 1), # Project directly to latent dim size
            nn.GroupNorm(32, latent_dim),
            nn.SiLU()
        )
        
        # Refinement MLP before VQ
        self.pre_vq_conv = nn.Linear(latent_dim, latent_dim)

        # --- Vector Quantizer ---
        # codebook size: num_embeddings (e.g. 1024), dim: latent_dim
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost=0.25)

        # --- Decoder ---
        # Note: Decoder now takes the Quantized Latent (z_q)
        self.decoder = ConditionalTransformerSpatialDecoder(
            latent_dim=latent_dim, 
            cond_dim=cond_dim, 
            out_channels=4,
            max_points=max_points
        )

    def encode(self, x, mask, e_init):
        B, _, N = x.shape
        
        # 1. Process Condition
        if e_init.dim() == 1: e_init = e_init.unsqueeze(1)
        e_emb = self.cond_net(e_init)
        e_feat = e_emb.unsqueeze(2).repeat(1, 1, N)
        
        # 2. Concat and Conv
        x_in = torch.cat([x, e_feat], dim=1)
        feat = self.enc_conv(x_in) # [B, latent_dim, N]
        
        # 3. Masked Max Pooling (Global Feature Extraction)
        # Compresses the whole cloud into a single vector representation
        m = mask.unsqueeze(1)
        feat = feat.masked_fill(m == 0, -1e9)
        global_feat = torch.max(feat, dim=2)[0] # [B, latent_dim]
        
        # Refine before VQ
        z_e = self.pre_vq_conv(global_feat)
        
        return z_e

    def forward(self, x, mask, e_init):
        # 1. Encode to continuous latent
        z_e = self.encode(x, mask, e_init)
        
        # 2. Vector Quantization
        # z_q: Quantized latent vector (used for decoding)
        # vq_loss: Must be added to total training loss
        # indices: Discrete tokens (useful for debugging codebook usage)
        z_q, vq_loss, indices = self.vq_layer(z_e)
        
        # 3. Decode
        # Pass z_q (quantized) instead of random z
        recon = self.decoder(z_q, e_init, x.shape[2])
        
        return recon, vq_loss, indices

# --- Decoder (Kept mostly same, adjusted for inputs) ---
class ConditionalTransformerSpatialDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, out_channels=4, max_points=1700):
        super().__init__()
        self.d_model = latent_dim + cond_dim
        self.max_points = max_points
        
        # Noise Projection (Random Gaussian Queries)
        self.noise_proj = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU()
        )
        
        # Condition projection
        self.cond_net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=2048, 
            batch_first=True, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Output Heads (Coords, Energy, Probability)
        self.fc_coords = nn.Linear(self.d_model, 3) 
        self.fc_energy = nn.Linear(self.d_model, 1)
        self.fc_hit = nn.Linear(self.d_model, 1)

    def forward(self, z, e_init, num_points=None):
        # z: [B, latent_dim] (This is now z_q from VQ)
        B = z.shape[0]
        N = self.max_points if num_points is None else num_points
        
        # 1. Prepare Memory
        if e_init.dim() == 1: e_init = e_init.unsqueeze(1)
        e_emb = self.cond_net(e_init)
        
        # Combine Quantized Latent + Global Energy Condition
        memory = torch.cat([z, e_emb], dim=1).unsqueeze(1) # [B, 1, d_model]

        # 2. Prepare Random Gaussian Queries
        # "Where should points be?" (Random guess -> Refined by Transformer)
        noise_xyz = torch.randn(B, N, 3, device=z.device)
        queries = self.noise_proj(noise_xyz) # [B, N, d_model]
        
        # 3. Transformer Decoding
        latent_points = self.transformer(queries, memory)

        # 4. Heads
        # Residual Coordinate Prediction
        coords = self.fc_coords(latent_points) + noise_xyz
        
        # Energy (Softplus for positivity)
        energy = F.softplus(self.fc_energy(latent_points)) + 1e-6
        
        # Hit Probability (Sigmoid for 0-1)
        hit_prob = torch.sigmoid(self.fc_hit(latent_points))
        
        return torch.cat([coords, energy, hit_prob], dim=-1).transpose(1, 2)

#NOTE loss from omni_alpha_c paper
class PointCloudVQVAELoss(nn.Module):
    def __init__(self, factor_vq=1.0, factor_hit=1.0):
        """
        Loss class for VQ-VAE with Occupancy Head.

        Parameters
        ----------
        factor_vq : float
            Weight for the Vector Quantization loss.
        factor_hit : float
            Weight for the Hit Probability (Occupancy) loss.
        """
        super().__init__()
        self.factor_vq = factor_vq
        self.factor_hit = factor_hit

    def forward(self, model_output, target, mask):
        """
        Parameters
        ----------
        model_output : tuple
            (recon, vq_loss, indices) where recon is [B, 5, N]
        target : Tensor
            Ground truth [B, 4, N]
        mask : Tensor
            Binary mask [B, N]
        """
        # 1. Unpack Output
        recon, vq_loss, _ = model_output
        
        # Transpose to [B, N, C] for easier slicing and masking
        recon = recon.transpose(1, 2)   # [B, N, 5]
        target = target.transpose(1, 2) # [B, N, 4]

        # 2. Split Channels
        # First 4 channels: Physics (x, y, z, E)
        recon_phys = recon[..., :4] 
        # 5th channel: Hit Probability (Logits or Sigmoid)
        # Note: If your model outputs Sigmoid, use BCELoss. If logits, use BCEWithLogitsLoss.
        # Assuming your model has torch.sigmoid() at the end:
        recon_prob = recon[..., 4]  

        # 3. Masked Reconstruction Loss (Physics)
        if mask is not None:
            mask_bool = mask.bool()
            
            # Select valid points [Total_Valid, 4]
            r_valid_phys = recon_phys[mask_bool]
            x_valid = target[mask_bool]
            
            mse_reco = F.mse_loss(r_valid_phys, x_valid)
        else:
            mse_reco = F.mse_loss(recon_phys, target)

        # 4. Occupancy Loss (Hit Probability)
        # We train this on ALL points (both real and padding) so the model learns
        # to predict 1 for real and 0 for padding.
        # recon_prob: [B, N], mask: [B, N]
        bce_hit = F.binary_cross_entropy(recon_prob, mask.float())

        # 5. Total Loss
        loss = mse_reco + (self.factor_vq * vq_loss) + (self.factor_hit * bce_hit)
        
        return {
            "loss": loss,
            "mse_reco": mse_reco.item(),
            "bce_hit": bce_hit.item(),
            "vq_loss": vq_loss.item()
        }

#NOTE old (similar to the one used for PCVAELoss)    
# class PointCloudVQVAELoss(nn.Module):
#     def __init__(self, lambda_e_sum=1.0, lambda_hit=1.0, lambda_chamfer=1.0, lambda_vq=1.0):
#         super().__init__()        
#         self.lambda_e_sum = lambda_e_sum    # Global Energy conservation
#         self.lambda_hit = lambda_hit        # Occupancy (MSE/BCE)
#         self.lambda_chamfer = lambda_chamfer # Spatial Geometry
#         self.lambda_vq = lambda_vq          # Commitment + Codebook Loss
        
#     def forward(self, preds, target, target_mask, vq_loss, e_init):
#         """
#         preds:       [B, 5, N_pred] -> (x, y, z, E, hit_prob)
#         target:      [B, 4, N_gt]   -> (x, y, z, E)
#         target_mask: [B, N_gt]      -> 1.0 for real points, 0.0 for padding
#         vq_loss:     Scalar tensor from the VQ layer
#         e_init:      [B, 1] Total energy condition
#         """
#         target_mask = target_mask.float() 
#         e_init = e_init.float().view(-1)
#         batch_size = preds.shape[0]

#         # --- 1. Unpack Predictions ---
#         preds = preds.transpose(1, 2)   # [B, N, 5]
#         target = target.transpose(1, 2) # [B, N, 4]
        
#         pred_xyz = preds[..., :3]       # [B, N, 3]
#         pred_E   = preds[..., 3]        # [B, N]
#         pred_hit = preds[..., 4]        # [B, N] 

#         target_xyz = target[..., :3]    # [B, N, 3]
#         target_E   = target[..., 3]     # [B, N]
        
#         # --- 2. Masked Chamfer Distance (Geometry) ---
#         # Calculate squared distances
#         dist_sq = torch.cdist(pred_xyz, target_xyz, p=2) ** 2 
        
#         # A. Precision: Pred -> Nearest Valid Target
#         # Mask out padding columns in target so predictions don't match with ghosts
#         inf_mask = (target_mask.unsqueeze(1) == 0) # [B, 1, N_gt]
#         dist_sq_masked = dist_sq.masked_fill(inf_mask, 1e18)
        
#         min_dist_pred, idx_pred = torch.min(dist_sq_masked, dim=2) # [B, N_pred]
        
#         # B. Recall: Target -> Nearest Pred
#         # Every real target point must have a match
#         min_dist_target, _ = torch.min(dist_sq, dim=1)     # [B, N_gt]
        
#         # Weighted average for Chamfer
#         # Note: We weight the "pred" term by hit_prob to allow the model to "turn off" points
#         # efficiently without incurring heavy penalties for points it considers "ghosts"
#         loss_chamfer_target = (min_dist_target * target_mask).sum() / (target_mask.sum() + 1e-6)
#         loss_chamfer_pred   = (min_dist_pred * pred_hit).sum() / (pred_hit.sum() + 1e-6)
        
#         loss_chamfer = (loss_chamfer_target + loss_chamfer_pred) * self.lambda_chamfer

#         # --- 3. Local Energy Match (Precision) ---
#         # Match energy of prediction to the energy of the nearest target point it found
#         batch_indices = torch.arange(batch_size, device=idx_pred.device).unsqueeze(1).expand(-1, idx_pred.shape[1])
#         matched_target_E = target_E[batch_indices, idx_pred] 
        
#         # Only penalize energy error if the point is predicted to exist (hit_prob > 0.5)
#         # using soft weighting with pred_hit
#         loss_local_E = (F.l1_loss(pred_E, matched_target_E, reduction='none') * pred_hit).mean()

#         # --- 4. Occupancy / Hit Loss ---
#         # A. Count Match (MSE)
#         n_hits_pred = pred_hit.sum(dim=1)
#         n_hits_target = target_mask.sum(dim=1)
#         loss_hit_count = F.mse_loss(n_hits_pred, n_hits_target)
        
#         # B. Entropy (Encourage confident 0 or 1 predictions)
#         loss_hit_entropy = -(pred_hit * torch.log(pred_hit + 1e-6) + \
#                              (1-pred_hit) * torch.log(1-pred_hit + 1e-6)).mean()

#         loss_hit_total = (self.lambda_hit * loss_hit_count) + (0.1 * loss_hit_entropy)

#         # --- 5. Global Energy Conservation ---
#         # Sum of Energy * Hit_Prob should equal Total Ground Truth Energy
#         total_pred_E = (pred_E * pred_hit).sum(dim=1)
#         total_target_E = (target_E * target_mask).sum(dim=1)
        
#         loss_global_E_sum = F.mse_loss(total_pred_E, total_target_E) * self.lambda_e_sum

#         # --- 6. Total Loss ---
#         total_loss = loss_chamfer + loss_local_E + loss_hit_total + loss_global_E_sum + (vq_loss * self.lambda_vq)
        
#         return {
#             "loss": total_loss,
#             "chamfer": loss_chamfer.item(),
#             "local_E": loss_local_E.item(),
#             "global_E": loss_global_E_sum.item(),
#             "loss_hit": loss_hit_total.item(),
#             "vq_loss": vq_loss.item()
#         }


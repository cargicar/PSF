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

# --- Updated Conditional Decoder ---
class ConditionalTransformerSpatialDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, out_channels=4):
        super().__init__()
        self.d_model = latent_dim + cond_dim # Combined context size
        
        self.cond_net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # Positional Encoding projection
        self.pe_projection = nn.Linear(128, self.d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)

        #self.fc_out = nn.Linear(self.d_model, out_channels) 
        # Instead Split the output heads
        # out_channels-1 is for (x, y, z)
        self.fc_coords = nn.Linear(self.d_model, out_channels - 1) 
        
        # The EnergyHead plugged in here
        self.fc_energy = nn.Linear(self.d_model, 1)
        # To fix the "number of hits" discrepancy where your model predicts a constant 2048 points regardless of the physics, you need to add an Occupancy Head. This allows the model to predict which points in the grid are "real" and which should be discarded as "padding."
        # You will add a third branch  that predicts the probability (p∈[0,1]) of a hit being valid.
        #self.fc_hit = nn.Linear(self.d_model, 1)


    def generate_2d_grid(self, n, device):
        # Logic same as before: generates [n, 2] grid
        #side = int(math.sqrt(n))
        side = 30
        x = torch.linspace(-1, 1, side, device=device)
        y = torch.linspace(-1, 1, side, device=device)
        grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
        if grid.shape[0] < n:
            pad = torch.zeros((n - grid.shape[0], 2), device=device)
            grid = torch.cat([grid, pad], 0)
        return grid[:n]

    def get_pe(self, grid):
        # Sinusoidal PE logic same as before (returns [n, 128])
        freqs = torch.exp(torch.arange(0, 64, 2, device=grid.device) * -(math.log(10000.0) / 64))
        u_enc = grid[:, 0:1] * freqs
        v_enc = grid[:, 1:2] * freqs
        return torch.cat([torch.sin(u_enc), torch.cos(u_enc), torch.sin(v_enc), torch.cos(v_enc)], -1)

    def forward(self, z, e_init, num_points):
        B = z.shape[0]
        
        #  Combine z and e_init into a single memory block
        if e_init.dim() == 1:#TODO is this needed?
            e_init = e_init.unsqueeze(1) # [B] -> [B, 1]
        e_emb = self.cond_net(e_init) # [B, cond_dim]
        memory = torch.cat([z, e_emb], dim=1).unsqueeze(1) # [B, 1, d_model]
        
        #  Dynamic Spatial Queries with PE
        grid = self.generate_2d_grid(num_points, z.device)
        grid = grid + torch.randn_like(grid) * 0.05 #  5% jitter
        queries = self.pe_projection(self.get_pe(grid)).unsqueeze(0).repeat(B, 1, 1)
        
        #  Cross-Attention
        latent_points = self.transformer(queries, memory)
        #old out
        #full_recon = self.fc_out(latent_points)
        # instead we have two heads: coord head and energy head
        #Coordinate Prediction (x, y, z) (Head Coordinates)
        coords = self.fc_coords(latent_points) # [B, N, 3]
        #  CONSTRAINED Energy Prediction (Head energy)
        # # Apply Softmax across the N dimension so weights sum to 1.0
        energy_logits = self.fc_energy(latent_points) # [B, N, 1]
        full_recon = torch.cat([coords, energy_logits], dim=-1)
        #energy_weights = torch.softmax(energy_logits, dim=1)
        # # Scale by e_init so the total sum matches your condition exactly
        # # e_init is [B, 1], energy_weights is [B, N, 1]
        #actual_energy = energy_weights * e_init.unsqueeze(1) # [B, N, 1]
        #full_recon = torch.cat([coords, actual_energy], dim=-1)
        # # # Head Hit Probability (Occupancy)
        #hit_prob = torch.sigmoid(self.fc_hit(latent_points)) # [B, N, 1]
        # Combine to [B, 5, N] (x, y, z, E, hit_prob)
        #full_recon = torch.cat([coords, actual_energy, hit_prob], dim=-1)
        
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

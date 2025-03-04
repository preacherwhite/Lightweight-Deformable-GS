import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
import spconv.pytorch as spconv
import numpy as np
import torch.nn.functional as F
from scene.forecast_ode_transformer import TransformerLatentODEWrapper
from scene.forecast_ode_var_rnn import LatentODERNN

# Use original BasicBlock implementation from SpUNet-v1m1
class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out

def offset2batch(offset):
    """Convert offsets to batch indices"""
    batch = torch.zeros(offset[-1], dtype=torch.long, device=offset.device)
    for i in range(1, len(offset)):
        batch[offset[i-1]:offset[i]] = i-1
    return batch

class PointPreservingSpUNetLatentODE(nn.Module):
    """
    SpUNet with ODE processing for each feature level,
    using the original SpUNet-v1m1 architecture for encoder-decoder.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        # ODE parameters
        latent_dim=16,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        ode_nhidden=32,
        decoder_nhidden=128,
        noise_std=0.1,
        ode_layers=2,
        reg_weight=1e-3,
        variational_inference=True,
        use_torchode=False,
        use_rnn=False,  # Whether to use LatentODERNN (True) or TransformerLatentODEWrapper (False)
        voxel_size=0.05,
        spatial_shape=[128, 128, 128],
        rtol=1e-1, 
        atol=1e-1, 
        use_tanh=False
    ):
        super(PointPreservingSpUNetLatentODE, self).__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.voxel_size = voxel_size
        self.spatial_shape = spatial_shape
        
        # ODE model parameters
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.ode_nhidden = ode_nhidden
        self.decoder_nhidden = decoder_nhidden
        self.noise_std = noise_std
        self.ode_layers = ode_layers
        self.reg_weight = reg_weight
        self.variational_inference = variational_inference
        self.use_torchode = use_torchode
        self.use_rnn = use_rnn
        self.rtol = rtol
        self.atol = atol
        self.use_tanh = use_tanh
        
        # Initialize SpUNet components using the original SpUNet-v1m1 architecture
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # Initial convolution
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )
        
        # Build encoder-decoder following SpUNet-v1m1 architecture
        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        
        for s in range(self.num_stages):
            # Encoder: Downsample and residual blocks
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                BasicBlock(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            
            # Decoder: Inverse conv and residual blocks
            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        channels[len(channels) - s - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
                )
            )
            self.dec.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                BasicBlock(
                                    dec_channels + enc_channels
                                    if i == 0
                                    else dec_channels,
                                    dec_channels,
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s}",
                                ),
                            )
                            for i in range(layers[len(channels) - s - 1])
                        ]
                    )
                )
            )
            
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]
        
        # Final layer
        self.final = spconv.SubMConv3d(
            channels[-1], out_channels, kernel_size=1, padding=1, bias=True
        )
        
        # Create ODE models for each feature level - INPUT LEVEL + ENCODER LEVELS
        self.ode_models = nn.ModuleList()
        
        # ODE for initial input level
        self._add_ode_model(base_channels)  # Initial convolution output
        
        # ODE for each encoder level
        for s in range(self.num_stages):
            self._add_ode_model(channels[s])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _add_ode_model(self, feature_dim):
        """Add appropriate ODE model based on configuration"""
        if self.use_rnn:
            # Use LatentODERNN
            ode_model = LatentODERNN(
                latent_dim=self.latent_dim,
                rec_dim=self.d_model,
                num_decoder_layers=self.num_decoder_layers,
                ode_nhidden=self.ode_nhidden,
                decoder_nhidden=self.decoder_nhidden,
                obs_dim=feature_dim,
                noise_std=self.noise_std,
                ode_layers=self.ode_layers,
                reg_weight=self.reg_weight,
                variational_inference=self.variational_inference,
                use_torchode=self.use_torchode,
                rtol=self.rtol,
                atol=self.atol,
                use_tanh=self.use_tanh
            )
        else:
            # Use TransformerLatentODEWrapper
            ode_model = TransformerLatentODEWrapper(
                latent_dim=self.latent_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                ode_nhidden=self.ode_nhidden,
                decoder_nhidden=self.decoder_nhidden,
                obs_dim=feature_dim,
                noise_std=self.noise_std,
                ode_layers=self.ode_layers,
                reg_weight=self.reg_weight,
                variational_inference=self.variational_inference,
                use_torchode=self.use_torchode,
                rtol=self.rtol,
                atol=self.atol,
                use_tanh=self.use_tanh
            )
        self.ode_models.append(ode_model)
    
    @staticmethod
    def _init_weights(m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _process_sparse_features_batch(self, sparse_features_batch):
        """Process a batch of sparse features to get per-batch feature vectors"""
        # This function extracts features per batch item from sparse tensor
        # For each batch item, we'll average the features
        batch_size = sparse_features_batch.batch_size
        features = sparse_features_batch.features  # [N, C]
        indices = sparse_features_batch.indices   # [N, 4] (batch, x, y, z)
        batch_indices = indices[:, 0]     # [N]
        
        # Aggregate features per batch using mean pooling
        batch_features = []
        for b in range(batch_size):
            mask = batch_indices == b
            if mask.any():
                batch_features.append(features[mask].mean(dim=0))
            else:
                # If no points in this batch, use zeros
                batch_features.append(torch.zeros(features.shape[1], device=features.device))
        
        batch_features = torch.stack(batch_features)  # [B, C]
        return batch_features
    
    def create_combined_sparse_tensor(self, trajectories):
        """
        Create a single sparse tensor with time incorporated into batch dimension.
        """
        B, T, F = trajectories.shape
        # Create indices where batch dimension = t*B + b
        # This uniquely identifies each (point, time) combination
        indices = []
        features = []
        
        for b in range(B):
            for t in range(T):
                # Calculate combined batch index
                batch_idx = t * B + b
                
                # Get point features for this time step
                point = trajectories[b, t]
                
                # Get spatial coordinates for sparse tensor
                if F >= 3:
                    coords = point[:3]
                else:
                    coords = torch.zeros(3, device=point.device)
                
                # Quantize coordinates
                quantized_coords = (coords / self.voxel_size).int()
                
                # Add to lists
                indices.append([batch_idx, *quantized_coords])
                features.append(point)
        
        # Convert to tensors
        indices = torch.tensor(indices, device=trajectories.device).int()
        features = torch.stack(features)
        
        # Create combined sparse tensor
        sparse_tensor = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=self.spatial_shape,
            batch_size=B*T
        )
        
        return sparse_tensor, {"B": B, "T": T, "F": F}
    
    def forward_encoder(self, combined_sparse_tensor):
        """
        Forward pass through SpUNet encoder with parallel time processing.
        
        Args:
            combined_sparse_tensor: Sparse tensor with batch dimension B*T
            
        Returns:
            List of feature tensors at each encoder level
        """
        x = self.conv_input(combined_sparse_tensor)
        features = [x]
        
        # Process through encoder blocks
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            features.append(x)
        
        return features
    
    def reshape_for_ode_processing(self, feature_tensor, B, T):
        """
        Reshape encoded features for ODE processing.
        
        Args:
            feature_tensor: Sparse tensor with batch dimension B*T
            B, T: Original batch and time dimensions
            
        Returns:
            Tensor of shape [B, T, F'] ready for ODE processing
        """
        features = feature_tensor.features  # [N, C]
        indices = feature_tensor.indices    # [N, 4]
        
        # Extract batch indices
        batch_indices = indices[:, 0]  # [N]
        
        # Get feature dimension
        C = features.shape[1]
        
        # Initialize output tensor
        reshaped_features = torch.zeros(B, T, C, device=features.device)
        
        # Process each point
        for i in range(len(batch_indices)):
            combined_idx = batch_indices[i].item()
            t = combined_idx // B
            b = combined_idx % B
            if t < T:  # Safety check
                reshaped_features[b, t] = features[i]
        
        return reshaped_features
    
    def reshape_for_parallel_decoding(self, ode_features, sparse_tensor, B, T):
        """
        Reshape ODE processed features back to sparse tensor format.
        
        Args:
            ode_features: Tensor of shape [B, T, F']
            sparse_tensor: Original sparse tensor for structure
            B, T: Original batch and time dimensions
            
        Returns:
            Updated sparse tensor with processed features
        """
        indices = sparse_tensor.indices  # [N, 4]
        features = sparse_tensor.features  # [N, C]
        
        # Create new features tensor with same shape
        new_features = torch.zeros_like(features)
        
        # Map processed features back to sparse tensor
        for i in range(len(indices)):
            combined_idx = indices[i, 0].item()
            t = combined_idx // B
            b = combined_idx % B
            if t < T:  # Safety check
                new_features[i] = ode_features[b, t]
        
        # Create new sparse tensor with updated features
        new_tensor = sparse_tensor.replace_feature(new_features)
        return new_tensor
    
    def process_features_with_ode(self, encoded_features, obs_time, target_time=None):
        """
        Process each level's features with corresponding ODE model.
        Modified to handle gradient detachment when encoder is frozen.
        
        Args:
            encoded_features: List of feature tensors for each level from the encoder
            obs_time: Observation time points
            target_time: Target time points for extrapolation (optional)
            
        Returns:
            Processed features for each level ready for decoder
        """
        num_levels = len(encoded_features)
        processed_features = []
        
        # Determine B (points) and T (time steps)
        first_tensor = encoded_features[0]
        batch_size = first_tensor.batch_size
        T = len(obs_time)
        B = batch_size // T
        
        # Process each feature level with its corresponding ODE model
        for level, feature_tensor in enumerate(encoded_features):
            # Check if we need to detach gradients (when encoder is frozen)
            for param in self.conv_input.parameters():
                if not param.requires_grad:
                    # Encoder is frozen, detach feature tensor
                    feature_tensor = feature_tensor.replace_feature(feature_tensor.features.detach())
                    break
            
            # Reshape features for ODE processing
            reshaped_features = self.reshape_for_ode_processing(feature_tensor, B, T)
            
            # Get corresponding ODE model
            ode_model = self.ode_models[level]
            
            if target_time is not None:
                # Extrapolation mode
                full_time = torch.cat([obs_time, target_time])
                
                # Process with ODE model - extrapolate
                processed_features_level = ode_model.extrapolate(
                    reshaped_features, obs_time, target_time
                )
            else:
                # Reconstruction mode
                processed_features_level = ode_model.reconstruct(
                    reshaped_features, obs_time
                )
            
            # Reshape back to sparse tensor
            processed_tensor = self.reshape_for_parallel_decoding(
                processed_features_level, feature_tensor, B, 
                len(obs_time) if target_time is None else len(obs_time) + len(target_time)
            )
            
            processed_features.append(processed_tensor)
        
        return processed_features
    
    def forward_decoder(self, processed_features):
        """
        Decoder that processes all time steps in parallel.
        
        Args:
            processed_features: List of processed feature tensors for each level
            
        Returns:
            Output sparse tensor
        """
        # Get bottleneck features
        x = processed_features[-1]
        skips = processed_features[:-1]

        # Process through decoder
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips[s]
            
            # Concatenate features for skip connection
            x = x.replace_feature(torch.cat([x.features, skip.features], dim=1))
            x = self.dec[s](x)
        
        # Final layer
        x = self.final(x)
        return x
    
    def extract_trajectories(self, output_tensor, B, T):
        """
        Extract and reshape output back to trajectory format.
        
        Args:
            output_tensor: Sparse tensor from decoder
            B, T: Original batch and time dimensions
            
        Returns:
            Tensor of shape [B, T, F] with trajectories
        """
        features = output_tensor.features  # [N, C]
        indices = output_tensor.indices    # [N, 4]
        batch_indices = indices[:, 0]      # [N]
        
        # Get feature dimension
        F = features.shape[1]
        
        # Initialize output trajectories
        trajectories = torch.zeros(B, T, F, device=features.device)
        
        # Map sparse features back to dense trajectories
        for i in range(len(batch_indices)):
            combined_idx = batch_indices[i].item()
            t = combined_idx // B
            b = combined_idx % B
            if t < T:  # Safety check
                trajectories[b, t] = features[i]
        
        return trajectories
    
    def forward(self, trajectories, obs_time, target_time=None):
        """
        Forward pass with fully parallelized time-step processing.
        
        Args:
            trajectories: Input trajectories of shape [B, T, F]
            obs_time: Observation time points
            target_time: Target time points for extrapolation (optional)
            
        Returns:
            Output trajectories of shape [B, T, F] or [B, T+T_target, F]
        """
        # 1. Create a single sparse tensor with combined batch-time dimension
        print("forward")
        sparse_tensor, metadata = self.create_combined_sparse_tensor(trajectories)
        B, T, F = metadata["B"], metadata["T"], metadata["F"]
        print("created sparse tensor")
        # 2. Forward through encoder
        encoded_features = self.forward_encoder(sparse_tensor)
        print("forwarded encoder")
        # 3. Process with ODE models
        processed_features = self.process_features_with_ode(encoded_features, obs_time, target_time)
        print("processed features")
        # 4. Forward through decoder
        output_tensor = self.forward_decoder(processed_features)
        print("forwarded decoder")
        # 5. Extract trajectories
        output_trajectories = self.extract_trajectories(
            output_tensor, B, 
            T if target_time is None else T + len(target_time)
        )
        print("extracted trajectories")
        return output_trajectories
    
    def extrapolate(self, obs_traj, obs_time, extrapolate_time):
        """
        Extrapolate trajectory into the future.
        
        Args:
            obs_traj: Observed trajectories of shape [B, T_obs, F]
            obs_time: Observation time points
            extrapolate_time: Time points to extrapolate to
            
        Returns:
            Extrapolated trajectories of shape [B, T_obs+T_extrapolate, F]
        """
        return self.forward(obs_traj, obs_time, extrapolate_time)
    
    def reconstruct(self, obs_traj, obs_time):
        """
        Reconstruct trajectories.
        
        Args:
            obs_traj: Observed trajectories of shape [B, T, F]
            obs_time: Observation time points
            
        Returns:
            Reconstructed trajectories of shape [B, T, F]
        """
        return self.forward(obs_traj, obs_time)
    
    def transformer_only_reconstruction(self, obs_traj):
        """
        Directly reconstruct point cloud using only the encoder-decoder backbone 
        without ODE dynamics or transformer processing.
        
        Args:
            obs_traj: [B, T, F] tensor of observed trajectories
        
        Returns:
            loss: Reconstruction loss
            pred_first: [B, F] tensor of predicted first frame
        """
        # Get dimensions
        B, T, _ = obs_traj.shape
        device = obs_traj.device
        
       # reverse B and T, since B is the number of points and T is time steps which we interperet as the batch size
        obs_traj = obs_traj.transpose(0, 1)
        B, T, _ = obs_traj.shape
        
        obs_traj = obs_traj.reshape(B*T, -1)
        # Create sparse tensor for the first frame
        coords = obs_traj[:, :3]

        quantized_coords = torch.div(
                coords - coords.min(0)[0], self.voxel_size, rounding_mode="trunc"
            ).int()

        # batch should look like this: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
        batch_indices = torch.tensor(np.repeat(np.arange(B), T), device=device)
        
        # Create indices tensor for sparse representation
        indices = torch.cat([batch_indices.unsqueeze(-1).int(), quantized_coords], dim=1).contiguous()
        
        # Create sparse tensor
        sparse_tensor = spconv.SparseConvTensor(
            features=obs_traj,
            indices=indices,
            spatial_shape=self.spatial_shape,
            batch_size=B
        )
        
        # Forward through encoder
        x = self.conv_input(sparse_tensor)
        skips = [x]
        
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        
        # Bottleneck features
        x = skips.pop(-1)
        
        # Forward through decoder
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat([x.features, skip.features], dim=1))
            x = self.dec[s](x)
        
        # Final layer
        output_tensor = self.final(x)
        
        # Extract features and match with input points
        output_features = output_tensor.features
        loss = F.l1_loss(output_features, obs_traj)
        pred_trajectory = output_features.reshape(B, T, -1)
        return loss, pred_trajectory



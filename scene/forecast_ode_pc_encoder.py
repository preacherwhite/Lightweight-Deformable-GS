import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.sparse_unet import PointPreservingSpUNetLatentODE

class PointPreservingSpUNetODEWrapper(nn.Module):
    """
    Wrapper for PointPreservingSpUNetLatentODE that maintains compatibility with other ODE models.
    Modified to support variable number of active ODE models.
    """
    def __init__(
        self,
        latent_dim,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        ode_nhidden,
        decoder_nhidden,
        obs_dim,
        noise_std,
        ode_layers,
        reg_weight,
        variational_inference=True,
        use_torchode=False,
        use_rnn=False,
        base_channels=32,
        channels=(16, 32, 64, 64, 64, 64, 32, 16),
        layers=(2, 3, 4, 6, 1, 1, 1, 1),
        voxel_size=0.05,
        spatial_shape=[128, 128, 128],
        rtol=1e-1,
        atol=1e-1,
        use_tanh=False,
        num_active_odes=-1  # New parameter: -1 means all ODEs, 1 means only bottleneck
    ):
        super(PointPreservingSpUNetODEWrapper, self).__init__()
        
        # Store configuration
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.noise_std = noise_std
        self.reg_weight = reg_weight
        self.variational_inference = variational_inference
        self.voxel_size = voxel_size
        self.spatial_shape = spatial_shape
        self.num_active_odes = num_active_odes
        
        # Initialize the PointPreservingSpUNetLatentODE model
        self.model = PointPreservingSpUNetLatentODE(
            in_channels=obs_dim,
            out_channels=obs_dim,
            base_channels=base_channels,
            channels=channels,
            layers=layers,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            ode_nhidden=ode_nhidden,
            decoder_nhidden=decoder_nhidden,
            noise_std=noise_std,
            ode_layers=ode_layers,
            reg_weight=reg_weight,
            variational_inference=variational_inference,
            use_torchode=use_torchode,
            use_rnn=use_rnn,
            voxel_size=voxel_size,
            spatial_shape=spatial_shape,
            rtol=rtol,
            atol=atol,
            use_tanh=use_tanh
        )
        
        # The number of stages in the SpUNet (encoder or decoder)
        self.num_stages = self.model.num_stages
        
        # Total possible ODEs = num_stages + 1 (base level + each encoder level)
        self.total_odes = self.num_stages + 1
        
        # Determine which ODEs are active
        if self.num_active_odes == -1 or self.num_active_odes > self.total_odes:
            # All ODEs are active
            self.num_active_odes = self.total_odes
            self.active_ode_indices = list(range(self.total_odes))
        elif self.num_active_odes == 1:
            # Only bottleneck ODE is active (last encoder level)
            self.active_ode_indices = [self.num_stages]
        else:
            # Start with the bottleneck and work backwards
            start_idx = max(0, self.total_odes - self.num_active_odes)
            self.active_ode_indices = list(range(start_idx, self.total_odes))
        
        print("Initializing PointPreservingSpUNetODEWrapper")
        print(f"Total possible ODEs: {self.total_odes}")
        print(f"Number of active ODEs: {self.num_active_odes}")
        print(f"Active ODE indices: {self.active_ode_indices}")
        
        # Modify the process_features_with_ode method of the model to respect active ODEs
        self._patch_ode_processing()
    
    def _patch_ode_processing(self):
        """
        Monkey patch the process_features_with_ode method to respect active ODEs
        """
        original_process_features = self.model.process_features_with_ode
        active_ode_indices = self.active_ode_indices
        
        def patched_process_features(encoded_features, obs_time, target_time=None):
            """
            Modified to only process features with active ODEs.
            """
            num_levels = len(encoded_features)
            processed_features = []
            
            # Determine B (points) and T (time steps)
            first_tensor = encoded_features[0]
            batch_size = first_tensor.batch_size
            T = len(obs_time)
            B = batch_size // T
            
            # Process each feature level
            for level, feature_tensor in enumerate(encoded_features):
                if level in active_ode_indices:
                    # Process with ODE if this level is active
                    ode_index = active_ode_indices.index(level)
                    ode_model = self.model.ode_models[ode_index]
                    
                    # Reshape features for ODE processing
                    reshaped_features = self.model.reshape_for_ode_processing(feature_tensor, B, T)
                    
                    # Process with ODE model
                    if target_time is not None:
                        processed_features_level = ode_model.extrapolate(
                            reshaped_features, obs_time, target_time
                        )
                    else:
                        processed_features_level = ode_model.reconstruct(
                            reshaped_features, obs_time
                        )
                    
                    # Reshape back to sparse tensor
                    processed_tensor = self.model.reshape_for_parallel_decoding(
                        processed_features_level, feature_tensor, B, 
                        len(obs_time) if target_time is None else len(obs_time) + len(target_time)
                    )
                else:
                    # Skip ODE processing for inactive levels
                    processed_tensor = feature_tensor
                
                processed_features.append(processed_tensor)
            
            return processed_features
        
        # Replace the original method with our patched version
        self.model.process_features_with_ode = patched_process_features
    
    def freeze_encoder_decoder(self):
        """
        Freezes the encoder and decoder components of the SpUNet model while
        keeping the ODE components trainable. This function should be called
        after the warmup phase to transition to ODE-focused training.
        
        Returns:
            dict: Information about the number of parameters before and after freezing
        """
        # Count trainable parameters before freezing
        params_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Freeze SpUNet encoder parameters
        print("Freezing encoder parameters...")
        for param in self.model.conv_input.parameters():
            param.requires_grad = False
        
        for down_layer in self.model.down:
            for param in down_layer.parameters():
                param.requires_grad = False
        
        for enc_layer in self.model.enc:
            for param in enc_layer.parameters():
                param.requires_grad = False
        
        # Freeze SpUNet decoder parameters
        print("Freezing decoder parameters...")
        for up_layer in self.model.up:
            for param in up_layer.parameters():
                param.requires_grad = False
        
        for dec_layer in self.model.dec:
            for param in dec_layer.parameters():
                param.requires_grad = False
        
        # Freeze final layer
        for param in self.model.final.parameters():
            param.requires_grad = False
        
        # Make sure active ODE models remain trainable
        print("Ensuring active ODE models remain trainable...")
        for i, level in enumerate(self.active_ode_indices):
            ode_model = self.model.ode_models[i]
            for param in ode_model.parameters():
                param.requires_grad = True
            print(f"  ODE Model {i} (level {level}): {sum(p.numel() for p in ode_model.parameters() if p.requires_grad)} trainable parameters")
        
        # Count trainable parameters after freezing
        params_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Print summary
        print(f"Freezing complete!")
        print(f"Trainable parameters before: {params_before:,}")
        print(f"Trainable parameters after: {params_after:,}")
        print(f"Reduced by: {params_before - params_after:,} parameters ({100 * (1 - params_after / params_before):.2f}% reduction)")
        
        # Also add a flag to mark that the model has frozen components
        self.encoder_decoder_frozen = True
        
        return {
            "params_before": params_before,
            "params_after": params_after,
            "reduction": params_before - params_after,
            "reduction_percent": 100 * (1 - params_after / params_before)
        }
    
    def forward(self, obs_traj, target_traj, full_time):
        """Forward pass through the model for training."""
        # Extract time information
        full_time = full_time[0]  # Remove batch dimension if present
        obs_time = full_time[:obs_traj.shape[1]]
        target_time = full_time[obs_traj.shape[1]:]
        
        # Call the inner model
        pred_traj = self.model(obs_traj, obs_time, target_time)
        
        # Split for computing losses
        pred_obs = pred_traj[:, :obs_traj.shape[1]]
        pred_target = pred_traj[:, obs_traj.shape[1]:]
        
        # Calculate losses
        recon_loss = F.l1_loss(pred_obs, obs_traj)
        pred_loss = F.l1_loss(pred_target, target_traj)
        
        # Placeholder for KL loss and regularization loss
        kl_loss = torch.tensor(0.0, device=obs_traj.device)
        reg_loss = torch.tensor(0.0, device=obs_traj.device)
        
        # Compute KL loss from active ODE models
        if self.variational_inference:
            for i, level in enumerate(self.active_ode_indices):
                ode_model = self.model.ode_models[i]
                if hasattr(ode_model, 'kl_loss'):
                    kl_loss = kl_loss + ode_model.kl_loss
        
        # Compute regularization loss from active ODE models
        for i, level in enumerate(self.active_ode_indices):
            ode_model = self.model.ode_models[i]
            if hasattr(ode_model, 'regularization_loss'):
                reg_loss = reg_loss + ode_model.regularization_loss
        
        # Total loss
        loss = recon_loss + pred_loss
        if self.variational_inference:
            loss = loss + self.reg_weight * (kl_loss + reg_loss)
        else:
            loss = loss + self.reg_weight * reg_loss
        
        return loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_traj
    
    def extrapolate(self, obs_traj, obs_time, extrapolate_time):
        """Extrapolate trajectory into the future."""
        return self.model.extrapolate(obs_traj, obs_time, extrapolate_time)
    
    def reconstruct(self, obs_traj, obs_time):
        """Reconstruct the observed trajectory."""
        return self.model.reconstruct(obs_traj, obs_time)
    
    def transformer_only_reconstruction(self, obs_traj):
        """
        Perform reconstruction using only the encoder-decoder backbone without ODE.
        Used for warmup training phase.
        """
        return self.model.transformer_only_reconstruction(obs_traj)
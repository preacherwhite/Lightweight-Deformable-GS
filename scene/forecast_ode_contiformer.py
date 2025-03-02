import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchcde
from physiopro.network.contiformer import ContiFormer, AttrDict

class ContiformerLatentODEWrapper(nn.Module):
    def __init__(self, latent_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 ode_nhidden, decoder_nhidden, obs_dim, noise_std, ode_layers):
        super(ContiformerLatentODEWrapper, self).__init__()
        print("initialzing ContiformerLatentOdeWrapper")
        print("latent_dim: ", latent_dim)
        print("d_model: ", d_model)
        print("nhead: ", nhead)
        print("num_encoder_layers: ", num_encoder_layers)
        print("num_decoder_layers: ", num_decoder_layers)
        print("ode_nhidden: ", ode_nhidden)
        print("decoder_nhidden: ", decoder_nhidden)
        print("obs_dim: ", obs_dim)
        print("noise_std: ", noise_std)
        print("ode_layers: ", ode_layers)

        
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.obs_dim = obs_dim
        self.d_model = d_model
        
        # Always use deterministic approach as in original Contiformer
        self.variational_inference = False  # Override to follow Contiformer algorithm
        
        # Input projection for observations
        self.input_projection = nn.Linear(obs_dim, d_model)
        
        # Projection to latent space (deterministic)
        self.latent_projection = nn.Linear(d_model, latent_dim)
        
        # Initialize the ContiFormer model
        # ContiFormer handles multiple encoder layers and ODE integration internally
        self.contiformer = ContiFormer(
            input_size=None,  # We'll handle input projection separately
            d_model=d_model,
            d_inner=ode_nhidden,
            n_layers=num_encoder_layers,
            n_head=nhead,
            d_k=d_model // nhead,
            d_v=d_model // nhead,
            dropout=0.1,
            actfn_ode="softplus",
            layer_type_ode="concat",
            zero_init_ode=True,
            atol_ode=0.1,
            rtol_ode=0.1,
            method_ode="rk4",
            linear_type_ode="before",
            interpolate_ode="linear",
            add_pe=True,  # We'll handle positional encoding separately
            normalize_before=True,
            nlinspace=1
        )
        
        # Decoder to map latent state to observation space
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, decoder_nhidden))
        decoder_layers.append(nn.ReLU())
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(decoder_nhidden, decoder_nhidden))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_nhidden, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, obs_traj, target_traj, full_time):
        """
        Args:
            obs_traj: Tensor (B, obs_length, obs_dim)
            target_traj: Tensor (B, target_length, obs_dim)
            full_time: 1D tensor of length window_length (obs_length + target_length)
        Returns:
            loss and predicted full trajectory (B, window_length, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Ensure time is properly formatted
        if full_time.dim() == 1:
            full_time = full_time.unsqueeze(0).expand(B, -1)
            
        # Process input through linear projection
        x = self.input_projection(obs_traj)  # (B, T_obs, d_model)
        
        # Create a mask for ContiFormer - all visible
        mask = torch.ones(B, full_time.size(1), 1).to(device).bool()
        
        # Run the full sequence through ContiFormer
        # First, we need to pad the observations to match the full time length
        pad_length = full_time.size(1) - x.size(1)
        
        # Create a zero tensor of the right shape
        padding = torch.zeros(B, pad_length, self.d_model).to(device)
        
        # Concatenate the real observations with the padding
        padded_x = torch.cat([x, padding], dim=1)
        
        # Process through ContiFormer
        # print("passing through contiformer")
        # print("padded_x: ", padded_x.shape)
        # print("full_time: ", full_time.shape)
        # print("mask: ", mask.shape)
        # extend full time to have batch size dimension
        full_time = full_time.expand(B, -1)
        encoded_output, _ = self.contiformer(padded_x, full_time, mask)
        # Project to latent space
        latent = self.latent_projection(encoded_output)
        
        # Decode latent to observation space
        pred_x = self.decoder(latent)  # (B, window_length, obs_dim)
        # Split predictions into observed and target parts
        pred_x_obs = pred_x[:, :T_obs, :]
        pred_x_target = pred_x[:, T_obs:, :]
        
        # Simple MSE losses following Contiformer's deterministic approach
        recon_loss = F.mse_loss(pred_x_obs, obs_traj)
        pred_loss = F.mse_loss(pred_x_target, target_traj)
        kl_loss = torch.tensor(0.0, device=device)  # No KL loss in deterministic model
        reg_loss = torch.tensor(0.0, device=device)  # No additional regularization
        loss = recon_loss + pred_loss
        
        return loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_x
    
    def extrapolate(self, obs_traj, obs_time, extrapolate_time):
        """
        Deterministically extrapolates the trajectory.
        Args:
            obs_traj: (B, obs_length, obs_dim)
            obs_time: 1D tensor of length obs_length
            extrapolate_time: 1D tensor for prediction horizon
        Returns:
            Full predicted trajectory (B, obs_length + prediction_length, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Ensure time is in the right format
        if obs_time.dim() == 1:
            obs_time = obs_time.unsqueeze(0).expand(B, -1)
        if extrapolate_time.dim() == 1:
            extrapolate_time = extrapolate_time.unsqueeze(0).expand(B, -1)
            
        # Combine all time points
        full_time = torch.cat([obs_time, extrapolate_time], dim=1)
        
        # Process input through linear projection
        x = self.input_projection(obs_traj)
        
        # Create a mask for ContiFormer - all visible
        mask = torch.ones(B, full_time.size(1), 1).to(device).bool()
        
        # We need to provide an input tensor that matches the full_time dimensions
        # The model will handle extrapolation internally
        pad_length = extrapolate_time.size(1)
        padding = torch.zeros(B, pad_length, self.d_model).to(device)
        padded_x = torch.cat([x, padding], dim=1)
        
        # Process through ContiFormer
        encoded_output, _ = self.contiformer(padded_x, full_time, mask)
        
        # Project to latent space
        latent = self.latent_projection(encoded_output)
        
        # Decode latent to observation space
        pred_x = self.decoder(latent)
        return pred_x
    
    def reconstruct(self, obs_traj, obs_time):
        """
        Reconstructs the trajectory for the input time.
        Args:
            obs_traj: (B, obs_length, obs_dim)
            obs_time: 1D tensor of length obs_length    
        Returns:
            pred_x: (B, obs_length, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Ensure time is in the right format
        if obs_time.dim() == 1:
            obs_time = obs_time.unsqueeze(0).expand(B, -1)
            
        # Process input through linear projection
        x = self.input_projection(obs_traj)
        
        # Create a mask for ContiFormer
        mask = torch.ones(B, obs_time.size(1), 1).to(device).bool()
        
        # Process through ContiFormer
        encoded_output, _ = self.contiformer(x, obs_time, mask)
        
        # Project to latent space
        latent = self.latent_projection(encoded_output)
        
        # Decode latent to observation space
        pred_x = self.decoder(latent)
        
        return pred_x
    
    def transformer_only_reconstruction(self, obs_traj):
        """
        Uses only the ContiFormer to produce reconstruction for the observation.
        Args:
            obs_traj: Tensor (B, obs_length, obs_dim)
        Returns:
            recon_loss: Scalar reconstruction loss
            pred_x_first: Predicted first frame (B, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Create artificial time points if not provided
        time_steps = torch.arange(T_obs, device=device).float().unsqueeze(0).expand(B, -1)
        
        # Process input through linear projection
        x = self.input_projection(obs_traj)
        
        # Create a mask for ContiFormer
        mask = torch.ones(B, time_steps.size(1), 1).to(device).bool()
        
        # Process through ContiFormer
        encoded_output, _ = self.contiformer(x, time_steps, mask)
        
        # Project to latent space
        latent = self.latent_projection(encoded_output)
        
        # Decode latent to observations
        pred_x = self.decoder(latent)
        
        # We'll focus on the first frame for consistency with the original API
        pred_x_first = pred_x[:, 0, :]
        
        # Compute MSE loss
        total_loss = F.mse_loss(pred_x, obs_traj)
        
        return total_loss, pred_x_first
        
    def compute_derivative_regularization(self, trajectory, t):
        """Compatibility method for regularization interface"""
        # Return placeholder values since regularization is handled differently in ContiFormer
        reg_loss = torch.tensor(0.0, device=trajectory.device)
        derivatives = torch.zeros_like(trajectory)
        return reg_loss, derivatives
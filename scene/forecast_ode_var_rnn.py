import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchdiffeq import odeint
import torch.nn.functional as F
import torchode as to

USE_ADAPTIVE = True

def log_normal_pdf(x, mean, logvar):
    const = torch.log(torch.tensor(2. * np.pi, device=x.device))
    return -0.5 * (const + logvar + (x - mean) ** 2 / torch.exp(logvar))

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    kl = lv2 - lv1 + (v1 + (mu1 - mu2) ** 2) / (2 * v2) - 0.5
    return kl

# --------------------------
# ODE Function
# --------------------------
class ODEFunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20, num_layers=2):
        super(ODEFunc, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.middle_layers = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(num_layers - 2)])
        self.fc_final = nn.Linear(nhidden, latent_dim)

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.relu(out)
        for layer in self.middle_layers:
            out = layer(out)
            out = self.relu(out)
        out = self.fc_final(out)
        return out

# --------------------------
# GRU Cell for Hidden State Updates
# --------------------------
class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
    
    def forward(self, x, h):
        return self.gru_cell(x, h)

# --------------------------
# ODE-RNN Encoder for Latent ODE
# --------------------------
class ODERNNEncoder(nn.Module):
    def __init__(self, latent_dim, rec_dim, obs_dim, variational_inference):
        super(ODERNNEncoder, self).__init__()
        self.rec_dim = rec_dim
        self.latent_dim = latent_dim
        self.variational_inference = variational_inference
        
        # Input embedding
        self.input_embedding = nn.Linear(obs_dim, rec_dim)
        
        # GRU cell for updating hidden state
        self.gru_cell = GRUCell(rec_dim, rec_dim)
        
        # ODE function for evolving hidden state
        self.ode_func = ODEFunc(rec_dim, rec_dim*2, num_layers=2)
        
        # Projection to latent space
        if variational_inference:
            self.latent_projection = nn.Linear(rec_dim, 2 * latent_dim)  # For mean and logvar
        else:
            self.latent_projection = nn.Linear(rec_dim, latent_dim)

    def evolve_hidden_state(self, h0, time_points, device):
        """Evolve hidden state using ODE"""
        if len(time_points) <= 1:
            return h0.unsqueeze(0)
            
        if USE_ADAPTIVE:
            return odeint(self.ode_func, h0, time_points, rtol=1e-1, atol=1e-1)
        else:
            time_span = time_points[-1] - time_points[0]
            step_size = time_span / 20
            return odeint(self.ode_func, h0, time_points, method='rk4', options=dict(step_size=step_size))
            
    def forward(self, obs_traj, time_points):
        """
        Encode trajectory into latent initial state z0
        Args:
            obs_traj: (B, T, obs_dim) - Observed trajectory
            time_points: (T) - Time points of observations
        Returns:
            z0_mean, z0_logvar: Initial latent state parameters
        """
        B, T, _ = obs_traj.size()
        device = obs_traj.device
        
        # Initialize hidden state
        h = torch.zeros(B, self.rec_dim, device=device)
        
        # Process observed trajectory with ODE-RNN
        for t in range(T):
            # Get current observation and embed it
            x_t = obs_traj[:, t, :]
            x_embedded = self.input_embedding(x_t)
            
            # If not the first step, evolve h from previous time to current time
            if t > 0:
                dt = time_points[t] - time_points[t-1]
                if dt > 0:
                    t_span = torch.tensor([0, dt], device=device)
                    h = self.evolve_hidden_state(h, t_span, device)[-1]
            
            # Update hidden state with GRU
            h = self.gru_cell(x_embedded, h)
        
        # Project to latent space
        if self.variational_inference:
            z0_params = self.latent_projection(h)
            z0_mean, z0_logvar = torch.chunk(z0_params, 2, dim=-1)
            return z0_mean, z0_logvar
        else:
            z0 = self.latent_projection(h)
            return z0

# --------------------------
# Latent ODE Model with ODE-RNN Encoder
# --------------------------
class LatentODERNN(nn.Module):
    def __init__(self, latent_dim, rec_dim, num_decoder_layers,
                 ode_nhidden, decoder_nhidden, obs_dim, noise_std, ode_layers, 
                 reg_weight, variational_inference, use_torchode):
        super(LatentODERNN, self).__init__()
        print("Initializing Latent ODE with ODE-RNN Encoder")
        print("latent_dim:", latent_dim)
        print("rec_dim:", rec_dim)
        print("ode_nhidden:", ode_nhidden)
        print("decoder_nhidden:", decoder_nhidden)
        print("obs_dim:", obs_dim)
        print("noise_std:", noise_std)
        print("ode_layers:", ode_layers)
        print("reg_weight:", reg_weight)
        print("variational_inference:", variational_inference)
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.reg_weight = reg_weight
        self.variational_inference = variational_inference
        self.noise_std = noise_std
        
        # ODE-RNN Encoder
        self.encoder = ODERNNEncoder(latent_dim, rec_dim, obs_dim, variational_inference)
        
        # ODE function for latent dynamics
        self.func = ODEFunc(latent_dim, ode_nhidden, num_layers=ode_layers)
        
        # Setup ODE solver
        if use_torchode:
            term = to.ODETerm(self.func)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.IntegralController(atol=1e-1, rtol=1e-1, term=term)
            self.adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
        self.use_torchode = use_torchode
        
        # Decoder to map latent state to observation space
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, decoder_nhidden))
        decoder_layers.append(nn.ReLU())
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(decoder_nhidden, decoder_nhidden))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_nhidden, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _filter_unique_times(self, times):
        """
        Filter out duplicate timestamps and return unique, strictly increasing times along with mapping indices.
        Args:
            times: 1D tensor of timestamps
        Returns:
            unique_times: 1D tensor of unique, strictly increasing timestamps
            mapping_indices: List mapping original indices to unique time indices
        """
        unique_times = torch.unique(times)
        
        mapping_indices = []
        j = 0  # Index into unique_times
        for i in range(len(times)):
            while j < len(unique_times) and times[i] > unique_times[j]:
                j += 1
            mapping_indices.append(j if j < len(unique_times) else j - 1)
        
        return unique_times, mapping_indices

    def _reconstruct_trajectory(self, pred_z_unique, mapping_indices, original_length):
        """
        Reconstruct the full trajectory by duplicating predictions for duplicate timestamps.
        Args:
            pred_z_unique: (unique_length, B, latent_dim) tensor from ODE solver
            mapping_indices: List mapping original indices to unique time indices
            original_length: Original length of the time sequence
        Returns:
            pred_z: (original_length, B, latent_dim) tensor with duplicates reinserted
        """
        _, B, latent_dim = pred_z_unique.shape
        pred_z = torch.zeros(original_length, B, latent_dim, device=pred_z_unique.device)
        for i, idx in enumerate(mapping_indices):
            pred_z[i] = pred_z_unique[idx]
        return pred_z
        
    def forward(self, obs_traj, target_traj, full_time):
        """
        Forward pass through the model
        Args:
            obs_traj: (B, obs_length, obs_dim) - Observed trajectory
            target_traj: (B, target_length, obs_dim) - Target trajectory
            full_time: (B, obs_length + target_length) - Full time sequence
        Returns:
            loss: Total loss
            recon_loss: Reconstruction loss
            pred_loss: Prediction loss
            kl_loss: KL divergence loss (if variational)
            reg_loss: Regularization loss
            pred_x: (B, obs_length + target_length, obs_dim) - Predicted trajectory
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        full_time = full_time[0]  # Remove batch dimension if present
        
        # Encode trajectory into latent initial state z0
        obs_time = full_time[:T_obs]
        
        if self.variational_inference:
            z0_mean, z0_logvar = self.encoder(obs_traj, obs_time)
            
            # Sample z0 from variational distribution
            epsilon = torch.randn_like(z0_mean)
            z0 = z0_mean + epsilon * torch.exp(0.5 * z0_logvar)
        else:
            z0 = self.encoder(obs_traj, obs_time)
        
        # Filter unique times for ODE integration
        unique_times, mapping_indices = self._filter_unique_times(full_time)
        
        # Integrate ODE to get latent trajectory
        if USE_ADAPTIVE:
            if self.use_torchode:
                problem = to.InitialValueProblem(y0=z0, t_eval=unique_times.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                pred_z_unique = torch.transpose(sol.ys, 0, 1)  # (unique_length, B, latent_dim)
            else:
                pred_z_unique = odeint(self.func, z0, unique_times, rtol=1e-1, atol=1e-1)  # (unique_length, B, latent_dim)
        else:
            time_span = unique_times[-1] - unique_times[0]
            step_size = time_span / 20
            pred_z_unique = odeint(self.func, z0, unique_times, method='rk4', options=dict(step_size=step_size))
        
        # Calculate regularization loss if needed
        reg_loss = torch.tensor(0.0, device=device)
        if self.reg_weight > 0:
            detached_states = [state.detach() for state in pred_z_unique]
            derivatives = []
            for i, time in enumerate(unique_times):
                derivative = self.func(time, detached_states[i])
                derivatives.append(derivative)
            derivatives = torch.stack(derivatives)
            if len(derivatives) > 1:
                derivative_changes = torch.diff(derivatives, dim=0)
                reg_loss = torch.mean(torch.square(derivative_changes)) * self.reg_weight
        
        # Reconstruct full trajectory with duplicate time points
        pred_z = self._reconstruct_trajectory(pred_z_unique, mapping_indices, len(full_time))
        pred_z = pred_z.permute(1, 0, 2)  # (B, full_length, latent_dim)
        
        # Decode to observation space
        pred_x = self.decoder(pred_z)  # (B, full_length, obs_dim)
        
        # Split into observed and target parts for loss calculation
        pred_x_obs = pred_x[:, :T_obs, :]
        pred_x_target = pred_x[:, T_obs:, :]
        
        # Calculate losses
        if self.variational_inference:
            noise_logvar = 2 * torch.log(torch.tensor(self.noise_std, device=device))
            
            # Reconstruction loss (observed part)
            logpx_obs = log_normal_pdf(obs_traj, pred_x_obs, noise_logvar)
            recon_loss = -logpx_obs.sum(dim=2).sum(dim=1).mean()
            
            # Prediction loss (target part)
            logpx_target = log_normal_pdf(target_traj, pred_x_target, noise_logvar)
            pred_loss = -logpx_target.sum(dim=2).sum(dim=1).mean()
            
            # KL divergence loss
            kl_loss = normal_kl(z0_mean, z0_logvar,
                               torch.zeros_like(z0_mean),
                               torch.zeros_like(z0_logvar)).sum(dim=1).mean()
            
            # Total loss
            loss = recon_loss + pred_loss + kl_loss + reg_loss
        else:
            # Deterministic losses using MSE
            recon_loss = F.mse_loss(pred_x_obs, obs_traj)
            pred_loss = F.mse_loss(pred_x_target, target_traj)
            kl_loss = torch.tensor(0.0, device=device)
            loss = recon_loss + pred_loss + reg_loss
            
        return loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_x
    
    def extrapolate(self, obs_traj, obs_time, extrapolate_time):
        """
        Extrapolate trajectory into the future
        Args:
            obs_traj: (B, obs_length, obs_dim) - Observed trajectory
            obs_time: (obs_length) - Time points of observations
            extrapolate_time: (extrapolate_length) - Future time points to predict
        Returns:
            pred_x: (B, obs_length + extrapolate_length, obs_dim) - Predicted trajectory
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Encode trajectory into latent initial state z0
        if self.variational_inference:
            z0_mean, z0_logvar = self.encoder(obs_traj, obs_time)
            z0 = z0_mean  # Use mean for deterministic prediction
        else:
            z0 = self.encoder(obs_traj, obs_time)
        
        # Concatenate observation and extrapolation times
        full_time = torch.cat([obs_time, extrapolate_time])
        
        # Filter unique times for ODE integration
        unique_times, mapping_indices = self._filter_unique_times(full_time)
        
        # Integrate ODE to get latent trajectory
        if USE_ADAPTIVE:
            if self.use_torchode:
                problem = to.InitialValueProblem(y0=z0, t_eval=unique_times.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                pred_z_unique = torch.transpose(sol.ys, 0, 1)  # (unique_length, B, latent_dim)
            else:
                pred_z_unique = odeint(self.func, z0, unique_times, rtol=1e-1, atol=1e-1)
        else:
            time_span = unique_times[-1] - unique_times[0]
            step_size = time_span / 20
            pred_z_unique = odeint(self.func, z0, unique_times, method='rk4', options=dict(step_size=step_size))
        
        # Reconstruct full trajectory with duplicate time points
        pred_z = self._reconstruct_trajectory(pred_z_unique, mapping_indices, len(full_time))
        pred_z = pred_z.permute(1, 0, 2)  # (B, full_length, latent_dim)
        
        # Decode to observation space
        pred_x = self.decoder(pred_z)  # (B, full_length, obs_dim)
        
        return pred_x
    
    def reconstruct(self, obs_traj, obs_time):
        """
        Reconstruct the observed trajectory
        Args:
            obs_traj: (B, obs_length, obs_dim) - Observed trajectory
            obs_time: (obs_length) - Time points of observations
        Returns:
            pred_x: (B, obs_length, obs_dim) - Reconstructed trajectory
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Encode trajectory into latent initial state z0
        if self.variational_inference:
            z0_mean, z0_logvar = self.encoder(obs_traj, obs_time)
            z0 = z0_mean  # Use mean for deterministic prediction
        else:
            z0 = self.encoder(obs_traj, obs_time)
        
        # Filter unique times for ODE integration
        unique_times, mapping_indices = self._filter_unique_times(obs_time)
        
        # Integrate ODE to get latent trajectory
        if USE_ADAPTIVE:
            if self.use_torchode:
                problem = to.InitialValueProblem(y0=z0, t_eval=unique_times.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                pred_z_unique = torch.transpose(sol.ys, 0, 1)  # (unique_length, B, latent_dim)
            else:
                pred_z_unique = odeint(self.func, z0, unique_times, rtol=1e-1, atol=1e-1)
        else:
            time_span = unique_times[-1] - unique_times[0]
            step_size = time_span / 20
            pred_z_unique = odeint(self.func, z0, unique_times, method='rk4', options=dict(step_size=step_size))
        
        # Reconstruct full trajectory with duplicate time points
        pred_z = self._reconstruct_trajectory(pred_z_unique, mapping_indices, len(obs_time))
        pred_z = pred_z.permute(1, 0, 2)  # (B, full_length, latent_dim)
        
        # Decode to observation space
        pred_x = self.decoder(pred_z)  # (B, obs_length, obs_dim)
        
        return pred_x
    
    def transformer_only_reconstruction(self, obs_traj):
        """
        Reconstruct just using the first frame (adapter for interface compatibility)
        Args:
            obs_traj: (B, obs_length, obs_dim) - Observed trajectory
        Returns:
            loss: Reconstruction loss for the first frame
            pred_x_first: (B, obs_dim) - Predicted first frame
        """
        B, _, _ = obs_traj.size()
        device = obs_traj.device
        
        # For compatibility, we'll just reconstruct the first frame
        first_frame = obs_traj[:, 0:1, :]
        dummy_time = torch.zeros(1, device=device)
        
        # Get latent representation of first frame
        if self.variational_inference:
            z0_mean, z0_logvar = self.encoder(first_frame, dummy_time)
            epsilon = torch.randn_like(z0_mean)
            z0 = z0_mean + epsilon * torch.exp(0.5 * z0_logvar)
            kl_loss = normal_kl(z0_mean, z0_logvar,
                               torch.zeros_like(z0_mean),
                               torch.zeros_like(z0_logvar)).sum(dim=1).mean()
        else:
            z0 = self.encoder(first_frame, dummy_time)
            kl_loss = torch.tensor(0.0, device=device)
        
        # Decode first frame
        pred_x_first = self.decoder(z0)
        
        # Compute reconstruction loss
        if self.variational_inference:
            noise_logvar = 2 * torch.log(torch.tensor(self.noise_std, device=device))
            logpx = log_normal_pdf(obs_traj[:, 0, :], pred_x_first, noise_logvar)
            recon_loss = -logpx.sum(dim=1).mean()
            loss = recon_loss + kl_loss
        else:
            recon_loss = F.mse_loss(pred_x_first, obs_traj[:, 0, :])
            loss = recon_loss
        
        return loss, pred_x_first
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchdiffeq import odeint
import torch.nn.functional as F
import torchode as to

USE_ADAPTIVE = True

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
# ODE-RNN Model with TransformerODE Interface
# --------------------------
class ODE_RNN_Model(nn.Module):
    def __init__(self, latent_dim, d_model, num_decoder_layers,
                 ode_nhidden, decoder_nhidden, obs_dim, ode_layers, reg_weight, 
                 use_torchode):
        super(ODE_RNN_Model, self).__init__()
        print("Initializing ODE-RNN Model")
        print("latent_dim: ", latent_dim)
        print("d_model: ", d_model)
        print("obs_dim: ", obs_dim)
        print("ode_nhidden: ", ode_nhidden)
        print("decoder_nhidden: ", decoder_nhidden)
        print("ode_layers: ", ode_layers)
        print("reg_weight: ", reg_weight)
        
        # Note: We're ignoring certain parameters that don't apply to ODE-RNN:
        # - variational_inference (we're making a deterministic model)
        # - nhead, num_encoder_layers (transformer-specific)
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.reg_weight = reg_weight
        
        # ODE-RNN consists of:
        # 1. Input embedding 
        self.input_embedding = nn.Linear(obs_dim, latent_dim)
        
        # 2. GRU cell for updating hidden state when new observations arrive
        self.gru_cell = GRUCell(latent_dim, latent_dim)
        
        # 3. ODE function for evolving hidden state between observations
        self.func = ODEFunc(latent_dim, ode_nhidden, num_layers=ode_layers)
        
        # Setup ODE solver
        if use_torchode:
            term = to.ODETerm(self.func)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.IntegralController(atol=1e-3, rtol=1e-3, term=term)
            self.adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
        self.use_torchode = use_torchode
        
        # 4. Decoder to map latent state to observation space
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
        # Convert to CPU numpy for unique operation, then back to tensor
        unique_times = torch.unique(times)
        
        # Create mapping: for each original time, find the index in unique_times it corresponds to
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
    
    def evolve_hidden_state(self, h0, time_points):
        """
        Evolve the hidden state using the ODE from time_points[0] to time_points[-1]
        """
        B = h0.size(0)
        if USE_ADAPTIVE:
            if self.use_torchode:
                problem = to.InitialValueProblem(y0=h0, t_eval=time_points.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                return torch.transpose(sol.ys, 0, 1)
            else:
                return odeint(self.func, h0, time_points, rtol=1e-1, atol=1e-1)
        else:
            time_span = time_points[-1] - time_points[0]
            step_size = time_span / 20
            return odeint(self.func, h0, time_points, method='rk4', options=dict(step_size=step_size))
    
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
        full_time = full_time[0]  # Remove batch dimension if present
        
        # Initialize hidden state
        h = torch.zeros(B, self.latent_dim, device=device)
        all_h = []
        
        # Process observed trajectory with ODE-RNN
        for t in range(T_obs):
            # 1. Get current observation and embed it
            x_t = obs_traj[:, t, :]
            x_embedded = self.input_embedding(x_t)
            
            # 2. If not the first step, evolve h from previous time to current time
            if t > 0:
                dt = full_time[t] - full_time[t-1]
                if dt > 0:
                    t_span = torch.tensor([0, dt], device=device)
                    h = self.evolve_hidden_state(h, t_span)[-1]
            
            # 3. Update hidden state with GRU
            h = self.gru_cell(x_embedded, h)
            all_h.append(h)
            
        # Evolve the hidden state through target time points
        target_length = len(full_time) - T_obs
        
        if target_length > 0:
            # Filter unique times for evolving the ODE
            target_times = full_time[T_obs:]
            unique_target_times, mapping_indices = self._filter_unique_times(target_times)
            
            # Time span from last observation to each unique target time
            # time_spans = torch.cat([torch.tensor([0.0], device=device), 
            #                       unique_target_times - full_time[T_obs-1]])
            time_spans = unique_target_times - full_time[T_obs-1]
            # Evolve h through all unique target times
            #print("time_spans: ", time_spans)
            evolved_h = self.evolve_hidden_state(h, time_spans)
            
            # Reconstruct full trajectory of hidden states using mapping
            #target_h = evolved_h[1:]  # Skip the initial state (t=0)
            target_h = evolved_h
            if len(target_h) > 0:
                target_h_reconstructed = self._reconstruct_trajectory(
                    target_h, mapping_indices, target_length)
                target_h_list = [target_h_reconstructed[i] for i in range(target_h_reconstructed.shape[0])]
                all_h.extend(target_h_list)
        
        # Convert list of hidden states to tensor
        all_h = torch.stack(all_h, dim=1)  # (B, T_full, latent_dim)
        
        # Decode to observation space
        pred_x = self.decoder(all_h)  # (B, T_full, obs_dim)
        
        # Calculate losses
        pred_x_obs = pred_x[:, :T_obs, :]
        pred_x_target = pred_x[:, T_obs:, :]
        
        recon_loss = F.mse_loss(pred_x_obs, obs_traj)
        pred_loss = F.mse_loss(pred_x_target, target_traj)
        
        reg_loss = torch.tensor(0.0, device=device)
        if self.reg_weight > 0:
            # Sample a few trajectory points to enforce regularity in ODE function
            sample_indices = torch.randint(0, all_h.size(1), (min(10, all_h.size(1)),))
            sample_h = all_h[:, sample_indices, :]
            sample_h_detached = sample_h.detach()
            
            derivatives = []
            dummy_time = torch.zeros(1, device=device)
            for i in range(sample_h_detached.size(1)):
                derivative = self.func(dummy_time, sample_h_detached[:, i, :])
                derivatives.append(derivative)
            
            derivatives = torch.stack(derivatives, dim=1)
            if derivatives.size(1) > 1:
                derivative_changes = torch.diff(derivatives, dim=1)
                reg_loss = torch.mean(torch.square(derivative_changes)) * self.reg_weight
        
        # Placeholder values for interface compatibility
        kl_loss = torch.tensor(0.0, device=device)
        
        loss = recon_loss + pred_loss + reg_loss
        
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
        
        # Initialize hidden state
        h = torch.zeros(B, self.latent_dim, device=device)
        all_h = []
        
        # Process observed trajectory with ODE-RNN
        for t in range(T_obs):
            # 1. Get current observation and embed it
            x_t = obs_traj[:, t, :]
            x_embedded = self.input_embedding(x_t)
            
            # 2. If not the first step, evolve h from previous time to current time
            if t > 0:
                dt = obs_time[t] - obs_time[t-1]
                if dt > 0:
                    t_span = torch.tensor([0, dt], device=device)
                    h = self.evolve_hidden_state(h, t_span)[-1]
            
            # 3. Update hidden state with GRU
            h = self.gru_cell(x_embedded, h)
            all_h.append(h)
            
        # Evolve the hidden state through extrapolation time points
        if len(extrapolate_time) > 0:
            # Filter unique times for evolving the ODE
            unique_extrap_times, mapping_indices = self._filter_unique_times(extrapolate_time)
            
            # Time span from last observation to each unique extrapolation time
            # time_spans = torch.cat([torch.tensor([0.0], device=device), 
            #                       unique_extrap_times - obs_time[-1]])
            time_spans = unique_extrap_times - obs_time[-1]
            # Evolve h through all unique extrapolation times
            evolved_h = self.evolve_hidden_state(h, time_spans)
            
            # Reconstruct full trajectory of hidden states using mapping
            #extrap_h = evolved_h[1:]  # Skip the initial state (t=0)
            extrap_h = evolved_h
            if len(extrap_h) > 0:
                extrap_h_reconstructed = self._reconstruct_trajectory(
                    extrap_h, mapping_indices, len(extrapolate_time))
                extrap_h_list = [extrap_h_reconstructed[i] for i in range(extrap_h_reconstructed.shape[0])]
                all_h.extend(extrap_h_list)
        
        # Convert list of hidden states to tensor
        all_h = torch.stack(all_h, dim=1)  # (B, T_full, latent_dim)
        
        # Decode to observation space
        pred_x = self.decoder(all_h)  # (B, T_full, obs_dim)
        
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
        
        # Initialize hidden state
        h = torch.zeros(B, self.latent_dim, device=device)
        all_h = []
        
        # Process observed trajectory with ODE-RNN
        for t in range(T_obs):
            # 1. Get current observation and embed it
            x_t = obs_traj[:, t, :]
            x_embedded = self.input_embedding(x_t)
            
            # 2. If not the first step, evolve h from previous time to current time
            if t > 0:
                dt = obs_time[t] - obs_time[t-1]
                if dt > 0:
                    t_span = torch.tensor([0, dt], device=device)
                    h = self.evolve_hidden_state(h, t_span)[-1]
            
            # 3. Update hidden state with GRU
            h = self.gru_cell(x_embedded, h)
            all_h.append(h)
        
        # Convert list of hidden states to tensor
        all_h = torch.stack(all_h, dim=1)  # (B, T_obs, latent_dim)
        
        # Decode to observation space
        pred_x = self.decoder(all_h)  # (B, T_obs, obs_dim)
        
        return pred_x
    
    def transformer_only_reconstruction(self, obs_traj):
        """
        This method can't be implemented with ODE-RNN architecture as it's specific to Transformer.
        Instead, we'll reconstruct the first frame based on the first observation only.
        
        Args:
            obs_traj: Tensor (B, obs_length, obs_dim)
        Returns:
            recon_loss: Scalar reconstruction loss for the first frame
            pred_x_first: Predicted first frame (B, obs_dim)
        """
        # Note: This is a simplified adaptation - not a true transformer_only_reconstruction
        B, _, _ = obs_traj.size()
        device = obs_traj.device
        
        # Get first observation and embed it
        x_first = obs_traj[:, 0, :]
        x_embedded = self.input_embedding(x_first)
        
        # Initialize hidden state with the embedded observation
        h = torch.zeros(B, self.latent_dim, device=device)
        h = self.gru_cell(x_embedded, h)
        
        # Decode to get prediction for first frame
        pred_x_first = self.decoder(h)
        
        # Compute reconstruction loss for first frame
        recon_loss = F.mse_loss(pred_x_first, x_first)
        
        return recon_loss, pred_x_first
    
    def compute_derivative_regularization(self, trajectory, t):
        """
        Computes regularization on the ODE function derivatives.
        Args:
            trajectory: Tensor of latent states
            t: Time points
        Returns:
            reg_loss: Regularization loss
            derivatives: Computed derivatives
        """
        # Detach trajectory points to focus regularization on function behavior only
        detached_states = [state.detach() for state in trajectory]
        
        # Calculate derivatives at each point
        derivatives = []
        for i, time in enumerate(t):
            derivative = self.func(time, detached_states[i])
            derivatives.append(derivative)
        
        derivatives = torch.stack(derivatives)
        
        # Measure consistency between consecutive derivative evaluations
        derivative_changes = torch.diff(derivatives, dim=0)
        reg_loss = torch.mean(torch.square(derivative_changes))
        
        return reg_loss, derivatives
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
# Positional Embedding (Sinusoidal)
# --------------------------
class TimeSeriesSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int = None):
        super().__init__(num_positions, embedding_dim, padding_idx=padding_idx)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        n_pos, dim = out.shape
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, :sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        bsz, seq_len = input_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len,
            dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

# --------------------------
# Latent ODE Function
# --------------------------
class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20, num_layers=2, use_tanh=False):
        super(LatentODEfunc, self).__init__()
        self.activation = nn.Tanh() if use_tanh else nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.middle_layers = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(num_layers - 2)])
        self.fc_final = nn.Linear(nhidden, latent_dim)

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.activation(out)
        for layer in self.middle_layers:
            out = layer(out)
            out = self.activation(out)
        out = self.fc_final(out)
        return out

# --------------------------
# Dataset: Sliding Window over Trajectories
# --------------------------
class TrajectoryWindowDataset(Dataset):
    def __init__(self, trajectories, window_length, obs_length):
        self.trajectories = trajectories
        self.window_length = window_length
        self.obs_length = obs_length
        self.target_length = window_length - obs_length
        self.num_time = trajectories.shape[0]
        self.num_space = trajectories.shape[1]
        self.num_features = trajectories.shape[2]
        self.indices = []
        for s in range(self.num_space):
            for t in range(0, self.num_time - window_length + 1):
                self.indices.append((t, s))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, s = self.indices[idx]
        traj = self.trajectories[t:t+self.window_length, s, :]  # (window_length, num_features)
        obs_traj = traj[:self.obs_length, :]
        target_traj = traj[self.obs_length:, :]
        time_steps = np.arange(self.window_length)
        obs_time = time_steps[:self.obs_length]
        target_time = time_steps[self.obs_length:]
        return {
            'obs_traj': torch.tensor(obs_traj, dtype=torch.float),
            'target_traj': torch.tensor(target_traj, dtype=torch.float),
            'obs_time': torch.tensor(obs_time, dtype=torch.float),
            'target_time': torch.tensor(target_time, dtype=torch.float),
            'full_time': torch.tensor(time_steps, dtype=torch.float)
        }

# --------------------------
# Transformer-based Latent ODE Wrapper Model
# --------------------------
class TransformerLatentODEWrapper(nn.Module):
    def __init__(self, latent_dim, d_model, nhead, num_encoder_layers,num_decoder_layers,
                 ode_nhidden, decoder_nhidden, obs_dim, noise_std, ode_layers, reg_weight, variational_inference, use_torchode, rtol=1e-1, atol=1e-1, use_tanh=False):
        super(TransformerLatentODEWrapper, self).__init__()
        print("initialzing TransformerLatentOdeWrapper")
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
        print("reg_weight: ", reg_weight)
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.obs_dim = obs_dim
        self.reg_weight = reg_weight
        self.variational_inference = variational_inference
        self.use_tanh = use_tanh
        # Recognition: transformer encoder to encode the observed trajectory
        self.value_embedding = nn.Linear(obs_dim, d_model)
        self.positional_embedding = TimeSeriesSinusoidalPositionalEmbedding(num_positions=500, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        if self.variational_inference:
            self.initial_state_projection = nn.Linear(d_model, 2 * latent_dim)
        else:
            self.initial_state_projection = nn.Linear(d_model, latent_dim)
        
        # Latent ODE dynamics
        self.func = LatentODEfunc(latent_dim, ode_nhidden, num_layers=ode_layers, use_tanh=use_tanh)
        if use_torchode:
            term = to.ODETerm(self.func)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
            self.adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
        self.use_torchode = use_torchode
        # Decoder to map latent state to observation space
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, decoder_nhidden))
        if use_tanh:
            decoder_layers.append(nn.Tanh())
        else:
            decoder_layers.append(nn.ReLU())
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(decoder_nhidden, decoder_nhidden))
            if use_tanh:
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_nhidden, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.rtol = rtol
        self.atol = atol
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
        unique_times= torch.unique(times)
        
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

        # Recognition: embed and encode observed trajectory
        x = self.value_embedding(obs_traj)  # (B, T_obs, d_model)
        pos_emb = self.positional_embedding(obs_traj.shape, past_key_values_length=0)  # (T_obs, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        x = x.transpose(0, 1)  # (T_obs, B, d_model)
        encoded = self.transformer_encoder(x)  # (T_obs, B, d_model)
        h = encoded[-1]  # (B, d_model)
        #h = encoded.mean(dim=0)
        z_params = self.initial_state_projection(h)  # (B, 2 * latent_dim)
        if self.variational_inference:
            qz0_mean, qz0_logvar = z_params.chunk(2, dim=-1)
            epsilon = torch.randn_like(qz0_mean)
            z0 = qz0_mean + epsilon * torch.exp(0.5 * qz0_logvar)
        else:
            z0 = z_params
        
        # Filter unique times and get mapping
        full_time = full_time[0]  # Remove batch dimension if present
        unique_times, mapping_indices = self._filter_unique_times(full_time)
        
        # ODE integration with unique times
        if USE_ADAPTIVE:
            if self.use_torchode:
                # expand unique_times to match the shape of z0
                problem = to.InitialValueProblem(y0=z0, t_eval=unique_times.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                pred_z_unique = torch.transpose(sol.ys, 0, 1)  # (B, unique_length, latent_dim)
                
            else:
                pred_z_unique = odeint(self.func, z0, unique_times, rtol=self.rtol, atol=self.atol)  # (unique_length, B, latent_dim)
        else:
            time_span = unique_times[-1] - unique_times[0]
            step_size = time_span / 20
            pred_z_unique = odeint(self.func, z0, unique_times, method='rk4', rtol=self.rtol, atol=self.atol, options=dict(step_size=step_size))  # (unique_length, B, latent_dim)
        
        reg_loss = torch.tensor(0.0, device=device)
        if self.reg_weight > 0:
            reg_loss, _ = self.compute_derivative_regularization(pred_z_unique, unique_times)
            reg_loss = reg_loss * self.reg_weight

        # Reconstruct full trajectory with duplicates
        pred_z = self._reconstruct_trajectory(pred_z_unique, mapping_indices, len(full_time))
        pred_z = pred_z.permute(1, 0, 2)  # (B, window_length, latent_dim)
        pred_x = self.decoder(pred_z)  # (B, window_length, obs_dim)
        
        # Split predictions into observed and target parts
        pred_x_obs = pred_x[:, :T_obs, :]
        pred_x_target = pred_x[:, T_obs:, :]
        
        # Losses
        if self.variational_inference:
            noise_logvar = 2 * torch.log(torch.tensor(self.noise_std, device=device))
            logpx_obs = log_normal_pdf(obs_traj, pred_x_obs, noise_logvar)
            recon_loss = -logpx_obs.sum(dim=2).sum(dim=1).mean()
            logpx_target = log_normal_pdf(target_traj, pred_x_target, noise_logvar)
            pred_loss = -logpx_target.sum(dim=2).sum(dim=1).mean()
            kl_loss = normal_kl(qz0_mean, qz0_logvar,
                                torch.zeros_like(qz0_mean),
                                torch.zeros_like(qz0_logvar)).sum(dim=1).mean()
            loss = recon_loss + pred_loss + kl_loss + reg_loss
        else:
            # deterministic prediction, use mse loss
            recon_loss = F.mse_loss(pred_x_target, target_traj)
            pred_loss = F.mse_loss(pred_x_obs, obs_traj)
            loss = recon_loss + pred_loss + reg_loss
            kl_loss = 0
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
        x = self.value_embedding(obs_traj)
        pos_emb = self.positional_embedding(obs_traj.shape, past_key_values_length=0)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)
        h = encoded[-1]
        if self.variational_inference:
            z_params = self.initial_state_projection(h)
            qz0_mean, _ = z_params.chunk(2, dim=-1)
            z0 = qz0_mean  # use mean for deterministic prediction
        else:
            z0 = self.initial_state_projection(h)
        
        # Filter unique times and get mapping
        full_time = torch.cat([obs_time, extrapolate_time])
        unique_times, mapping_indices = self._filter_unique_times(full_time)
        
        # ODE integration with unique times
        if USE_ADAPTIVE:
            if self.use_torchode:
                # expand unique_times to match the shape of z0
                problem = to.InitialValueProblem(y0=z0, t_eval=unique_times.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                pred_z_unique = torch.transpose(sol.ys, 0, 1)  # (B, unique_length, latent_dim)
            else:
                pred_z_unique = odeint(self.func, z0, unique_times, rtol=self.rtol, atol=self.atol)  # (unique_length, B, latent_dim)
        else:
            time_span = unique_times[-1] - unique_times[0]
            step_size = time_span / 20
            pred_z_unique = odeint(self.func, z0, unique_times, method='rk4', rtol=self.rtol, atol=self.atol, options=dict(step_size=step_size))
        
        # Reconstruct full trajectory with duplicates
        pred_z = self._reconstruct_trajectory(pred_z_unique, mapping_indices, len(full_time))
        pred_z = pred_z.permute(1, 0, 2)
        pred_x = self.decoder(pred_z)
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
        B, _, _ = obs_traj.size()
        x = self.value_embedding(obs_traj)
        pos_emb = self.positional_embedding(obs_traj.shape, past_key_values_length=0)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)
        h = encoded[-1]
        if self.variational_inference:
            z_params = self.initial_state_projection(h)
            qz0_mean, _ = z_params.chunk(2, dim=-1)
            z0 = qz0_mean  # use mean for deterministic prediction
        else:
            z0 = self.initial_state_projection(h)
        
        # Filter unique times and get mapping
        full_time = obs_time
        unique_times, mapping_indices = self._filter_unique_times(full_time)
        
        # ODE integration with unique times
        if USE_ADAPTIVE:
            if self.use_torchode:
                # expand unique_times to match the shape of z0
                problem = to.InitialValueProblem(y0=z0, t_eval=unique_times.unsqueeze(0).expand(B, -1))
                sol = self.adjoint.solve(problem)
                pred_z_unique = torch.transpose(sol.ys, 0, 1)  # (B, unique_length, latent_dim)
            else:
                pred_z_unique = odeint(self.func, z0, unique_times, rtol=self.rtol, atol=self.atol)  # (unique_length, B, latent_dim)
        else:
            time_span = unique_times[-1] - unique_times[0]
            step_size = time_span / 20
            pred_z_unique = odeint(self.func, z0, unique_times, method='rk4', rtol=self.rtol, atol=self.atol, options=dict(step_size=step_size))
        
        # Reconstruct full trajectory with duplicates
        pred_z = self._reconstruct_trajectory(pred_z_unique, mapping_indices, len(full_time))
        pred_z = pred_z.permute(1, 0, 2)
        pred_x = self.decoder(pred_z)
        return pred_x
        

    def transformer_only_reconstruction(self, obs_traj):
        """
        Uses only the transformer encoder-decoder to produce latent from the first frame and compute reconstruction loss
        for the first frame only.
        Args:
            obs_traj: Tensor (B, obs_length, obs_dim)
        Returns:
            recon_loss: Scalar reconstruction loss for the first frame
            pred_x_first: Predicted first frame (B, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device

        # Embed and encode observed trajectory
        x = self.value_embedding(obs_traj)  # (B, T_obs, d_model)
        pos_emb = self.positional_embedding(obs_traj.shape, past_key_values_length=0)  # (T_obs, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        x = x.transpose(0, 1)  # (T_obs, B, d_model)
        encoded = self.transformer_encoder(x)  # (T_obs, B, d_model)
        
        # Use only the first frame's encoding
        h_first = encoded[0]  # (B, d_model) - first time step
        z_params = self.initial_state_projection(h_first)  # (B, 2 * latent_dim)
        if self.variational_inference:
            qz0_mean, qz0_logvar = z_params.chunk(2, dim=-1)
            epsilon = torch.randn_like(qz0_mean)
            z0 = qz0_mean + epsilon * torch.exp(0.5 * qz0_logvar)  # (B, latent_dim)
        else:
            z0 = z_params
        
        # Decode only the first frame's latent
        pred_x_first = self.decoder(z0)  # (B, obs_dim)
        
        # Compute reconstruction loss for the first frame only
        if self.variational_inference:
            noise_logvar = 2 * torch.log(torch.tensor(self.noise_std, device=device))
            logpx = log_normal_pdf(obs_traj[:, 0, :], pred_x_first, noise_logvar)  # Compare with first frame of obs_traj
            recon_loss = -logpx.sum(dim=1).mean()  # Sum over features, mean over batch
            kl_loss = normal_kl(qz0_mean, qz0_logvar,
                               torch.zeros_like(qz0_mean),
                               torch.zeros_like(qz0_logvar)).sum(dim=1).mean()
        else:
            recon_loss = F.mse_loss(pred_x_first, obs_traj[:, 0, :])
            kl_loss = 0
        
        total_loss = recon_loss + kl_loss
        return total_loss, pred_x_first
    
    def compute_derivative_regularization(self, trajectory, t):
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
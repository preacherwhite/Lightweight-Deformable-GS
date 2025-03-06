import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

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
# Autoregressive Transformer Model
# --------------------------
class AutoregressiveTransformer(nn.Module):
    def __init__(self, latent_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 ode_nhidden, decoder_nhidden, obs_dim, noise_std, ode_layers, reg_weight, 
                 variational_inference=False, use_torchode=False, rtol=1e-1, atol=1e-1, use_tanh=False):
        super(AutoregressiveTransformer, self).__init__()
        print("initializing AutoregressiveTransformer")
        print("latent_dim: ", latent_dim)
        print("d_model: ", d_model)
        print("nhead: ", nhead)
        print("num_encoder_layers: ", num_encoder_layers)
        print("num_decoder_layers: ", num_decoder_layers)
        print("decoder_nhidden: ", decoder_nhidden)
        print("obs_dim: ", obs_dim)
        print("noise_std: ", noise_std)
        
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.reg_weight = 0  # Not used in autoregressive model
        
        # Embeddings
        self.value_embedding = nn.Linear(obs_dim, d_model)
        self.positional_embedding = TimeSeriesSinusoidalPositionalEmbedding(num_positions=500, embedding_dim=d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, obs_dim)
        
        # Used for interface compatibility
        self.rtol = rtol
        self.atol = atol
        self.use_tanh = use_tanh
        
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
        _, T_target, _ = target_traj.size()
        device = obs_traj.device
        
        # Get time sequences for observation and target
        full_time = full_time[0]  # Remove batch dimension if present
        obs_time = full_time[:T_obs]
        target_time = full_time[T_obs:]
        
        # Filter unique observation times and get mapping
        unique_obs_times, obs_mapping_indices = self._filter_unique_times(obs_time)
        
        # Get unique observation trajectory (eliminate duplicates)
        unique_obs_indices = []
        for i, idx in enumerate(obs_mapping_indices):
            if i == 0 or obs_mapping_indices[i-1] != idx:
                unique_obs_indices.append(i)
        
        unique_obs_traj = obs_traj[:, unique_obs_indices, :]  # (B, unique_T_obs, obs_dim)
        unique_T_obs = unique_obs_traj.size(1)
        
        # Filter unique target times and get mapping
        unique_target_times, target_mapping_indices = self._filter_unique_times(target_time)
        unique_T_target = len(unique_target_times)
        
        # Encode observed trajectory (use only unique timepoints)
        encoder_input = self.value_embedding(unique_obs_traj)  # (B, unique_T_obs, d_model)
        pos_emb_encoder = self.positional_embedding(unique_obs_traj.shape, past_key_values_length=0)  # (unique_T_obs, d_model)
        pos_emb_encoder = pos_emb_encoder.unsqueeze(0).expand(B, -1, -1)
        encoder_input = encoder_input + pos_emb_encoder
        encoder_input = encoder_input.transpose(0, 1)  # (unique_T_obs, B, d_model)
        memory = self.transformer_encoder(encoder_input)  # (unique_T_obs, B, d_model)
        
        # For training, use true autoregressive approach (no teacher forcing)
        unique_pred_target = []
        current_input = unique_obs_traj[:, -1, :]  # Start with last observed frame
        
        # Generate predictions one step at a time
        for t in range(unique_T_target):
            # Embed current input
            decoder_input = self.value_embedding(current_input).unsqueeze(0)  # (1, B, d_model)
            
            # Add positional encoding
            pos_emb = self.positional_embedding(
                (B, 1), 
                past_key_values_length=unique_T_obs + t
            )  # (1, d_model)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1).transpose(0, 1)
            decoder_input = decoder_input + pos_emb
            
            # Decode
            decoder_output = self.transformer_decoder(
                decoder_input,
                memory
            )  # (1, B, d_model)
            
            # Project to observation space and get prediction
            decoder_output = decoder_output.transpose(0, 1)  # (B, 1, d_model)
            pred = self.output_projection(decoder_output).squeeze(1)  # (B, obs_dim)
            
            # Store prediction
            unique_pred_target.append(pred)
            
            # Use prediction as next input (fully autoregressive)
            current_input = pred
        
        # Stack unique predictions
        unique_pred_target = torch.stack(unique_pred_target, dim=1)  # (B, unique_T_target, obs_dim)
        
        # Reconstruct full target prediction with duplicates
        pred_target = torch.zeros(B, T_target, self.obs_dim, device=device)
        
        # Map unique predictions back to full timeline
        for j in range(T_target):
            # Get the unique time index for this target point
            unique_idx = target_mapping_indices[j] 
            # Use the prediction for that unique time
            pred_target[:, j, :] = unique_pred_target[:, unique_idx, :]
        
        # Create full prediction by concatenating observed and predicted
        pred_x = torch.cat([obs_traj, pred_target], dim=1)  # (B, T_obs + T_target, obs_dim)
        
        # Calculate losses (using full reconstructed target)
        pred_loss = F.l1_loss(pred_target, target_traj)
        recon_loss = torch.tensor(0.0, device=device)  # Not used in autoregressive model
        kl_loss = torch.tensor(0.0, device=device)    # Not used in autoregressive model
        reg_loss = torch.tensor(0.0, device=device)   # Not used in autoregressive model
        
        # Total loss is just the prediction loss for autoregressive model
        loss = pred_loss
        
        return loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_x
    
    def extrapolate(self, obs_traj, obs_time, extrapolate_time):
        """
        Deterministically extrapolates the trajectory, handling duplicate time points.
        Args:
            obs_traj: (B, obs_length, obs_dim)
            obs_time: 1D tensor of length obs_length
            extrapolate_time: 1D tensor for prediction horizon
        Returns:
            Full predicted trajectory (B, obs_length + prediction_length, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Filter unique observation times and get mapping
        unique_obs_times, obs_mapping_indices = self._filter_unique_times(obs_time)
        
        # Get unique observation trajectory (eliminate duplicates)
        unique_obs_traj = []
        for i, idx in enumerate(obs_mapping_indices):
            if i == 0 or obs_mapping_indices[i-1] != idx:
                unique_obs_traj.append(obs_traj[:, i, :])
        unique_obs_traj = torch.stack(unique_obs_traj, dim=1)  # (B, unique_T_obs, obs_dim)
        unique_T_obs = unique_obs_traj.size(1)
        
        # Filter unique extrapolation times and get mapping
        unique_extrapolate_times, extrapolate_mapping_indices = self._filter_unique_times(extrapolate_time)
        unique_T_target = len(unique_extrapolate_times)
        
        # Encode observed trajectory
        encoder_input = self.value_embedding(unique_obs_traj)  # (B, unique_T_obs, d_model)
        pos_emb_encoder = self.positional_embedding(unique_obs_traj.shape, past_key_values_length=0)
        pos_emb_encoder = pos_emb_encoder.unsqueeze(0).expand(B, -1, -1)
        encoder_input = encoder_input + pos_emb_encoder
        encoder_input = encoder_input.transpose(0, 1)  # (unique_T_obs, B, d_model)
        memory = self.transformer_encoder(encoder_input)  # (unique_T_obs, B, d_model)
        
        # Autoregressive generation for unique time points
        pred_targets = []
        
        # Start with the last unique observation
        current_input = unique_obs_traj[:, -1, :]
        
        # Generate predictions one step at a time
        for t in range(unique_T_target):
            # Embed current input
            decoder_input = self.value_embedding(current_input).unsqueeze(0)  # (1, B, d_model)
            
            # Add positional encoding
            pos_emb = self.positional_embedding(
                (B, 1), 
                past_key_values_length=unique_T_obs + t
            )  # (1, d_model)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1).transpose(0, 1)
            decoder_input = decoder_input + pos_emb
            
            # Decode
            decoder_output = self.transformer_decoder(
                decoder_input,
                memory
            )  # (1, B, d_model)
            
            # Project to observation space and get prediction
            decoder_output = decoder_output.transpose(0, 1)  # (B, 1, d_model)
            pred = self.output_projection(decoder_output).squeeze(1)  # (B, obs_dim)
            
            # Store prediction
            pred_targets.append(pred)
            
            # Use prediction as next input
            current_input = pred
        
        # Stack unique predictions
        unique_pred_target = torch.stack(pred_targets, dim=1)  # (B, unique_T_target, obs_dim)
        
        # Reconstruct full target trajectory with duplicates
        target_traj = torch.zeros(len(extrapolate_time), B, self.obs_dim, device=device)
        for i, idx in enumerate(extrapolate_mapping_indices):
            target_traj[i] = unique_pred_target[:, idx, :]
        target_traj = target_traj.transpose(0, 1)  # (B, T_target, obs_dim)
        
        # Reconstruct full observed trajectory (should be identical to input)
        reconst_obs_traj = torch.zeros(len(obs_time), B, self.obs_dim, device=device)
        for i, idx in enumerate(obs_mapping_indices):
            reconst_obs_traj[i] = unique_obs_traj[:, idx, :]
        reconst_obs_traj = reconst_obs_traj.transpose(0, 1)  # (B, T_obs, obs_dim)
        
        # Concatenate reconstructed observed and target trajectories
        pred_x = torch.cat([reconst_obs_traj, target_traj], dim=1)  # (B, T_obs + T_target, obs_dim)
        
        return pred_x
    
    def reconstruct(self, obs_traj, obs_time):
        """
        Reconstructs the trajectory for the input time, handling duplicate timestamps.
        Args:
            obs_traj: (B, obs_length, obs_dim)
            obs_time: 1D tensor of length obs_length    
        Returns:
            pred_x: (B, obs_length, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device
        
        # Filter unique observation times and get mapping
        unique_obs_times, obs_mapping_indices = self._filter_unique_times(obs_time)
        
        # Get unique observation trajectory (eliminate duplicates)
        unique_obs_indices = []
        for i, idx in enumerate(obs_mapping_indices):
            if i == 0 or obs_mapping_indices[i-1] != idx:
                unique_obs_indices.append(i)
        
        unique_obs_traj = obs_traj[:, unique_obs_indices, :]  # (B, unique_T_obs, obs_dim)
        unique_T_obs = unique_obs_traj.size(1)
        
        # For reconstruction, we simply need to map the unique observations back to the full timeline
        reconstructed_traj = torch.zeros(B, T_obs, self.obs_dim, device=device)
        for i, idx in enumerate(obs_mapping_indices):
            unique_idx = 0
            for j, orig_idx in enumerate(unique_obs_indices):
                if orig_idx <= i and (j == len(unique_obs_indices) - 1 or unique_obs_indices[j+1] > i):
                    unique_idx = j
                    break
            reconstructed_traj[:, i, :] = unique_obs_traj[:, unique_idx, :]
            
        return reconstructed_traj

    def transformer_only_reconstruction(self, obs_traj):
        """
        Uses only the transformer encoder-decoder to produce output for the first frame.
        Args:
            obs_traj: Tensor (B, obs_length, obs_dim)
        Returns:
            recon_loss: Scalar reconstruction loss for the first frame
            pred_x_first: Predicted first frame (B, obs_dim)
        """
        B, T_obs, _ = obs_traj.size()
        device = obs_traj.device

        # Encode observed trajectory
        encoder_input = self.value_embedding(obs_traj)  # (B, T_obs, d_model)
        pos_emb_encoder = self.positional_embedding(obs_traj.shape, past_key_values_length=0)
        pos_emb_encoder = pos_emb_encoder.unsqueeze(0).expand(B, -1, -1)
        encoder_input = encoder_input + pos_emb_encoder
        encoder_input = encoder_input.transpose(0, 1)  # (T_obs, B, d_model)
        memory = self.transformer_encoder(encoder_input)  # (T_obs, B, d_model)
        
        # Use the embedding of the first frame as decoder input
        decoder_input = self.value_embedding(obs_traj[:, 0, :]).unsqueeze(0)  # (1, B, d_model)
        
        # Add positional encoding
        pos_emb = self.positional_embedding((B, 1), past_key_values_length=0)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1).transpose(0, 1)
        decoder_input = decoder_input + pos_emb
        
        # Decode
        decoder_output = self.transformer_decoder(decoder_input, memory)
        
        # Project to observation space
        decoder_output = decoder_output.transpose(0, 1)
        pred_x_first = self.output_projection(decoder_output).squeeze(1)
        
        # Compute reconstruction loss for the first frame only
        recon_loss = F.l1_loss(pred_x_first, obs_traj[:, 0, :])
        
        return recon_loss, pred_x_first
    
    # These methods are added for interface compatibility with the ODE model
    def compute_derivative_regularization(self, trajectory, t):
        # Not used in autoregressive model, but included for interface compatibility
        device = trajectory.device if hasattr(trajectory, 'device') else torch.device('cpu')
        return torch.tensor(0.0, device=device), None
        
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
        unique_times, inverse_indices = torch.unique(times, return_inverse=True)
        
        # Create mapping from original indices to unique indices
        mapping_indices = inverse_indices.tolist()
        
        return unique_times, mapping_indices

    def _reconstruct_trajectory(self, predictions, mapping_indices, original_length):
        """
        Reconstruct the full trajectory by duplicating predictions for duplicate timestamps.
        Args:
            predictions: tensor of predictions for unique timestamps
            mapping_indices: List mapping original indices to unique time indices
            original_length: Original length of the time sequence
        Returns:
            complete_trajectory: tensor with duplicates properly handled
        """
        if len(predictions.shape) == 3:  # (T, B, D)
            T, B, D = predictions.shape
            complete_trajectory = torch.zeros(original_length, B, D, device=predictions.device)
            for i, idx in enumerate(mapping_indices):
                complete_trajectory[i] = predictions[idx]
        elif len(predictions.shape) == 2:  # (B, D)
            B, D = predictions.shape
            complete_trajectory = torch.zeros(original_length, B, D, device=predictions.device)
            for i, idx in enumerate(mapping_indices):
                complete_trajectory[i] = predictions
                
        return complete_trajectory
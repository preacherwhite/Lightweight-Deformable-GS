import torch
from random import randint, sample

def sample_ode_only_fids(min_fid, max_fid, window_length):
    """Sample continuous fids for ODE-only training."""
    fid_range = max_fid - min_fid
    start_fid = min_fid + torch.rand(1).item() * (fid_range - window_length * 0.01)
    fids = torch.linspace(start_fid, start_fid + window_length * 0.01, window_length, device="cuda")
    return fids

def sample_rendering_included_fids(train_cameras, min_fid, max_fid, window_length, obs_length):
    """Sample continuous fids that include some discrete camera frames."""
    # Randomly select camera frames that will be included
    if len(train_cameras) <= window_length // 2:
        selected_cam_indices = list(range(len(train_cameras)))
    else:
        num_cams_to_include = min(window_length // 2, len(train_cameras))
        selected_cam_indices = sorted(sample(range(len(train_cameras)), num_cams_to_include))
    
    # Get their fids
    selected_fids = [train_cameras[i].fid.item() for i in selected_cam_indices]
    
    # Now create a continuous time window that includes these discrete fids
    # First determine the window boundaries
    min_selected_fid = min(selected_fids)
    max_selected_fid = max(selected_fids)
    window_range = max(max_selected_fid - min_selected_fid, 0.01)  # Avoid div by zero
    
    # Ensure window has enough space on both sides
    padding = window_range * 0.1  # 10% padding
    start_fid = max(min_fid, min_selected_fid - padding)
    end_fid = min(max_fid, max_selected_fid + padding)
    
    # Generate continuous fids, ensuring the selected discrete fids are included
    continuous_fids = []
    
    # Add points before first selected fid
    if start_fid < min_selected_fid:
        num_points_before = max(1, int((min_selected_fid - start_fid) / window_range * window_length // 4))
        pre_fids = torch.linspace(start_fid, min_selected_fid, num_points_before, device="cuda")[:-1]
        continuous_fids.extend(pre_fids.tolist())
    
    # Add all selected fids
    continuous_fids.extend(selected_fids)
    
    # Add points after last selected fid
    if end_fid > max_selected_fid:
        num_points_after = max(1, int((end_fid - max_selected_fid) / window_range * window_length // 4))
        post_fids = torch.linspace(max_selected_fid, end_fid, num_points_after, device="cuda")[1:]
        continuous_fids.extend(post_fids.tolist())
    
    # Ensure we have the right number of points by interpolating if needed
    if len(continuous_fids) < window_length:
        # Need more points - interpolate between existing points
        dense_fids = []
        for i in range(len(continuous_fids) - 1):
            dense_fids.append(continuous_fids[i])
            # Add interpolated points between current and next
            num_to_add = max(1, int((window_length - len(continuous_fids)) / (len(continuous_fids) - 1)))
            interp_fids = torch.linspace(continuous_fids[i], continuous_fids[i+1], num_to_add + 2, device="cuda")[1:-1]
            dense_fids.extend(interp_fids.tolist())
        dense_fids.append(continuous_fids[-1])
        continuous_fids = dense_fids
    
    # Ensure exact window length
    if len(continuous_fids) > window_length:
        # Too many points - need to sample
        indices = torch.linspace(0, len(continuous_fids) - 1, window_length).long()
        continuous_fids = [continuous_fids[i] for i in indices]
    
    # Convert to tensor
    fids = torch.tensor(continuous_fids, device="cuda")
    
    return fids, selected_fids, selected_cam_indices

def sample_discrete_trajectories(train_cameras, train_trajectories, window_length, obs_length, batch_size, total_gaussians):
    """Sample discrete trajectories from pre-computed data."""
    if len(train_cameras) <= window_length:
        start_idx = 0
    else:
        start_idx = randint(obs_length, len(train_cameras) - (window_length - obs_length) - 1)
    
    viewpoint_indices = list(range(start_idx - obs_length, start_idx + (window_length - obs_length)))
    fids = torch.tensor([train_cameras[i].fid for i in viewpoint_indices], device="cuda")
    batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
    trajectories = train_trajectories[viewpoint_indices]
    traj_gt = trajectories[:, batch_indices]
    
    obs_traj = traj_gt[:obs_length].permute(1, 0, 2)
    target_traj = traj_gt[obs_length:].permute(1, 0, 2)
    full_time = fids.unsqueeze(0)
    
    return obs_traj, target_traj, full_time, fids, viewpoint_indices, batch_indices

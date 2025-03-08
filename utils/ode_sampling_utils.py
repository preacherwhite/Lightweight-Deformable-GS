import torch
from random import randint, sample

def sample_ode_only_fids(train_cameras, min_fid, max_fid, window_length, obs_length, max_extrapolation_time=10.0):
    """Sample continuous fids for ODE-only training with discrete observation segment.
    
    Args:
        train_cameras: List of camera objects with fid attributes
        min_fid: Minimum frame ID in dataset
        max_fid: Maximum frame ID in dataset
        window_length: Total number of points in the output sequence
        obs_length: Number of points in the observed part of trajectory
        max_extrapolation_time: Maximum time distance for extrapolation after last observed input
    """
    # First ensure we have enough cameras to work with
    if len(train_cameras) <= obs_length:
        start_idx = 0
    else:
        # Make sure we have room for extrapolation after observed sequence
        start_idx = randint(0, len(train_cameras) - obs_length - 1)
    
    # Get the observed trajectory camera indices and their fids
    obs_cam_indices = list(range(start_idx, start_idx + obs_length))
    obs_fids = [train_cameras[i].fid.item() for i in obs_cam_indices]
    
    # Create continuous sequence for observation part by using linspace between camera frames
    continuous_fids_obs = torch.linspace(obs_fids[0], obs_fids[-1], obs_length, device="cuda")
    
    # Get the last observed time
    last_obs_fid = obs_fids[-1]
    
    # Sample a random extrapolation end time within max_extrapolation_time
    max_possible_fid = min(max_fid, last_obs_fid + max_extrapolation_time)
    if max_possible_fid <= last_obs_fid:
        extrapolation_end_fid = last_obs_fid
    else:
        extrapolation_end_fid = last_obs_fid + torch.rand(1).item() * (max_possible_fid - last_obs_fid)
    
    # Create continuous sequence for extrapolation part
    remaining_points = window_length - obs_length
    if remaining_points > 0 and extrapolation_end_fid > last_obs_fid:
        continuous_fids_extra = torch.linspace(last_obs_fid, extrapolation_end_fid, remaining_points + 1, device="cuda")[1:]
    else:
        continuous_fids_extra = torch.tensor([], device="cuda")
    
    # Combine observed and extrapolation points
    continuous_fids = torch.cat([continuous_fids_obs, continuous_fids_extra])
    
    # Ensure we have exactly window_length points
    if len(continuous_fids) != window_length:
        indices = torch.linspace(0, len(continuous_fids) - 1, window_length, device="cuda").long()
        continuous_fids = continuous_fids[indices]
    
    return continuous_fids

def sample_rendering_included_fids(train_cameras, min_fid, max_fid, window_length, obs_length, max_extrapolation_time=0.5):
    """Sample continuous fids with observed trajectory first, then extrapolation period including camera frames.
    
    Args:
        train_cameras: List of camera objects with fid attributes
        min_fid: Minimum frame ID in dataset
        max_fid: Maximum frame ID in dataset
        window_length: Total number of points in the output sequence
        obs_length: Number of points in the observed part of trajectory
        max_extrapolation_time: Maximum time distance for extrapolation after last observed input
    """
    # First ensure we have enough cameras to work with
    if len(train_cameras) <= obs_length:
        start_idx = 0
    else:
        # Make sure we have room for extrapolation after observed sequence
        start_idx = randint(0, len(train_cameras) - obs_length - 1)
    
    # Get the observed trajectory camera indices and their fids
    obs_cam_indices = list(range(start_idx, start_idx + obs_length))
    obs_fids = [train_cameras[i].fid.item() for i in obs_cam_indices]
    
    # Get the last observed time
    last_obs_fid = obs_fids[-1]
    
    # Sample a random extrapolation end time
    max_possible_fid = min(max_fid, last_obs_fid + max_extrapolation_time)
    if max_possible_fid <= last_obs_fid:
        extrapolation_end_fid = last_obs_fid
    else:
        extrapolation_end_fid = last_obs_fid + torch.rand(1).item() * (max_possible_fid - last_obs_fid)
    
    # Find all camera indices that fall within the extrapolation period
    extrapolation_cam_indices = []
    extrapolation_cam_fids = []
    for i, camera in enumerate(train_cameras):
        if i not in obs_cam_indices:  # Skip observed cameras
            camera_fid = camera.fid.item()
            if last_obs_fid < camera_fid <= extrapolation_end_fid:
                extrapolation_cam_indices.append(i)
                extrapolation_cam_fids.append(camera_fid)
    
    # Create continuous sequence
    # 1. First obs_length points are from the observed segment
    continuous_fids_obs = torch.linspace(obs_fids[0], obs_fids[-1], obs_length, device="cuda")
    
    # 2. Calculate remaining points needed
    remaining_points = window_length - obs_length - len(extrapolation_cam_fids)
    
    # 3. Create additional points in the extrapolation period using linspace
    if remaining_points > 0 and extrapolation_end_fid > last_obs_fid:
        continuous_fids_extra = torch.linspace(last_obs_fid, extrapolation_end_fid, remaining_points + 2, device="cuda")[1:-1]
    else:
        continuous_fids_extra = torch.tensor([], device="cuda")
    
    # 4. Combine all points and sort them
    if extrapolation_cam_fids:
        all_extra_fids = torch.cat([
            torch.tensor(extrapolation_cam_fids, device="cuda"),
            continuous_fids_extra
        ])
    else:
        all_extra_fids = continuous_fids_extra
    
    # 5. Sort all extrapolation points
    all_extra_fids = torch.sort(all_extra_fids).values
    
    # 6. Combine observed and extrapolation points
    continuous_fids = torch.cat([continuous_fids_obs, all_extra_fids])
    
    # Combine all camera indices
    selected_cam_indices = obs_cam_indices + extrapolation_cam_indices
    selected_fids = [train_cameras[i].fid.item() for i in selected_cam_indices]
    
    # Ensure we have exactly window_length points
    if len(continuous_fids) != window_length:
        indices = torch.linspace(0, len(continuous_fids) - 1, window_length, device="cuda").long()
        continuous_fids = continuous_fids[indices]
    
    return continuous_fids, selected_fids, selected_cam_indices

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

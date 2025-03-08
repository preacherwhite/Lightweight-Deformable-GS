import torch
from utils.loss_utils import l1_loss, ssim
from pytorch_msssim import ms_ssim
from gaussian_renderer import render_ode
from utils.ode_sampling_utils import sample_discrete_trajectories
def _compute_rendering_loss(viewpoint_cam, pred_values, batch_indices, trajectories_or_traj_gt, 
                           frame_idx, gaussians, pipe, background, xyz_only, dataset, opt, 
                           render_weight, deform=None, fid_val=None, total_gaussians=None, scales = None):
    """Common helper for computing rendering loss for a single frame."""
    # Handle parameter updates based on prediction type
    if xyz_only:
        if deform is not None and fid_val is not None:
            # Continuous case with deform model
            time_input = torch.tensor([fid_val], device="cuda").expand(total_gaussians, -1)
            with torch.no_grad():
                d_xyz, _, _ = deform.step(gaussians.get_xyz.detach(), time_input)
            new_xyz_render = (d_xyz + gaussians.get_xyz).clone()
            new_xyz_render[batch_indices] = pred_values
            new_rotation_render = gaussians.get_rotation
            new_scaling_render = gaussians.get_scaling
        else:
            # Discrete case
            new_xyz_render = trajectories_or_traj_gt[frame_idx].clone()
            new_xyz_render[batch_indices] = pred_values
            new_rotation_render = gaussians.get_rotation
            new_scaling_render = gaussians.get_scaling
    else:
        if deform is not None and fid_val is not None:
            # Continuous case with full parameters
            time_input = torch.tensor([fid_val], device="cuda").expand(total_gaussians, -1)
            with torch.no_grad():
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
            new_xyz_render = (d_xyz + gaussians.get_xyz).clone()
            new_rotation_render = (d_rotation + gaussians.get_rotation).clone()
            new_scaling_render = (d_scaling + gaussians.get_scaling).clone()
            new_xyz_render[batch_indices] = pred_values[:, :3]
            new_rotation_render[batch_indices] = pred_values[:, 3:7]
            new_scaling_render[batch_indices] = pred_values[:, 7:]
        else:
            # Discrete case with full parameters
            new_xyz_render = trajectories_or_traj_gt[frame_idx, :, :3].clone()
            new_rotation_render = trajectories_or_traj_gt[frame_idx, :, 3:7].clone()
            new_scaling_render = trajectories_or_traj_gt[frame_idx, :, 7:].clone()
            new_xyz_render[batch_indices] = pred_values[:, :3]
            new_rotation_render[batch_indices] = pred_values[:, 3:7]
            new_scaling_render[batch_indices] = pred_values[:, 7:]
    
    if scales is None:
        render_pkg = render_ode(
            viewpoint_cam, gaussians, pipe, background,
            new_xyz_render, new_rotation_render, new_scaling_render,
            dataset.is_6dof
        )
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        l1 = l1_loss(image, gt_image)
        rendering_loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        scaled_loss = rendering_loss * render_weight
        
        return scaled_loss, render_pkg
    
    elif len(scales) == 1 and scales[0] == 0:
        render_pkg = render_ode(
            viewpoint_cam, gaussians, pipe, background,
            new_xyz_render, new_rotation_render, new_scaling_render,
            dataset.is_6dof
        )
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        l1 = l1_loss(image, gt_image)
        rendering_loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ms_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        scaled_loss = rendering_loss * render_weight
        return scaled_loss, render_pkg
    
    else:
        if len(scales) == 1 and scales[0] < 0   :
            scales = []
            image_height = viewpoint_cam.image_height
            image_width = viewpoint_cam.image_width
            # compute times it takes to half the image size each time until 1x1
            scale = 1.0
            while image_height > 4 and image_width > 4:
                scale /= 4
                scales.append(scale)
                image_height /= 4
                image_width /= 4
        
        stacked_images = []
        for i in range(len(scales)):
            scale = scales[i]
            render_pkg = render_ode(
                viewpoint_cam, gaussians, pipe, background,
                new_xyz_render, new_rotation_render, new_scaling_render,
                dataset.is_6dof, resize_factor=scale
            )
            image = render_pkg["render"]
            stacked_images.append(image)
        
        stacked_images = torch.stack(stacked_images, dim=0)
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = gt_image.repeat(stacked_images.shape[0], 1, 1, 1)
        l1 = l1_loss(stacked_images, gt_image)
        rendering_loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ssim(stacked_images, gt_image))
        scaled_loss = rendering_loss * render_weight
        return scaled_loss, render_pkg

def compute_warmup_losses(loss, pred_traj_first, warmup_weight, render_weight, opt, 
                         train_cameras, viewpoint_indices, batch_indices, trajectories, 
                         gaussians, pipe, background, xyz_only, dataset, scales = None):
    """Compute losses during warmup phase."""
    # Transformer-only reconstruction
    warmup_loss = loss * warmup_weight
    
    # Rendering for the first frame only
    render_frame_idx = 0
    viewpoint_idx = viewpoint_indices[render_frame_idx]
    viewpoint_cam_k = train_cameras[viewpoint_idx]
    
    scaled_render_loss, render_pkg_k = _compute_rendering_loss(
        viewpoint_cam_k, pred_traj_first, batch_indices, trajectories,
        render_frame_idx, gaussians, pipe, background, xyz_only, dataset, opt, render_weight, scales=scales
    )
    
    total_loss = warmup_loss + scaled_render_loss
    render_pkgs = [render_pkg_k]
    
    return total_loss, warmup_loss, scaled_render_loss, render_pkgs

def compute_continuous_warmup_losses(loss,pred_traj_first, warmup_weight, render_weight, opt,
                                    train_cameras, fids, selected_fids, selected_cam_indices,
                                    batch_indices, traj_gt, gaussians, pipe, background, xyz_only,
                                    dataset, deform, scales = None):
    """Compute losses during warmup phase for continuous sampling."""
    # Sample observation trajectory to use the batch indices
    warmup_loss = loss * warmup_weight
    
    # Find a camera frame to render, if available
    render_frame_idx = None
    viewpoint_cam_k = None
    fid_val = None
    
    for cam_idx, curr_fid_val in enumerate(selected_fids):
        for i, current_fid in enumerate(fids):
            if abs(current_fid.item() - curr_fid_val) < 1e-5:
                render_frame_idx = i
                viewpoint_cam_k = train_cameras[selected_cam_indices[cam_idx]]
                fid_val = curr_fid_val
                break
        if viewpoint_cam_k is not None:
            break
    
    if viewpoint_cam_k is not None and render_frame_idx is not None:
        total_gaussians = gaussians.get_xyz.shape[0]
        scaled_render_loss, render_pkg_k = _compute_rendering_loss(
            viewpoint_cam_k, pred_traj_first, batch_indices, traj_gt,
            render_frame_idx, gaussians, pipe, background, xyz_only, dataset, opt, render_weight,
            deform, fid_val, total_gaussians, scales=scales
        )
        total_loss = warmup_loss + scaled_render_loss
        render_pkgs = [render_pkg_k]
    else:
        total_loss = warmup_loss
        scaled_render_loss = torch.tensor(0.0)
        render_pkgs = []
    
    return total_loss, warmup_loss, scaled_render_loss, render_pkgs

def compute_normal_losses(loss, pred_traj, ode_weight, render_weight, opt,
                         train_cameras, viewpoint_indices, batch_indices, trajectories, gaussians, pipe,
                         background, xyz_only, dataset, render_interval, window_length, scales = None):
    """Compute losses during normal training phase."""
    pred_traj = pred_traj.permute(1, 0, 2)
    
    # Rendering for training
    render_indices = list(range(0, window_length, render_interval))
    if window_length - 1 not in render_indices:
        render_indices.append(window_length - 1)
    
    rendering_losses = []
    render_pkgs = []
    for render_frame_idx in render_indices:
        viewpoint_idx = viewpoint_indices[render_frame_idx]
        viewpoint_cam_k = train_cameras[viewpoint_idx]
        pred_at_k = pred_traj[render_frame_idx]
        
        scaled_render_loss, render_pkg_k = _compute_rendering_loss(
            viewpoint_cam_k, pred_at_k, batch_indices, trajectories,
            render_frame_idx, gaussians, pipe, background, xyz_only, dataset, opt, 1.0, scales=scales  # Use 1.0 as we average later
        )
        
        render_pkgs.append(render_pkg_k)
        rendering_losses.append(scaled_render_loss)
    
    avg_rendering_loss = torch.mean(torch.stack(rendering_losses)) if rendering_losses else torch.tensor(0.0)
    scaled_ode_loss = loss * ode_weight
    scaled_render_loss = avg_rendering_loss * render_weight  # Apply render_weight to the average
    total_loss = scaled_ode_loss + scaled_render_loss
    
    return total_loss, scaled_ode_loss, scaled_render_loss, render_pkgs

def compute_continuous_normal_losses(loss, pred_traj, fids, 
                                    selected_fids, selected_cam_indices, train_cameras, batch_indices, 
                                    traj_gt, total_gaussians, gaussians, deform, pipe, background, 
                                    xyz_only, dataset, ode_weight, render_weight, opt, scales = None):
    """Compute losses during normal training phase for continuous sampling."""
    # Process observation trajectory
    pred_traj = pred_traj.permute(1, 0, 2)
    
    # Rendering for camera frames
    rendering_losses = []
    render_pkgs = []
    
    # Find all camera frames in our continuous sequence
    for cam_idx, fid_val in enumerate(selected_fids):
        # Find closest matching fid in our continuous sequence
        distances = torch.abs(fids - fid_val)
        render_idx = torch.argmin(distances).item()
        
        if distances[render_idx].item() < 1e-5:  # Only use exact matches
            viewpoint_cam_k = train_cameras[selected_cam_indices[cam_idx]]
            pred_at_k = pred_traj[render_idx]
            
            scaled_render_loss, render_pkg_k = _compute_rendering_loss(
                viewpoint_cam_k, pred_at_k, batch_indices, traj_gt,
                render_idx, gaussians, pipe, background, xyz_only, dataset, opt, 1.0,  # Use 1.0 as we average later
                deform, fid_val, total_gaussians, scales=scales
            )
            
            render_pkgs.append(render_pkg_k)
            rendering_losses.append(scaled_render_loss)
    
    # Calculate losses
    if rendering_losses:
        avg_rendering_loss = torch.mean(torch.stack(rendering_losses))
        scaled_render_loss = avg_rendering_loss * render_weight
        scaled_ode_loss = loss * ode_weight
        total_loss = scaled_ode_loss + scaled_render_loss
    else:
        scaled_ode_loss = loss * ode_weight
        scaled_render_loss = torch.tensor(0.0)
        total_loss = scaled_ode_loss
    
    return total_loss, scaled_ode_loss, scaled_render_loss, render_pkgs


def compute_render_only_loss(transformer_ode, train_cameras, train_trajectories, gaussians, pipe, background, 
                             xyz_only, dataset, opt, render_weight, total_gaussians, window_length, obs_length, 
                             batch_size, rendering_scales=None):
    """
    Compute rendering loss using extrapolation over a sampled discrete trajectory.
    
    Args:
        transformer_ode: The Transformer-ODE model
        train_cameras: List of training cameras
        train_trajectories: Ground truth trajectories
        gaussians: Gaussian model
        pipe: Pipeline parameters
        background: Background tensor
        xyz_only: Whether to use only XYZ coordinates
        dataset: Dataset parameters
        opt: Optimization parameters
        render_weight: Weight for rendering loss
        total_gaussians: Total number of gaussians
        window_length: Length of the trajectory window
        obs_length: Length of the observed trajectory
        batch_size: Batch size for sampling
        rendering_scales: Scales for multi-scale rendering
    
    Returns:
        total_loss: Total rendering loss
        avg_rendering_loss: Average rendering loss across frames
        render_pkgs: List of rendering packages
    """
    # Sample discrete trajectories
    obs_traj, target_traj, full_time, fids, viewpoint_indices, batch_indices = sample_discrete_trajectories(
        train_cameras, train_trajectories, window_length, obs_length, batch_size, total_gaussians
    )
    
    # Define observation and extrapolation times
    obs_time = full_time[:, :obs_length]  # (1, obs_length)
    extrapolate_time = full_time[:, obs_length:]  # (1, window_length - obs_length)
    
    # Extrapolate the full trajectory
    pred_traj = transformer_ode.extrapolate(obs_traj, obs_time[0], extrapolate_time[0])  # (B, window_length, obs_dim)
    
    # Compute rendering loss for all predicted frames with corresponding viewpoints
    rendering_losses = []
    render_pkgs = []
    
    for t in range(window_length):
        pred_at_t = pred_traj[:, t, :]  # (B, obs_dim)
        viewpoint_idx = viewpoint_indices[t]
        viewpoint_cam = train_cameras[viewpoint_idx]
        
        # Compute rendering loss for this frame
        render_loss, render_pkg = _compute_rendering_loss(
            viewpoint_cam, pred_at_t, batch_indices, train_trajectories,
            t, gaussians, pipe, background, xyz_only, dataset, opt, 1.0,  # Use 1.0 as we average later
            scales=rendering_scales
        )
        rendering_losses.append(render_loss)
        render_pkgs.append(render_pkg)
    
    # Average rendering losses
    avg_rendering_loss = torch.mean(torch.stack(rendering_losses)) if rendering_losses else torch.tensor(0.0, device="cuda")
    total_loss = avg_rendering_loss * render_weight
    
    return total_loss, avg_rendering_loss, render_pkgs

def compute_render_only_loss_warmup(transformer_ode, train_cameras, fids, batch_indices,
                                   gaussians, deform, pipe, background, xyz_only,
                                   dataset, opt, warmup_weight, render_weight, total_gaussians, rendering_scales=None):
    """
    Warmup version of rendering-only loss, using simpler extrapolation and fewer frames.
    """
    num_frames = min(3, len(train_cameras))
    camera_indices = torch.randperm(len(train_cameras))[:num_frames]
    selected_cameras = [train_cameras[idx] for idx in camera_indices]
    selected_fids = [cam.fid.item() for cam in selected_cameras]

    rendering_losses = []
    render_pkgs = []
    warmup_loss = 0.0

    for cam_idx, (camera, fid_val) in enumerate(zip(selected_cameras, selected_fids)):
        # Generate observation trajectory for extrapolation
        time_input = torch.tensor([fid_val], device="cuda").expand(total_gaussians, -1)
        with torch.no_grad():
            if xyz_only:
                d_xyz, _, _ = deform.step(gaussians.get_xyz.detach(), time_input)
                obs_traj = (d_xyz + gaussians.get_xyz)[batch_indices].unsqueeze(1)
            else:
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
                obs_traj = torch.cat([
                    d_xyz + gaussians.get_xyz,
                    d_rotation + gaussians.get_rotation,
                    d_scaling + gaussians.get_scaling
                ], dim=-1)[batch_indices].unsqueeze(1)

        obs_time = torch.tensor([0.0], device="cuda")
        extrapolate_time = torch.tensor([fid_val], device="cuda")
        pred_traj = transformer_ode.extrapolate(obs_traj, obs_time, extrapolate_time)
        frame_pred = pred_traj[:, -1, :]

        # Compute warmup loss using transformer_only_reconstruction
        warmup_loss_temp, _ = transformer_ode.transformer_only_reconstruction(obs_traj)
        warmup_loss += warmup_loss_temp

        # Render the frame
        render_loss, render_pkg = _compute_rendering_loss(
            camera, frame_pred, batch_indices, None, None,
            gaussians, pipe, background, xyz_only, dataset, opt,
            1.0, deform, fid_val, total_gaussians, scales=rendering_scales
        )
        rendering_losses.append(render_loss)
        render_pkgs.append(render_pkg)

    avg_rendering_loss = torch.mean(torch.stack(rendering_losses)) if rendering_losses else torch.tensor(0.0, device="cuda")
    avg_warmup_loss = warmup_loss / num_frames if num_frames > 0 else torch.tensor(0.0, device="cuda")
    total_loss = (avg_warmup_loss * warmup_weight) + (avg_rendering_loss * render_weight)

    return total_loss, avg_warmup_loss, avg_rendering_loss, render_pkgs
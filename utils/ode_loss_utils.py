import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_ode
def _compute_rendering_loss(viewpoint_cam, pred_values, batch_indices, trajectories_or_traj_gt, 
                           frame_idx, gaussians, pipe, background, xyz_only, dataset, opt, 
                           render_weight, deform=None, fid_val=None, total_gaussians=None):
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
    
    # Render and compute loss
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

def compute_warmup_losses(loss, pred_traj_first, warmup_weight, render_weight, opt, 
                         train_cameras, viewpoint_indices, batch_indices, trajectories, 
                         gaussians, pipe, background, xyz_only, dataset):
    """Compute losses during warmup phase."""
    # Transformer-only reconstruction
    warmup_loss = loss * warmup_weight
    
    # Rendering for the first frame only
    render_frame_idx = 0
    viewpoint_idx = viewpoint_indices[render_frame_idx]
    viewpoint_cam_k = train_cameras[viewpoint_idx]
    
    scaled_render_loss, render_pkg_k = _compute_rendering_loss(
        viewpoint_cam_k, pred_traj_first, batch_indices, trajectories,
        render_frame_idx, gaussians, pipe, background, xyz_only, dataset, opt, render_weight
    )
    
    total_loss = warmup_loss + scaled_render_loss
    render_pkgs = [render_pkg_k]
    
    return total_loss, warmup_loss, scaled_render_loss, render_pkgs

def compute_continuous_warmup_losses(loss,pred_traj_first, warmup_weight, render_weight, opt,
                                    train_cameras, fids, selected_fids, selected_cam_indices,
                                    batch_indices, traj_gt, gaussians, pipe, background, xyz_only,
                                    dataset, deform):
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
            deform, fid_val, total_gaussians
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
                         background, xyz_only, dataset, render_interval, window_length):
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
            render_frame_idx, gaussians, pipe, background, xyz_only, dataset, opt, 1.0  # Use 1.0 as we average later
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
                                    xyz_only, dataset, ode_weight, render_weight, opt):
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
                deform, fid_val, total_gaussians
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
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_ode

def compute_warmup_losses(transformer_ode, obs_traj, warmup_weight, render_weight, opt, 
                         train_cameras, viewpoint_indices, batch_indices, trajectories, 
                         gaussians, pipe, background, xyz_only, dataset):
    """Compute losses during warmup phase."""
    # Transformer-only reconstruction
    loss, pred_traj_first = transformer_ode.transformer_only_reconstruction(obs_traj)
    warmup_loss = loss * warmup_weight
    
    # Rendering for the first frame only
    render_frame_idx = 0  # First frame of the window
    viewpoint_idx = viewpoint_indices[render_frame_idx]
    viewpoint_cam_k = train_cameras[viewpoint_idx]
    
    if xyz_only:
        new_xyz_render_k = trajectories[render_frame_idx].clone()
        new_xyz_render_k[batch_indices] = pred_traj_first
        new_rotation_render_k = gaussians.get_rotation
        new_scaling_render_k = gaussians.get_scaling
    else:
        new_xyz_render_k = trajectories[render_frame_idx, :, :3].clone()
        new_rotation_render_k = trajectories[render_frame_idx, :, 3:7].clone()
        new_scaling_render_k = trajectories[render_frame_idx, :, 7:].clone()
        new_xyz_render_k[batch_indices] = pred_traj_first[:, :3]
        new_rotation_render_k[batch_indices] = pred_traj_first[:, 3:7]
        new_scaling_render_k[batch_indices] = pred_traj_first[:, 7:]
    
    render_pkg_k = render_ode(
        viewpoint_cam_k, gaussians, pipe, background,
        new_xyz_render_k, new_rotation_render_k, new_scaling_render_k,
        dataset.is_6dof
    )
    image_k = render_pkg_k["render"]
    gt_image_k = viewpoint_cam_k.original_image.cuda()
    Ll1_k = l1_loss(image_k, gt_image_k)
    rendering_loss_k = (1.0 - opt.lambda_dssim) * Ll1_k + opt.lambda_dssim * (1.0 - ssim(image_k, gt_image_k))
    scaled_render_loss = rendering_loss_k * render_weight
    
    total_loss = warmup_loss + scaled_render_loss
    render_pkgs = [render_pkg_k]
    
    return total_loss, warmup_loss, scaled_render_loss, render_pkgs

def compute_continuous_warmup_losses(transformer_ode, obs_traj, warmup_weight, render_weight, opt,
                                    train_cameras, fids, selected_fids, selected_cam_indices,
                                    batch_indices, traj_gt, gaussians, pipe, background, xyz_only,
                                    dataset, deform):
    """Compute losses during warmup phase for continuous sampling."""
    # Transformer-only reconstruction
    # First sample observation trajectory to use the batch indices
    obs_traj = obs_traj[batch_indices,:]
    loss, pred_traj_first = transformer_ode.transformer_only_reconstruction(obs_traj)
    warmup_loss = loss * warmup_weight
    
    # Find a camera frame to render, if available
    render_frame_idx = None
    viewpoint_cam_k = None
    
    for cam_idx, fid_val in enumerate(selected_fids):
        for i, current_fid in enumerate(fids):
            if abs(current_fid.item() - fid_val) < 1e-5:
                render_frame_idx = i
                viewpoint_cam_k = train_cameras[selected_cam_indices[cam_idx]]
                break
        if viewpoint_cam_k is not None:
            break
    
    if viewpoint_cam_k is not None and render_frame_idx is not None:
        if xyz_only:
            new_xyz_render_k = traj_gt[render_frame_idx].clone()
            new_xyz_render_k[batch_indices] = pred_traj_first
            new_rotation_render_k = gaussians.get_rotation
            new_scaling_render_k = gaussians.get_scaling
        else:
            new_xyz_render_k = traj_gt[render_frame_idx, :, :3].clone()
            new_rotation_render_k = traj_gt[render_frame_idx, :, 3:7].clone()
            new_scaling_render_k = traj_gt[render_frame_idx, :, 7:].clone()
            new_xyz_render_k[batch_indices] = pred_traj_first[:, :3]
            new_scaling_render_k[batch_indices] = pred_traj_first[:, 7:]
            new_rotation_render_k[batch_indices] = pred_traj_first[:, 3:7]
            
        
        render_pkg_k = render_ode(
            viewpoint_cam_k, gaussians, pipe, background,
            new_xyz_render_k, new_rotation_render_k, new_scaling_render_k,
            dataset.is_6dof
        )
        image_k = render_pkg_k["render"]
        gt_image_k = viewpoint_cam_k.original_image.cuda()
        Ll1_k = l1_loss(image_k, gt_image_k)
        rendering_loss_k = (1.0 - opt.lambda_dssim) * Ll1_k + opt.lambda_dssim * (1.0 - ssim(image_k, gt_image_k))
        scaled_render_loss = rendering_loss_k * render_weight
        total_loss = warmup_loss + scaled_render_loss
        render_pkgs = [render_pkg_k]
    else:
        total_loss = warmup_loss
        scaled_render_loss = torch.tensor(0.0)
        render_pkgs = []
    
    return total_loss, warmup_loss, scaled_render_loss, render_pkgs

def compute_normal_losses(transformer_ode, obs_traj, target_traj, full_time, ode_weight, render_weight, opt,
                         train_cameras, viewpoint_indices, batch_indices, trajectories, gaussians, pipe,
                         background, xyz_only, dataset, render_interval, window_length):
    """Compute losses during normal training phase."""
    loss, pred_traj = transformer_ode(obs_traj, target_traj, full_time)
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
        
        if xyz_only:
            new_xyz_render_k = trajectories[render_frame_idx].clone()
            new_xyz_render_k[batch_indices] = pred_at_k
            new_rotation_render_k = gaussians.get_rotation
            new_scaling_render_k = gaussians.get_scaling
        else:
            new_xyz_render_k = trajectories[render_frame_idx, :, :3].clone()
            new_rotation_render_k = trajectories[render_frame_idx, :, 3:7].clone()
            new_scaling_render_k = trajectories[render_frame_idx, :, 7:].clone()
            new_xyz_render_k[batch_indices] = pred_at_k[:, :3]
            new_rotation_render_k[batch_indices] = pred_at_k[:, 3:7]
            new_scaling_render_k[batch_indices] = pred_at_k[:, 7:]
        
        render_pkg_k = render_ode(
            viewpoint_cam_k, gaussians, pipe, background,
            new_xyz_render_k, new_rotation_render_k, new_scaling_render_k,
            dataset.is_6dof
        )
        render_pkgs.append(render_pkg_k)
        image_k = render_pkg_k["render"]
        gt_image_k = viewpoint_cam_k.original_image.cuda()
        Ll1_k = l1_loss(image_k, gt_image_k)
        rendering_loss_k = (1.0 - opt.lambda_dssim) * Ll1_k + opt.lambda_dssim * (1.0 - ssim(image_k, gt_image_k))
        rendering_losses.append(rendering_loss_k)
    
    avg_rendering_loss = torch.mean(torch.stack(rendering_losses)) if rendering_losses else torch.tensor(0.0)
    scaled_ode_loss = loss * ode_weight
    scaled_render_loss = avg_rendering_loss * render_weight
    total_loss = scaled_ode_loss + scaled_render_loss
    
    return total_loss, scaled_ode_loss, scaled_render_loss, render_pkgs

def compute_continuous_normal_losses(transformer_ode, obs_traj, target_traj, full_time, fids, 
                                    selected_fids, selected_cam_indices, train_cameras, batch_indices, 
                                    traj_gt, total_gaussians, gaussians, deform, pipe, background, 
                                    xyz_only, dataset, ode_weight, render_weight, opt):
    """Compute losses during normal training phase for continuous sampling."""
    # First sample observation trajectory to use the batch indices
    obs_traj = obs_traj[batch_indices,:]
    loss, pred_traj = transformer_ode(obs_traj, target_traj, full_time)
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
            
            if xyz_only:
                # Get the full Gaussian state for the current frame
                time_input = torch.tensor([fid_val], device="cuda").expand(total_gaussians, -1)
                with torch.no_grad():
                    d_xyz, _, _ = deform.step(gaussians.get_xyz.detach(), time_input)
                full_xyz = d_xyz + gaussians.get_xyz
                
                # Replace the batch indices with predicted values
                new_xyz_render_k = full_xyz.clone()
                new_xyz_render_k[batch_indices] = pred_at_k
                new_rotation_render_k = gaussians.get_rotation
                new_scaling_render_k = gaussians.get_scaling
            else:
                # Get the full Gaussian state for the current frame
                time_input = torch.tensor([fid_val], device="cuda").expand(total_gaussians, -1)
                with torch.no_grad():
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
                full_xyz = d_xyz + gaussians.get_xyz
                full_rotation = d_rotation + gaussians.get_rotation
                full_scaling = d_scaling + gaussians.get_scaling
                
                # Replace the batch indices with predicted values
                new_xyz_render_k = full_xyz.clone()
                new_rotation_render_k = full_rotation.clone()
                new_scaling_render_k = full_scaling.clone()
                new_xyz_render_k[batch_indices] = pred_at_k[:, :3]
                new_rotation_render_k[batch_indices] = pred_at_k[:, 3:7]
                new_scaling_render_k[batch_indices] = pred_at_k[:, 7:]
            
            render_pkg_k = render_ode(
                viewpoint_cam_k, gaussians, pipe, background,
                new_xyz_render_k, new_rotation_render_k, new_scaling_render_k,
                dataset.is_6dof
            )
            render_pkgs.append(render_pkg_k)
            image_k = render_pkg_k["render"]
            gt_image_k = viewpoint_cam_k.original_image.cuda()
            Ll1_k = l1_loss(image_k, gt_image_k)
            rendering_loss_k = (1.0 - opt.lambda_dssim) * Ll1_k + opt.lambda_dssim * (1.0 - ssim(image_k, gt_image_k))
            rendering_losses.append(rendering_loss_k)
    
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

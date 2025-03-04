import torch
import lpips
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render_ode

def perform_model_evaluation(transformer_ode, gaussians, pipe, background, dataset, 
                           train_cameras, val_cameras, train_trajectories, val_trajectories, 
                           val_batch_size, obs_length, fixed_val_indices, fixed_train_indices, tb_writer, 
                           iteration, testing_iterations, xyz_only, total_gaussians, window_length):
    """Evaluate model on training and validation data with multiple configurations."""
    torch.cuda.empty_cache()
    validation_configs = [
        {'name': 'val_full', 'cameras': val_cameras},
        #{'name': 'val_auto', 'cameras': val_cameras},
        # {'name': 'train_reconstruction', 'cameras': train_cameras},
        {'name': 'train_sliding_window', 'cameras': train_cameras}
    ]
    psnr_sum = 0.0
    psnr_count = 0
    ssim_sum = 0.0
    lpips_sum = 0.0
    
    # Initialize LPIPS model if not already defined in global scope
    device = torch.device("cuda:0")
    lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    
    with torch.no_grad():
        for config in validation_configs:
            name = config['name']
            cameras = config['cameras']
            if not cameras:
                continue
            
            images = []
            gts = []
            num_batches = (total_gaussians + val_batch_size - 1) // val_batch_size
            
            if name == 'val_full':
                obs_start = len(train_cameras) - obs_length
                obs_fids = torch.tensor([train_cameras[i].fid for i in range(obs_start, len(train_cameras))], device="cuda")
                val_fids = torch.tensor([cam.fid for cam in val_cameras], device="cuda")
                pred_traj_full = []
                for batch_idx in range(num_batches):
                    start = batch_idx * val_batch_size
                    end = min((batch_idx + 1) * val_batch_size, total_gaussians)
                    batch_indices = slice(start, end)
                    obs_traj = train_trajectories[obs_start:, batch_indices].permute(1, 0, 2)
                    
                    pred_traj = transformer_ode.extrapolate(obs_traj, obs_fids, val_fids)
                    pred_traj_full.append(pred_traj[:, obs_length:, :])
                pred_traj_full = torch.cat(pred_traj_full, dim=0).permute(1, 0, 2)
                
                for idx, viewpoint in enumerate(cameras):
                    pred_at_k = pred_traj_full[idx]
                    if xyz_only:
                        new_xyz = pred_at_k
                        new_rotation = gaussians.get_rotation
                        new_scaling = gaussians.get_scaling
                    else:
                        new_xyz = pred_at_k[:, :3]
                        new_rotation = pred_at_k[:, 3:7]
                        new_scaling = pred_at_k[:, 7:]
                    render_pkg = render_ode(viewpoint, gaussians, pipe, background, new_xyz, new_rotation, new_scaling, dataset.is_6dof)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda()
                    images.append(image.unsqueeze(0))
                    gts.append(gt_image.unsqueeze(0))
                    if idx in fixed_val_indices and tb_writer:
                        tb_writer.add_images(f"val_full_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"val_full_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
            
            elif name == 'val_auto':
                obs_fids = torch.tensor([train_cameras[i].fid for i in range(len(train_cameras) - obs_length, len(train_cameras))], device="cuda")
                val_fids = torch.tensor([cam.fid for cam in val_cameras], device="cuda")
                val_length = len(val_cameras)
                step_size = window_length - obs_length
                pred_traj_auto = []
                current_input = train_trajectories[-obs_length:].permute(1, 0, 2)
                obs_time = obs_fids
                for t in range(0, val_length, step_size):
                    t_end = min(t + step_size, val_length)
                    extrapolate_time = val_fids[t:t_end]
                    pred_x = []
                    for batch_idx in range(num_batches):
                        start = batch_idx * val_batch_size
                        end = min((batch_idx + 1) * val_batch_size, total_gaussians)
                        batch_indices = slice(start, end)
                        batch_input = current_input[batch_indices]
                        pred_batch = transformer_ode.extrapolate(batch_input, obs_time, extrapolate_time)
                        pred_x.append(pred_batch[:, -step_size:])
                    pred_x = torch.cat(pred_x, dim=0)
                    pred_traj_auto.append(pred_x)
                    current_input = pred_x[:, -obs_length:]
                    obs_time = extrapolate_time[-obs_length:]
                pred_traj_auto = torch.cat(pred_traj_auto, dim=1).permute(1, 0, 2)
                
                for idx, viewpoint in enumerate(cameras):
                    pred_at_k = pred_traj_auto[idx]
                    if xyz_only:
                        new_xyz = pred_at_k
                        new_rotation = gaussians.get_rotation
                        new_scaling = gaussians.get_scaling
                    else:
                        new_xyz = pred_at_k[:, :3]
                        new_rotation = pred_at_k[:, 3:7]
                        new_scaling = pred_at_k[:, 7:]
                    render_pkg = render_ode(viewpoint, gaussians, pipe, background, new_xyz, new_rotation, new_scaling, dataset.is_6dof)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda()
                    images.append(image.unsqueeze(0))
                    gts.append(gt_image.unsqueeze(0))
                    if idx in fixed_val_indices and tb_writer:
                        tb_writer.add_images(f"val_auto_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"val_auto_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                            
            elif name == 'train_reconstruction':
                # Train reconstruction mode - sliding window with step size obs_length
                train_length = len(train_cameras)
                train_fids = torch.tensor([cam.fid for cam in train_cameras], device="cuda")
                
                pred_traj_recon = [None] * train_length  # Initialize list for all training frames
                
                # Slide window with step size obs_length
                for t_start in range(0, train_length - obs_length + 1, obs_length):
                    t_end = t_start + obs_length
                    obs_time = train_fids[t_start:t_end]
                    
                    # Get observation trajectories for this window
                    obs_traj = train_trajectories[t_start:t_end].permute(1, 0, 2)  # (num_gaussians, obs_length, obs_dim)
                    
                    # Reconstruct using transformer_ode.reconstruct
                    num_batches = (total_gaussians + val_batch_size - 1) // val_batch_size
                    window_preds = []
                    for batch_idx in range(num_batches):
                        start = batch_idx * val_batch_size
                        end = min((batch_idx + 1) * val_batch_size, total_gaussians)
                        batch_indices = slice(start, end)
                        batch_obs_traj = obs_traj[batch_indices]  # (batch_size, obs_length, obs_dim)
                        batch_preds = transformer_ode.reconstruct(batch_obs_traj, obs_time)  # (batch_size, obs_length, obs_dim)
                        window_preds.append(batch_preds)
                    window_preds = torch.cat(window_preds, dim=0)  # (num_gaussians, obs_length, obs_dim)
                    
                    # Store predictions for this window
                    for i, idx in enumerate(range(t_start, t_end)):
                        pred_traj_recon[idx] = window_preds[:, i, :]  # (num_gaussians, obs_dim)
            
                # Render and evaluate
                for idx, viewpoint in enumerate(cameras):
                    if pred_traj_recon[idx] is None:
                        continue  # Skip frames that weren't predicted (e.g., at the end)
                    
                    pred_at_k = pred_traj_recon[idx]
                    if xyz_only:
                        new_xyz = pred_at_k
                        new_rotation = gaussians.get_rotation
                        new_scaling = gaussians.get_scaling
                    else:
                        new_xyz = pred_at_k[:, :3]
                        new_rotation = pred_at_k[:, 3:7]
                        new_scaling = pred_at_k[:, 7:]
                        
                    render_pkg = render_ode(viewpoint, gaussians, pipe, background, new_xyz, new_rotation, new_scaling, dataset.is_6dof)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda()
                    images.append(image.unsqueeze(0))
                    gts.append(gt_image.unsqueeze(0))
                    
                    if idx in fixed_train_indices and tb_writer:
                        tb_writer.add_images(f"train_reconstruction_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"train_reconstruction_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
            
            elif name == 'train_sliding_window':
                # Train sliding window mode - with extrapolation
                train_length = len(train_cameras)
                predict_length = window_length - obs_length
                train_fids = torch.tensor([cam.fid for cam in train_cameras], device="cuda")
                
                pred_traj_sliding = [None] * train_length  # Fix: Initialize with full training length
                
                # Process each prediction window
                for t_end in range(obs_length, train_length, predict_length):
                    t_start = t_end - obs_length
                    t_extrap_end = min(t_end + predict_length, train_length)
                    
                    obs_time = train_fids[t_start:t_end]
                    extrap_time = train_fids[t_end:t_extrap_end]
                    
                    if len(extrap_time) == 0:
                        continue  # Skip if no frames to extrapolate
                    
                    window_preds = None
                    
                    for batch_idx in range(num_batches):
                        start = batch_idx * val_batch_size
                        end = min((batch_idx + 1) * val_batch_size, total_gaussians)
                        batch_indices = slice(start, end)
                        
                        obs_traj = train_trajectories[t_start:t_end, batch_indices].permute(1, 0, 2)
                        batch_preds = transformer_ode.extrapolate(obs_traj, obs_time, extrap_time)
                        
                        if window_preds is None:
                            window_preds = batch_preds
                        else:
                            window_preds = torch.cat([window_preds, batch_preds], dim=0)
                    
                    # Add predictions to our list (extrapolation only)
                    for i, idx in enumerate(range(t_end, t_extrap_end)):
                        pred_traj_sliding[idx] = window_preds[:, i, :].clone()
                
                # Render and evaluate for frames we were able to predict
                for idx, viewpoint in enumerate(cameras):
                    if idx < obs_length or pred_traj_sliding[idx] is None:
                        continue  # Skip frames we couldn't predict
                    
                    pred_at_k = pred_traj_sliding[idx]
                    if xyz_only:
                        new_xyz = pred_at_k
                        new_rotation = gaussians.get_rotation
                        new_scaling = gaussians.get_scaling
                    else:
                        new_xyz = pred_at_k[:, :3]
                        new_rotation = pred_at_k[:, 3:7]
                        new_scaling = pred_at_k[:, 7:]
                        
                    render_pkg = render_ode(viewpoint, gaussians, pipe, background, new_xyz, new_rotation, new_scaling, dataset.is_6dof)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda()
                    images.append(image.unsqueeze(0))
                    gts.append(gt_image.unsqueeze(0))
                    
                    if idx in fixed_train_indices and tb_writer:
                        tb_writer.add_images(f"train_sliding_window_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"train_sliding_window_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
            
            if images:
                images = torch.cat(images, dim=0)
                gts = torch.cat(gts, dim=0)
                
                # Calculate L1 loss
                l1_test = l1_loss(images, gts)
                
                # Calculate PSNR
                psnr_values = []
                for image, gt in zip(images, gts):
                    psnr_values.append(psnr(image, gt))
                psnr_test = torch.mean(torch.stack(psnr_values))
                
                # Calculate SSIM
                ssim_values = []
                for image, gt in zip(images, gts):
                    ssim_values.append(ssim(image.unsqueeze(0), gt.unsqueeze(0)))
                ssim_test = torch.mean(torch.stack(ssim_values))
                
                # Calculate LPIPS
                lpips_values = []
                for image, gt in zip(images, gts):
                    lpips_values.append(lpips_vgg(image.unsqueeze(0), gt.unsqueeze(0)).detach())
                lpips_test = torch.mean(torch.stack(lpips_values))
                
                if tb_writer:
                    tb_writer.add_scalar(f'{name}/loss_viewpoint_l1', l1_test.item(), iteration)
                    tb_writer.add_scalar(f'{name}/loss_viewpoint_psnr', psnr_test.item(), iteration)
                    tb_writer.add_scalar(f'{name}/loss_viewpoint_ssim', ssim_test.item(), iteration)
                    tb_writer.add_scalar(f'{name}/loss_viewpoint_lpips', lpips_test.item(), iteration)
                
                print(f"\n[ITER {iteration}] {name}: L1 {l1_test.item():.4f}, PSNR {psnr_test.item():.4f}, SSIM {ssim_test.item():.4f}, LPIPS {lpips_test.item():.4f}")
                
                psnr_sum += psnr_test.item()
                psnr_count += 1
                ssim_sum += ssim_test.item()
                lpips_sum += lpips_test.item()
        
        avg_psnr = psnr_sum / psnr_count if psnr_count > 0 else -float('inf')
        avg_ssim = ssim_sum / psnr_count if psnr_count > 0 else -float('inf')
        avg_lpips = lpips_sum / psnr_count if psnr_count > 0 else float('inf')
        
        # Log the average metrics
        if tb_writer:
            tb_writer.add_scalar('evaluation/avg_psnr', avg_psnr, iteration)
            tb_writer.add_scalar('evaluation/avg_ssim', avg_ssim, iteration)
            tb_writer.add_scalar('evaluation/avg_lpips', avg_lpips, iteration)
            
        print(f"\n[ITER {iteration}] Average metrics - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
        
        torch.cuda.empty_cache()
    
    # Return a dictionary of metrics instead of just PSNR
    metrics = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }
    
    return metrics
import torch
import os
from random import sample
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scene.forecast_ode_transformer import TransformerLatentODEWrapper
from scene.forecast_ode_contiformer import ContiformerLatentODEWrapper
from scene.forecast_transformer import AutoregressiveTransformer
from scene.forecast_ode_rnn import ODE_RNN_Model
from scene.forecast_ode_var_rnn import LatentODERNN
import numpy as np
from argparse import ArgumentParser, Namespace
import sys
import uuid
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.ode_loss_utils import compute_warmup_losses, compute_normal_losses, compute_continuous_warmup_losses, compute_continuous_normal_losses, compute_render_only_loss, compute_render_only_loss_warmup
from utils.ode_sampling_utils import sample_ode_only_fids, sample_rendering_included_fids, sample_discrete_trajectories
from utils.ode_load_utils import load_or_generate_trajectories, load_models
from utils.ode_eval_utils import perform_model_evaluation

class TrainingState:
    def __init__(self, starting_gap, min_gap, decrease_gap_factor=0.5):
        self.iteration = 0
        self.current_gap = starting_gap
        self.min_gap = min_gap
        self.is_trajectory_mode = True  # Start with trajectory-only
        self.decrease_gap_factor = decrease_gap_factor

    def update(self):
        self.iteration += 1
        if self.iteration >= self.current_gap:
            self.iteration = 0
            self.is_trajectory_mode = not self.is_trajectory_mode
            self.current_gap = max(self.current_gap * self.decrease_gap_factor, self.min_gap)
            print(f"Switching to {'trajectory' if self.is_trajectory_mode else 'rendering'} mode, new gap: {self.current_gap}")

    def should_train_simultaneously(self):
        return self.current_gap <= self.min_gap

def prepare_output_and_logger(args, log_directory=None):
    """Set up output directory and TensorBoard logger."""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    if log_directory is None:
        log_dir = os.path.join(args.model_path, "logs")
    else:
        log_dir = os.path.join(args.model_path, log_directory)
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

def setup_training(dataset, gaussians, train_gaussians, opt,
                   learning_rate, latent_dim, d_model, nhead,
                   num_encoder_layers, num_decoder_layers, 
                   ode_nhidden, decoder_nhidden, noise_std, 
                   ode_layers, reg_weight, voxel_size, spatial_shape,
                   variational_inference, use_torchode, use_contiformer, use_ode_rnn, use_latent_ode_rnn, use_pc_encoder, use_transformer_baseline,
                   rtol, atol, use_tanh, num_active_odes):
    """Set up training configurations and optimizers."""
    obs_dim = 3 if dataset.xyz_only else 10
    if use_pc_encoder:
        print("Using PointPreservingSpUNetODEWrapper")
        transformer_ode = PointPreservingSpUNetODEWrapper(
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            ode_nhidden=ode_nhidden,
            decoder_nhidden=decoder_nhidden,
            obs_dim=obs_dim,
            noise_std=noise_std,
            ode_layers=ode_layers,
            reg_weight=reg_weight,
            variational_inference=variational_inference,
            use_torchode=use_torchode,
            use_rnn=use_latent_ode_rnn,
            voxel_size=voxel_size,
            spatial_shape=spatial_shape,
            rtol=rtol,
            atol=atol,
            use_tanh=use_tanh,
            num_active_odes=num_active_odes
        ).cuda()
    elif use_contiformer:
        print("Using ContiformerLatentODEWrapper")
        transformer_ode = ContiformerLatentODEWrapper(
            latent_dim=latent_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden,
            obs_dim=obs_dim, noise_std=noise_std, ode_layers=ode_layers, rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    elif use_ode_rnn:
        print("Using ODE_RNN_Model")
        transformer_ode = ODE_RNN_Model(
            latent_dim=latent_dim, d_model=d_model, nhead=nhead, num_decoder_layers=num_decoder_layers,
            ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden, obs_dim=obs_dim, ode_layers=ode_layers, 
            reg_weight=reg_weight, variational_inference=variational_inference, use_torchode=use_torchode,
            rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    elif use_latent_ode_rnn:
        print("Using LatentODERNN")
        transformer_ode = LatentODERNN(
            latent_dim=latent_dim, rec_dim=d_model, num_decoder_layers=num_decoder_layers,
            ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden, obs_dim=obs_dim, 
            noise_std=noise_std, ode_layers=ode_layers, reg_weight=reg_weight,
            variational_inference=variational_inference, use_torchode=use_torchode, rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    elif use_transformer_baseline:
        print("Using AutoregressiveTransformer")
        transformer_ode = AutoregressiveTransformer(
            latent_dim=latent_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden,
            obs_dim=obs_dim, noise_std=noise_std, ode_layers=ode_layers, reg_weight=reg_weight,
        ).cuda()
    else:
        print("Using TransformerLatentODEWrapper")
        transformer_ode = TransformerLatentODEWrapper(
            latent_dim=latent_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden,
            obs_dim=obs_dim, noise_std=noise_std, ode_layers=ode_layers, reg_weight=reg_weight,
            variational_inference=variational_inference, use_torchode=use_torchode, rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    
    if train_gaussians:
        gaussians.training_setup(opt)
    
    transformer_ode_optimizer = torch.optim.Adam(transformer_ode.parameters(), lr=learning_rate)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    return transformer_ode, transformer_ode_optimizer, background

def split_train_val_data(scene, all_trajectories, time_split=0.8 ):
    """Split data into training and validation sets."""
    all_cameras = scene.getTrainCameras().copy()
    all_cameras.extend(scene.getTestCameras().copy())
    all_cameras.sort(key=lambda x: x.fid)
    total_frames = len(all_cameras)
    split_idx = int(total_frames * time_split)
    train_cameras = all_cameras[:split_idx]
    val_cameras = all_cameras[split_idx:]
    train_trajectories = all_trajectories[:split_idx]
    val_trajectories = all_trajectories[split_idx:]
    return train_cameras, val_cameras, train_trajectories, val_trajectories

def generate_continuous_trajectories(deform, fids, gaussians, xyz_only=False):
    """Generate trajectories on-the-fly using the deform model."""
    xyz = gaussians.get_xyz
    rotation = gaussians.get_rotation
    scaling = gaussians.get_scaling
    num_gaussians = xyz.shape[0]
    traj_gt = []
    
    for fid in fids:
        time_input = fid.unsqueeze(0).expand(num_gaussians, -1)
        with torch.no_grad():
            if xyz_only:
                d_xyz, _, _ = deform.step(xyz.detach(), time_input)
                t_xyz = d_xyz + xyz
                trajectories = t_xyz
            else:
                d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                t_xyz = d_xyz + xyz
                t_rotation = d_rotation + rotation
                t_scaling = d_scaling + scaling
                trajectories = torch.cat([t_xyz, t_rotation, t_scaling], dim=-1)
        traj_gt.append(trajectories)
    
    return torch.stack(traj_gt, dim=0)

def training(dataset, opt, pipe, num_testing_iterations, num_saving_iterations, train_gaussians=False, 
             window_length=40, obs_length=20, batch_size=1024, render_interval=4, 
             val_batch_size=8192, learning_rate=1e-3, 
             xyz_only=False, warmup_iterations=500, continuous_time_sampling=True, 
             ode_only_ratio=0.7, render_weight=1, warmup_weight=1, ode_weight=1e-3, reg_weight=1e-3,
             latent_dim=10, d_model=128, nhead=8, num_encoder_layers=5, num_decoder_layers=5, 
             ode_nhidden=20, decoder_nhidden=128, noise_std=0.1, ode_layers=1, variational_inference=True,
             log_directory=None, use_torchode=False, use_contiformer=False, use_ode_rnn=False, 
             use_latent_ode_rnn=False, use_pc_encoder=False, use_transformer_baseline=False,
             voxel_size=0.05, spatial_shape=[128, 128, 128],
             rtol=1e-1, atol=1e-1, use_tanh=False, num_active_odes=-1, min_lr=0, rendering_scales=None,
             starting_gap=100, min_gap=10, decrease_gap_factor=0.5, time_split=0.8):
    """
    Main training function with alternating trajectory-only and rendering-only modes.
    """
    testing_iterations = np.linspace(100, opt.iterations, num=num_testing_iterations, dtype=int).tolist()
    saving_iterations = np.linspace(100, opt.iterations, num=num_saving_iterations, dtype=int).tolist()

    print(f"Testing iterations: {testing_iterations}")
    print(f"Saving iterations: {saving_iterations}")

    # Setup
    tb_writer = prepare_output_and_logger(dataset, log_directory)
    gaussians, scene, deform = load_models(dataset)
    transformer_ode, transformer_ode_optimizer, background = setup_training(
        dataset, gaussians, train_gaussians, opt, learning_rate, latent_dim, d_model,
        nhead, num_encoder_layers, num_decoder_layers, ode_nhidden, decoder_nhidden, noise_std,
        ode_layers, reg_weight, voxel_size, spatial_shape, variational_inference, use_torchode,
        use_contiformer, use_ode_rnn, use_latent_ode_rnn, use_pc_encoder, use_transformer_baseline, rtol, atol, use_tanh, num_active_odes
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(transformer_ode_optimizer, T_max=opt.iterations, eta_min=min_lr)
    total_gaussians = gaussians.get_xyz.shape[0]
    print("Total Gaussians:", total_gaussians)

    # Load or generate trajectories for discrete sampling
    all_trajectories = load_or_generate_trajectories(dataset, scene, gaussians, deform, xyz_only)
    train_cameras, val_cameras, train_trajectories, val_trajectories = split_train_val_data(scene, all_trajectories,time_split=time_split)
    
    # Extract fid range for continuous sampling
    min_fid = train_cameras[0].fid.item()
    max_fid = train_cameras[-1].fid.item()
    print("Fid range:", min_fid, max_fid)
    
    # Progress tracking
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    num_samples = min(6, len(val_cameras))
    fixed_val_indices = np.linspace(0, len(val_cameras) - 1, num_samples, dtype=int).tolist()
    fixed_train_indices = np.linspace(0, len(train_cameras) - 1, num_samples, dtype=int).tolist()
    
    # Best model tracking
    best_metrics = {'psnr': -float('inf'), 'ssim': -float('inf'), 'lpips': float('inf')}
    best_iteration = None

    encoder_decoder_frozen = False
    training_state = TrainingState(starting_gap, min_gap, decrease_gap_factor)

    # Training loop
    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        if not training_state.should_train_simultaneously():
            training_state.update()

        # Check if we need to freeze encoder/decoder after warmup phase
        if iteration == warmup_iterations + 1 and not encoder_decoder_frozen and use_pc_encoder:
            print(f"\n[ITER {iteration}] Warmup complete. Freezing encoder/decoder components...")
            freeze_stats = transformer_ode.freeze_encoder_decoder()
            encoder_decoder_frozen = True
            transformer_ode_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, transformer_ode.parameters()), 
                lr=learning_rate
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(transformer_ode_optimizer, T_max=opt.iterations, eta_min=min_lr)
            scheduler.last_epoch = iteration - 1
            if tb_writer:
                tb_writer.add_scalar('training/trainable_params_before', freeze_stats["params_before"], iteration)
                tb_writer.add_scalar('training/trainable_params_after', freeze_stats["params_after"], iteration)
                tb_writer.add_scalar('training/param_reduction_percent', freeze_stats["reduction_percent"], iteration)

        # Training logic
        pred_traj = None
        recon_loss = torch.tensor(0.0)
        pred_loss = torch.tensor(0.0)
        kl_loss = torch.tensor(0.0)
        reg_loss = torch.tensor(0.0)
        
        if continuous_time_sampling:
            if training_state.should_train_simultaneously():
                fids, selected_fids, selected_cam_indices = sample_rendering_included_fids(
                    train_cameras, min_fid, max_fid, window_length, obs_length
                )
                traj_gt = generate_continuous_trajectories(deform, fids, gaussians, xyz_only)
                batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
                obs_traj = traj_gt[:obs_length].permute(1, 0, 2)
                target_traj = traj_gt[obs_length:].permute(1, 0, 2)
                full_time = fids.unsqueeze(0)
                if iteration <= warmup_iterations:
                    loss, pred_traj = transformer_ode.transformer_only_reconstruction(obs_traj[batch_indices])
                    total_loss, warmup_loss, scaled_render_loss, render_pkgs = compute_continuous_warmup_losses(
                        loss, pred_traj, warmup_weight, render_weight, opt, train_cameras, fids, 
                        selected_fids, selected_cam_indices, batch_indices, traj_gt, gaussians, 
                        pipe, background, xyz_only, dataset, deform, rendering_scales
                    )
                    scaled_ode_loss = warmup_loss
                else:
                    loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_traj = transformer_ode.forward(
                        obs_traj[batch_indices], target_traj[batch_indices], full_time
                    )
                    total_loss, scaled_ode_loss, scaled_render_loss, render_pkgs = compute_continuous_normal_losses(
                        loss, pred_traj, fids, selected_fids, selected_cam_indices, train_cameras, 
                        batch_indices, traj_gt, total_gaussians, gaussians, deform, pipe, background, 
                        xyz_only, dataset, ode_weight, render_weight, opt, rendering_scales
                    )
            else:
                if training_state.is_trajectory_mode:
                    fids = sample_ode_only_fids(min_fid, max_fid, window_length)
                    traj_gt = generate_continuous_trajectories(deform, fids, gaussians, xyz_only)
                    batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
                    traj_gt = traj_gt[:, batch_indices]
                    obs_traj = traj_gt[:obs_length].permute(1, 0, 2)
                    target_traj = traj_gt[obs_length:].permute(1, 0, 2)
                    full_time = fids.unsqueeze(0)
                    if iteration <= warmup_iterations:
                        loss, _ = transformer_ode.transformer_only_reconstruction(obs_traj)
                        total_loss = loss * warmup_weight
                        scaled_ode_loss = total_loss
                        scaled_render_loss = torch.tensor(0.0)
                    else:
                        loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_traj = transformer_ode.forward(obs_traj, target_traj, full_time)
                        total_loss = loss * ode_weight
                        scaled_ode_loss = total_loss
                        scaled_render_loss = torch.tensor(0.0)
                    render_pkgs = []
                else:
                    batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
                    if iteration <= warmup_iterations:
                        total_loss, warmup_loss, scaled_render_loss, render_pkgs = compute_render_only_loss_warmup(
                            transformer_ode, train_cameras, [cam.fid.item() for cam in train_cameras], 
                            batch_indices, gaussians, deform, pipe, background, xyz_only, dataset, 
                            opt, warmup_weight, render_weight, total_gaussians, rendering_scales
                        )
                        scaled_ode_loss = torch.tensor(0.0)
                    else:
                        total_loss, scaled_render_loss, render_pkgs = compute_render_only_loss(
                            transformer_ode, train_cameras, train_trajectories, gaussians, pipe, background, 
                            xyz_only, dataset, opt, render_weight, total_gaussians, window_length, obs_length, 
                            batch_size, rendering_scales
                        )
                        scaled_ode_loss = torch.tensor(0.0)
        else:
            if training_state.should_train_simultaneously():
                obs_traj, target_traj, full_time, fids, viewpoint_indices, batch_indices = sample_discrete_trajectories(
                    train_cameras, train_trajectories, window_length, obs_length, batch_size, total_gaussians
                )
                if iteration <= warmup_iterations:
                    loss, pred_traj = transformer_ode.transformer_only_reconstruction(obs_traj)
                    total_loss, warmup_loss, scaled_render_loss, render_pkgs = compute_warmup_losses(
                        loss, pred_traj, warmup_weight, render_weight, opt, train_cameras, viewpoint_indices, 
                        batch_indices, train_trajectories, gaussians, pipe, background, xyz_only, dataset, rendering_scales
                    )
                    scaled_ode_loss = warmup_loss
                else:
                    loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_traj = transformer_ode.forward(obs_traj, target_traj, full_time)
                    total_loss, scaled_ode_loss, scaled_render_loss, render_pkgs = compute_normal_losses(
                        loss, pred_traj, ode_weight, render_weight, opt, train_cameras, viewpoint_indices, 
                        batch_indices, train_trajectories, gaussians, pipe, background, xyz_only, dataset, 
                        render_interval, window_length, rendering_scales
                    )
            else:
                if training_state.is_trajectory_mode:
                    obs_traj, target_traj, full_time, fids, viewpoint_indices, batch_indices = sample_discrete_trajectories(
                        train_cameras, train_trajectories, window_length, obs_length, batch_size, total_gaussians
                    )
                    if iteration <= warmup_iterations:
                        loss, _ = transformer_ode.transformer_only_reconstruction(obs_traj)
                        total_loss = loss * warmup_weight
                        scaled_ode_loss = total_loss
                        scaled_render_loss = torch.tensor(0.0)
                    else:
                        loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_traj = transformer_ode.forward(obs_traj, target_traj, full_time)
                        total_loss = loss * ode_weight
                        scaled_ode_loss = total_loss
                        scaled_render_loss = torch.tensor(0.0)
                    render_pkgs = []
                else:
                    if iteration <= warmup_iterations:
                        batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
                        total_loss, warmup_loss, scaled_render_loss, render_pkgs = compute_render_only_loss_warmup(
                            transformer_ode, train_cameras, [cam.fid.item() for cam in train_cameras], 
                            batch_indices, gaussians, deform, pipe, background, xyz_only, dataset, 
                            opt, warmup_weight, render_weight, total_gaussians, rendering_scales
                        )
                        scaled_ode_loss = torch.tensor(0.0)
                    else:
                        total_loss, scaled_render_loss, render_pkgs = compute_render_only_loss(
                            transformer_ode, train_cameras, train_trajectories, gaussians, pipe, background, 
                            xyz_only, dataset, opt, render_weight, total_gaussians, window_length, obs_length, 
                            batch_size, rendering_scales
                        )
                        scaled_ode_loss = torch.tensor(0.0)

        # Backward pass
        total_loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{total_loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            if train_gaussians and iteration < opt.densify_until_iter and len(render_pkgs) > 0:
                for render_pkg in render_pkgs:
                    visibility_filter = render_pkg["visibility_filter"]
                    radii = render_pkg["radii"]
                    viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            
            if tb_writer:
                if iteration <= warmup_iterations:
                    tb_writer.add_scalar('train_loss/warmup_loss', scaled_ode_loss.item(), iteration)
                else:
                    tb_writer.add_scalar('train_loss/transformer_ode_loss', scaled_ode_loss.item(), iteration)
                    tb_writer.add_scalar('train_loss/ode_recon_loss', recon_loss.item() * ode_weight, iteration)
                    tb_writer.add_scalar('train_loss/ode_pred_loss', pred_loss.item() * ode_weight, iteration)
                    if variational_inference:
                        tb_writer.add_scalar('train_loss/ode_kl_loss', kl_loss.item() * ode_weight, iteration)
                    tb_writer.add_scalar('train_loss/ode_reg_loss', reg_loss.item() * ode_weight, iteration)
                tb_writer.add_scalar('train_loss/avg_rendering_loss', scaled_render_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/total_loss', total_loss.item(), iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)
                tb_writer.add_scalar('training_mode', 1 if training_state.is_trajectory_mode else 0, iteration)
                current_lr = transformer_ode_optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('learning_rate', current_lr, iteration)
            
            transformer_ode_optimizer.step()
            scheduler.step()
            transformer_ode_optimizer.zero_grad()
            if train_gaussians and len(render_pkgs) > 0:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)
        
        if iteration in saving_iterations:
            save_dir = os.path.join(dataset.model_path, "extrapolate_model", f"iteration_{iteration}") if log_directory is None else os.path.join(dataset.model_path, log_directory, f"iteration_{iteration}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "model.pth")
            torch.save({
                'iteration': iteration,
                'model_state_dict': transformer_ode.state_dict(),
                'optimizer_state_dict': transformer_ode_optimizer.state_dict(),
            }, save_path)
            print(f"\n[ITER {iteration}] Saved model to {save_path}")
        
        if iteration in testing_iterations:
            metrics = perform_model_evaluation(
                transformer_ode, gaussians, pipe, background, dataset,
                train_cameras, val_cameras, train_trajectories, val_trajectories,
                val_batch_size, obs_length, fixed_val_indices, fixed_train_indices, tb_writer,
                iteration, testing_iterations, xyz_only, total_gaussians, window_length
            )
            
            if metrics['psnr'] > best_metrics['psnr']:
                best_metrics = metrics
                best_iteration = iteration
                best_save_dir = os.path.join(dataset.model_path, "extrapolate_model")
                os.makedirs(best_save_dir, exist_ok=True)
                best_save_path = os.path.join(best_save_dir, "best_model.pth")
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': transformer_ode.state_dict(),
                    'optimizer_state_dict': transformer_ode_optimizer.state_dict(),
                    'best_metrics': best_metrics
                }, best_save_path)
                print(f"\n[ITER {iteration}] Saved best model with PSNR {metrics['psnr']:.4f}, SSIM {metrics['ssim']:.4f}, LPIPS {metrics['lpips']:.4f} to {best_save_path}")
    
    if tb_writer:
        tb_writer.close()
    print(f"\nTraining complete. Best metrics at iteration {best_iteration}:")
    print(f"  PSNR: {best_metrics['psnr']:.4f}")
    print(f"  SSIM: {best_metrics['ssim']:.4f}")
    print(f"  LPIPS: {best_metrics['lpips']:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # Training parameters
    parser.add_argument('--train_gaussians', action='store_true', default=False)
    parser.add_argument('--window_length', type=int, default=20)
    parser.add_argument('--obs_length', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--val_batch_size', type=int, default=2048)
    parser.add_argument('--trajectory_path', type=str, default=None)
    parser.add_argument('--render_interval', type=int, default=4)
    parser.add_argument('--test_iterations', nargs="+", type=int, default=20)
    parser.add_argument('--save_iterations', nargs="+", type=int, default=20)
    parser.add_argument('--custom_iterations', type=int, default=None)

    # Optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_iterations', type=int, default=1000)
    
    # Model configuration
    parser.add_argument('--xyz_only', action='store_true', default=False)
    parser.add_argument('--continuous_time_sampling', action='store_true', default=True)
    
    # Loss weights
    parser.add_argument('--ode_only_ratio', type=float, default=0.5)
    parser.add_argument('--render_weight', type=float, default=1)
    parser.add_argument('--warmup_weight', type=float, default=1)
    parser.add_argument('--ode_weight', type=float, default=1)
    parser.add_argument('--reg_weight', type=float, default=1e-3)
    parser.add_argument('--rendering_scales', type=list, default=[0])

    # Architecture parameters
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--ode_nhidden', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--ode_layers', type=int, default=2)
    parser.add_argument('--encoder_nhidden', type=int, default=32)
    parser.add_argument('--decoder_nhidden', type=int, default=32)

    # ODE solver parameters
    parser.add_argument('--rtol', type=float, default=1e-1)
    parser.add_argument('--atol', type=float, default=1e-1)
    parser.add_argument('--noise_std', type=float, default=0.001)
    
    # Model variants and features
    parser.add_argument('--var_inf', action='store_true', default=False)
    parser.add_argument('--use_torchode', action='store_true', default=False)
    parser.add_argument('--use_contiformer', action='store_true', default=False)
    parser.add_argument('--use_ode_rnn', action='store_true', default=False)
    parser.add_argument('--use_pc_encoder', action='store_true', default=False)
    parser.add_argument('--use_latent_ode_rnn', action='store_true', default=False)
    parser.add_argument('--use_transformer_baseline', action='store_true', default=False)
    parser.add_argument('--use_tanh', action='store_true', default=False)
    parser.add_argument('--num_active_odes', type=int, default=1)

    # Point cloud encoder parameters
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--spatial_shape', type=list, default=[128, 128, 128])

    # Alternating training parameters
    parser.add_argument('--starting_gap', type=int, default=40000)
    parser.add_argument('--min_gap', type=int, default=10)
    parser.add_argument('--decrease_gap_factor', type=float, default=0.95)
    parser.add_argument('--time_split', type=float, default=0.8)
    # Logging
    parser.add_argument('--log_directory', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    if args.custom_iterations is not None:
        args.save_iterations.append(args.custom_iterations)
    
    lp_args = lp.extract(args)
    lp_args.trajectory_path = args.trajectory_path
    op_original = op.extract(args)
    if args.custom_iterations is not None:
        op_original.iterations = args.custom_iterations
    
    dataset = lp_args
    dataset.xyz_only = args.xyz_only
    if args.use_transformer_baseline:
        args.use_torchode = False
        args.use_contiformer = False
        args.use_ode_rnn = False
        args.use_latent_ode_rnn = False
        args.use_pc_encoder = False
        args.use_var_inf = False
        args.continuous_time_sampling = False
    print("Beginning training...")
    training(
        dataset, op_original, pp.extract(args), 
        args.test_iterations, args.save_iterations,
        args.train_gaussians, args.window_length, args.obs_length, 
        args.batch_size, args.render_interval, args.val_batch_size, 
        args.learning_rate, args.xyz_only, 
        args.warmup_iterations, args.continuous_time_sampling, args.ode_only_ratio,
        args.render_weight, args.warmup_weight, args.ode_weight, args.reg_weight,
        args.latent_dim, args.encoder_nhidden, args.nhead, args.num_encoder_layers,
        args.num_decoder_layers, args.ode_nhidden, args.decoder_nhidden, args.noise_std, args.ode_layers, args.var_inf,
        args.log_directory, args.use_torchode, args.use_contiformer, args.use_ode_rnn, args.use_latent_ode_rnn, args.use_pc_encoder, args.use_transformer_baseline,
        args.voxel_size, args.spatial_shape,
        args.rtol, args.atol, args.use_tanh, args.num_active_odes,
        args.min_lr, args.rendering_scales,
        args.starting_gap, args.min_gap, args.decrease_gap_factor, args.time_split
    )
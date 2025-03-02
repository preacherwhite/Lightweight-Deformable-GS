import torch
import os
from random import randint, sample
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_ode
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scene.forecast_ode_transformer import TransformerLatentODEWrapper
from scene.forecast_ode_contiformer import ContiformerLatentODEWrapper
from scene.forecast_ode_rnn import ODE_RNN_Model
from scene.forecast_ode_var_rnn import LatentODERNN
import numpy as np
from argparse import ArgumentParser, Namespace
import sys
import uuid
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.image_utils import psnr
from utils.ode_loss_utils import compute_warmup_losses, compute_normal_losses, compute_continuous_warmup_losses, compute_continuous_normal_losses
from utils.ode_sampling_utils import sample_ode_only_fids, sample_rendering_included_fids, sample_discrete_trajectories
from utils.ode_load_utils import load_or_generate_trajectories, load_models

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
                      ode_layers, reg_weight,
                    variational_inference, use_torchode, use_contiformer, use_ode_rnn, use_latent_ode_rnn,
                    rtol, atol, use_tanh):
    """Set up training configurations and optimizers."""
    obs_dim = 3 if dataset.xyz_only else 10
    if use_contiformer:
        print("Using ContiformerLatentODEWrapper")
        transformer_ode = ContiformerLatentODEWrapper(
            latent_dim=latent_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,
        ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden, obs_dim=obs_dim, noise_std=noise_std,
            ode_layers=ode_layers, rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    elif use_ode_rnn:
        print("Using ODE_RNN_Model")
        transformer_ode = ODE_RNN_Model(
            latent_dim=latent_dim, d_model=d_model, nhead=nhead, num_decoder_layers=num_decoder_layers,
            ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden, obs_dim=obs_dim, ode_layers=ode_layers, 
            reg_weight=reg_weight, variational_inference=variational_inference, use_torchode=use_torchode, rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    elif use_latent_ode_rnn:
        print("Using LatentODERNN")
        # For LatentODERNN, d_model is used as rec_dim (recurrent hidden dimension)
        transformer_ode = LatentODERNN(
            latent_dim=latent_dim, rec_dim=d_model, num_decoder_layers=num_decoder_layers,
            ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden, obs_dim=obs_dim, 
            noise_std=noise_std, ode_layers=ode_layers, reg_weight=reg_weight,
            variational_inference=variational_inference, use_torchode=use_torchode, rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    else:
        print("Using TransformerLatentODEWrapper")
        transformer_ode = TransformerLatentODEWrapper(
            latent_dim=latent_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,
            ode_nhidden=ode_nhidden, decoder_nhidden=decoder_nhidden, obs_dim=obs_dim, noise_std=noise_std,
            ode_layers=ode_layers, reg_weight=reg_weight, variational_inference=variational_inference, use_torchode=use_torchode, rtol=rtol, atol=atol, use_tanh=use_tanh
        ).cuda()
    
    if train_gaussians:
        gaussians.training_setup(opt)
    
    transformer_ode_optimizer = torch.optim.Adam(transformer_ode.parameters(), lr=learning_rate)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    return transformer_ode, transformer_ode_optimizer, background

def split_train_val_data(scene, all_trajectories):
    """Split data into training and validation sets."""
    all_cameras = scene.getTrainCameras().copy()
    all_cameras.extend(scene.getTestCameras().copy())
    all_cameras.sort(key=lambda x: x.fid)
    total_frames = len(all_cameras)
    split_idx = int(total_frames * 0.8)
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
                l1_test = l1_loss(images, gts)
                psnr_test = []
                for image, gt in zip(images, gts):
                    psnr_test.append(psnr(image, gt))
                psnr_test = torch.mean(torch.stack(psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(f'{name}/loss_viewpoint_l1', l1_test.item(), iteration)
                    tb_writer.add_scalar(f'{name}/loss_viewpoint_psnr', psnr_test.item(), iteration)
                print(f"\n[ITER {iteration}] {name}: L1 {l1_test.item():.4f}, PSNR {psnr_test.item():.4f}")
                psnr_sum += psnr_test.item()
                psnr_count += 1
        
        avg_psnr = psnr_sum / psnr_count if psnr_count > 0 else -float('inf')
        torch.cuda.empty_cache()
    return avg_psnr
       

def training(dataset, opt, pipe, num_testing_iterations, num_saving_iterations, train_gaussians=False, 
          window_length=40, obs_length=20, batch_size=1024, render_interval=4, 
          val_batch_size=8192,learning_rate=1e-3, 
          xyz_only=False, warmup_iterations=500, continuous_time_sampling=True, 
          ode_only_ratio=0.7, render_weight=1, warmup_weight=1, ode_weight=1e-3, reg_weight=1e-3,
          latent_dim=10, d_model=128, nhead=8, num_encoder_layers=5, num_decoder_layers=5, ode_nhidden=20, decoder_nhidden=128, noise_std=0.1, ode_layers=1, variational_inference=True,
          log_directory=None, use_torchode=False, use_contiformer=False, use_ode_rnn=False, use_latent_ode_rnn=False,
          rtol=1e-1, atol=1e-1, use_tanh=False):
    """
    Main training function for Transformer-ODE model.
    
    Args:
        dataset: Dataset object containing scene data
        opt: Optimization parameters
        pipe: Pipeline parameters
        num_testing_iterations: Number of iterations for evaluation
        num_saving_iterations: Number of iterations for saving models
        train_gaussians: Whether to train Gaussian model
        window_length: Length of the prediction window
        obs_length: Length of the observation window
        batch_size: Batch size for training
        render_interval: Interval for rendering frames during training
        val_batch_size: Batch size for validation
        learning_rate: Learning rate for optimizer
        xyz_only: Whether to use only XYZ coordinates
        warmup_iterations: Number of iterations for warmup phase
        continuous_time_sampling: Whether to use continuous time sampling
        ode_only_ratio: Ratio of ODE-only training
        variational_inference: Whether to use variational inference for ODE
    """

    
    testing_iterations = np.linspace(100, opt.iterations, num=num_testing_iterations, dtype=int).tolist()
    saving_iterations = np.linspace(100, opt.iterations, num=num_saving_iterations, dtype=int).tolist()

    print(f"Testing iterations: {testing_iterations}")
    print(f"Saving iterations: {saving_iterations}")

    # Setup
    print("Preparing output and logger")
    tb_writer = prepare_output_and_logger(dataset, log_directory)
    
    # Load models
    print("Loading models")
    gaussians, scene, deform = load_models(dataset)
    
    # Set up optimizers and initialize Transformer-ODE
    print("Setting up optimizers and initializing Transformer-ODE")
    transformer_ode, transformer_ode_optimizer, background = setup_training(
        dataset, gaussians, train_gaussians,
          opt, learning_rate, latent_dim, d_model,
            nhead, num_encoder_layers, num_decoder_layers,
              ode_nhidden, decoder_nhidden, noise_std,
                ode_layers, reg_weight, variational_inference, use_torchode, use_contiformer, use_ode_rnn, use_latent_ode_rnn,
                rtol, atol, use_tanh
    )
    print("Setting up optimizers and initializing Transformer-ODE complete")
    total_gaussians = gaussians.get_xyz.shape[0]
    print("Total Gaussians:", total_gaussians)

    # Load or generate trajectories for discrete sampling
    print("Loading or generating trajectories for discrete sampling")
    all_trajectories = load_or_generate_trajectories(dataset, scene, gaussians, deform, xyz_only)
    train_cameras, val_cameras, train_trajectories, val_trajectories = split_train_val_data(scene, all_trajectories)
    
    # Extract fid range for continuous sampling
    print("Extracting fid range for continuous sampling")
    min_fid = train_cameras[0].fid.item()
    max_fid = train_cameras[-1].fid.item()
    print("Fid range:", min_fid, max_fid)
    
    # Progress tracking
    print("Setting up progress tracking")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    num_samples = min(6, len(val_cameras))  # Up to 5 samples
    fixed_val_indices = np.linspace(0, len(val_cameras) - 1, num_samples, dtype=int).tolist()
    fixed_train_indices = np.linspace(0, len(train_cameras) - 1, num_samples, dtype=int).tolist()
    # Best model tracking
    best_psnr = -float('inf')
    best_iteration = None

    # Training loop
    print(f"Training for {opt.iterations} iterations\n")
    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        
        # Decide training mode: ODE-only or rendering-included
        do_ode_only = torch.rand(1).item() < ode_only_ratio
        pred_traj = None
        if continuous_time_sampling:
            # Continuous time sampling approach
            if do_ode_only:
                # Pure ODE training with random continuous fids
                fids = sample_ode_only_fids(min_fid, max_fid, window_length)
                # Generate trajectories on-the-fly
                traj_gt = generate_continuous_trajectories(deform, fids, gaussians, xyz_only)
                # Sample batch of Gaussians
                batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
                traj_gt = traj_gt[:, batch_indices]
                # Split into observation and target
                obs_traj = traj_gt[:obs_length].permute(1, 0, 2)
                target_traj = traj_gt[obs_length:].permute(1, 0, 2)
                full_time = fids.unsqueeze(0)
                # Compute loss based on training phase
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
                
                render_pkgs = []  # No rendering in this case
                
            else:
                # Rendering-included training
                fids, selected_fids, selected_cam_indices = sample_rendering_included_fids(
                    train_cameras, min_fid, max_fid, window_length, obs_length
                )
                
                # Generate trajectories on-the-fly
                traj_gt = generate_continuous_trajectories(deform, fids, gaussians, xyz_only)
                
                # Sample batch of Gaussians
                batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
                #traj_gt = traj_gt[:, batch_indices]
                
                # Split into observation and target
                obs_traj = traj_gt[:obs_length].permute(1, 0, 2)
                target_traj = traj_gt[obs_length:].permute(1, 0, 2)
                full_time = fids.unsqueeze(0)
                
                # Compute loss based on training phase
                if iteration <= warmup_iterations:
                    loss, pred_traj = transformer_ode.transformer_only_reconstruction(obs_traj[batch_indices])
                    
                    total_loss, warmup_loss, scaled_render_loss, render_pkgs = compute_continuous_warmup_losses(
                        loss, pred_traj, warmup_weight, render_weight, opt,
                        train_cameras, fids, selected_fids, selected_cam_indices,
                        batch_indices, traj_gt, gaussians, pipe, background, xyz_only,
                        dataset, deform
                    )
                    scaled_ode_loss = warmup_loss
                else:
                    loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_traj = transformer_ode.forward(obs_traj[batch_indices], target_traj[batch_indices], full_time)
                    total_loss, scaled_ode_loss, scaled_render_loss, render_pkgs = compute_continuous_normal_losses(
                        loss, pred_traj, fids, 
                        selected_fids, selected_cam_indices, train_cameras, batch_indices, 
                        traj_gt, total_gaussians, gaussians, deform, pipe, background, 
                        xyz_only, dataset, ode_weight, render_weight, opt
                    )
        else:
            # Original discrete sampling logic from existing trajectories
            obs_traj, target_traj, full_time, fids, viewpoint_indices, batch_indices = sample_discrete_trajectories(
                train_cameras, train_trajectories, window_length, obs_length, batch_size, total_gaussians
            )
            
            # Compute loss based on training phase
            if iteration <= warmup_iterations:
                loss, pred_traj = transformer_ode.transformer_only_reconstruction(obs_traj[batch_indices])
                total_loss, warmup_loss, scaled_render_loss, render_pkgs = compute_warmup_losses(
                    loss, pred_traj, warmup_weight, render_weight, opt,
                    train_cameras, viewpoint_indices, batch_indices, train_trajectories,
                    gaussians, pipe, background, xyz_only, dataset
                )
                scaled_ode_loss = warmup_loss
            else:
                loss, recon_loss, pred_loss, kl_loss, reg_loss, pred_traj = transformer_ode.forward(obs_traj[batch_indices], target_traj[batch_indices], full_time)
                total_loss, scaled_ode_loss, scaled_render_loss, render_pkgs = compute_normal_losses(
                    loss, pred_traj, fids, 
                    selected_fids, selected_cam_indices, train_cameras, batch_indices, 
                    traj_gt, total_gaussians, gaussians, deform, pipe, background, 
                    xyz_only, dataset, ode_weight, render_weight, opt
                )
        # Backward pass
        total_loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Update progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{total_loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Densification logic for Gaussians if needed
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
            
            # TensorBoard logging
            if tb_writer:
                if iteration <= warmup_iterations:
                    tb_writer.add_scalar('train_loss/warmup_loss', scaled_ode_loss.item(), iteration)
                else:
                    tb_writer.add_scalar('train_loss/transformer_ode_loss', scaled_ode_loss.item(), iteration)
                    tb_writer.add_scalar('train_loss/ode_recon_loss', recon_loss.item()*ode_weight, iteration)
                    tb_writer.add_scalar('train_loss/ode_pred_loss', pred_loss.item()*ode_weight, iteration)
                    if variational_inference:
                        tb_writer.add_scalar('train_loss/ode_kl_loss', kl_loss.item()*ode_weight, iteration)
                    tb_writer.add_scalar('train_loss/ode_reg_loss', reg_loss.item()*ode_weight, iteration)
                tb_writer.add_scalar('train_loss/avg_rendering_loss', scaled_render_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/total_loss', total_loss.item(), iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)
                tb_writer.add_scalar('training_mode', 1 if do_ode_only else 0, iteration)  # Log which mode we're in
            
            # Optimizer steps
            transformer_ode_optimizer.step()
            transformer_ode_optimizer.zero_grad()
            if train_gaussians and len(render_pkgs) > 0:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)
        
        # Save model at specified iterations
        if iteration in saving_iterations:
            if log_directory is None:
                save_dir = os.path.join(dataset.model_path,"extrapolate_model", f"iteration_{iteration}")
            else:
                save_dir = os.path.join(dataset.model_path, log_directory, f"iteration_{iteration}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "model.pth")
            torch.save({
                'iteration': iteration,
                'model_state_dict': transformer_ode.state_dict(),
                'optimizer_state_dict': transformer_ode_optimizer.state_dict(),
            }, save_path)
            print(f"\n[ITER {iteration}] Saved model to {save_path}")
        
        # Evaluate model at specified iterations
        if iteration in testing_iterations:
            avg_psnr = perform_model_evaluation(
                transformer_ode, gaussians, pipe, background, dataset,
                train_cameras, val_cameras, train_trajectories, val_trajectories,
                val_batch_size, obs_length, fixed_val_indices, fixed_train_indices, tb_writer,
                iteration, testing_iterations, xyz_only, total_gaussians, window_length
            )
            
            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_iteration = iteration
                best_save_dir = os.path.join(dataset.model_path, "extrapolate_model")
                os.makedirs(best_save_dir, exist_ok=True)
                best_save_path = os.path.join(best_save_dir, "best_model.pth")
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': transformer_ode.state_dict(),
                    'optimizer_state_dict': transformer_ode_optimizer.state_dict(),
                    'best_psnr': best_psnr
                }, best_save_path)
                print(f"\n[ITER {iteration}] Saved best model with PSNR {best_psnr:.4f} to {best_save_path}")
    
    if tb_writer:
        tb_writer.close()
    print(f"\nTraining complete. Best PSNR: {best_psnr:.4f} at iteration {best_iteration}")



if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--train_gaussians', action='store_true', default=False)
    parser.add_argument('--window_length', type=int, default=40)
    parser.add_argument('--obs_length', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--val_batch_size', type=int, default=8192)
    parser.add_argument('--trajectory_path', type=str, default=None)
    parser.add_argument('--render_interval', type=int, default=4)
    parser.add_argument('--test_iterations', nargs="+", type=int, default= 20)
    parser.add_argument('--save_iterations', nargs="+", type=int, default= 20)
    parser.add_argument('--custom_iterations', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--xyz_only', action='store_true', default=False, help="Use only XYZ coordinates for ODE and DeformModel")
    parser.add_argument('--warmup_iterations', type=int, default=200, help="Number of iterations for warmup phase using transformer_only_reconstruction")
    parser.add_argument('--continuous_time_sampling', action='store_true', default=True, help="Use continuous time sampling instead of discrete")
    parser.add_argument('--ode_only_ratio', type=float, default=0.7, help="Ratio of ODE-only training vs rendering-included training")
    parser.add_argument('--render_weight', type=float, default=1, help="Weight for rendering loss")
    parser.add_argument('--warmup_weight', type=float, default=1, help="Weight for warmup loss")
    parser.add_argument('--ode_weight', type=float, default=1, help="Weight for ODE loss")
    parser.add_argument('--reg_weight', type=float, default=0, help="Weight for regularization loss")
    
    parser.add_argument('--nhead', type=int, default=4, help="Number of attention heads for ODE")
    parser.add_argument('--num_encoder_layers', type=int, default=2, help="Number of encoder layers for transformer")
    parser.add_argument('--num_decoder_layers', type=int, default=2, help="Number of decoder layers for transformer")
    parser.add_argument('--ode_nhidden', type=int, default=32, help="Number of hidden units for ODE")
    parser.add_argument('--latent_dim', type=int, default=32 , help="Latent dimension for ODE")
    parser.add_argument('--ode_layers', type=int, default=2, help="Number of ODE layers")
    parser.add_argument('--encoder_nhidden', type=int, default=64, help="Model dimension for encoder")
    parser.add_argument('--decoder_nhidden', type=int, default=64, help="Number of hidden units for decoder for ODE")

    parser.add_argument('--rtol', type=float, default=1e-1, help="Relative tolerance for ODE")
    parser.add_argument('--atol', type=float, default=1e-1, help="Absolute tolerance for ODE")

    parser.add_argument('--noise_std', type=float, default=0.01, help="Noise standard deviation for ODE")
    parser.add_argument('--var_inf', action='store_true', default=False, help="Use variational inference for ODE")

    parser.add_argument('--log_directory', type=str, default=None, help="Log directory")
    parser.add_argument('--use_torchode', action='store_true', default=False, help="Use torchode for ODE")
    parser.add_argument('--use_contiformer', action='store_true', default=False, help="Use contiformer for ODE")
    parser.add_argument('--use_ode_rnn', action='store_true', default=False, help="Use ode_rnn for ODE")
    # Add to the parser arguments at the bottom of run_models.py
    parser.add_argument('--use_latent_ode_rnn', action='store_true', default=False, help="Use Latent ODE with ODE-RNN encoder")
    parser.add_argument('--use_tanh', action='store_true', default=False, help="Use tanh for network activation")
    args = parser.parse_args(sys.argv[1:])
    if args.custom_iterations is not None:
        args.save_iterations.append(args.custom_iterations)
    
    # Make sure dataset has access to trajectory_path
    lp_args = lp.extract(args)
    lp_args.trajectory_path = args.trajectory_path
    
    op_original = op.extract(args)
    if args.custom_iterations is not None:
        op_original.iterations = args.custom_iterations
    
    dataset = lp_args
    dataset.xyz_only = args.xyz_only  # Pass xyz_only to dataset
    
    print("Beginning training...")
    # And modify the training function call to include the new parameter
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
        args.log_directory, args.use_torchode, args.use_contiformer, args.use_ode_rnn, args.use_latent_ode_rnn,
        args.rtol, args.atol, args.use_tanh
    )
import torch
import os
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_ode
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scene.forecast_ode_transformer import TransformerLatentODEWrapper
import numpy as np
from argparse import ArgumentParser, Namespace
import sys
import uuid
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.image_utils import psnr
from random import sample

def training(dataset, opt, pipe, testing_iterations, saving_iterations, train_gaussians=False, window_length=40, obs_length=20, batch_size=1024, render_interval=4, val_batch_size=8192, trajectory_path=None, learning_rate=1e-3, xyz_only=False, warmup_iterations=500):
    tb_writer = prepare_output_and_logger(dataset)
    
    # Load pre-trained models
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    print("loaded {} gaussians".format(gaussians.get_xyz.shape[0]))
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.load_weights(dataset.model_path, iteration=-1)
    deform.deform.eval()

    # Load or generate trajectories
    if trajectory_path is not None:
        all_trajectories = np.load(trajectory_path)
        print("loaded trajectories from {}".format(trajectory_path))
    else:
        trajectory_file = os.path.join(dataset.model_path, "trajectories.npy")
        if os.path.exists(trajectory_file):
            all_trajectories = np.load(trajectory_file)
            all_trajectories = torch.from_numpy(all_trajectories).to('cuda')
            print("loaded trajectories from {}".format(trajectory_file))
        else:
            viewpoint_stack_a = scene.getTrainCameras().copy()
            viewpoint_stack_b = scene.getTestCameras().copy()
            viewpoint_stack = viewpoint_stack_a + viewpoint_stack_b
            viewpoint_stack.sort(key=lambda x: x.fid)
            split_idx = int(len(viewpoint_stack) * 0.8)
            viewpoint_stack = viewpoint_stack[:split_idx]
            xyz = gaussians.get_xyz
            rotation = gaussians.get_rotation
            scaling = gaussians.get_scaling
            num_gaussians = xyz.shape[0]
            all_trajectories = []
            for viewpoint_cam in viewpoint_stack:
                fid = viewpoint_cam.fid
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
                    all_trajectories.append(trajectories)
            all_trajectories = torch.stack(all_trajectories, dim=0)
            np.save(trajectory_file, all_trajectories.cpu().numpy())

    if xyz_only and (trajectory_path is not None or os.path.exists(trajectory_file)):
        all_trajectories = all_trajectories[..., :3]

    # Initialize Transformer ODE
    obs_dim = 3 if xyz_only else 10
    transformer_ode = TransformerLatentODEWrapper(
        latent_dim=10, d_model=128, nhead=8, num_encoder_layers=5,
        ode_nhidden=20, decoder_nhidden=128, obs_dim=obs_dim, noise_std=0.1, ode_layers=1
    ).cuda()
    
    # Set up optimizers
    if train_gaussians:
        gaussians.training_setup(opt)
    transformer_ode_optimizer = torch.optim.Adam(transformer_ode.parameters(), lr=learning_rate)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    
    # Dataset split based on fid
    all_cameras = scene.getTrainCameras().copy()
    all_cameras.extend(scene.getTestCameras().copy())
    all_cameras.sort(key=lambda x: x.fid)
    total_frames = len(all_cameras)
    split_idx = int(total_frames * 0.8)
    train_cameras = all_cameras[:split_idx]
    val_cameras = all_cameras[split_idx:]
    train_trajectories = all_trajectories[:split_idx]
    val_trajectories = all_trajectories[split_idx:]
    total_gaussians = gaussians.get_xyz.shape[0]
    target_length = window_length - obs_length
    
    # Loss balancing parameters
    ode_weight = 1e-3
    render_weight = 1
    warmup_weight = 1  # Weight for warmup reconstruction loss
    fixed_val_indices = [0, 5, 10, 15, 20][:len(val_cameras)]
    
    # Best model tracking
    best_psnr = -float('inf')
    best_iteration = None

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()

        # Sample training data
        if len(train_cameras) <= window_length:
            start_idx = 0
        else:
            start_idx = randint(obs_length, len(train_cameras) - target_length - 1)
        viewpoint_indices = list(range(start_idx - obs_length, start_idx + target_length))
        fids = torch.tensor([train_cameras[i].fid for i in viewpoint_indices], device="cuda")
        batch_indices = sample(range(total_gaussians), min(batch_size, total_gaussians))
        trajectories = train_trajectories[viewpoint_indices]
        traj_gt = trajectories[:, batch_indices]
        
        obs_traj = traj_gt[:obs_length].permute(1, 0, 2)
        target_traj = traj_gt[obs_length:].permute(1, 0, 2)
        full_time = fids.unsqueeze(0)
        
        # Warmup phase: Use transformer_only_reconstruction
        if iteration <= warmup_iterations:
            # Use only the first frame of obs_traj
            loss, pred_traj_first = transformer_ode.transformer_only_reconstruction(obs_traj)  # pred_traj_first: (B, obs_dim)
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
            render_pkgs = [render_pkg_k]  # For densification if train_gaussians is True
        
        # Normal ODE-based training
        else:
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
            
            avg_rendering_loss = torch.mean(torch.stack(rendering_losses))
            scaled_ode_loss = loss * ode_weight
            scaled_render_loss = avg_rendering_loss * render_weight
            total_loss = scaled_ode_loss + scaled_render_loss
        
        total_loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{total_loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            if train_gaussians and iteration < opt.densify_until_iter:
                for render_pkg in render_pkgs:
                    visibility_filter = render_pkg["visibility_filter"]
                    radii = render_pkg["radii"]
                    viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            
            if tb_writer:
                if iteration <= warmup_iterations:
                    tb_writer.add_scalar('train_loss/warmup_loss', warmup_loss.item(), iteration)
                else:
                    tb_writer.add_scalar('train_loss/transformer_ode_loss', scaled_ode_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/avg_rendering_loss', scaled_render_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/total_loss', total_loss.item(), iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)
            
            transformer_ode_optimizer.step()
            transformer_ode_optimizer.zero_grad()
            if train_gaussians:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)
        
        if iteration in saving_iterations:
            save_dir = os.path.join(dataset.model_path, "extrapolate_model", f"iteration_{iteration}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "model.pth")
            torch.save({
                'iteration': iteration,
                'model_state_dict': transformer_ode.state_dict(),
                'optimizer_state_dict': transformer_ode_optimizer.state_dict(),
            }, save_path)
            print(f"\n[ITER {iteration}] Saved model to {save_path}")
        
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = [
                {'name': 'val_full', 'cameras': val_cameras},
                {'name': 'val_auto', 'cameras': val_cameras}
            ]
            psnr_sum = 0.0
            psnr_count = 0
            
            for config in validation_configs:
                name = config['name']
                cameras = config['cameras']
                if not cameras:
                    continue
                
                images = []
                gts = []
                total_gaussians = gaussians.get_xyz.shape[0]
                num_batches = (total_gaussians + val_batch_size - 1) // val_batch_size
                
                if name == 'val_full':
                    obs_start = len(train_cameras) - obs_length
                    obs_fids = torch.tensor([train_cameras[i].fid for i in range(obs_start, len(train_cameras))], device="cuda")
                    val_fids = torch.tensor([cam.fid for cam in val_cameras], device="cuda")
                    full_time = torch.cat([obs_fids, val_fids]).unsqueeze(0)
                    pred_traj_full = []
                    for batch_idx in range(num_batches):
                        start = batch_idx * val_batch_size
                        end = min((batch_idx + 1) * val_batch_size, total_gaussians)
                        batch_indices = slice(start, end)
                        obs_traj = train_trajectories[obs_start:, batch_indices].permute(1, 0, 2)
                        with torch.no_grad():
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
                    step_size = target_length
                    pred_traj_auto = []
                    current_input = train_trajectories[-obs_length:].permute(1, 0, 2)
                    obs_time = obs_fids
                    with torch.no_grad():
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
            
            torch.cuda.empty_cache()
    
    if tb_writer:
        tb_writer.close()
    print(f"\nTraining complete. Best PSNR: {best_psnr:.4f} at iteration {best_iteration}")

def prepare_output_and_logger(args):
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
    log_dir = os.path.join(args.model_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--train_gaussians', action='store_true', default=False)
    parser.add_argument('--window_length', type=int, default=20)
    parser.add_argument('--obs_length', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--val_batch_size', type=int, default=8192)
    parser.add_argument('--trajectory_path', type=str, default=None)
    parser.add_argument('--render_interval', type=int, default=2)
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    parser.add_argument('--save_iterations', nargs="+", type=int, default=[500, 1000, 1500, 2000])
    parser.add_argument('--custom_iterations', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--xyz_only', action='store_true', default=False, help="Use only XYZ coordinates for ODE and DeformModel")
    parser.add_argument('--warmup_iterations', type=int, default=500, help="Number of iterations for warmup phase using transformer_only_reconstruction")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.custom_iterations)
    op_original = op.extract(args)
    if args.custom_iterations is not None:
        op_original.iterations = args.custom_iterations
    training(
        lp.extract(args), op_original, pp.extract(args), args.test_iterations, args.save_iterations,
        args.train_gaussians, args.window_length, args.obs_length, args.batch_size, args.render_interval,
        args.val_batch_size, args.trajectory_path, args.learning_rate, args.xyz_only, args.warmup_iterations
    )
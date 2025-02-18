#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui, render_ode
import sys
from scene import Scene, GaussianModel, DeformModel, SetDeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.sh_utils import SH2RGB
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

PRE_TRAINED_GAUSSIANS = True
PRE_TRAINED_DEFORM = False
PRE_TRAINED_LOCATION = "/media/staging2/dhwang/Lightweight-Deformable-GS/output/set_pretrain_spunet_stand"
USE_COLOR= True
DETACH_DEFORM = False
num_cams_per_iter = 10
sequence_length = 20
SPREAD_OUT_SEQUENCE = False
def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = SetDeformModel()
    deform.train_setting(opt)
    if PRE_TRAINED_DEFORM:
        deform.load_weights(PRE_TRAINED_LOCATION)
    if PRE_TRAINED_GAUSSIANS:       
        #Test loading saved guassians
        model_path_tmp = dataset.model_path
        dataset.model_path = PRE_TRAINED_LOCATION
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
        print("loaded {} guassians".format(gaussians.get_xyz.shape[0]))
        scene.model_path = model_path_tmp
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    # Print frozen parameters in deform model
    print("Frozen parameters in deform model:")
    for name, param in deform.deform.named_parameters():
        if not param.requires_grad:
            print(f"  {name}")
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack or iteration == opt.warm_up:
            viewpoint_stack = scene.getTrainCameras().copy()
            # Print all fids in the viewpoint stack
            if iteration >= opt.warm_up:
                # Sort the viewpoints by fid
                # TODO: Remove the first viewpoint from the stack due to repeated fit, check later
                viewpoint_stack = sorted(viewpoint_stack, key=lambda x: x.fid)[1:]
                # Calculate the total number of viewpoints
                total_viewpoints = len(viewpoint_stack)
                
                # Calculate the step size for uniform distribution
                step = (total_viewpoints - 1) / (sequence_length - 1)
                
                # Select uniformly spread out viewpoints
                if SPREAD_OUT_SEQUENCE:
                    selected_indices = [int(round(i * step)) for i in range(sequence_length)]
                    viewpoint_stack = [viewpoint_stack[i] for i in selected_indices]
                else:
                    viewpoint_stack = viewpoint_stack[:sequence_length]

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        #start sampling from the viewpoint stack
        k = min(num_cams_per_iter, len(viewpoint_stack))
        sampled_indices = random.sample(range(len(viewpoint_stack)), k)
        sampled_indices.sort()  # Sort to maintain temporal order
        sampled_cams = [viewpoint_stack[i] for i in sampled_indices]
        if dataset.load2gpu_on_the_fly:
            for i in sampled_indices:
                sampled_cams[i].load2device()

        # Remove the sampled cameras from the stack
        viewpoint_stack = [cam for i, cam in enumerate(viewpoint_stack) if i not in sampled_indices]
        sampled_fids = [viewpoint_cam.fid for viewpoint_cam in sampled_cams]

        if iteration < opt.warm_up:
            d_xyz_list = [0.0 for _ in sampled_cams]
            d_rotation_list = [0.0 for _ in sampled_cams]
            d_scaling_list = [0.0 for _ in sampled_cams]
        else:
            N = gaussians.get_xyz.shape[0]
            # TODO: do we add time to the input? time_embed->encoder->latent->deform_ode, theoretically ode is dependent on time
            # However the ode directly operates on the latent space, which means it only takes in raw gaussian parameters + time on the initial position
            # when calculating ode values for intermediate steps, it uses the latent space directly without need for time input 
            # unless we sample initial values other than the first viewpoint, time likely is not making a difference. Here we'll still concat it to the input
            # for consistency

            # initial latent is calculated with the first frame, for the first version this will always be 0: canonical frame
            time_input = torch.zeros(N, 1, device='cuda')

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            time_sequence = torch.Tensor(sampled_fids).to('cuda')
            #check if 0 is the first frame, if not add it in the beginning
            if time_sequence[0] != 0:
                time_sequence = torch.cat((torch.zeros(1, device='cuda'), time_sequence), dim=0)
                add_zero = True
            else:
                add_zero = False
            if USE_COLOR:
                # here we use the first coefficient of the SH as the color
                gaussian_shs = gaussians.get_features
                sh0 = gaussian_shs[:,0, :]
                rgb = SH2RGB(sh0)
                # Now here the deform takes in the time_input which is the (potentially uncessary) time embedding. The ode version of deform
                # also needs to take in a time_sequence, which is our sampled_fids, this needs to be shaped in T*1
                
                # Deform model calculates the latents for all viewpoints and decodes each, so the returned values are a list of T*N*(Feature_dim)
                new_xyz_list, new_rotation_list, new_scaling_list = deform.step(gaussians.get_xyz, time_input + ast_noise, rgb, time_sequence)
            else:
                new_xyz_list, new_rotation_list, new_scaling_list = deform.step(gaussians.get_xyz, time_input + ast_noise, time_seq=time_sequence)
        if add_zero:
            # get rid of the first deformed instance, since it's the canonical frame
            new_xyz_list = new_xyz_list[1:]
            new_rotation_list = new_rotation_list[1:]
            new_scaling_list = new_scaling_list[1:]
        # Render, now operates on the list of deformations
        loss = 0.0
        Ll1 = 0.0
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        radii_list = []
        for viewpoint_cam, new_xyz, new_rotation, new_scaling in zip(sampled_cams, new_xyz_list, new_rotation_list, new_scaling_list):  
            # use renderode for direct computation
            render_pkg_re = render_ode(viewpoint_cam, gaussians, pipe, background, new_xyz, new_rotation, new_scaling, dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            # collect results
            visibility_filter_list.append(visibility_filter)
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            radii_list.append(radii)
            loss += Ll1
            Ll1 += Ll1
        
        # Backprop
        loss.backward()
        
        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            # unload all viewpoint cams
            for viewpoint_cam in sampled_cams:
                viewpoint_cam.load2device('cpu')
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if not PRE_TRAINED_GAUSSIANS:
                # Given we have a list of visibility filters, repeat this step for each visibility filter
                for visibility_filter, radii in zip(visibility_filter_list, radii_list):
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])

            # Log and save
            
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render_ode, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if not PRE_TRAINED_GAUSSIANS:
                if iteration < opt.densify_until_iter:
                    for viewspace_point_tensor, visibility_filter in zip(viewspace_point_tensor_list, visibility_filter_list):
                        # adding densificaiton in a loop since we compute multiple renders each iteration
                        print(viewspace_point_tensor.grad.shape)
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if not PRE_TRAINED_GAUSSIANS:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if not PRE_TRAINED_DEFORM:
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                    deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    # if fid is 0 then simply use the canonical frame
                    if fid == 0:
                        #for rotation and scaling, we use the canonical frame
                        rotation = scene.gaussians.get_rotation 
                        scaling = scene.gaussians.get_scaling
                        image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs, xyz, rotation, scaling, is_6dof)["render"],
                            0.0, 1.0)
                    else:
                        # since we're no longer feeding the target time but rather the current time, instead fid is used to create the time sequence
                        time_input = torch.zeros(xyz.shape[0], 1, device='cuda')
                        # also concat 0 in the beginning, since during validation we start from the first frame
                        time_sequence = torch.cat((torch.zeros(1, device='cuda'), torch.Tensor([fid]).to('cuda')), dim=0)
                        if USE_COLOR:
                            rgb = SH2RGB(scene.gaussians.get_features[:,0, :])
                            new_xyz, new_rotation, new_scaling = deform.step(xyz.detach(), time_input, rgb.detach(), time_sequence)
                        else:
                            new_xyz, new_rotation, new_scaling = deform.step(xyz.detach(), time_input, time_sequence)
                        # take the second deformed instance, since the first one is the canonical frame
                        image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs, new_xyz[1], new_rotation[1], new_scaling[1], is_6dof)["render"],
                            0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default= list(range(1000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")

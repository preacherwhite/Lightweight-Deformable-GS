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

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_ode
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
    TrainingArguments,
)


def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, train_views, gaussians, pipeline, background, forecast_results):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    renderings = []
    t_list = []
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    # Combine train and test views into full sequence
    views = []
    if train_views is not None:
        views.extend(train_views)
    # Sort views by frame ID
    views.sort(key=lambda x: x.fid)
    try:
        assert len(views) == 150
    except:
        print(f"Number of views: {len(views)}")


    # Take the last 20% of views for testing
    test_views = views[-int(len(views) * 0.2):]
    print(len(test_views))
    # Load forecast results
    for idx, view in enumerate(tqdm(test_views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        
        d_xyz = forecast_results[idx]
        # rotation and scaling are kept the same
        device = 'cuda'
        d_xyz = torch.from_numpy(d_xyz).to(device)
        d_rotation = gaussians.get_rotation
        d_scaling = gaussians.get_scaling
        results = render_ode(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

    #renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    #imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)

# TARGET_LENGTH = 30
# INPUT_LENGTH = 80
# FORECAST_MODEL_PATH = "/media/staging2/dhwang/Deformable-3D-Gaussians/forecast_exp/results/target30"
#FORECASE_RESULT_PATH = "/media/staging2/dhwang/Deformable-3D-Gaussians/forecast_exp/forecast_results.npy"
#FORECASE_RESULT_PATH = "/media/staging2/dhwang/Deformable-3D-Gaussians/forecast_exp/latent_ode_model_40_seq_20_3_5_extrapolate/figures/full_predicted_trajectories.npy"
FORECASE_RESULT_PATH = "/media/staging2/dhwang/Deformable-3D-Gaussians/forecast_exp/transformer_latent_ode_model/figures/full_predicted_trajectories.npy"
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        forecast_results = np.load(FORECASE_RESULT_PATH)
        forecast_results = forecast_results.transpose(1,0,2)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_func = render_set

        render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test_30_extrapolate_ode_transformer", scene.loaded_iter,scene.getTrainCameras(), gaussians, pipeline,
                    background, forecast_results)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))

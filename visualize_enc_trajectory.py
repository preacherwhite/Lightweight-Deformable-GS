# Randomly sample k Gaussians
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import torch
import numpy as np
import os
from os import makedirs
from tqdm import tqdm
from scene import Scene, GaussianModel, DeformModel, SetDeformModel, SetDeformModelSPUNet
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
import sys
from utils.system_utils import searchForMaxIteration
def render_trajectory(dataset):
    k = 50
    n_frames = 150

    gaussians = GaussianModel(3)
    model_path = "/media/staging2/dhwang/Lightweight-Deformable-GS/output/set_pretrain_scan_ppt"
    loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    gaussians.load_ply(os.path.join(model_path,
                                    "point_cloud",
                                    "iteration_" + str(loaded_iter),
                                    "point_cloud.ply"))
    print("Loading trained model at iteration {}".format(loaded_iter))

    deform = SetDeformModel()
    deform.load_weights("/media/staging2/dhwang/Lightweight-Deformable-GS/output/set_pretrain_scan_ppt")


    n_gaussians = gaussians.get_xyz.shape[0]
    # Filter out gaussians with low opacity
    opacity = gaussians._opacity.sigmoid().detach()
    high_opacity_mask = opacity.squeeze() > 0.95
    print(f"Number of gaussians with high opacity: {high_opacity_mask.sum().item()}")
    n_gaussians = high_opacity_mask.sum().item()
    
    # Update xyz tensor to only include high opacity gaussians
    gaussians._xyz = gaussians._xyz[high_opacity_mask]

    sampled_indices = np.random.choice(gaussians.get_xyz.shape[0], k, replace=False)
    # Initialize arrays to store trajectories
    trajectories = np.zeros((k, n_frames, 3))
    # Collect trajectory data
    times = np.linspace(0, 1, n_frames)
    for frame, time in enumerate(tqdm(times, desc="Collecting trajectory data")):
        fid = torch.Tensor([time]).unsqueeze(0).cuda().repeat(n_gaussians, 1)
        xyz = gaussians.get_xyz.detach()
        d_xyz, _, _ = deform.step(xyz, fid)

        trajectories[:, frame, :] = (xyz[sampled_indices] + d_xyz[sampled_indices]).cpu().detach().numpy()
        
    # Visualize trajectories
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors
    special_color = 'red'  # Color for the specified Gaussian
    other_color = 'blue'   # Color for other Gaussians

    for i in range(len(sampled_indices)):
        ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2], 
                color=other_color, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title(f'3D Trajectories of {len(sampled_indices)} Sampled Gaussians')

    # Save the plot
    trajectory_path = os.path.join('output/set_pretrain_scan_ppt', f"trajectories", f"trajectories_{k}")
    makedirs(trajectory_path, exist_ok=True)
    plt.savefig(os.path.join(trajectory_path, 'gaussian_trajectories_filtered.png'))
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])
    render_trajectory(lp.extract(args))
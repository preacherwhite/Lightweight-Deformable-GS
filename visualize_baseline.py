import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from matplotlib import cm
from tqdm import tqdm
from scene import Scene, GaussianModel, DeformModel
from argparse import ArgumentParser, Namespace
import sys
from arguments import ModelParams, PipelineParams, OptimizationParams

def visualize_gaussian_trajectories(
    gaussians,
    deform,
    num_frames=50,
    num_gaussians=20,
    specific_indices=None,
    save_path=None,
    viewpoint=(30, 45)
):
    """
    Visualize the trajectories of a subset of Gaussians.
    
    Args:
        gaussians: GaussianModel instance
        deform: DeformModel instance
        num_frames: Number of frames for trajectory visualization
        num_gaussians: Number of Gaussians to visualize
        specific_indices: Optional list of specific Gaussian indices to visualize
        save_path: Path to save the visualization
        viewpoint: (elevation, azimuth) for 3D viewpoint
    """
    # Get the total number of Gaussians
    total_gaussians = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    
    # Select Gaussians (either randomly or specified ones)
    if specific_indices is not None:
        if isinstance(specific_indices, list):
            random_indices = torch.tensor(specific_indices, device=device)
        else:
            random_indices = specific_indices.to(device)
        num_gaussians = len(random_indices)
    else:
        # Randomly select Gaussians
        random_indices = torch.randperm(total_gaussians)[:num_gaussians].to(device)
    
    # Get the initial positions
    xyz = gaussians.get_xyz[random_indices]
    
    # Prepare for trajectory visualization
    time_range = torch.linspace(0, 1, num_frames, device=device)
    trajectories = []
    
    # Generate trajectories for each time step
    print("Generating trajectories...")
    for t in tqdm(time_range):
        time_input = t.unsqueeze(0).expand(num_gaussians, -1)
        with torch.no_grad():
            d_xyz, _, _ = deform.step(xyz.detach(), time_input)
            t_xyz = d_xyz + xyz
            trajectories.append(t_xyz.cpu().numpy())
    
    # Convert to numpy arrays for plotting
    trajectories = np.array(trajectories)  # Shape: [num_frames, num_gaussians, 3]
    
    # Get a colormap for the trajectories
    colors = cm.rainbow(np.linspace(0, 1, num_gaussians))
    
    # Create visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the viewpoint
    ax.view_init(elev=viewpoint[0], azim=viewpoint[1])
    
    # Plot each Gaussian's trajectory
    for i in range(num_gaussians):
        ax.plot(trajectories[:, i, 0], trajectories[:, i, 1], trajectories[:, i, 2], 
                color=colors[i], linewidth=1.5, alpha=0.8)
        
        # # Plot the start point
        # ax.scatter(trajectories[0, i, 0], trajectories[0, i, 1], trajectories[0, i, 2], 
        #            color='green', s=50, marker='o')
        
        # # Plot the end point
        # ax.scatter(trajectories[-1, i, 0], trajectories[-1, i, 1], trajectories[-1, i, 2], 
        #            color='red', s=50, marker='x')
    
    # Create custom legend elements
    legend_elements = [
        plt.Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=8, label='Start'),
        plt.Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8, label='End')
    ]
    
    # Add a subset of Gaussian trajectories to the legend
    max_legend_gaussians = min(5, num_gaussians)
    for i in range(max_legend_gaussians):
        legend_elements.append(
            plt.Line2D([0], [0], color=colors[i], linestyle='-', 
                      label=f'Gaussian {random_indices[i].item()}')
        )
    
    ax.legend(handles=legend_elements, loc="upper right")
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gaussian Trajectories')
    
    # Set equal aspect ratio
    limits = []
    for dim in range(3):
        min_val = trajectories[:, :, dim].min()
        max_val = trajectories[:, :, dim].max()
        mid_val = (max_val + min_val) / 2
        diff = max_val - min_val
        limits.append((mid_val - diff/1.8, mid_val + diff/1.8))
    
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    return random_indices.cpu()

def main():
    parser = argparse.ArgumentParser(description='Visualize Gaussian trajectories')
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
        
    parser.add_argument('--num_gaussians', type=int, default=500, help='Number of Gaussians to visualize')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames for trajectory visualization')
    parser.add_argument('--output', type=str, default='trajectories.png', help='Output image path')
    parser.add_argument('--specific_indices', type=int, nargs='+', help='Specific Gaussian indices to visualize')
    args = parser.parse_args(sys.argv[1:])
    lp_args = lp.extract(args)
    dataset = lp_args
    
    try:
        # Load dataset and models
        # Note: Modify this section to match your specific dataset and model structure
        
        # Load Gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
        print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
        
        # Load Deform model
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path, iteration=-1)
        deform.deform.eval()
        
        # Visualize trajectories
        selected_indices = visualize_gaussian_trajectories(
            gaussians=gaussians,
            deform=deform,
            num_frames=args.num_frames,
            num_gaussians=args.num_gaussians,
            specific_indices=args.specific_indices,
            save_path=args.output
        )
        
        # Save selected indices for reproducibility
        np.save(os.path.splitext(args.output)[0] + '_indices.npy', selected_indices.numpy())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
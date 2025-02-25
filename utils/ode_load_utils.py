import torch
import numpy as np
import os
from scene.gaussian_model import GaussianModel
from scene import Scene
from scene.deform_model import DeformModel

def load_models(dataset):
    """Load pre-trained Gaussian and Deform models."""
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    print("loaded {} gaussians".format(gaussians.get_xyz.shape[0]))
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.load_weights(dataset.model_path, iteration=-1)
    deform.deform.eval()
    return gaussians, scene, deform

def load_or_generate_trajectories(dataset, scene, gaussians, deform, xyz_only=False):
    """Load pre-computed trajectories or generate them if not available."""
    trajectory_path = dataset.trajectory_path
    if trajectory_path is not None:
        all_trajectories = np.load(trajectory_path)
        print("loaded trajectories from {}".format(trajectory_path))
        if xyz_only:
            all_trajectories = all_trajectories[..., :3]
        return torch.from_numpy(all_trajectories).to('cuda')
    
    trajectory_file = os.path.join(dataset.model_path, "trajectories.npy")
    if os.path.exists(trajectory_file):
        all_trajectories = np.load(trajectory_file)
        all_trajectories = torch.from_numpy(all_trajectories).to('cuda')
        print("loaded trajectories from {}".format(trajectory_file))
        if xyz_only:
            all_trajectories = all_trajectories[..., :3]
        return all_trajectories
    
    # Generate trajectories if not available
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
    return all_trajectories

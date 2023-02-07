import os

from ros_edf.ros_interface import EdfRosInterface
from edf.pc_utils import draw_geometry, voxel_filter
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence
from edf.preprocess import Rescale, NormalizeColor, Downsample
from edf.agent import PickAgent

import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from pytorch3d.transforms import quaternion_multiply, axis_angle_to_quaternion

torch.set_printoptions(precision= 3, sci_mode=False, linewidth=120)


env_interface = EdfRosInterface(reference_frame = "scene")



device = 'cuda:0'
# device = 'cpu'
unit_len = 0.01
transforms = Compose([Rescale(rescale_factor=1/unit_len),
                      Downsample(voxel_size=1.7, coord_reduction="average"),
                      NormalizeColor(color_mean = torch.tensor([0.5, 0.5, 0.5]), color_std = torch.tensor([0.5, 0.5, 0.5])),
                     ])

inv_transforms = Compose([Rescale(rescale_factor=unit_len),
                         ])





pick_agent_config_dir = "config/agent_config/pick_agent.yaml"
pick_agent_param_dir = "checkpoint/mug_10_demo/pick/model_iter_600.pt"
max_N_query_pick = 1
langevin_dt_pick = 0.001

pick_agent = PickAgent(config_dir=pick_agent_config_dir, 
                       device = device,
                       max_N_query = max_N_query_pick, 
                       langevin_dt = langevin_dt_pick)
pick_agent.load(pick_agent_param_dir)

query_points = torch.tensor([[0.0, 0.0, 15.]])
pick_agent.query_model.query_points = query_points.to(device)






points, colors = env_interface.observe_ee(obs_type = 'pointcloud', update = True)
grasp_pc_raw = PointCloud.from_numpy(points=points, colors=colors, device=device)
points, colors = env_interface.observe_scene(obs_type = 'pointcloud', update = True)
scene_pc_raw = PointCloud.from_numpy(points=points, colors=colors, device=device)

scene_pc = transforms(scene_pc_raw)
grasp_pc = transforms(grasp_pc_raw)







T_seed = 100
pick_policy = 'sorted'
pick_mh_iter = 1000
pick_langevin_iter = 300
pick_dist_temp = 1.
pick_policy_temp = 1.
pick_optim_iter = 100
pick_optim_lr = 0.005

Ts, edf_outputs, logs = pick_agent.forward(pc=scene_pc, T_seed=T_seed, policy = pick_policy, mh_iter=pick_mh_iter, langevin_iter=pick_langevin_iter, 
                                            temperature=pick_dist_temp, policy_temperature=pick_policy_temp, optim_iter=pick_optim_iter, optim_lr=pick_optim_lr)




q, t = Ts[..., :4], Ts[..., 4:]
q = quaternion_multiply(axis_angle_to_quaternion(axis_angle=torch.tensor([0, 0, torch.pi/2], device=q.device)), q)
Ts = torch.cat((q,t), dim=-1)

Ts = SE3(poses=Ts)
target_poses = inv_transforms(Ts).poses.detach().cpu().numpy()
# print(torch.tensor(target_poses[:20]))

pick_poses = SE3.from_numpy(orns = target_poses[..., :4], positions = target_poses[..., 4:], versor_last_input = False)
env_interface.pick(target_poses=target_poses)
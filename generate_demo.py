from ros_edf.ros_interface import EdfRosInterface
from ros_edf.pc_utils import pcd_from_numpy, draw_geometry, reconstruct_surface
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, save_demos
from edf.pc_utils import optimize_pcd_collision

import torch
import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d



### Initialize EDF ROS Interface

env_interface = EdfRosInterface(reference_frame = "scene")
env_interface.reset()
env_interface.moveit_interface.arm_group.set_planning_time(seconds=5)

### Initialize dataset list

dataset = []

### Observe Point clouds

grasp_pc = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
scene_pc = env_interface.observe_scene(obs_type = 'pointcloud', update = True)

### Pick

target_poses = SE3([0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.275])
result = env_interface.pick(target_poses=target_poses)

pick_success = result[-1] is not None

if True: # pick_success:
    pick_demo = TargetPoseDemo(target_poses=target_poses, scene_pc=scene_pc, grasp_pc=grasp_pc)

### Move to observe gripper

grasp_pc = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
env_interface.attach(pcd = grasp_pc)

env_interface.move_to_target_pose(poses = SE3([0.0, 0.0, 1.0, 0.0, 0.00, 0.0, 0.6]))

### Observe gripper

env_interface.detach()
grasp_pc = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
env_interface.attach(pcd = grasp_pc)

### Observe Scene

env_interface.move_to_target_pose(SE3([0.0, 0.0, 1.0, 0.0, -0.4, 0.4, 0.6]))

scene_pc = env_interface.observe_scene(obs_type = 'pointcloud', update = True)

env_interface.move_to_target_pose(poses = SE3([0.0, 0.0, 1.0, 0.0, 0.00, 0.0, 0.6]))

### Place
target_poses = SE3([0.5000, -0.5000, -0.5000, -0.5000, 0.12, -0.18, 0.30])
_, pre_place_poses = optimize_pcd_collision(x=scene_pc, y=grasp_pc, 
                                            cutoff_r = 0.03, dt=0.01, eps=1., iters=5,
                                            rel_pose=target_poses)
results, result_pose = env_interface.move_to_target_pose(poses = pre_place_poses)
if results[-1] is True:
    result = env_interface.move_cartesian(poses=target_poses, cartesian_step=0.01, cspace_step_thr=10, avoid_collision=False)
else:
    result = False

assert result is True
env_interface.detach()
env_interface.release()
place_demo = TargetPoseDemo(target_poses=target_poses, scene_pc=scene_pc, grasp_pc=grasp_pc)

# ### Save

# demo_seq = DemoSequence(demo_seq = [pick_demo, place_demo])
# dataset.append(demo_seq)
# save_demos(demos=dataset, dir="demo/test_demo")
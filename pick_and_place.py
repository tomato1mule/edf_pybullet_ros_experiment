from ros_edf.ros_interface import EdfRosInterface
from ros_edf.pc_utils import pcd_from_numpy, draw_geometry, reconstruct_surface
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, save_demos
from edf.pc_utils import check_pcd_collision, optimize_pcd_collision

import torch
import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d


# ### Initialize EDF ROS Interface


env_interface = EdfRosInterface(reference_frame = "scene")
env_interface.reset()
env_interface.moveit_interface.arm_group.set_planning_time(seconds=5)


# ### Initialize dataset list


dataset = []


# ### Observe Point clouds


grasp_raw = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
scene_raw = env_interface.observe_scene(obs_type = 'pointcloud', update = True)


# ### Pick


# pick_poses = SE3([0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.275])
# pre_pick_poses = SE3.multiply(pick_poses, SE3([1., 0., 0., 0., 0., 0., -0.1]))
# post_pick_poses = SE3.multiply(SE3([1., 0., 0., 0., 0., 0., 0.2]), pick_poses)


pick_poses = SE3([0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.275])
_, pre_pick_poses = optimize_pcd_collision(x=scene_raw, y=grasp_raw, 
                                           cutoff_r = 0.03, dt=0.01, eps=1., iters=50,
                                           rel_pose=pick_poses)
post_pick_poses = pre_pick_poses


idx = 0
pick_pose, pre_pick_pose, post_pick_pose = pick_poses[idx], pre_pick_poses[idx], post_pick_poses[idx]

colcheck_r = 0.003 # Should be similar to voxel filter size
col_check = check_pcd_collision(x=scene_raw, y=grasp_raw.transformed(pick_pose)[0], r = colcheck_r)
print(f"Collision-free Pick Pose: {not col_check}")
assert not col_check, "No collision-free pick pose found!"


# draw_geometry([scene_raw] + grasp_raw.transformed(pre_pick_pose))


pick_result = env_interface.pick(pre_pick_pose, pick_pose, post_pick_pose)
print(pick_result)


assert pick_result == 'SUCCESS'
pick_demo = TargetPoseDemo(target_poses=pick_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)


env_interface.detach()
env_interface.attach_placeholder() # To avoid collsion with the grasped object


# ### Observe gripper


# Home position
result = env_interface.move_simple(target_poses = SE3([0., 0., 1., 0., -0.1, 0., 0.6]))
print(result)


# result = env_interface.move_simple(target_poses = SE3([[0., 0., 1., 0., -0.1, 0., 0.6], 
#                                                        [0.5, -0.5, -0.5, -0.5, 0.0, 0., 0.5], 
#                                                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]]), 
#                                   )
# print(result)


if result == 'SUCCESS':
    env_interface.detach()
    grasp_raw = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
    env_interface.attach(obj = grasp_raw)


# ### Observe Scene


result = env_interface.move_simple(target_poses = SE3([0.0, 0.0, 1.0, 0.0, -0.4, 0.4, 0.6]))
print(result)


if result == 'SUCCESS':
    scene_raw = env_interface.observe_scene(obs_type = 'pointcloud', update = True)


result = env_interface.move_simple(target_poses = SE3([0., 0., 1., 0., -0.1, 0., 0.6]))
print(result)


# ### Place


place_poses = SE3([0.5000, -0.5000, -0.5000, -0.5000, 0.13, -0.20, 0.32])
_, pre_place_poses = optimize_pcd_collision(x=scene_raw, y=grasp_raw, 
                                            cutoff_r = 0.03, dt=0.01, eps=1., iters=5,
                                            rel_pose=place_poses)
post_place_poses = place_poses * pick_pose.inv() * pre_pick_pose


idx = 0
place_pose, pre_place_pose, post_place_pose = place_poses[idx], pre_place_poses[idx], post_place_poses[idx]

colcheck_r = 0.0015 # Should be similar to voxel filter size
col_check = check_pcd_collision(x=scene_raw, y=grasp_raw.transformed(place_pose)[0], r = colcheck_r)
print(f"Collision-free Place Pose: {not col_check}")
assert not col_check, "No collision-free place pose found!"


result = env_interface.place(pre_place_pose, place_pose, post_place_pose)
print(result)


assert result == 'SUCCESS'
env_interface.detach()
env_interface.release()

place_demo = TargetPoseDemo(target_poses=place_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)


# ### Save


# demo_seq = DemoSequence(demo_seq = [pick_demo, place_demo])
# dataset.append(demo_seq)
# save_demos(demos=dataset, dir="demo/test_demo")



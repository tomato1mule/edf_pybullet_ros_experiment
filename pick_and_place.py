from ros_edf.ros_interface import EdfRosInterface
from ros_edf.pc_utils import pcd_from_numpy, draw_geometry, reconstruct_surface
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, save_demos
from edf.pc_utils import check_pcd_collision, optimize_pcd_collision
from edf.server import DashEdfDemoServer

import torch
import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d


###### Initialize Robot Interface ######
env_interface = EdfRosInterface(reference_frame = "scene")
env_interface.reset()
env_interface.moveit_interface.arm_group.set_planning_time(seconds=5)


###### Initialize Demo Server ######
save_demo = True
demo_from_server = True
scene_ranges = np.array([[-0.23, 0.23],
                         [-0.23, 0.23],
                         [-0.01, 0.4]])
point_size = 3.


demo_list = []
if demo_from_server:
    demo_server = DashEdfDemoServer(scene_ranges = scene_ranges,
                                    name = "dash_edf_demo_server", 
                                    point_size = point_size, 
                                    host_id = '127.0.0.1',
                                    server_debug=False)
    demo_server.run()


###### Observe ######
grasp_raw = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
scene_raw = env_interface.observe_scene(obs_type = 'pointcloud', update = True)


###### Sample Pick Pose ######
if demo_from_server:
    demo_server.update_scene_pcd(pcd=scene_raw)
    demo_server.update_grasp_pcd(pcd=grasp_raw)
    pick_poses =demo_server.get_user_response()
else:
    pick_poses = SE3([0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.275])
_, pre_pick_poses = optimize_pcd_collision(x=scene_raw, y=grasp_raw, 
                                           cutoff_r = 0.03, dt=0.01, eps=1., iters=50,
                                           rel_pose=pick_poses)
post_pick_poses = pre_pick_poses


###### Execute Pick ######
for idx in [0]:
    pick_pose, pre_pick_pose, post_pick_pose = pick_poses[idx], pre_pick_poses[idx], post_pick_poses[idx]
    
    colcheck_r = 0.003 # Should be similar to voxel filter size
    col_check = check_pcd_collision(x=scene_raw, y=grasp_raw.transformed(pick_pose)[0], r = colcheck_r)
    print(f"Pick Pose_{idx} collision-free: {not col_check}")
    if not col_check:
        break
    
if not col_check:
    print("Found collision-free pick pose!")
else:
    raise NotImplementedError("No collision-free pick pose found!")

# DEBUG
# draw_geometry([scene_raw] + grasp_raw.transformed(post_pick_pose))

pick_result = env_interface.pick(pre_pick_pose, pick_pose, post_pick_pose)
print(f"Pick result: {pick_result}")
if pick_result == "SUCCESS":
    pick_demo = TargetPoseDemo(target_poses=pick_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)
    env_interface.detach()
    env_interface.attach_placeholder() # To avoid collsion with the grasped object
else:
    raise NotImplementedError("Pick failed")


###### Observe for Place ######

# Observe EEF
for _ in range(3):
    result = env_interface.move_to_named_target("init")
    if result == 'SUCCESS':
        break
print(f"Move to End-Effector observation pose: {result}")
if result == 'SUCCESS':
    env_interface.detach()
    grasp_raw = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
    env_interface.attach(obj = grasp_raw)
else:
    raise NotImplementedError(f"Failed to move to End-Effector observation pose.")

# Observe Scene
result = env_interface.move_to_named_target("observe")
print(f"Move to Scene observation pose: {result}")
if result == 'SUCCESS':
    scene_raw = env_interface.observe_scene(obs_type = 'pointcloud', update = True)
else:
    raise NotImplementedError(f"Failed to move to Scene observation pose.")


result = env_interface.move_to_named_target("init")
print(f"Come back to default pose: {result}")
if result == 'SUCCESS':
    pass
else:
    raise NotImplementedError(f"Failed to come back to default pose.")


###### Sample Place Poses ######
if demo_from_server:
    demo_server.update_scene_pcd(pcd=scene_raw)
    demo_server.update_grasp_pcd(pcd=grasp_raw)
    place_poses = demo_server.get_user_response()
else:
    place_poses = SE3([0.5000, -0.5000, -0.5000, -0.5000, 0.13, -0.20, 0.32])
_, pre_place_poses = optimize_pcd_collision(x=scene_raw, y=grasp_raw, 
                                            cutoff_r = 0.03, dt=0.01, eps=1., iters=5,
                                            rel_pose=place_poses)
post_place_poses = place_poses * pick_pose.inv() * pre_pick_pose


###### Execute Place ######
for idx in [0]:
    place_pose, pre_place_pose, post_place_pose = place_poses[idx], pre_place_poses[idx], post_place_poses[idx]
    
    colcheck_r = 0.0015 # Should be similar to voxel filter size
    col_check = check_pcd_collision(x=scene_raw, y=grasp_raw.transformed(place_pose)[0], r = colcheck_r)
    print(f"Place Pose_{idx} collision-free: {not col_check}")
    if not col_check:
        break
    
if not col_check:
    print("Found collision-free place pose!")
else:
    raise NotImplementedError("No collision-free place pose found!")

# DEBUG
# draw_geometry([scene_raw] + grasp_raw.transformed(place_pose))

place_result = env_interface.place(pre_place_pose, place_pose, post_place_pose)
print(f"Place result: {place_result}")
if place_result == "SUCCESS":
    place_demo = TargetPoseDemo(target_poses=place_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)
    env_interface.detach()
    env_interface.release()
else:
    raise NotImplementedError("Place failed")

# Go back home
for _ in range(3):
    result = env_interface.move_to_named_target("init")
    if result == 'SUCCESS':
        break
print(f"Move to End-Effector observation pose: {result}")

###### Save Collected Demonstrations ######

if save_demo:
    demo_seq = DemoSequence(demo_seq = [pick_demo, place_demo])
    demo_list.append(demo_seq)
    save_demos(demos=demo_list, dir="demo/test_demo")

if demo_from_server:
    demo_server.close()


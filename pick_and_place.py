
from ros_edf.ros_interface import EdfRosInterface
from edf.pc_utils import pcd_from_numpy, draw_geometry, reconstruct_surface
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, save_demos

import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d


env_interface = EdfRosInterface(reference_frame = "scene")


dataset = []


points, colors = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
grasp_pc = PointCloud.from_numpy(points=points, colors=colors)
points, colors = env_interface.observe_scene(obs_type = 'pointcloud', update = True)
scene_pc = PointCloud.from_numpy(points=points, colors=colors)


target_poses = np.array([[0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.275]])
pick_poses = SE3.from_numpy(orns = target_poses[..., :4], positions = target_poses[..., 4:], versor_last_input = False)
result = env_interface.pick(target_poses=target_poses)

pick_success = result[-1] is not None


if True: # pick_success:
    pick_demo = TargetPoseDemo(target_poses=pick_poses, scene_pc=scene_pc, grasp_pc=grasp_pc)


points, colors = env_interface.move_and_observe()
grasp_pc = PointCloud.from_numpy(points=points, colors=colors)
points, colors = env_interface.observe_scene(obs_type = 'pointcloud', update = True)
scene_pc = PointCloud.from_numpy(points=points, colors=colors)


env_interface.attach(pcd = grasp_pc.to_pcd())


target_poses = np.array([[0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.25]])
place_poses = SE3.from_numpy(orns = target_poses[..., :4], positions = target_poses[..., 4:], versor_last_input = False)
pre_grasp_results, grasp_pose = env_interface.move_to_target_pose(poses = target_poses)


target_poses = np.array([[0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.3]])
place_poses = SE3.from_numpy(orns = target_poses[..., :4], positions = target_poses[..., 4:], versor_last_input = False)
pre_grasp_results, grasp_pose = env_interface.move_to_target_pose(poses = target_poses)


env_interface.detach()


env_interface.release()


target_poses = np.array([[0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.6]])
place_poses = SE3.from_numpy(orns = target_poses[..., :4], positions = target_poses[..., 4:], versor_last_input = False)
pre_grasp_results, grasp_pose = env_interface.move_to_target_pose(poses = target_poses)


place_demo = TargetPoseDemo(target_poses=place_poses, scene_pc=scene_pc, grasp_pc=grasp_pc)


points, colors = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
grasp_pc = PointCloud.from_numpy(points=points, colors=colors)
points, colors = env_interface.observe_scene(obs_type = 'pointcloud', update = True)
scene_pc = PointCloud.from_numpy(points=points, colors=colors)


demo_seq = DemoSequence(demo_seq = [pick_demo, place_demo])
dataset.append(demo_seq)
save_demos(demos=dataset, dir="demo/test_demo")



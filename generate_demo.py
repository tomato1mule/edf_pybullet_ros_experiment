import edf_env
from edf_env.env import UR5Env
from ros_edf.ros_interface import EdfRosInterface
from edf_env.pybullet_pc_utils import pb_cams_to_pc
from edf_env.pc_utils import pcd_from_numpy, draw_geometry
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, save_demos

import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d

env_interface = EdfRosInterface(reference_frame = "scene")





dataset = []




points, colors = env_interface.observe_ee(obs_type = 'pointcloud', update = True)
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

target_poses = np.array([[0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.275]])
place_poses = SE3.from_numpy(orns = target_poses[..., :4], positions = target_poses[..., 4:], versor_last_input = False)
# result = env_interface.pick(target_poses=target_poses)

place_demo = TargetPoseDemo(target_poses=place_poses, scene_pc=scene_pc, grasp_pc=grasp_pc)



demo_seq = DemoSequence(demo_seq = [pick_demo, place_demo])
dataset.append(demo_seq)
save_demos(demos=dataset, dir="demo/test_demo")
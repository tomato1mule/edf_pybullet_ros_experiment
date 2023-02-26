import os

import torch
import numpy as np

from edf.server import DashEdfDemoServer
from edf.data import gzip_load, PointCloud, SE3


log_id = 100

agent_config_dir = "config/agent_config/place_agent_dev.yaml"
train_config_dir = "config/train_config/train_place_dev.yaml"
agent_param_dir = "checkpoint/mug_1_demo_dev/place"
log_name = f"trainlog_iter_{log_id}.gzip"
print(f"Visualizing {os.path.join(agent_param_dir,log_name)}")

train_logs = gzip_load(dir=agent_param_dir, filename=log_name)
scene_raw: PointCloud = train_logs['scene_raw']
grasp_raw: PointCloud = train_logs['grasp_raw']
# query_points = train_logs['edf_outputs']['query_points']
# query_attention = train_logs['edf_outputs']['query_attention']
# target_pose = SE3(train_logs['target_T'])
# best_pose = SE3(train_logs['best_neg_T'])
# sampled_poses= SE3(train_logs['sampled_Ts'])




server_debug = False
scene_ranges = np.array([[-23., 23.],
                         [-23., 23.],
                         [-1., 40.]])
point_size = 3.


demo_server = DashEdfDemoServer(scene_ranges = scene_ranges,
                                name = "dash_edf_demo_server", 
                                point_size = point_size, 
                                host_id = '127.0.0.1',
                                server_debug=server_debug)


if not server_debug:
    demo_server.run()





demo_server.update_scene_pcd(pcd=scene_raw)
demo_server.update_grasp_pcd(pcd=grasp_raw)


if server_debug:
    demo_server.run()

target_poses = demo_server.get_user_response()
print(target_poses)
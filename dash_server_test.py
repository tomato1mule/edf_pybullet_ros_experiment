import os
import argparse

from edf.pc_utils import draw_geometry, create_o3d_points, get_plotly_fig
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, DemoSeqDataset, gzip_load
from edf.preprocess import Rescale, NormalizeColor, Downsample, PointJitter, ColorJitter

from dash import Dash, html, dcc, Input, Output
import dash_vtk
import numpy as np
import torch

log_id = 100

agent_config_dir = "config/agent_config/place_agent_dev.yaml"
train_config_dir = "config/train_config/train_place_dev.yaml"
agent_param_dir = "checkpoint/mug_1_demo_dev/place"
log_name = f"trainlog_iter_{log_id}.gzip"
print(f"Visualizing {os.path.join(agent_param_dir,log_name)}")

train_logs = gzip_load(dir=agent_param_dir, filename=log_name)
scene_raw: PointCloud = train_logs['scene_raw']
grasp_raw: PointCloud = train_logs['grasp_raw']
query_points = train_logs['edf_outputs']['query_points']
query_attention = train_logs['edf_outputs']['query_attention']
target_pose = SE3(train_logs['target_T'])
best_pose = SE3(train_logs['best_neg_T'])
sampled_poses= SE3(train_logs['sampled_Ts'])

scene_repr = dash_vtk.PointCloudRepresentation(xyz=scene_raw.points.ravel(), rgb=scene_raw.colors.ravel() * 255, property={"pointSize": 2})
scene_repr.id = 'scene'
grasp_repr = dash_vtk.PointCloudRepresentation(xyz=grasp_raw.points.ravel(), rgb=grasp_raw.colors.ravel() * 255, property={"pointSize": 2})
grasp_repr.id = 'grasp'

vtk_view = dash_vtk.View(children=[scene_repr, grasp_repr], id="vtk-view1")


grasp_repr2 = dash_vtk.PointCloudRepresentation(xyz=grasp_raw.points.ravel(), rgb=grasp_raw.colors.ravel() * 255, property={"pointSize": 2})
grasp_repr2.id = 'grasp2'
vtk_view2 = dash_vtk.View(children=[grasp_repr2])



app = Dash(__name__)
server = app.server

# app.layout = html.Div(
#     style={"height": "calc(100vh - 16px)"},
#     children=[html.Div(vtk_view, style={"height": "100%", "width": "100%"})],
# )
rows = []
first_row = html.Div(children=[html.Div(children=[vtk_view], style={'padding': 10, 'flex': 1, "height": "30%", "width": "30%"}),
                               html.Div(children=[vtk_view2], style={'padding': 10, 'flex': 1, "height": "30%", "width": "30%"}),
                               html.Div([html.H6("Change the value in the text box to see callbacks in action!"),
                                         html.Div(["Input: ", dcc.Input(id='my-input', value='initial value', type='text')]),
                                         html.Br(),
                                         html.Div(id='my-output'),]),
                               ],
                     style={'display': 'flex', 'flex-direction': 'row', "height": "calc(100vh - 16px)"})
rows.append(first_row)
# second_row = html.Div(children=["Input: ", dcc.Input(id='my-input', value='initial value', type='text')])
# rows.append(second_row)
app.layout = html.Div(children=rows, style={'display': 'flex', 'flex-direction': 'column', "height": "calc(100vh - 16px)"})



@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value}'










app.run_server(debug=True, host='127.0.0.1')
import os
import argparse

from edf.pc_utils import draw_geometry, create_o3d_points, get_plotly_fig
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, DemoSeqDataset, gzip_load
from edf.preprocess import Rescale, NormalizeColor, Downsample, PointJitter, ColorJitter

from dash import Dash, html, dcc, Input, Output
import dash_vtk
import dash_daq
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



scene_ranges = np.array([[-23., 23.],
                         [-23., 23.],
                         [-1., 32.]])
n_trans_step = 100
n_rot_step = 100
slider_size = 500
point_size = 3


def get_pcd_repr(id, points, colors):
    return dash_vtk.GeometryRepresentation(id=id, 
                                           property={"pointSize": point_size},
                                           children=[dash_vtk.PolyData(points=points.ravel(), 
                                                                       connectivity='points', 
                                                                       children=[dash_vtk.PointData([dash_vtk.DataArray(registration='setScalars',
                                                                                                                        type='Uint8Array',
                                                                                                                        numberOfComponents=3,
                                                                                                                        values=colors.ravel() * 255)])],)],)


scene_repr = get_pcd_repr(id = 'scene-pcd', points = scene_raw.points, colors=scene_raw.colors)
grasp_repr = get_pcd_repr(id = 'grasp-pcd', points = grasp_raw.points, colors=grasp_raw.colors)
vtk_view = dash_vtk.View(children=[scene_repr, grasp_repr], id="vtk-view1", background=[1., 1., 1.])




app = Dash(__name__)
server = app.server


rows = []
first_row = html.Div(children=[html.Div(children=[html.Div(["Show Scene: ", dash_daq.BooleanSwitch(id='scene_visible_toggle', on=True),]),
                                                  html.Div(["Show Gripper: ", dash_daq.BooleanSwitch(id='grasp_visible_toggle', on=True),])
                                                  ], style={"height": "1000px", "width": "500px"}, id="left-panel"),

                               html.Div(children=[vtk_view], style={"height": "1000px", "width": "1000px"}, id="vtk-view1-panel"),

                               html.Div([html.H3("Target Pose"),
                                         #html.Div(["Input: ", dcc.Input(id='my-input', value=0., type='number')]),
                                         html.Div(["x: ", 
                                                   dash_daq.Slider(id='x-slider', min=scene_ranges[0,0], max=scene_ranges[0,1], value=scene_ranges[0,0], step=(scene_ranges[0,1]-scene_ranges[0,0])/n_trans_step, size=slider_size),
                                                   html.Div(id='current-x')
                                                   ]),
                                         html.Div(["y: ", 
                                                   dash_daq.Slider(id='y-slider', min=scene_ranges[1,0], max=scene_ranges[1,1], value=scene_ranges[1,0], step=(scene_ranges[1,1]-scene_ranges[1,0])/n_trans_step, size=slider_size),
                                                   html.Div(id='current-y')
                                                   ]),
                                         html.Div(["z: ", 
                                                   dash_daq.Slider(id='z-slider', min=scene_ranges[2,0], max=scene_ranges[2,1], value=scene_ranges[2,0], step=(scene_ranges[2,1]-scene_ranges[2,0])/n_trans_step, size=slider_size),
                                                   html.Div(id='current-z')
                                                   ]),
                                         html.Div(["Rx: ", 
                                                   dash_daq.Slider(id='Rx-slider', min=-180, max=180, value=-180, step=2*180/n_rot_step, size=slider_size),
                                                   html.Div(id='current-Rx')
                                                   ]),
                                         html.Div(["Ry: ", 
                                                   dash_daq.Slider(id='Ry-slider', min=-90, max=90, value=-90, step=180/n_rot_step, size=slider_size),
                                                   html.Div(id='current-Ry')
                                                   ]),
                                         html.Div(["Rz: ", 
                                                   dash_daq.Slider(id='Rz-slider', min=-180, max=180, value=-180, step=2*180/n_rot_step, size=slider_size),
                                                   html.Div(id='current-Rz')
                                                   ]),
                                         html.Br(),
                                         html.Div(id='my-output'),] , id="right-panel"),                            
                               ],
                     style={'display': 'flex', 'flex-direction': 'row', "height": "1200px", })
rows.append(first_row)
# second_row = html.Div(children=["Input: ", dcc.Input(id='my-input', value='initial value', type='text')])
# rows.append(second_row)
app.layout = html.Div(children=rows, style={'display': 'flex', 'flex-direction': 'column'}, id='main_panel')

# Scene Visibility Toggle
@app.callback(
    Output(component_id="scene-pcd", component_property="actor"),
    Input(component_id='scene_visible_toggle', component_property='on')
)
def update_output_div(input_value):
    return {"visibility": input_value}
# Equivalent to....
# app.callback(
#     Output(component_id="scene-pcd", component_property="actor"),
#     Input(component_id='scene_visible_toggle', component_property='on')
# )(update_output_div)



# Grasp
@app.callback(
    Output(component_id='my-output', component_property='children'),
    Output(component_id="grasp-pcd", component_property="actor"),
    Input(component_id='grasp_visible_toggle', component_property='on'),
    Input(component_id='x-slider', component_property='value'),
    Input(component_id='y-slider', component_property='value'),
    Input(component_id='z-slider', component_property='value'),
    Input(component_id='Rx-slider', component_property='value'),
    Input(component_id='Ry-slider', component_property='value'),
    Input(component_id='Rz-slider', component_property='value'),
)
def update_output_div(visible, x, y, z, Rx, Ry, Rz):
    output = [f'Target Pose: {x,y,z}']
    output.append({"position": [x, y, z], 'orientation': [Rx, Ry, Rz], "visibility": visible})
    return tuple(output)


# Get slider numbers
@app.callback(
    Output(component_id='current-x', component_property='children'),
    Output(component_id='current-y', component_property='children'),
    Output(component_id='current-z', component_property='children'),
    Output(component_id='current-Rx', component_property='children'),
    Output(component_id='current-Ry', component_property='children'),
    Output(component_id='current-Rz', component_property='children'),
    Input(component_id='x-slider', component_property='value'),
    Input(component_id='y-slider', component_property='value'),
    Input(component_id='z-slider', component_property='value'),
    Input(component_id='Rx-slider', component_property='value'),
    Input(component_id='Ry-slider', component_property='value'),
    Input(component_id='Rz-slider', component_property='value'),
)
def update_output_div(x,y,z,Rx,Ry,Rz):
    return f"{x}", f"{y}", f"{z}", f"{Rx}", f"{Ry}", f"{Rz}"





app.run_server(debug=True, host='127.0.0.1')
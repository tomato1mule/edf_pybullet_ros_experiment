import os
import argparse
from dash import Dash, html, dcc

from edf.pc_utils import draw_geometry, create_o3d_points, get_plotly_fig
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, DemoSeqDataset, gzip_load
from edf.preprocess import Rescale, NormalizeColor, Downsample, PointJitter, ColorJitter
from edf.agent import PickAgent, PlaceAgent

app = Dash(__name__)

def main_func(log_id):
    agent_param_dir = "checkpoint/mug_10_demo/pick"
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




    grasp_pl = grasp_raw.plotly(point_size=1.0, name="grasp")
    query_opacity = query_attention ** 1
    query_pl = PointCloud.points_to_plotly(pcd=query_points, point_size=15.0, opacity=query_opacity / query_opacity.max())#, custom_data={'attention': query_attention.cpu()})
    fig_grasp = get_plotly_fig("Grasp")
    fig_grasp = fig_grasp.add_traces([grasp_pl, query_pl])



    target_pcd = PointCloud.merge(scene_raw, grasp_raw.transformed(target_pose)[0])
    target_pl = target_pcd.plotly(point_size=1.0)
    fig_target = get_plotly_fig("Target Placement")
    fig_target = fig_target.add_traces([target_pl])



    best_sample_pcd = PointCloud.merge(scene_raw, grasp_raw.transformed(best_pose)[0])
    best_sample_pl = best_sample_pcd.plotly(point_size=1.0)
    sample_pl = PointCloud.points_to_plotly(pcd=sampled_poses.points, point_size=7.0, colors=[0.2, 0.5, 0.8])
    fig_sample = get_plotly_fig("Sampled Placement")
    fig_sample = fig_sample.add_traces([best_sample_pl, sample_pl])

    app.layout = html.Div(children=[
                                    html.Div(children=[
                                                    dcc.Graph(id='target',
                                                            figure=fig_target)
                                    ], style={'padding': 10, 'flex': 1}),

                                    html.Div(children=[
                                                    dcc.Graph(id='grasp',
                                                            figure=fig_grasp)
                                    ], style={'padding': 10, 'flex': 1}),

                                    html.Div(children=[
                                                    dcc.Graph(id='sample',
                                                            figure=fig_sample)
                                    ], style={'padding': 10, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row'})




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization webserver for EDF place training')
    # parser.add_argument('--eval-config-dir', type=str, default='config/eval_config/eval.yaml',
    #                     help='')
    # parser.add_argument('--save-plot', action='store_true',
    #                     help='')
    # parser.add_argument('--place-max-distance-plan', type=float, default=[0.05, 1.5], nargs='+',
    #                     help='')
    parser.add_argument('-i', '--log-id', type=int, default=1,
                        help='')
    args = parser.parse_args()
    
    log_id = args.log_id
    main_func(log_id=log_id)

    app.run_server(debug=True, host='127.0.0.1')
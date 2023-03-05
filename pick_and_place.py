from typing import Optional, Union, List, Tuple, Dict, Any
import time

from ros_edf.ros_interface import EdfRosInterface
from ros_edf.pc_utils import pcd_from_numpy, draw_geometry, reconstruct_surface
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, save_demos
from edf.pc_utils import check_pcd_collision, optimize_pcd_collision
from edf.server import DashEdfDemoServer
from edf.env_interface import SUCCESS, PLAN_FAIL, EXECUTION_FAIL, RESET, INFEASIBLE, FEASIBLE

import torch
import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d


###### Initialize Demo Server ######
save_demo = True
n_episodes = 10
count_reset_episodes = False

scene_ranges = np.array([[-0.23, 0.23],
                         [-0.23, 0.23],
                         [-0.01, 0.4]])
point_size = 3.


demo_server = DashEdfDemoServer(scene_ranges = scene_ranges,
                                name = "dash_edf_demo_server", 
                                point_size = point_size, 
                                host_id = '127.0.0.1',
                                server_debug=False)
demo_server.run()


###### Define Primitives ######
pick_checklist = [('collision_check', {'colcheck_r': 0.003})]
place_checklist = [('collision_check', {'colcheck_r': 0.0015})]


def get_pick(scene: PointCloud, grasp: PointCloud) -> Union[str, SE3]:
    # return SE3([0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 0.275])
    demo_server.update_scene_pcd(pcd=scene)
    demo_server.update_grasp_pcd(pcd=grasp)
    user_response = demo_server.get_user_response()
    if user_response == RESET:
        return user_response
    elif isinstance(user_response, SE3):
        return user_response
    else:
        raise ValueError(f"Unknown user response: {user_response}")


def get_place(scene: PointCloud, grasp: PointCloud) -> Union[str, SE3]:
    # return SE3([0.5000, -0.5000, -0.5000, -0.5000, 0.13, -0.20, 0.32])
    demo_server.update_scene_pcd(pcd=scene)
    demo_server.update_grasp_pcd(pcd=grasp)
    user_response = demo_server.get_user_response()
    if user_response == RESET:
        return user_response
    elif isinstance(user_response, SE3):
        return user_response
    else:
        raise ValueError(f"Unknown user response: {user_response}")


def update_system_msg(msg: str, wait_sec: float = 0.):
    # print(msg)
    demo_server.update_robot_state(msg)
    if wait_sec:
        time.sleep(wait_sec)

def cleanup():
    demo_server.close()











def move_robot_near_target(pose: SE3, env_interface: EdfRosInterface):
    assert len(pose) == 1

    rel_pos = torch.tensor([-0.7, 0.], device=pose.device, dtype=pose.poses.dtype)
    pos = pose.poses[0,4:6] + rel_pos
    if pos[0] > -0.6:
        pos[0] = -0.6

    env_interface.move_robot_base(pos=pos) # x,y

def check_collision(pose: SE3, 
                    scene: PointCloud, 
                    grasp: PointCloud, 
                    colcheck_r: float # Should be similar to voxel filter size
                    ) -> bool:
    assert len(pose) == 1

    col_check = check_pcd_collision(x=scene, y=grasp.transformed(pose)[0], r = colcheck_r)

    return col_check

def feasibility_check(context: Dict[str, Any], check_list: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, str]:
    available_check_types = ['collision_check']
    
    feasibility, msg = FEASIBLE, 'FEASIBLE'
    for check in check_list:
        check_type, check_kwarg = check
        assert check_type in available_check_types

        if check_type == 'collision_check':
            col_check = check_collision(pose=context['pose'], 
                                        scene=context['scene'], 
                                        grasp=context['grasp'], **check_kwarg)
            if col_check:
                return INFEASIBLE, 'COLLISION_DETECTED'
            
    return feasibility, msg

def get_pre_post_pick(scene: PointCloud, grasp: PointCloud, pick_poses: SE3) -> Tuple[SE3, SE3]:
    # _, pre_pick_poses = optimize_pcd_collision(x=scene, y=grasp, 
    #                                             cutoff_r = 0.03, dt=0.01, eps=1., iters=50,
    #                                             rel_pose=pick_poses)
    pre_pick_poses = pick_poses * SE3(torch.tensor([1., 0., 0., 0., 0., 0., -0.05], device=pick_poses.device))
    #post_pick_poses = pre_pick_poses
    post_pick_poses = SE3(pick_poses.poses + torch.tensor([0., 0., 0., 0., 0., 0., 0.1], device=pick_poses.device))

    return pre_pick_poses, post_pick_poses


def get_pre_post_place(scene: PointCloud, grasp: PointCloud, place_poses: SE3, pre_pick_pose: SE3, pick_pose: SE3) -> Tuple[SE3, SE3]:
    assert len(pick_pose) == len(pre_pick_pose) == 1

    _, pre_place_poses = optimize_pcd_collision(x=scene, y=grasp, 
                                                cutoff_r = 0.03, dt=0.01, eps=1., iters=5,
                                                rel_pose=place_poses)
    post_place_poses = place_poses * pick_pose.inv() * pre_pick_pose

    return pre_place_poses, post_place_poses


def observe(env_interface, max_try: int, attach: bool) -> bool:
    success = True
    update_system_msg("Move to Observe...")


    # Move to default pose before moving to observation pose
    for _ in range(max_try):
        move_result, _info = env_interface.move_to_named_target("init")
        if move_result == SUCCESS:
            break
        else:
            continue

    # Move to observation pose
    if move_result == SUCCESS:
        env_interface.move_robot_base(pos = torch.tensor([-1.5, 0.]))
        for _ in range(max_try):
            move_result, _info = env_interface.move_to_named_target("observe")
            if move_result == SUCCESS:
                break
            else:
                continue
    
    # Observe
    if move_result == SUCCESS:
        if attach:
            env_interface.detach()
        grasp_raw = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
        if attach:
            env_interface.attach(obj = grasp_raw)
        scene_raw = env_interface.observe_scene(obs_type = 'pointcloud', update = True)


    # Come back to default pose
    if move_result == SUCCESS:
        for _ in range(max_try):
            move_result, _info = env_interface.move_to_named_target("init")
            if move_result == SUCCESS:
                break
            else:
                continue
    if move_result == SUCCESS:
        env_interface.move_robot_base(pos = torch.tensor([-0.7, 0.0]))
    
    if move_result != SUCCESS:
        update_system_msg(f"Cannot Move to Observation Pose ({move_result}). Resetting env...", wait_sec=2.0)
        success = False
        
    return success, (scene_raw, grasp_raw)




















###### Initialize Robot Interface ######
env_interface = EdfRosInterface(reference_frame = "scene")
env_interface.moveit_interface.arm_group.set_planning_time(seconds=1)
env_interface.moveit_interface.arm_group.allow_replanning(True)



demo_list = []
episode_count = 0
reset_signal = False

try:
    while True:
        if episode_count >= n_episodes:
            break
        ###### Reset Env ######
        update_system_msg('Resetting Environment...')
        env_interface.reset()
        if reset_signal and not count_reset_episodes:
            pass
        else:
            episode_count += 1
        reset_signal = False

        ###### Observe ######
        success, (scene_raw, grasp_raw) = observe(env_interface=env_interface, max_try = 10, attach=False)
        if not success:
            reset_signal = True
            continue

        ###### Sample Pick Pose ######
        pick_max_try = 100000
        for n_trial in range(pick_max_try):
            ###### Infer pick poses ######
            if n_trial == 0:
                update_system_msg('Waiting for pick poses...')
            pick_inference_result = get_pick(scene=scene_raw, grasp=grasp_raw)
            
            ###### Infer pre-pick and post-pick poses ######
            if isinstance(pick_inference_result, SE3):
                update_system_msg('Looking for feasible pick poses...')
                pick_poses: SE3 = pick_inference_result
                pre_pick_poses, post_pick_poses = get_pre_post_pick(scene=scene_raw, grasp=grasp_raw, pick_poses=pick_poses)

                ###### Check Feasiblity ######
                for idx in range(len(pick_poses)):
                    pick_pose, pre_pick_pose, post_pick_pose = pick_poses[idx], pre_pick_poses[idx], post_pick_poses[idx]
                    context = {'pose': pick_pose, 'scene': scene_raw, 'grasp': grasp_raw}
                    feasibility, _info = feasibility_check(context=context, check_list=pick_checklist)
                    if feasibility == FEASIBLE:
                        move_robot_near_target(pose=pick_pose, env_interface=env_interface)
                        pick_plan_result, pick_plans = env_interface.pick_plan(pre_pick_pose=pre_pick_pose, pick_pose=pick_pose)
                        if pick_plan_result == SUCCESS:
                            break
                        else:
                            _info = pick_plans
                            feasibility = INFEASIBLE
                            continue
                    else:
                        continue
                        
                if feasibility == FEASIBLE:
                    update_system_msg("Found feasible pick-pose! Executing")
                    break
                else:
                    if len(pick_pose) == 1:
                        update_system_msg(f"No feasible pick-pose found. Try again! (Reason: {_info})")
                    else:
                        update_system_msg("No feasible pick-pose found. Try again!")
                    continue
            ###### Reset Signal ######
            elif pick_inference_result == RESET:
                reset_signal = True
                break
            else:
                raise NotImplementedError(f"Unknown pick_inference_result: {pick_inference_result}")
            
        if reset_signal:
            continue
        elif n_trial == pick_max_try - 1:
            reset_signal = True
            continue
        else:
            pass

        ###### Execute Pick ######
        pick_result, _info = env_interface.pick_execute(plans=pick_plans, post_pick_pose=post_pick_pose)
        if pick_result == SUCCESS:
            update_system_msg(f"Pick result: {pick_result}")
            pick_demo = TargetPoseDemo(target_poses=pick_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)
            env_interface.detach()
            env_interface.attach_placeholder(size=0.15) # To avoid collsion with the grasped object
        else:
            update_system_msg(f"Pick result: {pick_result}, Resetting env...", wait_sec=2.0)
            reset_signal = True
            continue

        

        ###### Observe for Place ######
        success, (scene_raw, grasp_raw) = observe(env_interface=env_interface, max_try = 10, attach=True)
        if not success:
            reset_signal = True
            continue



        ###### Sample Place Pose ######
        place_max_try = 100000
        for n_trial in range(place_max_try):
            ###### Infer place poses ######
            if n_trial == 0:
                update_system_msg('Waiting for place poses...')
            place_inference_result = get_place(scene=scene_raw, grasp=grasp_raw)
            
            ###### Infer pre-place and post-place poses ######
            if isinstance(place_inference_result, SE3):
                update_system_msg('Looking for feasible place poses...')
                place_poses: SE3 = place_inference_result
                pre_place_poses, post_place_poses = get_pre_post_place(scene=scene_raw, grasp=grasp_raw, place_poses=place_poses, pre_pick_pose=pre_pick_pose, pick_pose=pick_pose)

                ###### Check Feasiblity ######
                for idx in range(len(place_poses)):
                    place_pose, pre_place_pose, post_place_pose = place_poses[idx], pre_place_poses[idx], post_place_poses[idx]
                    context = {'pose': place_pose, 'scene': scene_raw, 'grasp': grasp_raw}
                    feasibility, _info = feasibility_check(context=context, check_list=place_checklist)
                    if feasibility == FEASIBLE:
                        move_robot_near_target(pose=place_pose, env_interface=env_interface)
                        place_plan_result, place_plans = env_interface.place_plan(pre_place_pose=pre_place_pose, place_pose=place_pose)
                        if place_plan_result == SUCCESS:
                            break
                        else:
                            _info = place_plans
                            feasibility = INFEASIBLE
                            continue
                    else:
                        continue
                        
                if feasibility == FEASIBLE:
                    update_system_msg("Found feasible place-pose! Executing")
                    break
                else:
                    if len(place_pose) == 1:
                        update_system_msg(f"No feasible place-pose found. Try again! (Reason: {_info})")
                    else:
                        update_system_msg("No feasible place-pose found. Try again!")
                    continue
            ###### Reset Signal ######
            elif place_inference_result == RESET:
                reset_signal = True
                break
            else:
                raise NotImplementedError(f"Unknown place_inference_result: {place_inference_result}")
            
        if reset_signal:
            reset_signal = True
            continue
        elif n_trial == place_max_try - 1:
            reset_signal = True
            continue
        else:
            pass

        ###### Execute place ######
        place_result, _info = env_interface.place_execute(plans=place_plans, post_place_pose=post_place_pose)
        if place_result == SUCCESS:
            update_system_msg(f"Place result: {place_result}")
            place_demo = TargetPoseDemo(target_poses=place_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)
            env_interface.detach()
            env_interface.release()
        else:
            update_system_msg(f"Place result: {place_result}, Resetting env...", wait_sec=2.0)
            reset_signal = True
            continue


        demo_seq = DemoSequence(demo_seq = [pick_demo, place_demo])
        demo_list.append(demo_seq)

        # # Go back home
        # for _ in range(3):
        #     result = env_interface.move_to_named_target("init")
        #     if result == 'SUCCESS':
        #         break
        # print(f"Move to End-Effector observation pose: {result}")
except Exception as e:
    update_system_msg(f"Error occured. Saving previously collected demonstrations. \n ERROR: {e}")
    if save_demo and len(demo_list > 1):
        save_dir = "demo/test_demo"
        update_system_msg(f"Saving to {save_dir}")
        save_demos(demos=demo_list, dir=save_dir)


###### Save Collected Demonstrations ######

if save_demo:
    save_dir = "demo/test_demo"
    update_system_msg(f"Saving to {save_dir}")
    save_demos(demos=demo_list, dir=save_dir)


cleanup()


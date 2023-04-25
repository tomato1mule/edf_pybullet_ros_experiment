target_obj_pose=random # random, upright, lying
target_obj_name=random # train/mug0, test/mug1~10
n_distractors=4
use_support=true

roslaunch ros_edf ros_edf_env.launch target_obj_pose:=$target_obj_pose target_obj_name:=$target_obj_name n_distractors:=$n_distractors use_support:=$use_support
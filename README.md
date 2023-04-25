# Equivariant Descriptor Fields with ROS MoveIt and PyBullet
Example codes for Equivariant Descriptor Fields with and ROS MoveIt in PyBullet Simulation Environment.\
Please visit https://github.com/tomato1mule/edf for more information on Equivariant Descriptor Fields.
# Installation

**Step 1.** Clone Github repository.
It is important to clone **RECURSIVELY**.
```shell
git clone --recursive https://github.com/tomato1mule/edf_pybullet_ros_experiment
```

**Step 2.** Setup Conda/Mamba environment. We recommend using Mamba for faster installation.
```shell
# if you don't have mamba yet, install it first (not needed when using mambaforge):
conda install mamba -c conda-forge

# now create a new environment
mamba create -n edf_exp python=3.8
conda activate edf_exp
```

**Step 3.** Install EDF.
```shell
# go to edf folder
cd edf


# install edf and dependencies
CUDA=cu113
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install iopath fvcore
pip install --no-index --no-cache-dir pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_${CUDA}_pyt1110/download.html
pip install -e .


# come back to root folder
cd ..
```

**Step 4.** Install ROS (Noetic) on conda via robostack
```shell
# Visit these links if you have trouble installing robostack.
# 1) https://robostack.github.io/GettingStarted.html
# 2) https://github.com/RoboStack/ros-noetic


# this adds the conda-forge channel to the new created environment configuration and the robostack channels
conda config --env --add channels conda-forge
conda config --env --add channels robostack
conda config --env --add channels robostack-experimental
conda config --env --add channels robostack-humble

# There is a bug with cryptography==39.0.0, so please downgrade it if you face OpenSSL related issues.
# https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
pip install cryptography==38.0.4

# Install the version of ROS you are interested in:
mamba install ros-noetic-desktop

# optionally, install some compiler packages if you want to e.g. build packages in a colcon_ws:
mamba install compilers cmake pkg-config make ninja

# on linux and osx (but not Windows) you might want to:
mamba install catkin_tools

# only on linux, if you are having issues finding GL/OpenGL, also do:
mamba install mesa-libgl-devel-cos7-x86_64 mesa-dri-drivers-cos7-x86_64 libselinux-cos7-x86_64 libxdamage-cos7-x86_64 libxxf86vm-cos7-x86_64 libxext-cos7-x86_64 xorg-libxfixes



# Reboot Conda env
conda deactivate
conda activate edf_exp

# if you want to use rosdep, also do:
mamba install rosdep
rosdep init  # note: do not use sudo!
rosdep update
```

**Step 5.** Install Moveit and other ROS packages
```shell
mamba install ros-noetic-moveit=1.1.0 ros-noetic-moveit-ros-perception=1.1.0 ros-noetic-ros-controllers=0.18.1 ros-noetic-ros-control=0.19.4 ros-noetic-ros-numpy=0.0.4
```

**Step 6.** Configure Catkin workspace for ros_edf_interface package
```shell
# Go to ros_edf_interface folder to setup ros workspace environment
cd ros_edf_interface
catkin_make

# This script will make ros workspace setups to be auto-loaded when conda env is activated
bash conda_env_setup.sh

# Come back to root folder
cd ..

# Reboot Conda env
conda deactivate
conda activate edf_exp
```

**Step 7.** Install PyBullet Simulation Environment
```shell
# Move to install edf_env
cd edf_env
pip install -e .

# Come back to root folder
cd ..

# Reboot Conda env
conda deactivate
conda activate edf_exp
```

**Step 8. (Optional)** Set .env file for Jupyter notebook
```shell
bash create_jupyter_dotenv.sh
pip install python-dotenv
```

# Usage
## A) Collect Human Demo
**1.** First, run Pybullet environment
```shell
bash bringup_edf_env_ros.sh
```
**2.** In another terminal, run demo-collection web server
```shell
# flag --dont-save if you don't want to save demonstrations.
# flag -n=<number> to set number of demonstrations to collect.
python collect_human_demo.py -n=10 --save-dir="demo/mug_demo"
```
**3.** Go to http://127.0.0.1:8050/ in your web browser and collect demonstrations.

* Drag the screen to rotate
* Shift drag the screen to pan
* Drag the sliders to adjust target poses
* Press submit button to execute and save demonstrations.
* Press reset button to reset the environment. However, it will not immediately reset, so please be patient.
* See the 'Robot State' information in the left side of the panel.

**Troubleshooter**
* See the 'Robot State' information in the left side of the panel.
* Only provide demonstrations when the 'Robot State' is 'Waiting for pick/place poses'.
* If the robot does not move even if you submit a pose, please check the 'Robot State' message to see if the submitted pose has collision, or reach plan cannot be found.
* Do not connect to the web server in more than one browsers/tabs. Otherwise, you will see no point cloud.
* Point cloud will appear once the observation is done and robot is ready to receive demonstrations. 
* If the web server is not launching, please check you have run 'bash bringup_edf_env_ros.sh'.

**4.** Once all the demonstrations are collected, the 'Robot State' will inform that the demonstrations are saved to certain directions.

## B) Train
```shell
python pick_train.py
python place_train.py
```

## C) Evaluation
**0 (Optional).** If you want to use already trained weights, just rename 'checkpoint_example' folder to 'checkpoint'

**1.** First, run Pybullet environment in an evaluation setup (Unseen pose/instance/distractors)
```shell
bash bringup_edf_env_ros_eval.sh
```
**2.** In another terminal, run the inference code
```shell
python edf_pick_and_place.py
```

> **Note** Our motion planning pipeline is currently very unstable. Therefore, even if EDFs sampled correct pick-and-place poses, these poses may get rejected as no motion plans can be found by the planner. Plus, we did not fine-tune hyperparameters for this new environments (The robot and perception pipelines are different from the original experiments). 
>
>As a result, the success rate in this implementation is lower than the rate reported in the paper. Still, the success rate should be reasonably high (about 7~80%). If the success rate is lower than this rate, something must have gone wrong. 
>
>To reproduce the original experimental results in the paper, please check the following branch: https://github.com/tomato1mule/edf/tree/iclr2023_rebuttal_ver

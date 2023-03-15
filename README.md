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

```

**Step 8. (Optional)** Set .env file for Jupyter notebook
```shell
bash create_jupyter_dotenv.sh
pip install python-dotenv
```

# Usage
## Collect Human Demo
```shell
python collect_human_demo.py
```

## Train
```shell
python pick_train.py
python place_train.py
```
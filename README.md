# 1. edf_pybullet_ros_experiment
Experiments for Equivariant Descriptor Fields with PyBullet and ROS MoveIt.
# 2. Installation
It is important to clone **RECURSIVELY**.
```shell
git clone --recursive https://github.com/tomato1mule/edf_pybullet_ros_experiment
```

# 2.1 Setup conda env
```shell
# if you don't have mamba yet, install it first (not needed when using mambaforge):
conda install mamba -c conda-forge

# now create a new environment
mamba create -n edf_exp python=3.8
conda activate edf_exp

# Install dependencies

pip install -e .
```

## 2.2 Install ROS (Noetic) on conda via robostack
https://robostack.github.io/GettingStarted.html
https://github.com/RoboStack/ros-humble
https://github.com/RoboStack/ros-noetic
```shell


# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channels
conda config --env --add channels robostack
conda config --env --add channels robostack-experimental
conda config --env --add channels robostack-humble

# There is a bug with cryptography==39.0.0, so please downgrade it if you face OpenSSL related issues.
# https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
pip install cryptography==38.0.4

# Install the version of ROS you are interested in:
#mamba install ros-humble-desktop  # (or "mamba install ros-noetic-desktop" or "mamba install ros-galactic-desktop")
mamba install ros-noetic-desktop

# optionally, install some compiler packages if you want to e.g. build packages in a colcon_ws:
mamba install compilers cmake pkg-config make ninja #colcon-common-extensions

# on linux and osx (but not Windows) you might want to:
mamba install catkin_tools
# on Windows, install Visual Studio 2017 or 2019 with C++ support 
# see https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-160

# only on linux, if you are having issues finding GL/OpenGL, also do:
mamba install mesa-libgl-devel-cos7-x86_64 mesa-dri-drivers-cos7-x86_64 libselinux-cos7-x86_64 libxdamage-cos7-x86_64 libxxf86vm-cos7-x86_64 libxext-cos7-x86_64 xorg-libxfixes

# on Windows, install the Visual Studio command prompt via Conda:
# mamba install vs2019_win-64

# note that in this case, you should also install the necessary dependencies with conda/mamba, if possible

# reload environment to activate required scripts before running anything
# on Windows, please restart the Anaconda Prompt / Command Prompt!
conda deactivate
conda activate edf_exp

# if you want to use rosdep, also do:
mamba install rosdep
rosdep init  # note: do not use sudo!
rosdep update
```

# 2.2 Install Moveit2 and other packages
```shell
mamba install ros-noetic-moveit=1.1.0
mamba install ros-noetic-moveit-ros-perception=1.1.0
mamba install ros-noetic-ros-controllers=0.18.1
mamba install ros-noetic-ros-control=0.19.4
mamba install ros-noetic-ros-numpy=0.0.4
```

# 2.3 Configure catkin
```shell
cd catkin_ws
catkin_make
```

# 2.5 
```shell
bash conda_env_setup.sh
```

# 3
```shell
pip install jupyterthemes
jt -t onedork -fs 115 -nfs 125 -tfs 115 -dfs 115 -ofs 115 -cursc r -cellw 80% -lineh 115 -altmd  -kl -T -N
```
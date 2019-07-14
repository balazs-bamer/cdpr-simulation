# cdpr-simulation
Cable-driven parallel robot simulation

## Invocation

### Gazebo plugin:

```bash
source /opt/ros/melodic/setup.bash
source devel/setup.bash
cd src/cdpr_gazebo
roslaunch launch/cdpr_gazebo.launch
```

### Simple sine velocity command publisher


```bash
rosrun cdpr_gazebo sinevelocitytest
```

The Gazebo model and its generation, and partially the Gazebo plugin is based on this work: [Franklin Okoli, Yuchuan Lang, Olivier Kermorgant, Stéphane Caro, "Cable-Driven Parallel Robot simulation using Gazebo and ROS", The 22nd CISM IFToMM Symposium on Robot Design, Dynamics and Control, Jun 2018, Rennes, France ](https://hal.archives-ouvertes.fr/hal-01757531v1)

### Compile Tensorflow 1.13.1 using Cuda 10.0

sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i ~/Letöltések/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-toolkit-10-0
sudo ./configure 
sudo bazel build -c opt --config=cuda --copt="-mtune=native" --copt="-O3" tensorflow:libtensorflow_cc.so tensorflow:libtensorflow.so --genrule_strategy=standalone --spawn_strategy=standalone


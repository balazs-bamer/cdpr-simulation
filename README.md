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

The Gazebo model its generation, and partially the Gazebo plugin is based on this work: [Franklin Okoli, Yuchuan Lang, Olivier Kermorgant, Stéphane Caro, "Cable-Driven Parallel Robot simulation using Gazebo and ROS", The 22nd CISM IFToMM Symposium on Robot Design, Dynamics and Control, Jun 2018, Rennes, France ](https://hal.archives-ouvertes.fr/hal-01757531v1)


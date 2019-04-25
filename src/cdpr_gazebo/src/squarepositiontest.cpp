#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <cmath>

int main(int argc, char **argv) {
  const double cPublishFrequency  = 10.0; // Hz
  const double cPositionAmplitude =  0.05; // m
  const double cPositionBias      =  0.0; // m
  const double cPositionFrequency =  0.1; // Hz
  const size_t cWireCount         =  4u;

  ros::init(argc, argv, "squarepositiontest");
  ros::NodeHandle nodeHandle;
  ros::Publisher publisher = nodeHandle.advertise<sensor_msgs::Joy>("jointPositions", 100);
  ros::Rate loopRate(cPublishFrequency);

  sensor_msgs::Joy positionCommand;
  positionCommand.axes.resize(cWireCount);

  double time = 0.0;
  while (ros::ok()) {
    double sine = sin(time * cPositionFrequency * 2 * M_PI);
    double position = cPositionBias + copysign(cPositionAmplitude, sine);
    for(size_t i = 0; i < cWireCount; ++i) {
      positionCommand.axes[i] = position;
    }
    ROS_INFO("%f", position);

    publisher.publish(positionCommand);

    ros::spinOnce();

    loopRate.sleep();
    time += 1.0 / cPublishFrequency;
  }
  return 0;
}

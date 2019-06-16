#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <cmath>

int main(int argc, char **argv) {
  const double cGiga              =   1e9;
  const double cPublishFrequency  = 100.0; // Hz
  const double cVelocityAmplitude =   0.05; // m/s
  const double cVelocityFrequency =   0.1; // Hz
  const size_t cWireCount         =   4u;

  ros::init(argc, argv, "sinevelocitytest");
  ros::NodeHandle nodeHandle;
  ros::NodeHandle paramNodeHandle("~");
  double velocityAmplitude;
  double velocityFrequency;
  if(!paramNodeHandle.getParam("freq", velocityFrequency)) {
    velocityFrequency = cVelocityFrequency;
  }
  else { // nothing to do
  }
  if(!paramNodeHandle.getParam("amp", velocityAmplitude)) {
    velocityAmplitude = cVelocityAmplitude;
  }
  else { // nothing to do
  }
  ros::Publisher publisher = nodeHandle.advertise<sensor_msgs::Joy>("jointVelocities", 100);
  ros::Rate loopRate(cPublishFrequency);

  sensor_msgs::Joy velocityCommand;
  velocityCommand.axes.resize(cWireCount);

  double time = 0.0;
  while (ros::ok()) {
    double velocity = velocityAmplitude * sin(time * velocityFrequency * 2 * M_PI);
    for(size_t i = 0; i < cWireCount; ++i) {
      velocityCommand.axes[i] = velocity;
    }
    velocityCommand.header.stamp.sec = static_cast<int64_t>(floor(time));
    velocityCommand.header.stamp.nsec = static_cast<int64_t>((time - floor(time)) * cGiga);
    ROS_INFO("%f %f", time, velocity);

    publisher.publish(velocityCommand);

    ros::spinOnce();

    loopRate.sleep();
    time += 1.0 / cPublishFrequency;
  }
  return 0;
}

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <cmath>

int main(int argc, char **argv) {
  const double cPublishFrequency  = 10.0; // Hz
  const double cVelocityAmplitude =  0.06; // m/s
  const double cVelocityFrequency =  0.05; // Hz
  const size_t cWireCount         =  4u;

  ros::init(argc, argv, "squarevelocitytest");
  ros::NodeHandle nodeHandle;
  ros::Publisher publisher = nodeHandle.advertise<sensor_msgs::Joy>("jointVelocities", 100);
  ros::Rate loopRate(cPublishFrequency);

  sensor_msgs::Joy velocityCommand;
  velocityCommand.axes.resize(cWireCount);

  double time = 0.0;
  while (ros::ok()) {
    double sine = sin(time * cVelocityFrequency * 2 * M_PI);
    double velocity = abs(sine) >= sqrt(0.5) ? copysign(cVelocityAmplitude, sine) : 0.0;
    for(size_t i = 0; i < cWireCount; ++i) {
      velocityCommand.axes[i] = velocity;
    }
    ROS_INFO("%f", velocity);

    publisher.publish(velocityCommand);

    ros::spinOnce();

    loopRate.sleep();
    time += 1.0 / cPublishFrequency;
  }
  return 0;
}

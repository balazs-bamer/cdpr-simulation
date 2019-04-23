#ifndef CDPR_GAZEBO_PLUGIN_H
#define CDPR_GAZEBO_PLUGIN_H

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/JointController.hh>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>

namespace gazebo {

class CdprGazeboPlugin : public ModelPlugin {
private:
  static constexpr size_t        cWireCount                        =   4u;
  static constexpr size_t        cSubscriberQueueSize              = 256u;
  static constexpr size_t        cPublisherQueueSize               = 256u;
  static constexpr char          cVelocityTopic[]                  = "jointVelocities";
  static constexpr char          cPositionTopic[]                  = "jointPositions";
  static constexpr char          cCableStatesTopic[]               = "jointStates";
  static constexpr char          cWireStatesTopic[]                = "wireStates";
  static constexpr char          cPlatformPoseTopic[]              = "platformPose";
  static constexpr char          cSdfNameCable[]                   = "cable";
  static constexpr char          cSdfNameFrame[]                   = "frame";
  static constexpr char          cSdfNamePlatform[]                = "platform";
  static constexpr char          cLaunchParamPublishPeriod[]       = "/cdpr_gazebo_simulator/publishPeriod";
  static constexpr char          cLaunchParamVelocityEpsilon[]     = "/cdpr_gazebo_simulator/velocityEpsilon";
  static constexpr char          cLaunchParamVelocityControllerP[] = "/cdpr_gazebo_simulator/velocityControllerP";
  static constexpr char          cLaunchParamVelocityControllerI[] = "/cdpr_gazebo_simulator/velocityControllerI";
  static constexpr char          cLaunchParamVelocityControllerD[] = "/cdpr_gazebo_simulator/velocityControllerD";
  static constexpr char          cLaunchParamPositionControllerP[] = "/cdpr_gazebo_simulator/positionControllerP";
  static constexpr char          cLaunchParamPositionControllerI[] = "/cdpr_gazebo_simulator/positionControllerI";
  static constexpr char          cLaunchParamPositionControllerD[] = "/cdpr_gazebo_simulator/positionControllerD";

  ros::NodeHandle                mRosNode;
  physics::ModelPtr              mPhysicsModel;
  event::ConnectionPtr           mUpdateEvent;
  double                         mPublishPeriod;
  double                         mPreviousProcessingTime;
  double                         mVelocityEpsilon;

  std::vector<std::string>       mJointNames;
  std::vector<physics::JointPtr> mJoints;

  ros::CallbackQueue             mVelocityCallbackQueue;
  ros::Subscriber                mVelocityCommandSubscriber;
  sensor_msgs::Joy               mVelocityCommand;

  ros::CallbackQueue             mPositionCallbackQueue;
  ros::Subscriber                mPositionCommandSubscriber; // only for adjusting PID or debugging
  sensor_msgs::Joy               mPositionCommand;
  bool                           mPositionCommandReceived;

  ros::Publisher                 mJointStatePublisher;
  sensor_msgs::JointState        mJointStates;               // we keep it here to avoid re-creation on stack
  ros::Publisher                 mWireStatePublisher;

  ros::Publisher                 mPlatformStatePublisher;
  physics::LinkPtr               mFrameLink;
  physics::LinkPtr               mPlatformLink;

public:
  CdprGazeboPlugin() {}

  ~CdprGazeboPlugin() {
    mRosNode.shutdown();
  }

  virtual void Load(physics::ModelPtr aModel, sdf::ElementPtr aSdf);

private:
  void cableVelocityCommandCallback(const sensor_msgs::JoyConstPtr &aMsg);
  void cablePositionCommandCallback(const sensor_msgs::JoyConstPtr &aMsg);
  void obtainLinks();
  void initJointsAndController();
  void initCommunication();
  void update();
  void updateJointVelocities();
  void updateJointPositions();
  void publishJointStates(ros::Time const &aNow);
  void publishPlatformState(ros::Time const &aNow);
};

GZ_REGISTER_MODEL_PLUGIN(CdprGazeboPlugin)

}
#endif // CDPR_GAZEBO_PLUGIN_H

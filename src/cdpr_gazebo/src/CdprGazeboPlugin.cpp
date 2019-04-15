#include "cdpr_gazebo/CdprGazeboPlugin.h"

#include <gazebo/gazebo.hh>
#include <gazebo/common/Exception.hh>
#include <gazebo/physics/PhysicsIface.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/PhysicsEngine.hh>
#include <ros/node_handle.h>
#include "cdpr_gazebo/WireStates.h"
#include "cdpr_gazebo/PlatformState.h"

#include <string>

constexpr char gazebo::CdprGazeboPlugin::cVelocityTopic[];
constexpr char gazebo::CdprGazeboPlugin::cCableStatesTopic[];
constexpr char gazebo::CdprGazeboPlugin::cWireStatesTopic[];
constexpr char gazebo::CdprGazeboPlugin::cPlatformPoseTopic[];
constexpr char gazebo::CdprGazeboPlugin::cSdfNameCable[];
constexpr char gazebo::CdprGazeboPlugin::cSdfNameFrame[];
constexpr char gazebo::CdprGazeboPlugin::cSdfNamePlatform[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPublishPeriod[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerP[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerI[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerD[];

void gazebo::CdprGazeboPlugin::Load(physics::ModelPtr aModel, sdf::ElementPtr) {
  mPhysicsModel = aModel;
  mRosNode = ros::NodeHandle();
  
  obtainLinks();
  initJointsAndController();
  initCommunication();

  mRosNode.getParam(cLaunchParamPublishPeriod, mPublishPeriod);
  gzdbg << "Publish period=" << mPublishPeriod << std::endl;
  mPreviousProcessingTime = 0.0;

  mUpdateEvent = event::Events::ConnectWorldUpdateBegin(boost::bind(&CdprGazeboPlugin::update, this));

  ros::spinOnce();
  ROS_INFO("Started CDPR Plugin for %s.", aModel->GetName().c_str());
}

void gazebo::CdprGazeboPlugin::obtainLinks() {
  for(auto &link : mPhysicsModel->GetLinks()) {
    if(link->GetName() == cSdfNameFrame) {
      mFrameLink = link;
    }
    else if(link->GetName() == cSdfNamePlatform) {
      mPlatformLink = link;
    }
    else { // nothing to do
    }
  }
}
  
void gazebo::CdprGazeboPlugin::initJointsAndController() {
  double pidP;
  mRosNode.getParam(cLaunchParamVelocityControllerP, pidP);
  double pidI;
  mRosNode.getParam(cLaunchParamVelocityControllerI, pidI);
  double pidD;
  mRosNode.getParam(cLaunchParamVelocityControllerD, pidD);
  auto pidController = common::PID(pidP, pidI, pidD);
  gzdbg << "PID: P=" << pidP << " I=" << pidI << " D=" << pidD << std::endl;

  mJoints.resize(cWireCount);
  mJointNames.resize(cWireCount);
  auto jointController = mPhysicsModel->GetJointController();
  size_t jointsRead = 0u; 
  for(size_t i = 0u; i < mPhysicsModel->GetJointCount(); ++i) {
    std::string name;
    physics::JointPtr joint;
    joint = mPhysicsModel->GetJoints()[i];
    name = joint->GetName();

    if(name.find(cSdfNameCable) == 0u) {
      size_t index = stoul(name.substr(std::strlen(cSdfNameCable)));
      if(index < cWireCount) {
        ++jointsRead;
        mJoints[index] = joint;
        mJointNames[index] = name;
        jointController->AddJoint(joint);
        jointController->SetVelocityPID(joint->GetScopedName(), pidController);
      }
      else { // nothing to do
      }
    }
    else { // nothing to do
    }
  }
  
  if(jointsRead != cWireCount) {
    throw gazebo::common::Exception(__FILE__, __LINE__, "invalid joint count");
  }
  else { // nothing to do
  }
}
  
void gazebo::CdprGazeboPlugin::initCommunication() {
  ros::SubscribeOptions subscribeOptions = ros::SubscribeOptions::create<sensor_msgs::Joy>(cVelocityTopic, cSubscriberQueueSize,
                  boost::bind(&CdprGazeboPlugin::cableVelocityCommandCallback, this, _1),
                  ros::VoidPtr(), &mCallbackQueue);
  mVelocityCommandSubscriber = mRosNode.subscribe(subscribeOptions);
  mCommandReceived = false;

  mJointStatePublisher = mRosNode.advertise<sensor_msgs::JointState>(cCableStatesTopic, cPublisherQueueSize);
  mJointStates.name = mJointNames;
  mJointStates.position.resize(cWireCount);
  mJointStates.velocity.resize(cWireCount);
  mJointStates.effort.resize(cWireCount);

  mWireStatePublisher = mRosNode.advertise<cdpr_gazebo::WireStates>(cWireStatesTopic, cPublisherQueueSize);
  mPlatformStatePublisher = mRosNode.advertise<cdpr_gazebo::PlatformState>(cPlatformPoseTopic, cPublisherQueueSize);
}

void gazebo::CdprGazeboPlugin::update() {
  mCallbackQueue.callAvailable();

  if(mCommandReceived) {
    updateJointVelocities();
  }

  // TODO develop logic for wire state publishing
  // this should definitely emit marker events even if they are entirely skipped during a simulation step

  auto now = ros::Time::now();
  double nowDouble = now.toSec();
  if((nowDouble - mPreviousProcessingTime) > mPublishPeriod) {
    mPreviousProcessingTime = nowDouble;

    publishJointStates(now);
    publishPlatformState(now);
  }
  else { // nothing to do
  }
  ros::spinOnce();
}
  
void gazebo::CdprGazeboPlugin::publishJointStates(ros::Time const &aNow) {
  mJointStates.header.stamp = aNow;
  for(size_t i = 0; i < mJoints.size(); ++i) {
    mJointStates.position[i] = mJoints[i]->Position();
    mJointStates.velocity[i] = mJoints[i]->GetVelocity(0);
    mJointStates.effort[i] = mJoints[i]->GetForce(0);
  }
  mJointStatePublisher.publish(mJointStates);
}

void gazebo::CdprGazeboPlugin::publishPlatformState(ros::Time const &aNow) {
  cdpr_gazebo::PlatformState platformState;
  platformState.header.stamp = aNow;

  auto platformPose = mPlatformLink->WorldPose() - mFrameLink->WorldPose();
  platformState.pose.position.x = platformPose.Pos().X();
  platformState.pose.position.y = platformPose.Pos().Y();
  platformState.pose.position.z = platformPose.Pos().Z();
  platformState.pose.orientation.x = platformPose.Rot().X();
  platformState.pose.orientation.y = platformPose.Rot().Y();
  platformState.pose.orientation.z = platformPose.Rot().Z();
  platformState.pose.orientation.w = platformPose.Rot().W();
  auto vel = platformPose.Rot().RotateVector(mPlatformLink->RelativeLinearVel());
  platformState.velocity.linear.x = vel.X();
  platformState.velocity.linear.y = vel.Y();
  platformState.velocity.linear.z = vel.Z();
  vel = platformPose.Rot().RotateVector(mPlatformLink->RelativeAngularVel());
  platformState.velocity.angular.x = vel.X();
  platformState.velocity.angular.y = vel.Y();
  platformState.velocity.angular.z = vel.Z();

  mPlatformStatePublisher.publish(platformState);
}

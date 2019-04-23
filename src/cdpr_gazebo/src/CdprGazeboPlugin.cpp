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
constexpr char gazebo::CdprGazeboPlugin::cPositionTopic[];
constexpr char gazebo::CdprGazeboPlugin::cCableStatesTopic[];
constexpr char gazebo::CdprGazeboPlugin::cWireStatesTopic[];
constexpr char gazebo::CdprGazeboPlugin::cPlatformPoseTopic[];
constexpr char gazebo::CdprGazeboPlugin::cSdfNameCable[];
constexpr char gazebo::CdprGazeboPlugin::cSdfNameFrame[];
constexpr char gazebo::CdprGazeboPlugin::cSdfNamePlatform[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPublishPeriod[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityEpsilon[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerP[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerI[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerD[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerP[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerI[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerD[];

void gazebo::CdprGazeboPlugin::Load(physics::ModelPtr aModel, sdf::ElementPtr) {
  mPhysicsModel = aModel;
  mRosNode = ros::NodeHandle();
  
  obtainLinks();
  initJointsAndController();
  initCommunication();

  mRosNode.getParam(cLaunchParamPublishPeriod, mPublishPeriod);
  gzdbg << "Publish period = " << mPublishPeriod << std::endl;
  mPreviousProcessingTime = 0.0;
  
  mRosNode.getParam(cLaunchParamVelocityEpsilon, mVelocityEpsilon);
  gzdbg << "Velocity epsilon below which position is held fix =" << mVelocityEpsilon << std::endl;

  mUpdateEvent = event::Events::ConnectWorldUpdateBegin(boost::bind(&CdprGazeboPlugin::update, this));

  ros::spinOnce();
  ROS_INFO("Started CDPR Plugin for %s.", aModel->GetName().c_str());
}

void gazebo::CdprGazeboPlugin::cableVelocityCommandCallback(const sensor_msgs::JoyConstPtr &aMsg) {
  if(aMsg->axes.size() == cWireCount) {
    mVelocityCommand = *aMsg;
    mVelocityCommandReceived = true;
    for(size_t i = 0; i < cWireCount; ++i) {
      if(abs(mVelocityCommand.axes[i]) < mVelocityEpsilon) {
        if(!mPositionHeld[i]) {
          mPositionHeld[i] = true;
          mLastPositionToHold[i] = mJoints[i]->Position();
        }
        else { // nothing to do
        }
      }
      else {
        mPositionHeld[i] = false;
      }
    }
  }
  else { // nothing to do
  }
}

void gazebo::CdprGazeboPlugin::cablePositionCommandCallback(const sensor_msgs::JoyConstPtr &aMsg) {
  if(aMsg->axes.size() == cWireCount) {
    mPositionCommand = *aMsg;
    mPositionCommandReceived = true;
  }
  else { // nothing to do
  }
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
  double pidI;
  double pidD;
  mRosNode.getParam(cLaunchParamVelocityControllerP, pidP);
  mRosNode.getParam(cLaunchParamVelocityControllerI, pidI);
  mRosNode.getParam(cLaunchParamVelocityControllerD, pidD);
  auto velocityPidController = common::PID(pidP, pidI, pidD);
  gzdbg << "Velocity controller: P = " << pidP << "  I = " << pidI << "  D = " << pidD << std::endl;
  mRosNode.getParam(cLaunchParamPositionControllerP, pidP);
  mRosNode.getParam(cLaunchParamPositionControllerI, pidI);
  mRosNode.getParam(cLaunchParamPositionControllerD, pidD);
  auto positionPidController = common::PID(pidP, pidI, pidD);
  gzdbg << "Position controller: P = " << pidP << "  I = " << pidI << "  D = " << pidD << std::endl;

  mJoints.resize(cWireCount);
  mJointNames.resize(cWireCount);
  mLastPositionToHold.resize(cWireCount);
  mPositionHeld.resize(cWireCount);
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
        mLastPositionToHold[index] = joint->Position();
        mPositionHeld[index] = true;
        jointController->AddJoint(joint);
        jointController->SetVelocityPID(joint->GetScopedName(), velocityPidController);
        jointController->SetPositionPID(joint->GetScopedName(), positionPidController);
        jointController->SetPositionTarget(joint->GetScopedName(), mLastPositionToHold[index]);
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
                  ros::VoidPtr(), &mVelocityCallbackQueue);
  mVelocityCommandSubscriber = mRosNode.subscribe(subscribeOptions);
  mVelocityCommandReceived = false;
  subscribeOptions = ros::SubscribeOptions::create<sensor_msgs::Joy>(cPositionTopic, cSubscriberQueueSize,
                  boost::bind(&CdprGazeboPlugin::cablePositionCommandCallback, this, _1),
                  ros::VoidPtr(), &mPositionCallbackQueue);
  mPositionCommandSubscriber = mRosNode.subscribe(subscribeOptions);
  mPositionCommandReceived = false;

  mJointStatePublisher = mRosNode.advertise<sensor_msgs::JointState>(cCableStatesTopic, cPublisherQueueSize);
  mJointStates.name = mJointNames;
  mJointStates.position.resize(cWireCount);
  mJointStates.velocity.resize(cWireCount);
  mJointStates.effort.resize(cWireCount);

  mWireStatePublisher = mRosNode.advertise<cdpr_gazebo::WireStates>(cWireStatesTopic, cPublisherQueueSize);
  mPlatformStatePublisher = mRosNode.advertise<cdpr_gazebo::PlatformState>(cPlatformPoseTopic, cPublisherQueueSize);
}

void gazebo::CdprGazeboPlugin::update() {
  mVelocityCallbackQueue.callAvailable();
  mPositionCallbackQueue.callAvailable();

  if(mVelocityCommandReceived) {
    updateJointVelocities();
  }
  else { // nothing to do
  }
  if(mPositionCommandReceived) {
    updateJointPositions();
  }
  else { // nothing to do
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

void gazebo::CdprGazeboPlugin::updateJointVelocities() {
  auto jointController = mPhysicsModel->GetJointController();
  for(size_t i = 0; i < cWireCount; ++i) {
    if(mPositionHeld[i]) {
      jointController->SetPositionTarget(mJoints[i]->GetScopedName(), mLastPositionToHold[i]);
    }
    else {
      jointController->SetVelocityTarget(mJoints[i]->GetScopedName(), mVelocityCommand.axes[i]);
    }
  }
}
  
void gazebo::CdprGazeboPlugin::updateJointPositions() {
  auto jointController = mPhysicsModel->GetJointController();
  for(size_t i = 0; i < cWireCount; ++i) {
    jointController->SetPositionTarget(mJoints[i]->GetScopedName(), mPositionCommand.axes[i]);
  }
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

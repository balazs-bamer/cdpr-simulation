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
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerMaxI[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamVelocityControllerMaxCmd[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerP[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerI[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerD[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerMaxI[];
constexpr char gazebo::CdprGazeboPlugin::cLaunchParamPositionControllerMaxCmd[];

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
  double pidMaxI;
  double pidMaxCmd;
  mRosNode.getParam(cLaunchParamVelocityControllerP, pidP);
  mRosNode.getParam(cLaunchParamVelocityControllerI, pidI);
  mRosNode.getParam(cLaunchParamVelocityControllerD, pidD);
  mRosNode.getParam(cLaunchParamVelocityControllerMaxI, pidMaxI);
  mRosNode.getParam(cLaunchParamVelocityControllerMaxCmd, pidMaxCmd);
  auto velocityPidController = common::PID(pidP, pidI, pidD, pidMaxI, -pidMaxI, pidMaxCmd, -pidMaxCmd);
mTestPid.Init(pidP, pidI, pidD, pidMaxI, -pidMaxI, pidMaxCmd, -pidMaxCmd);
  gzdbg << "Velocity controller: P = " << pidP << "  I = " << pidI << "  D = " << pidD << "  maxI = " << pidMaxI << "  maxCmd = " << pidMaxCmd<< std::endl;
  mRosNode.getParam(cLaunchParamPositionControllerP, pidP);
  mRosNode.getParam(cLaunchParamPositionControllerI, pidI);
  mRosNode.getParam(cLaunchParamPositionControllerD, pidD);
  mRosNode.getParam(cLaunchParamPositionControllerMaxI, pidMaxI);
  mRosNode.getParam(cLaunchParamPositionControllerMaxCmd, pidMaxCmd);
  auto positionPidController = common::PID(pidP, pidI, pidD, pidMaxI, -pidMaxI, pidMaxCmd, -pidMaxCmd);
  gzdbg << "Position controller: P = " << pidP << "  I = " << pidI << "  D = " << pidD << "  maxI = " << pidMaxI << "  maxCmd = " << pidMaxCmd<< std::endl;

  mJoints.resize(cWireCount);
  mJointNames.resize(cWireCount);
  mForceCalculators.resize(cWireCount);

  auto jointController = mPhysicsModel->GetJointController();
  size_t jointsRead = 0u; 
  for(size_t i = 0u; i < mPhysicsModel->GetJointCount(); ++i) {
    std::string name;
    physics::JointPtr joint;
    joint = mPhysicsModel->GetJoints()[i];
    name = joint->GetName();

    if(name.find(cSdfNameCable) == 0u) {
      size_t indexCableFound = stoul(name.substr(std::strlen(cSdfNameCable)));
      if(indexCableFound < cWireCount) {
        JointForceCalculator(mPhysicsModel, joint, positionPidController, velocityPidController);
        forceCalculator.setPositionTarget(joint->Position());
        mJoints[indexCableFound] = joint;
        mJointNames[indexCableFound] = name;
        mForceCalculators[indexCableFound] = forceCalculator;
        ++jointsRead;
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
    for(size_t i = 0; i < cWireCount; ++i) {
      forceCalculators[i].setVelocityTarget(mVelocityCommand.axes[i]);
    }
  }
  else { // nothing to do
  }
  if(mPositionCommandReceived) {
    for(size_t i = 0; i < cWireCount; ++i) {
      forceCalculators[i].setPositionTarget(mPositionCommand.axes[i]);
    }
  }
  else { // nothing to do
  }
  for(size_t i = 0; i < cWireCount; ++i) {
    mJoints[i].SetForce(mForceCalculators[i].update());
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

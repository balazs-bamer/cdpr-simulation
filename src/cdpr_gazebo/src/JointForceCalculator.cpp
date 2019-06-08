/*
 * Copyright (C) 2012 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#include "cdpr_gazebo/JointForceCalculator.h"
#include <gazebo/physics/World.hh>


gazebo::physics::JointForceCalculator::JointForceCalculator() noexcept
: mPhysicsModel(nullptr)
, mJoint(nullptr)
, mVelocityEpsilon(0.0)
, mLastUpdateTime(0.0) {
}
  
gazebo::physics::JointForceCalculator::JointForceCalculator(ModelPtr aModel, JointPtr aJoint, gazebo::common::Pid const &aPositionPid, gazebo::common::Pid const &aVelocityPid, double const aVelocityEpsilon) noexcept
: mPhysicsModel(aModel)
, mJoint(aJoint)
, mPositionPid(aPositionPid)
, mVelocityPid(aVelocityPid)
, mVelocityEpsilon(aVelocityEpsilon)
, mLastUpdateTime(mPhysicsModel->GetWorld()->SimTime()) {
}

gazebo::physics::JointForceCalculator& gazebo::physics::JointForceCalculator::operator=(gazebo::physics::JointForceCalculator const &aOther) noexcept {
  if(this != &aOther) {
    mPhysicsModel    = aOther.mPhysicsModel;
    mJoint           = aOther.mJoint;
    mPositionPid     = aOther.mPositionPid;
    mVelocityPid     = aOther.mVelocityPid;
    mUpdateMode      = aOther.mUpdateMode;
    mForce           = aOther.mForce;
    mPositionTarget  = aOther.mPositionTarget;
    mVelocityTarget  = aOther.mVelocityTarget;
    mVelocityEpsilon = aOther.mVelocityEpsilon;
    mLastUpdateTime  = mPhysicsModel->GetWorld()->SimTime();
    reset();
  }
  else { // nothing to do
  }
  return *this;
}

extern bool theZeroest;

double gazebo::physics::JointForceCalculator::update() {
  gazebo::common::Time currTime = mPhysicsModel->GetWorld()->SimTime();
  gazebo::common::Time stepTime = currTime - mLastUpdateTime;
  mLastUpdateTime = currTime;

  double force = 0.0;

  if (stepTime > 0) {
    if(mUpdateMode == UpdateMode::Force) {
      mLastPosition = mJoint->Position(0);
      force = mForce;
    }
    else if(mUpdateMode == UpdateMode::Velocity) {
      if(abs(mVelocityTarget) > mVelocityEpsilon) {
if(theZeroest)
gzdbg << "V " << mJoint->GetVelocity(0) << "  VT " << mVelocityTarget << "  ";
        mLastPosition = mJoint->Position(0);
        force = mVelocityPid.update(mVelocityTarget, mJoint->GetVelocity(0), currTime.Double());
      }
      else {
if(theZeroest)
gzdbg << "P " << mJoint->Position(0) << "  LP " << mLastPosition << "  ";
        force = mPositionPid.update(mLastPosition, mJoint->Position(0), currTime.Double());
      }
    }
    else if(mUpdateMode == UpdateMode::Position) {
if(theZeroest)
gzdbg << "P " << mJoint->Position(0) << "  PT " << mPositionTarget << "  ";
      mLastPosition = mJoint->Position(0);
      force = mPositionPid.update(mPositionTarget, mJoint->Position(0), currTime.Double());
    }
    else { // nothing to do
    }
  }
  else { // nothing to do
  }
  return force;
}


void gazebo::physics::JointForceCalculator::setPositionTarget(const double aTarget) noexcept {
  mPositionTarget = aTarget;
  if(mUpdateMode != UpdateMode::Position) {
    mPositionPid.reset();
  }
  else { // nothing to do
  }
  mUpdateMode = UpdateMode::Position;
}

/// \brief Set the target velocity for the velocity PID controller.
/// \param[in] _target Velocity target.
void gazebo::physics::JointForceCalculator::setVelocityTarget(const double aTarget) noexcept {
  mVelocityTarget = aTarget;
  if(mUpdateMode != UpdateMode::Velocity) {
    mVelocityPid.reset();
  }
  else { // nothing to do
  }
  mUpdateMode = UpdateMode::Velocity;
}


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


  
gazebo::physics::JointForceCalculator::JointForceCalculator(ModelPtr aModel, JointPtr aJoint, gazebo::common::Pid const &aPositionPid, gazebo::common::Pid const &aVelocityPid) noexcept
: mPhysicsModel(aModel)
, mJoint(aJoint)
, mPositionPid(aPositionPid)
, mVelocityPid(aVelocityPid)
, mLastUpdateTime(mPhysicsModel->GetWorld()->SimTime()) {
}

gazebo::physics::JointForceCalculator& gazebo::physics::JointForceCalculator::operator=(gazebo::physics::JointForceCalculator const &aOther) noexcept {
  if(this != &aOther) {
    mPhysicsModel   = aOther.mPhysicsModel;
    mJoint          = aOther.mJoint;
    mPositionPid    = aOther.mPositionPid;
    mVelocityPid    = aOther.mVelocityPid;
    mUpdateMode     = aOther.mUpdateMode;
    mForce          = aOther.mForce;
    mPositionTarget = aOther.mPositionTarget;
    mVelocityTarget = aOther.mVelocityTarget;
    mLastUpdateTime = mPhysicsModel->GetWorld()->SimTime();
    reset();
gzdbg << " F = " << mForce << "  P = " << mPositionTarget << "  V = " << mVelocityTarget << std::endl;
  }
  else { // nothing to do
  }
}

extern bool theZeroest;

double gazebo::physics::JointForceCalculator::update() noexcept {
  gazebo::common::Time currTime = mPhysicsModel->GetWorld()->SimTime();
  gazebo::common::Time stepTime = currTime - mLastUpdateTime;
  mLastUpdateTime = currTime;

  double force = 0.0;

  // Skip the update step if SimTime appears to have gone backward.
  // Negative update time wreaks havok on the integrators.
  // This happens when World::ResetTime is called.
  // TODO: fix this when World::ResetTime is improved
  if (stepTime > 0) {
    if(mUpdateMode == UpdateMode::Force) {
      force = mForce;
    }
    else if(mUpdateMode == UpdateMode::Velocity) {
if(theZeroest)
gzdbg << "V " << mJoint->GetVelocity(0) << "  VT " << mVelocityTarget << "  ";
      force = mVelocityPid.update(mVelocityTarget, mJoint->GetVelocity(0), stepTime);
    }
    if(mUpdateMode == UpdateMode::Position) {
      force = mPositionPid.update(mPositionTarget, mJoint->Position(0), stepTime);
//gzdbg << "position F = " << force << "  p = " << mJoint->Position(0) << "  pt = " << mPositionTarget << "  dt = " << stepTime << std::endl;
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


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
    mPhysicsModel = aOther.mPhysicsModel;
    mJoint = aOther.mJoint;
    mPositionPid = aOther.mPositionPid;
    mVelocityPid = aOther.mVelocityPid;
    mLastUpdateTime = mPhysicsModel->GetWorld()->SimTime();
    reset();
  }
  else { // nothing to do
  }
}

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
      force = mVelocityPid.update(mJoint->GetVelocity(0) - mVelocityTarget, stepTime);
    }
    if(mUpdateMode == UpdateMode::Position) {
      force = mVelocityPid.update(mJoint->Position(0) - mPositionTarget, stepTime);
    }
  }
  else { // nothing to do
  }

  return force;
}




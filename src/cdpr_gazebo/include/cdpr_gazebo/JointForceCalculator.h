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
#ifndef _GAZEBO_JOINTFORCECALCULATOR_HH_
#define _GAZEBO_JOINTFORCECALCULATOR_HH_

#include "cdpr_gazebo/Pid.h"
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/common/Time.hh>
//#include <gazebo/physics/PhysicsTypes.hh>
//#include "gazebo/util/system.hh"

namespace gazebo::physics {
/// \addtogroup gazebo_physics
/// \{

/// \brief A class for calculating forces for physics::JointController using position, velocity and force.
class JointForceCalculator {
  enum class UpdateMode {
    Force, Position, Velocity
  };

  ModelPtr              mPhysicsModel;
  JointPtr              mJoint;
  gazebo::common::Pid   mPositionPid;
  gazebo::common::Pid   mVelocityPid;
 
  UpdateMode            mUpdateMode = UpdateMode::Force;

  double                mVelocityEpsilon;
  double                mLastPosition   = 0.0;
  double                mForce          = 0.0;
  double                mPositionTarget = 0.0;
  double                mVelocityTarget = 0.0;

  gazebo::common::Time  mLastUpdateTime;

public:
  JointForceCalculator() noexcept;

  /// \brief Constructor
  /// \param[in] _model Model that uses this joint controller.
  JointForceCalculator(ModelPtr aModel, JointPtr aJoint, gazebo::common::Pid const &aPositionPid, gazebo::common::Pid const &aVelocityPid, double const aVelocityEpsilon) noexcept;

  /// \brief Destructor
  virtual ~JointForceCalculator() {
  }

  JointForceCalculator& operator=(JointForceCalculator const &aOther) noexcept;

  /// \brief Update the joint control using the last value set, returns the force.
  double update();

  /// \brief Reset all commands
  void reset() noexcept {
    mForce = mPositionTarget = mVelocityTarget = 0.0;
    mVelocityPid.reset();
    mPositionPid.reset();
  }

  /// \brief Get the last time the controller was updated.
  /// \return Last time the controller was updated.
  common::Time getLastUpdateTime() const noexcept {
    return mLastUpdateTime;
  }

  /// \brief Set the target position for the position PID controller.
  /// \param[in] _target Position target.
  void setPositionTarget(const double aTarget) noexcept;

  /// \brief Set the target velocity for the velocity PID controller.
  /// \param[in] _target Velocity target.
  void setVelocityTarget(const double aTarget) noexcept;

  /// \brief Set the applied effort for the specified joint.
  /// This force will persist across time steps.
  /// \param[in] _force Force to apply.
  void setForce(const double aForce) noexcept {
    mForce = aForce;
    mUpdateMode = UpdateMode::Force;
  }
};

}
#endif

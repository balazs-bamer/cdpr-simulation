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

#ifndef _GAZEBO_Pid_HH_
#define _GAZEBO_Pid_HH_

#include "gazebo/common/Time.hh"
#include "gazebo/util/system.hh"

namespace gazebo::common {
/// \addtogroup gazebo_common
/// \{

/// \class Pid Pid.hh common/common.hh
/// \brief Generic Pid controller class.
/// Generic proportiolnal-integral-derivative controller class that
/// keeps track of Pid-error states and control inputs given
/// the state of a system and a user specified target state.
class Pid {
private:
  class IirFilter1 {
  public:
    IirFilter1() noexcept : mCoefficient(0.0) {
    }

    IirFilter1(double const aCoefficient) noexcept : mCoefficient(aCoefficient) {
    }

    IirFilter1(IirFilter1 const &aOther) noexcept {
      mCoefficient = aOther.mCoefficient;
      reset();
    }

    void reset() noexcept {
      mLastOutput = 0.0;
    }

    double update(double const aInput) noexcept {
      mLastOutput = mLastOutput * mCoefficient + aInput * (1.0 - mCoefficient);
      return mLastOutput;
    } 

  private:
    double mCoefficient;
    double mLastOutput = 0.0;
  };

public:
  /// \brief Constructor, zeros out Pid values when created and
  /// initialize Pid-gains and integral term limits:[mImax:mImin]-[I1:I2].
  ///
  /// Disable command clamping by setting _mCmdMin to a value larger
  /// than _mCmdMax. Command clamping is disabled by default.
  ///
  /// \param[in] aPgain  The proportional gain.
  /// \param[in] aIgain  The integral gain.
  /// \param[in] aDgain  The derivative gain.
  /// \param[in] aImax The integral upper limit.
  /// \param[in] aImin The integral lower limit.
  /// \param[in] aCmdMax Output max value.
  /// \param[in] aCmdMin Output min value.
  Pid(double aPgain, double aIgain, double aDgain, double aImax, double aImin,
         double aCmdMax, double aCmdMin, double aPfilterCoefficient, double aDfilterCoefficient) noexcept;

  /// \brief Destructor
  virtual ~Pid() {
  }

  /// \brief Assignment operator
  /// \param[in] aOther a reference to a Pid to assign values from
  /// \return reference to this instance
  Pid &operator=(const Pid &aOther) noexcept;

  /// \brief reset the errors and command.
  void reset() noexcept {
    mErrLast = mPerr = mIerr = mDerr = mCmd = 0.0;
    mPfilter.reset();
    mDfilter.reset();
  }

  /// \brief update the Pid loop with nonuniform time step size.
  /// \param[_in] aError  Error since last call (p_state - p_target).
  /// \param[_in] aDt Change in time since last update call.
  /// Normally, this is called at every time step,
  /// The return value is an updated command to be passed
  /// to the object being controlled.
  /// \return the command value
  double update(double aError, common::Time aDt) noexcept;

private:
  /// \brief Error at a previous step.
  double mErrLast;

  /// \brief Current error.
  double mPerr;

  /// \brief Integral error.
  double mIerr;

  /// \brief Derivative error.
  double mDerr;

  /// \brief Gain for proportional control.
  double mPgain;

  /// \brief Gain for integral control.
  double mIgain;

  /// \brief Gain for derivative control.
  double mDgain;

  /// \brief Maximum clamping value for integral term.
  double mImax;

  /// \brief Minim clamping value for integral term.
  double mImin;

  /// \brief Command value.
  double mCmd;

  /// \brief Max command clamping value.
  double mCmdMax = -1.0;

  /// \brief Min command clamping value.
  double mCmdMin = 0.0;

  /// \brief First order IIR filter for P input
  IirFilter1 mPfilter;

  /// \brief First order IIR filter for D input
  IirFilter1 mDfilter;
};

}
#endif

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

#include <vector>
#include <eigen3/Eigen/Dense>
#include "cdpr_gazebo/Filter.h"
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
  class CascadeFilter {
  public:
    CascadeFilter() noexcept : CascadeFilter(0.0, 0.0, 0u) {
    }

    CascadeFilter(double const aRelCutoff, double const aQuality, size_t const aCascade) noexcept;

    CascadeFilter(CascadeFilter const &aOther) noexcept : CascadeFilter(aOther.mRelCutoff, aOther.mQuality, aOther.mCascade) {
    }

    void reset() noexcept {
      for(auto &filter : mFilters) {
        filter.SetValue(0.0);
      }
    }

    double update(double const aInput) noexcept;

  private:
    double mRelCutoff;
    double mQuality;
    size_t mCascade;
    std::vector<gazebo::math::BiQuad<double>> mFilters;
  };

public:
  struct FilterParameters {
    double relCutoff;
    double quality;
    size_t cascade;
  };

  struct PidParameters {
    double forwardGain;
    double pGain;
    double iGain;
    double dGain;
    size_t dDegree;
    size_t dBufferLength;
    double iLimit;
    double cmdLimit;
    FilterParameters pFilter;
    FilterParameters dFilter;
  };

  Pid() noexcept;
  Pid(PidParameters const &aParams) noexcept;

  /// \brief Destructor
  virtual ~Pid() {
  }

  /// \brief Assignment operator
  /// \param[in] aOther a reference to a Pid to assign values from
  /// \return reference to this instance
  Pid &operator=(Pid const &aOther) noexcept;

  /// \brief reset the errors and command.
  void reset() noexcept;

  /// \brief update the Pid loop with nonuniform time step size.
  /// \param[_in] aDesired Current desired value.
  /// \param[_in] aActual Current actual value.
  /// \param[_in] aDt Change in time since last update call.
  /// Normally, this is called at every time step,
  /// The return value is an updated command to be passed
  /// to the object being controlled.
  /// \return the command value
  double update(double const aDesired, double const aActual, double const aDt) noexcept;

  double derive(double const aValue, double const aNow) noexcept;
  void fitPolynomial() noexcept;

private:
  bool mWasLastTime;
  double mLastTime;

  /// \brief Current error.
  double mPerr;

  /// \brief Integral error.
  double mIerr;

  /// \brief Derivative error.
  double mDerr;

  double mForwardGain;

  /// \brief Gain for proportional control.
  double mPgain;

  /// \brief Gain for integral control.
  double mIgain;

  /// \brief Gain for derivative control.
  double mDgain;
  size_t mDpolynomialDegree;
  size_t mDbufferLength;

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
  CascadeFilter mPfilter;

  /// \brief First order IIR filter for D input
  CascadeFilter mDfilter;

  size_t mDbufferMissing;
  std::vector<double> mDbufferX;
  std::vector<double> mDbufferY;
  std::vector<double> mFitX;
  Eigen::MatrixXd mFitA;
  Eigen::VectorXd mFitB;
  Eigen::VectorXd mDpolynomCoefficients;
};

}
#endif

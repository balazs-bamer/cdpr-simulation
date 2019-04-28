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

#include <math.h>
#include <cmath>
#include <stdio.h>

#include <ignition/math/Helpers.hh>
#include "cdpr_gazebo/Pid.h"
#include <gazebo/gazebo.hh>


/////////////////////////////////////////////////
gazebo::common::Pid::Pid(double aPgain, double aIgain, double aDgain, double aImax, double aImin,
         double aCmdMax, double aCmdMin, double aPfilterCoefficient, double aDfilterCoefficient) noexcept
  : mPgain(aPgain)
  , mIgain(aIgain)
  , mDgain(aDgain)
  , mImax(aImax)
  , mImin(aImin)
  , mCmdMax(aCmdMax)
  , mCmdMin(aCmdMin)
  , mPfilter(aPfilterCoefficient)
  , mDfilter(aDfilterCoefficient) {
  reset();
}
  
Pid& gazebo::common::Pid::operator=(const Pid &aOther) noexcept {
  if(this != &aOther) {
    mPgain = aOther.mPgain;
    mIgain = aOther.mIgain;
    mDgain = aOther.mDgain;
    mImax = aOther.mImax;
    mImin = aOther.mImin;
    mCmdMax = aOther.mCmdMax;
    mCmdMin = aOther.mCmdMin;
    mPfilter = aOther.mPfilter;
    mDfilter = aOther.mDfilter;
    reset();
  }
  else { // nothing to do
  }
  return *this;
}

/////////////////////////////////////////////////
double gazebo::common::Pid::update(double aError, common::Time aDt) noexcept {
  if (aDt == common::Time(0, 0) || ignition::math::isnan(aError) ||  std::isinf(aError)) {
    return 0.0;
  }
  else {
    mPerr = mPfilter(aError);
    double pTerm = mPgain * mPerr;

    double prevIerr = mIerr;
    mIerr = mIerr + aDt.Double() * aError;
    double iTerm = mIgain * mIerr;
    if (iTerm > mImax) {
      iTerm = mImax;
      mIerr = iTerm / mIgain;
    }
    else if (iTerm < mImin) {
      iTerm = mImin;
      mIerr = iTerm / mIgain;
    }
    else { // nothing to do
    }

    if (aDt > common::Time(0, 0)) {
      mDerr = mDfilter((aError - mErrLast) / aDt.Double());
//gzdbg << "mPerr " << mPerr << "  mPerrLast " << mPerrLast << "  diff " << (mPerr - mPerrLast) << "  dt " << aDt.Double() << "  mDerr " << mDerr << " ";
      mErrLast = aError;
    }
    else {
//gzdbg << "dt was " << aDt.Double() << std::endl;
    }
    double dTerm = mDgain * mDerr;

    double cmd = -pTerm - iTerm - dTerm;

//gzdbg << "  P " << -pTerm << "  I " << -iTerm << "  D " << -dTerm << std::endl;

    if (mCmdMax >= mCmdMin) {
      mCmd = ignition::math::clamp(cmd, mCmdMin, mCmdMax);
    }
    else { // nothing to do
    }

    if(mCdm != cmd) {
      mIerr = prevIerr;
      mCmd += iTerm - mIgain * prevIerr;
    }
    else { // nothing to do
    }
  }

  return mCmd;
}



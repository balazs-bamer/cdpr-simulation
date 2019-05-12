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

#include <cmath>
#include <stdio.h>
#include <algorithm>

#include <ignition/math/Helpers.hh>
#include "cdpr_gazebo/Pid.h"
#include <gazebo/gazebo.hh>

gazebo::common::Pid::CascadeFilter::CascadeFilter(double const aRelCutoff, double const aQuality, size_t const aCascade) noexcept
: mRelCutoff(aRelCutoff)
, mQuality(aQuality)
, mCascade(aCascade) {
  mFilters.resize(aCascade);
  for(auto &filter : mFilters) {
    filter.SetValue(0.0);
    filter.SetFc(aRelCutoff, 1.0, aQuality);
  }
}

double gazebo::common::Pid::CascadeFilter::update(double const aInput) noexcept {
  double output = aInput;
  for(auto &filter : mFilters) {
    output = filter.process(output);
  }
  return output;
}

/////////////////////////////////////////////////
gazebo::common::Pid::Pid(PidParameters const &aPidParams) noexcept
: mForwardGain(aPidParams.forwardGain)
, mPgain(aPidParams.pGain)
, mIgain(aPidParams.iGain)
, mDgain(aPidParams.dGain)
, mDbufferLength(aPidParams.dBufferLength)
, mImax(abs(aPidParams.iLimit))
, mImin(-abs(aPidParams.iLimit))
, mCmdMax(abs(aPidParams.cmdLimit))
, mCmdMin(-abs(aPidParams.cmdLimit))
, mPfilter(aPidParams.pFilter.relCutoff, aPidParams.pFilter.quality, aPidParams.pFilter.cascade)
, mDfilter(aPidParams.dFilter.relCutoff, aPidParams.dFilter.quality, aPidParams.dFilter.cascade) {
  reset();
}
  
gazebo::common::Pid& gazebo::common::Pid::operator=(const gazebo::common::Pid &aOther) noexcept {
  if(this != &aOther) {
    mForwardGain = aOther.mForwardGain;
    mPgain   = aOther.mPgain;
    mIgain   = aOther.mIgain;
    mDgain   = aOther.mDgain;
    mDbufferLength = aOther.mDbufferLength;
    mImax    = aOther.mImax;
    mImin    = aOther.mImin;
    mCmdMax  = aOther.mCmdMax;
    mCmdMin  = aOther.mCmdMin;
    mPfilter = aOther.mPfilter;
    mDfilter = aOther.mDfilter;
    reset();
  }
  else { // nothing to do
  }
  return *this;
}
  
void gazebo::common::Pid::reset() noexcept {
  mErrPrev1 = mErrPrev2 = mPerr = mIerr = mDerr = mCmd = 0.0;
  mPfilter.reset();
  mDfilter.reset();
  mDbuffer.resize(mDbufferLength);
  std::fill(mDbuffer.begin(), mDbuffer.end(), 0.0);
  mDbufferIndex = 0u;
}

#include <sensor_msgs/Joy.h>
extern bool theZeroest;
extern sensor_msgs::Joy pidMsg;

/////////////////////////////////////////////////
double gazebo::common::Pid::update(double const aDesired, double const aActual, common::Time const aDt) noexcept {
  if (aDt == common::Time(0, 0)) {
    return 0.0;
  }
  else {
    double fTerm = mForwardGain * aDesired;
    double error = aDesired - aActual;

    mPerr = mPfilter.update(error);
    double pTerm = mPgain * mPerr;

    double prevIerr = mIerr;
    mIerr = mIerr + aDt.Double() * error;
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
      double derived = derive(error, aDt.Double());
      mDerr = mDfilter.update(derived);
if(theZeroest) {
pidMsg.axes[0] = derived;
pidMsg.axes[1] = mDerr;
gzdbg << "E " << error << "  diff " << (3.0 * error - 4.0 * mErrPrev1 + mErrPrev2) << "  mDerr " << mDerr << " ";
}
    }
    else {
if(theZeroest)
gzdbg << "dt was " << aDt.Double() << std::endl;
    }
    double dTerm = mDgain * mDerr;

    double cmd = fTerm + pTerm + iTerm + dTerm;

if(theZeroest)
gzdbg << "  F " << fTerm << "  P " << pTerm << "  I " << iTerm << "  D " << dTerm << "  C " << cmd << std::endl;

    if (mCmdMax > mCmdMin) {
      mCmd = ignition::math::clamp(cmd, mCmdMin, mCmdMax);
    }
    else { // nothing to do
    }

    if(mCmd != cmd) {
      mIerr = prevIerr;
      mCmd += iTerm - mIgain * prevIerr;
    }
    else { // nothing to do
    }
  }

  return mCmd;
}

double gazebo::common::Pid::derive(double const aValue, double const aDt) noexcept {
  double period = aDt * mDbufferLength;
  double sumSimple = (mDbuffer[mDbufferIndex] + aValue) / 2.0;
  double sumComplicated = (mDbuffer[mDbufferIndex] * (-period) + aValue * 2.0 * period) / 2.0;
  for(size_t i = 0; i < mDbufferLength; ++i) {
    if(i != mDbufferIndex) {
      sumSimple += mDbuffer[i];
      sumComplicated += mDbuffer[i] * period * (2.0 - 3.0 * static_cast<double>((mDbufferLength - i)) / mDbufferLength);
    }
    else { // nothing to do
    }
  }
  mDbuffer[mDbufferIndex] = aValue;
  mDbufferIndex = (mDbufferIndex + 1u) % mDbufferLength;
  return 2.0 / (period * period) * (2.0 / period * sumComplicated - sumSimple);
}

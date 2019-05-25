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
 * I use the polynomial regression algorithm of Chris Engelsma.
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
gazebo::common::Pid::Pid() noexcept
: mForwardGain(0.0)
, mPgain(0.0)
, mIgain(0.0)
, mDgain(0.0)
, mDpolynomialDegree(1u)
, mDbufferLength(2u)
, mImax(0.0)
, mImin(0.0)
, mCmdMax(0.0)
, mCmdMin(0.0)
, mPfilter(0.0, 0.0, 0u)
, mDfilter(0.0, 0.0, 0u) {
  reset();
}

gazebo::common::Pid::Pid(PidParameters const &aPidParams) noexcept
: mForwardGain(aPidParams.forwardGain)
, mPgain(aPidParams.pGain)
, mIgain(aPidParams.iGain)
, mDgain(aPidParams.dGain)
, mDpolynomialDegree(aPidParams.dDegree)
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
    mDpolynomialDegree = aOther.mDpolynomialDegree;
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
  mWasLastTime = false;
  mPerr = mIerr = mDerr = mCmd = 0.0;
  mPfilter.reset();
  mDfilter.reset();
  mDbufferX.resize(mDbufferLength);
  mDbufferY.resize(mDbufferLength);
  mFitX.resize(mDpolynomialDegree * 2u + 1u, 0.0);
  mDpolynomCoefficients.resize(mDpolynomialDegree + 1u, 0.0);
  std::vector<double> tmp(mDpolynomialDegree + 2u, 0.0);
  mFitB.resize(mDpolynomialDegree + 1u, tmp);
  mFitY.resize(mDpolynomialDegree + 1u, 0.0);
  for(size_t i = 0; i < mDbufferLength; ++i) {
    mDbufferX[i] = mDbufferY[i] = 0.0;
  }
  mDbufferFill = 0u;
}

#include <sensor_msgs/Joy.h>
extern bool theZeroest;
extern sensor_msgs::Joy pidMsg;

/////////////////////////////////////////////////
double gazebo::common::Pid::update(double const aDesired, double const aActual, double const aNow) noexcept {
  if (!mWasLastTime) {
    mWasLastTime = true;
    mCmd = 0.0;
  }
  else {
    double fTerm = mForwardGain * aDesired;
    double error = aDesired - aActual;
    double dt = aNow - mLastTime;
    mLastTime = aNow;

    mPerr = mPfilter.update(error);
    double pTerm = mPgain * mPerr;

    double prevIerr = mIerr;
    mIerr = mIerr + dt * error;
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

    if(dt > 0.0) {
      double derived;
      derived = derive(error, aNow);
      mDerr = mDfilter.update(derived);
if(theZeroest) {
pidMsg.axes[0] = derived;
pidMsg.axes[1] = mDerr;
gzdbg << "E " << error << "  derived " << derived << "  mDerr " << mDerr;
}
    }
    else {
if(theZeroest)
gzdbg << "dt was " << dt;
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
  mLastTime = aNow;

  return mCmd;
}

double gazebo::common::Pid::derive(double const aValue, double const aNow) noexcept {
  for(size_t i = 1; i < mDbufferLength; ++i) {
    mDbufferX[i - 1] = mDbufferX[i];
    mDbufferY[i - 1] = mDbufferY[i];
  }
  mDbufferX[mDbufferLength - 1] = aNow;
  mDbufferY[mDbufferLength - 1] = aValue;

  double derived = 0;
  if(mDbufferFill >= mDbufferLength) {
    fitPolynomial();

    for(size_t i = 1; i <= mDpolynomialDegree; ++i) {
      mDpolynomCoefficients[i - 1] = i * mDpolynomCoefficients[i];
    }
    for(size_t i = mDpolynomialDegree - 1; i > 0; --i) {
      derived = aNow * (derived + mDpolynomCoefficients[i]);
    }
    derived += mDpolynomCoefficients[0];
  }
  else {
    ++mDbufferFill;
  }
  return derived;
}

void gazebo::common::Pid::fitPolynomial() noexcept {
  size_t degreePlus1 = mDpolynomialDegree + 1u;
  size_t degreePlus2 = mDpolynomialDegree + 2u;
  size_t degreeDoublePlus1 = 2u * mDpolynomialDegree + 1u;

  // X = vector that stores values of sigma(xi^2n)
  for (size_t i = 0u; i < degreeDoublePlus1; ++i) {
    mFitX[i] = 0.0;
    for (size_t j = 0u; j < mDbufferLength; ++j) {
      mFitX[i] += pow(mDbufferX[j], i);
    }
  }

  for (size_t i = 0u; i < degreePlus1; ++i) {
    for (size_t j = 0u; j < degreePlus1; ++j) { 
      mFitB[i][j] = mFitX[i + j];
    }
  }

  // Y = vector to store values of sigma(xi^n * yi)
  for (size_t i = 0u; i < degreePlus1; ++i) {
    mFitY[i] = 0.0;
    for (size_t j = 0u; j < mDbufferLength; ++j) {
      mFitY[i] += (double)pow(mDbufferX[j], i) * mDbufferY[j];
    }
  }

  // Load values of Y as last column of B
  for (size_t i = 0u; i < degreePlus1; ++i) {
    mFitB[i][degreePlus1] = mFitY[i];
  }

  // Pivotisation of the B matrix.
  for (size_t i = 0u; i < degreePlus1; ++i) {
    for (size_t k = i + 1u; k < degreePlus1; ++k) {
      if (mFitB[i][i] < mFitB[k][i]) {
        for (size_t j = 0u; j <= degreePlus1; ++j) {
          double tmp = mFitB[i][j];
          mFitB[i][j] = mFitB[k][j];
          mFitB[k][j] = tmp;
        }
      }
      else { // nothing to do
      }
    }
  }

  // Performs the Gaussian elimination.
  // (1) Make all elements below the pivot equals to zero
  //     or eliminate the variable.
  for (size_t i = 0u; i < mDpolynomialDegree; ++i) {
    for (size_t k = i + 1u; k < degreePlus1; ++k) {
      double t = mFitB[k][i] / mFitB[i][i];
      for (size_t j = 0u; j <= degreePlus1; ++j) {
        mFitB[k][j] -= t * mFitB[i][j];         // (1)
      }
    }
  }

  // Back substitution.
  // (1) Set the variable as the rhs of last equation
  // (2) Subtract all lhs values except the target coefficient.
  // (3) Divide rhs by coefficient of variable being calculated.
  for (size_t i = mDpolynomialDegree; i < degreePlus1; --i) {
    mDpolynomCoefficients[i] = mFitB[i][degreePlus1];                   // (1)
    for (size_t j = 0; j<degreePlus1; ++j) {
      if (j != i) {
        mDpolynomCoefficients[i] -= mFitB[i][j] * mDpolynomCoefficients[j];       // (2)
      }
      else { // nothing to do
      }
    }
    mDpolynomCoefficients[i] /= mFitB[i][i];                  // (3)
  }
}

//
// Created by jhwangbo on 02.05.16.
//

#ifndef PoleOnCart_HPP
#define PoleOnCart_HPP


// system inclusion
#include <math.h>
#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <ctime>
#include <string>
#include <fstream>
#include "limits.h"

// commons
#include "enumeration.hpp"

// custom inclusion- Modify for your task
#include "rai/tasks/common/Task.hpp"
#include "math/RandomNumberGenerator.hpp"

#ifdef VIS_ON
// visualization
#include "rai/tasks/poleOnCart/visualizer/visualizer.hpp"
#endif

namespace rai {
namespace Task {

constexpr int ActionDim = 1;
constexpr int StateDim = 5;
constexpr int CommandDim = 0;

using Eigen::Vector4d;
using Eigen::Vector2d;

template<typename Dtype>
class PoleOnCart : public Task<Dtype, StateDim, ActionDim, CommandDim> {

 public:
  using TaskBase = Task<Dtype, StateDim, ActionDim, CommandDim>;
  typedef typename TaskBase::Action Action;
  typedef typename TaskBase::State State;
  typedef typename TaskBase::Command Command;
  typedef typename TaskBase::VectorXD VectorXD;
  typedef typename TaskBase::MatrixXD MatrixXD;
  typedef typename TaskBase::JacobianStateResAct Jacobian;
  typedef typename TaskBase::JacobianCostResAct JacobianCostResAct;

  PoleOnCart() {
////// set default parameters//////////////
    this->valueAtTermination_ = 1.0;
    this->discountFactor_ = 0.99;
    this->timeLimit_ = 5.0;
    this->controlUpdate_dt_ = 0.05;

    q_ << 0.0, M_PI;
    u_ << 0.0, 0.0;

/////// adding constraints////////////////////
//    State upperStateBound, lowerStateBound;
//    upperStateBound << 3, 1e50, 1e50, 1e50, 1e50;
//    lowerStateBound << -3, -1e50, -1e50, -1e50, -1e50;
//    this->setBoxConstraints(lowerStateBound, upperStateBound);
///////////////////////////////////////////////

#ifdef VIS_ON
    visualizer_ = new Visualizer;
    this->visualization_ON_ = false;
#endif
  }

  ~PoleOnCart() {
#ifdef VIS_ON
    delete visualizer_;
#endif
  }

  void takeOneStep(const Action &action_t,
                   State &state_tp1,
                   TerminationType &termType,
                   Dtype &costOUT) {
    double Sq = -std::sin(q_(1)), Cq = -std::cos(q_(1));
    Dtype limitedControl = clip(action_t(0), -10, 10);
    ddq_(0) = 1.0 / (mc_ + mp_ * Sq * Sq)
        * ((limitedControl - 0.0005 * u_(0)) + mp_ * Sq * (lp_ * u_(1) * u_(1) + g_ * Cq));
    ddq_(1) = 1.0 / (lp_ * (mc_ + mp_ * Sq * Sq))
        * (-(limitedControl - 0.0005 * u_(0)) * Cq - mp_ * lp_ * u_(1) * u_(1) * Sq * Cq - (mc_ + mp_) * g_ * Sq - 0.000002 * u_(1));
    u_ += (ddq_ - 0.3 * u_) * this->controlUpdate_dt_;
    u_(1) = clip(u_(1), -8, 8);
    q_ += u_ * this->controlUpdate_dt_;
    getState(state_tp1);
//    if (this->isViolatingBoxConstraint(state_tp1))
//      termType = TerminationType::terminalState;

    costOUT = 0.01 * ( pow(normalizeAngle(q_(1)), 2.0) + 1e-5 * action_t(0) * action_t(0) + std::abs(q_(0)));

    // visualization
#ifdef VIS_ON
    if (this->visualization_ON_) {
      visualizer_->drawWorld(q_, this->visualization_info_);
      usleep(this->controlUpdate_dt_ * 1e6 / 1.5);
    }
#endif
  }

  double normalizeAngle(double x) {
    return modeOpPython(x + M_PI, 2 * M_PI) - M_PI;
  }

  double modeOpPython(double x, double y) {
    double result = fmod(x, y);
    if (result < 0.0) {
      result += y;
    }
    return result;
  }

  double clip(double input, double lower, double upper) {
    input = (input > lower) * input + !(input > lower) * lower;
    return (input < upper) * input + !(input < upper) * upper;
  }

  bool isTerminalState(State state){
    return this->isViolatingBoxConstraint(state);
  }

  void setInitialState(const State &state) {
    LOG(FATAL)<<"initial state is random";
  }

  void setToInitialState() {
//    q_(0) = rn_.sampleUniform();
//    q_(1) = rn_.sampleUniform() + M_PI;
//    u_(0) = rn_.sampleUniform() * 2.0;
//    u_(1) = rn_.sampleUniform() * 3.0;
    q_(0) = 0.0;
    q_(1) = M_PI;
    u_(0) = 0.0;
    u_(1) = 0.0;
  }

  void setToParticularState(const State &state) {
    state4Learning_ = state;
    q_ = state4Learning_.topRows(2).template cast<double>();
    u_ = state4Learning_.bottomRows(2).template cast<double>();
  }

  void getState(State &state) {
    state(0) = q_(0);
    state(1) = std::cos(q_(1));
    state(2) = std::sin(q_(1));
    state(3) = u_(0);
    state(4) = u_(1);
  }

  void getInitialState(State &state){
    setToInitialState();
    getState(state);
  }

  bool isTerminalState(State& state){
    return this->isViolatingBoxConstraint(state);
  }

 private:

  Vector2d q_, u_; // generalized state and velocity
  Vector2d ddq_; // generalized acceleration
  State stateScale_, stateBias_;
  State state4Learning_;
  RandomNumberGenerator rn_;

  /// parameters
  double mc_ = 1.0, mp_ = 0.1, lp_ = 1.0, g_ = 9.81;

#ifdef VIS_ON
  Visualizer *visualizer_;
#endif
};

}
} /// namespaces


#endif //QPOLONCART_TEMPLATE4TASK_HPP

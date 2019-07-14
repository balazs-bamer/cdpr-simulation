//
// Created by jhwangbo on 02.05.16.
//

#ifndef PoleBalancing1D_HPP
#define PoleBalancing1D_HPP

// custom inclusion- Modify for your task
#include "rai/tasks/common/Task.hpp"

#ifdef VIS_ON
// visualization
#include "rai/tasks/poleBalancing1D/visualizer/visualizer.hpp"
#endif

namespace rai {
namespace Task{

constexpr int ActionDim = 1;
constexpr int StateDim = 1;
constexpr int CommandDim = 0;

template<typename Dtype>
class PoleBalancing1D : public Task<Dtype, StateDim, ActionDim, CommandDim> {

public:
  using TaskBase = Task<Dtype, StateDim, ActionDim, CommandDim>;
  typedef typename TaskBase::Action Action;
  typedef typename TaskBase::State State;
  typedef typename TaskBase::Command Command;
  typedef typename TaskBase::VectorXD VectorXD;
  typedef typename TaskBase::MatrixXD MatrixXD;
  typedef typename TaskBase::JacobianStateResAct MatrixJacobian;
  typedef typename TaskBase::JacobianCostResAct MatrixJacobianCostResAct;

  PoleBalancing1D() :
      q_(0), q0_(0.5) {
    //// set default parameters
    this->valueAtTermination_ = 1.5;
    this->discountFactor_ = 0.93;
    this->timeLimit_ = 15.0;
    this->controlUpdate_dt_ = 0.1;

    /////// adding constraints////////////////////
    State upperStateBound, lowerStateBound;
    upperStateBound << 1.0;
    lowerStateBound << -1.0;
    this->setBoxConstraints(lowerStateBound, upperStateBound);

#ifdef VIS_ON
    visualizer_ = new Visualizer;
    this->visualization_ON_ = false;
#endif
  }

  ~PoleBalancing1D() {
#ifdef VIS_ON
    delete visualizer_; //// this is giving an error!!!!!!
#endif
  }

  void step(const Action &action_t,
                    State &state_tp1,
                    TerminationType &termType,
                    Dtype &costOUT){
    q_ += 1.0 * action_t(0) * this->controlUpdate_dt_;
    costOUT = 0.03 * q_ * q_ + 0.01 * action_t(0) * action_t(0);
    state_tp1(0) = q_;

    if (this->isViolatingBoxConstraint(state_tp1))
      termType = TerminationType::terminalState;

    // visualization
#ifdef VIS_ON
    if (this->visualization_ON_) {
      visualizer_->drawWorld(q_, this->visualization_info_);
      usleep(this->controlUpdate_dt_ * 1e6 / 8);
    }
#endif
  }

  void init(){
    q_ = q0_;
  }

  void setInitialState(const State &in){
    q0_ = in(0);
  }

  void getInitialState(State &out){
    out(0) = q0_;
  }

  void initTo(const State& state){
    q_ = state(0);
  }

  void getState(State& state){
    state(0) = q_;
  }

  bool isTerminalState(State& state){
    return this->isViolatingBoxConstraint(state);
  }

  // Misc implementations
  void getGradientStateResAct(const State& stateIN, const Action& actionIN, MatrixJacobian& gradientOUT){
    gradientOUT(0, 0) = 3.0 * this->controlUpdate_dt_;
  };
  void getGradientCostResAct(const State& stateIN, const Action& actionIN, MatrixJacobianCostResAct& gradientOUT){
    gradientOUT(0, 0) = 0.02 * actionIN(0);
  }

private:

  double q_; // generalized state and velocity
  double q0_;

#ifdef VIS_ON
  Visualizer *visualizer_;
#endif
};

}} /// namespaces


#endif //QLEARNING_TEMPLATE4TASK_HPP

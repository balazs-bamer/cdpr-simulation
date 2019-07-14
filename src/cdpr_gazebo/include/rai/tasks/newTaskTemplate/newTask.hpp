#include <rai/tasks/common/Task.hpp>
#include <raiCommon/enumeration.hpp>

#pragma once

namespace rai {
namespace Task {

// change them according to your task
constexpr int StateDim = 3;
constexpr int ActionDim = 1;
constexpr int CommandDim = 0;

template<typename Dtype>
class newTask : public Task<Dtype, StateDim, ActionDim, CommandDim> {

 public:
  using TaskBase = Task<Dtype, StateDim, ActionDim, CommandDim>;
  using State = typename TaskBase::State;
  using Action = typename TaskBase::Action;
  using Command = typename TaskBase::Command;
  using MatrixXD = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorXD = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
  using TrueState = VectorXD; // this is the true underlying state used in simulation


  newTask() {
    // set your task parameters. for e.g.
    this->discountFactor_ = 0.99;
    this->controlUpdate_dt_ = 0.05;
    this->timeLimit_ = 25.0;
  }

  ~newTask() {};

  void step(const Action &action_t, State &state_tp1, TerminationType &termType, Dtype &cost) {
    // transition dynamics
  }

  void init() {
    // set your state to initial state
  }

  void setInitialState(const State &state) {
    // method to change initial state
  }

  void getInitialState(State &state) {
    // returns initial state
  }

  void initTo(const State &state) {
    // initialize task to a specific state. Use LOG(FATAL)<<"not implemented"; if you do not have freedom to change the state
  }

  void getState(State &state) {
    // returns the current state
  }

  bool isTerminalState(State &state) {
    // returns if the given state is terminal state
  }

 protected:

};
}
}

/// create your visualizer as static and initialize it outside.

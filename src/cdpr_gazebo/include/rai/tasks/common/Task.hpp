/*
 * EpisodeDyn.hpp
 *
 *  Created on: Mar 12, 2016
 *      Author: jhwangbo
 */



#ifndef Task_HPP_
#define Task_HPP_

// system inclusion
#include <math.h>
#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <ctime>
#include <string>
#include <fstream>
#include <memory>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/SVD>


#include "raiCommon/enumeration.hpp"
#include "raiCommon/TypeDef.hpp"



namespace rai {
namespace Task {

template<typename Dtype, int state_dimension, int action_dimension, int command_dimension>
class Task {
  //////////////// simulation_dt is the simulation time step!
  //////////////// controlUpdate_dt is the decision update time step

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<Dtype, state_dimension, 1> State;
  typedef Eigen::Matrix<Dtype, state_dimension, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, action_dimension, 1> Action;
  typedef Eigen::Matrix<Dtype, action_dimension, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> CostBatch;
  typedef Eigen::Matrix<Dtype, state_dimension, action_dimension> JacobianStateResAct;
  typedef Eigen::Matrix<Dtype, 1, action_dimension> JacobianCostResAct;
  typedef Eigen::Matrix<Dtype, command_dimension, 1> Command;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;

  ////////////// normal constructor
  Task() {};
  virtual ~Task(){};

  /// Core Methods!! Must be implemented///////////////////////////////////
  void takeOneStep(const Action &action_t,
                   State &state_tp1,
                   TerminationType &termType,
                   Dtype &costOUT) {
    step(action_t, state_tp1, termType, costOUT);
    episodeTime += controlUpdate_dt_;
    if (episodeTime > timeLimit_)
      termType = TerminationType::timeout;
  };

  // move the state to the initial state set by setInitialState()
  void setToInitialState() {
    episodeTime = 0.0;
    init();
  };

  // setting the default initial state
  virtual void setInitialState(const State &in) = 0;
  virtual void getInitialState(State &in) = 0;
  virtual void setToParticularState(const State &state) {
    episodeTime = 0.0;
    initTo(state);
  };
  virtual void getState(State &state) = 0;
  virtual bool isTerminalState(State &state) = 0;
  bool isTerminalState() {
    State state;
    getState(state);
    return isTerminalState(state);
  }

 private:
  virtual void initTo(const State &state) = 0;
  virtual void init() = 0;
  virtual void step(const Action &action_t,
                    State &state_tp1,
                    TerminationType &termType,
                    Dtype &costOUT) = 0;

  /////////////////////////////////////////////////////////////////////////
  /// Non-core methods!! might be useful for some algorithms //////////////
 public:
  virtual void getGradientStateResAct(const State &stateIN, const Action &actionIN, JacobianStateResAct &gradientOUT) {
    std::cout << " wrong task calling this method" << std::endl;
  };
  virtual void getGradientCostResAct(const State &stateIN, const Action &actionIN, JacobianCostResAct &gradientOUT) {};
  virtual void getTransition(State &state,
                             Action &action_t,
                             State &state_tp1,
                             TerminationType &termType,
                             Dtype &costOUT) {};
  virtual Command getCommand(const Command &commandIN) { return Command::Zero(); };
  virtual void setCommand(const Command &commandIN) {};
  virtual void noisifyState(StateBatch& stateBatch) {};

  //////////////////////////////////////////////////////////////////////////
  // task definition parameters --- gets
  virtual Dtype termValue() const { return valueAtTermination_; };
  virtual Dtype discountFtr() const { return discountFactor_; };
  virtual double timeLimit() const { return timeLimit_; };
  virtual int getMaximumNumberOfStepsPerEpisode() const { return timeLimit_ / controlUpdate_dt_; };
  virtual double dt() const { return controlUpdate_dt_; };

  // task definition parameter --- sets
  virtual void setValueAtTerminalState(Dtype valueIN) { valueAtTermination_ = valueIN; };
  virtual void setDiscountFactor(Dtype factorIN) { discountFactor_ = factorIN; };
  virtual void setTimeLimitPerEpisode(double timeLimit) { timeLimit_ = timeLimit; };
  virtual void setControlUpdate_dt(double timeIN) { controlUpdate_dt_ = timeIN; };

  ////////////////////// Methods related to constraints ////////////////////////////////////////////
 protected:
  /* linear constraints are defined as {Normal^T * State  > offset} */
  inline virtual void setLinearConstraint(const State &normal, Dtype offset) {
    constraintsNormal_.push_back(normal);
    constraintsOffset_.push_back(offset);
  }

  /* box constraints are defined as lower < state < upper */
  inline virtual void setBoxConstraints(const State &lower, const State &upper) {
    constraintBoxUpper_ = upper;
    constraintBoxLower_ = lower;
    isBoxConstrained_ = true;
  }

  inline virtual bool isViolatingBoxConstraint(const State &state) const {
    if (!isBoxConstrained_) false;
    for (int ID = 0; ID < constraintsNormal_.size(); ID++)
      if (constraintsNormal_[ID].transpose() * state <= constraintsOffset_[ID]) return true;
    if (isBoxConstrained_) {
      if (((constraintBoxUpper_ - state).array() < 0).any())
        return true;
      if (((state - constraintBoxLower_).array() < 0).any())
        return true;
    }
    return false;
  }

 public:
  void turnOnVisualization(std::string info) {
    visualization_info_ = info;
    visualization_ON_ = true;
  };

  bool shouldRecordVideo() {
    return this->videoRecordingEnabled_;
  }

  virtual void enableVideoRecording() {
    videoRecordingEnabled_ = true;
  }

  virtual void disableRecording() {
    videoRecordingEnabled_ = false;
  }

  virtual void startRecordingVideo(std::string dir, std::string fileName) { }

  virtual void endRecordingVideo(){ }

  void turnOffVisualization() {
    visualization_ON_ = false;
  };

  std::string visualization_info_;

 protected:
  bool visualization_ON_ = false;
  bool videoRecordingEnabled_ = false;

  std::string videoDir_;
  std::vector<State> constraintsNormal_;
  std::vector<Dtype> constraintsOffset_;
  State constraintBoxUpper_;
  State constraintBoxLower_;
  bool isBoxConstrained_ = false;
  Dtype discountFactor_;
  Dtype valueAtTermination_;
  double timeLimit_;
  double controlUpdate_dt_;
  double episodeTime = 0;

 public:
  TaskStochasticity taskStochasticity_ = TaskStochasticity::determinisitic;

};

}
} // namespaces

#endif

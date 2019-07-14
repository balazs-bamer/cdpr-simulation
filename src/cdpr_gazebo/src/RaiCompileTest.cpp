// This file here serves only as a test for RAI and Tensorflow compilation. It is the stripped-down version of poleBalwithPPO.cpp

//
// Created by joonho on 03.04.17.
//


#include <rai/RAI_core>

// Eigen
#include <eigen3/Eigen/Dense>

// task
#include "rai/tasks/poleBalancing/PoleBalancing.hpp"

// noise model
#include "rai/noiseModel/NormalDistributionNoise.hpp"

// Neural network
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"

// algorithm
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include "rai/algorithm/PPO.hpp"


#include <thread>
#include <rai/tasks/common/Task.hpp>
#include <raiCommon/enumeration.hpp>
#include <raiCommon/utils/RandomNumberGenerator.hpp>
#include <rai/function/common/Policy.hpp>
#include <rai/function/common/Qfunction.hpp>
#include <rai/RAI_core>
#include "raiCommon/TypeDef.hpp"


using namespace std;
using namespace boost;

namespace rai {
namespace Task {

constexpr int StateDim = 3;
constexpr int ActionDim = 1;
constexpr int CommandDim = 0;

template<typename Dtype>
class PoleBalancing : public Task<Dtype, StateDim, ActionDim, CommandDim> {

  /*
   * Implements a pole that is fixed at one end and can freely rotate otherwise
   *
   * When setting initialStateType = random and taskVersion = easy, it is equivalent to Pendulum-v0 from OpenAI Gym:
   * https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
   * (except for a factor of 0.01 in the reward)
   */

 public:
  using TaskBase = Task<Dtype, StateDim, ActionDim, CommandDim>;
  using State = typename TaskBase::State;
  using Action = typename TaskBase::Action;
  using Command = typename TaskBase::Command;
  using MatrixXD = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorXD = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
  using PhysicalState = VectorXD;

  enum InitialStateType {
    random,
    fixed
  };

  enum TaskVersion {
    easy,
    hard
  };
  PoleBalancing(InitialStateType initialStateType = random, TaskVersion taskVersion = hard) {
    this->discountFactor_ = 0.99;
    this->controlUpdate_dt_ = 0.05;
    this->timeLimit_ = 25.0;

    generalizedCoordinates_ << M_PI, 0;
    initialState_ << M_PI, 0;
    setVersion(taskVersion);
    initialStateType_ = fixed;
    initialStateType_ = initialStateType;

    /// visualization
    realTimeRatio = 0.8; ///visualization

    visualizeFrame.row(3).setZero();
    visualizeFrame(3, 3) = 1.0;
  };

  ~PoleBalancing() {};

  void step(const Action &action_t, State &state_tp1, TerminationType &termType, Dtype &cost) {
    double th = generalizedCoordinates_(0);
    double thdot = generalizedCoordinates_(1);
    double u = clip(action_t(0), -maxTorque_, maxTorque_);
    lastAction_ = action_t(0);
    cost = normAngle(th) * normAngle(th) + .1 * thdot * thdot + .001 * (u * u);
    cost *= 0.01;
    double newthdot = thdot + (-3.0 * gravity_ / (2.0 * lp_) * sin(th + M_PI)
        + 3. / (massPendulum_ * lp_ * lp_) * u) * this->controlUpdate_dt_;
    double newth = th + newthdot * this->controlUpdate_dt_;
    newthdot = clip(newthdot, -maxSpeed_, maxSpeed_);
    generalizedCoordinates_ << newth, newthdot;
    getState(state_tp1);
    termType = TerminationType::not_terminated;

    if (this->visualization_ON_){
      updateVisualizationFrames();
      visualizer_.drawWorld(visualizeFrame,lastAction_/maxTorque_);
      double waitTime = std::max(0.0, this->controlUpdate_dt_ / realTimeRatio - watch.measure("sim", true));
      watch.start("sim");
      usleep(waitTime * 1e6);
    }
  }

  virtual void init() {
    if (initialStateType_ == fixed) {
      generalizedCoordinates_ = initialState_;
    } else if (initialStateType_ == random) {
      generalizedCoordinates_ << rn_.sampleUniform() * M_PI, rn_.sampleUniform();
    }
  }
//
//  bool isTerminalState() { return false; }

  virtual void setInitialState(const State &state) {
    // TODO: implementation
  }

  void getInitialState(State &state) {
    init();
    getState(state);
  }

  void initTo(const State &state) {
    generalizedCoordinates_ << std::atan2(state(1), state(0)), state(2) * 2.0;
  }

  void getState(State &state) {
    Dtype theta = generalizedCoordinates_(0);
    Dtype thetadot = generalizedCoordinates_(1) / 2.0;
    state << cos(theta), sin(theta), thetadot;
  }

  virtual void setInitialStateWithGeneralizedCoordinate(const Eigen::VectorXd &in) {
    if (initialStateType_ == random) {
      LOG(WARNING)
          << "You are trying to set the initial state to a particular value, but the task is in random initial state mode.";
    }
    initialState_ = in.cast<Dtype>();
  };

  void setInitialStateType(InitialStateType initialStateType) {
    initialStateType_ = initialStateType;
  }

  void setVersion(TaskVersion taskVersion) {
    if (taskVersion == easy) {
      maxTorque_ = 2.0;
    } else {
      maxTorque_ = 0.5;
    }
    taskVersion_ = taskVersion;
  }

  void setToPendulumv0Mode() {
    setInitialStateType(random);
    setVersion(easy);
  }

  double clip(double input, double lower, double upper) {
    input = (input > lower) * input + !(input > lower) * lower;
    return (input < upper) * input + !(input < upper) * upper;
  }

  static double normAngle(double x) {
    return pythonModOp(x + M_PI, 2 * M_PI) - M_PI;
  }

  static double pythonModOp(double x, double y) {
    double result = fmod(x, y);
    if (result < 0.0) {
      result += y;
    }
    return result;
  }

 public:
  void setRealTimeFactor(double fctr) {
    realTimeRatio = fctr;
  }

  void scaleRealtimeFactor(double in) {
    realTimeRatio *= in;
  }

 private:
  virtual bool isTerminalState(State &state) { return false; }

  double lastAction_ = 0;

  Eigen::Matrix<Dtype, 2, 1> generalizedCoordinates_;
  Eigen::Matrix<Dtype, 2, 1> initialState_;
  RandomNumberGenerator<Dtype> rn_;
  InitialStateType initialStateType_;
  TaskVersion taskVersion_;

  // Task params
  Dtype maxTorque_;
  static constexpr Dtype maxSpeed_ = 8.0;
  static constexpr Dtype gravity_ = 10.0;
  static constexpr Dtype massPendulum_ = 1.0;
  static constexpr Dtype lp_ = 1.0;
};
}
}


/// learning states
using Dtype = float;

/// shortcuts
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;
using Task = rai::Task::PoleBalancing<Dtype>;
using State = Task::State;
using Action = Task::Action;
using VectorXD = Task::VectorXD;
using MatrixXD = Task::MatrixXD;
using Policy_TensorFlow = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Vfunction_TensorFlow = rai::FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
//using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Sequential<Dtype, StateDim, ActionDim>;

using Noise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

#define nThread 10

int main(int argc, char *argv[]) {

  RAI_init();
  omp_set_num_threads(nThread);

  ////////////////////////// Define task ////////////////////////////
  std::vector<Task> taskVec(nThread, Task(Task::fixed, Task::easy));
  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector;

  for (auto &task : taskVec) {
    task.setControlUpdate_dt(0.05);
    task.setDiscountFactor(0.995);
    task.setRealTimeFactor(1.5);
    task.setTimeLimitPerEpisode(25.0);
    taskVector.push_back(&task);
  }

  ////////////////////////// Define Noise Model //////////////////////
  Dtype Stdev = 1;
  NoiseCovariance covariance = NoiseCovariance::Identity() * Stdev;
  std::vector<Noise> noiseVec(nThread, Noise(covariance));
  std::vector<Noise *> noiseVector;
  for (auto &noise : noiseVec)
    noiseVector.push_back(&noise);


  ////////////////////////// Define Function approximations //////////
  Vfunction_TensorFlow Vfunction("cpu", "MLP", "relu 1e-3 3 32 32 1", 0.001);
  Policy_TensorFlow policy("cpu", "MLP", "relu 1e-3 3 32 32 1", 0.001);

  ////////////////////////// Acquisitor
  Acquisitor_ acquisitor;

  ////////////////////////// Algorithm and Hyperparameters /////////////////////////
  rai::Algorithm::PPO<Dtype, StateDim, ActionDim>
      algorithm(taskVector, &Vfunction, &policy, noiseVector, &acquisitor, 0.97, 0, 0, 1, 4, 4, true);

  policy.setLearningRateDecay(0.99,50);
  policy.setMaxGradientNorm(0.1);
  Vfunction.setLearningRateDecay(0.99,50);
  Vfunction.setMaxGradientNorm(0.1);

  algorithm.setVisualizationLevel(0);

  ////////////////////////// Choose the computation mode //////////////
  rai::Tensor<Dtype,2> state_plot({3, 2601}, "state");
  rai::Tensor<Dtype,2> action_plot({1, 2601}, "action");
  rai::Tensor<Dtype,2> value_plot({1, 2601}, "value");
  MatrixXD minimal_X_extended(1, 2601);
  MatrixXD minimal_Y_extended(1, 2601);

  MatrixXD minimal_X_sampled(1, 2601);
  MatrixXD minimal_Y_sampled(1, 2601);
  MatrixXD arrowTip(1, 2601);
  MatrixXD zeros2601(1, 5601);
  zeros2601.setZero();

  for (int i = 0; i < 51; i++) {
    for (int j = 0; j < 51; j++) {
      minimal_X_extended(i * 51 + j) = -M_PI + M_PI * i / 25.0;
      minimal_Y_extended(i * 51 + j) = -5.0 + j / 25.0 * 5.0;
      state_plot.eMat()(0, i * 51 + j) = cos(minimal_X_extended(i * 51 + j));
      state_plot.eMat()(1, i * 51 + j) = sin(minimal_X_extended(i * 51 + j));
      state_plot.eMat()(2, i * 51 + j) = minimal_Y_extended(i * 51 + j);
    }
  }

  ////////////////////////// Learning /////////////////////////////////
  constexpr int loggingInterval = 50;
  constexpr int iteration = 100;
  for (int iterationNumber = 1; iterationNumber <= iteration; iterationNumber++) {
  //  LOG(INFO) << iterationNumber << "th Iteration";
  //  LOG(INFO) << "Learning rate:"<<policy.getLearningRate();

    algorithm.runOneLoop(6000);
  }
//  policy.dumpParam(RAI_LOG_PATH + "/policy.txt");

}

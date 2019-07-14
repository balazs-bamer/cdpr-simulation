#include <thread>
#include <rai/tasks/common/Task.hpp>
#include <raiCommon/enumeration.hpp>
#include <raiCommon/utils/RandomNumberGenerator.hpp>
#include <rai/function/common/Policy.hpp>
#include <rai/function/common/Qfunction.hpp>
#include <rai/RAI_core>
#include "raiCommon/TypeDef.hpp"
#include "rai/tasks/poleBalancing/visualizer/Pole_Visualizer.hpp"
#include "raiCommon/utils/StopWatch.hpp"

#pragma once

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

  void startRecordingVideo(std::string dir, std::string fileName) {
    mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    visualizer_.getGraphics()->savingSnapshots(dir, fileName);
  }

  void endRecordingVideo() {
    visualizer_.getGraphics()->images2Video();
  }

 public:
  void setRealTimeFactor(double fctr) {
    realTimeRatio = fctr;
  }

  void scaleRealtimeFactor(double in) {
    realTimeRatio *= in;
  }

 private:

  void updateVisualizationFrames() {
    visualizeFrame.row(3).setZero();
    visualizeFrame(3, 3) = 1.0;
    Quaternion orientation;

    double Ang = generalizedCoordinates_(0);
    Eigen::Vector3d axis;
    axis << 1,0,0;
    orientation = rai::Math::MathFunc::angleAxisToQuat(Ang, axis);
    visualizeFrame.topLeftCorner(3,3) = rai::Math::MathFunc::quatToRotMat(orientation);
  }
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

  // Visualization
  StopWatch watch;
  const unsigned int windowSize_ = 500;
  const int pHeightVis_ = 150;
  const int p_WidthVis_ = 20;
  double realTimeRatio;
  static rai::Vis::Pole_Visualizer visualizer_;
  HomogeneousTransform visualizeFrame;
};
}
}
template<typename Dtype>
rai::Vis::Pole_Visualizer rai::Task::PoleBalancing<Dtype>::visualizer_;
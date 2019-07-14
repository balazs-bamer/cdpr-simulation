//
// Created by joonho on 08.05.17.
//

#ifndef RAI_SPG_GAE_HPP
#define RAI_SPG_GAE_HPP

#include <iostream>

#include "rai/tasks/common/Task.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Jacobi>
#include <Eigen/Cholesky>
#include <boost/bind.hpp>
#include <math.h>
#include "rai/RAI_core"
#include <vector>
#include <math/RAI_math.hpp>

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"
#include "rai/function/common/StochasticPolicy.hpp"
#include "VectorHelper.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>


// common
#include "enumeration.hpp"
#include "math/inverseUsingCholesky.hpp"
#include "math/ConjugateGradient.hpp"
#include "math.h"
#include "rai/RAI_core"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class SPG_gae {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, StateDim, ActionDim> JacobianStateResAct;
  typedef Eigen::Matrix<Dtype, 1, ActionDim> JacobianCostResAct;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> JacobianActResParam;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, 1> Parameter;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::StochasticPolicy<Dtype, StateDim, ActionDim>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;

  SPG_gae(std::vector<Task_ *> &tasks,
          ValueFunc_ *vfunction,
          Policy_ *policy,
          std::vector<Noise_ *> &noises,
          Acquisitor_ *acquisitor,
          Dtype lambda,
          int K,
          int numofjunctions,
          unsigned testingTrajN) :
      task_(tasks),
      vfunction_(vfunction),
      policy_(policy),
      noise_(noises),
      acquisitor_(acquisitor),
      lambda_(lambda),
      testingTrajN_(testingTrajN),
      testTraj_(testingTrajN, Trajectory_()),
      numofjunct_(numofjunctions),
      K_(K),
      stepsTaken(0),
      DataN(0),
      cg_damping(0.1),
      klD_threshold(0.01) {
    parameter_.setZero(policy_->getLPSize());
    policy_->getLP(parameter_);
    Utils::logger->addVariableToLog(2, "Nominal performance", "");
    termCost = task_[0]->termValue();
    discFactor = task_[0]->discountFtr();
    dt = task_[0]->dt();
    timeLimit = task_[0]->timeLimit();

    noiseBasePtr_.resize(task_.size());

    ///update input stdev
    stdev_o = noise_[0]->getCovariance().diagonal();
    policy_->setStdev(stdev_o);

  };

  ~SPG_gae() {};

  void runOneLoop(int numOfSteps) {
    iterNumber_++;
    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_[i] = noise_[i];

    //////////////// testing (not part of the algorithm) //////////////
    timer->disable();

    for (auto &tra : testTraj_)
      tra.clear();
    for (auto &noise : noiseBasePtr_)
      noise->initializeNoise();
    for (auto &task : task_)
      task->setToInitialState();

    StateBatch startState(StateDim, testingTrajN_);
    sampleBatchOfInitial(startState);

    if (vis_lv_ > 0) {
      task_[0]->turnOnVisualization("");
      if (task_[0]->shouldRecordVideo())
        task_[0]->startRecordingVideo(RAI_LOG_PATH + "/" + std::to_string(iterNumber_), "nominalPolicy");
    }
    Dtype averageCost = acquisitor_->acquire(task_,
                                             policy_,
                                             noiseBasePtr_,
                                             testTraj_,
                                             startState,
                                             timeLimit,
                                             false);
    if (vis_lv_ > 0) task_[0]->turnOffVisualization();
    if (task_[0]->shouldRecordVideo()) { task_[0]->endRecordingVideo(); }

    Utils::logger->appendData("Nominal performance",
                              float(acquisitor_->stepsTaken()),
                              float(averageCost));

    LOG(INFO) << "steps taken " << logger->getData("Nominal performance")->at(0).back()
              << ", average cost " << logger->getData("Nominal performance")->at(1).back();

    timer->enable();

    ////////////////////////////////Algorithm////////////////////////////
    LOG(INFO) << "Simulation";
    get_trajs(numOfSteps); /// run at least "numOfSteps" steps
    LOG(INFO) << "Vfunction update";
    VFupdate();
    LOG(INFO) << "Policy update";
    PolicyUpdater();
  }

  void set_cg_daming(Dtype cgd) { cg_damping = cgd; }
  void set_kl_thres(Dtype thres) { klD_threshold = thres; }
  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:
  void get_trajs(int numOfSteps) {
    std::vector<Trajectory_> rollouts;
    Utils::timer->startTimer("Simulation");
    numOfTra_ = std::ceil(1.1 * numOfSteps * dt / timeLimit);
    traj_.resize(numOfTra_);
    StateBatch startState(StateDim, numOfTra_);
    sampleBatchOfInitial(startState);
    for (auto &noise : noiseBasePtr_)
      noise->initializeNoise();
    for (auto &task : task_)
      task->setToInitialState();
    for (auto &tra : traj_)
      tra.clear();
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
//    Utils::timer->startTimer("Initial Trajectory Acquisition");
    Dtype cost = acquisitor_->acquire(task_,
                                      policy_,
                                      noiseBasePtr_,
                                      traj_,
                                      startState,
                                      timeLimit,
                                      true);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();

    int stepsInThisLoop = int(acquisitor_->stepsTaken() - stepsTaken);
//    Utils::timer->stopTimer("Initial Trajectory Acquisition");

    ///////////////////////////////////////VINE//////////////////////////////
    StateBatch VineStartPosition(StateDim, numofjunct_);
    StateBatch rolloutstartState(StateDim, numofjunct_ * K_);
    rollouts.resize(numofjunct_ * K_);
    rolloutstartState.setOnes();
    std::vector<std::pair<int, int> > indx;
    rai::Op::VectorHelper::sampleRandomStates(traj_, VineStartPosition, int(0.3 * timeLimit / dt), indx);

//    Utils::timer->startTimer("Rollout Trajectory Acquisition");

    for (int dataID = 0; dataID < numofjunct_; dataID++) {
      rolloutstartState.block(0, dataID * K_, StateDim, K_) =
          rolloutstartState.block(0, dataID * K_, StateDim, K_).array().colwise()
              * VineStartPosition.col(dataID).array();
    }

    for (auto &tra : rollouts)
      tra.clear();

    ///acquire K start state(With different noise)
    acquisitor_->acquire(task_, policy_, noiseBasePtr_, rollouts, rolloutstartState, dt, true);

    noise_[0]->initializeNoise();
    for (auto &noise : noiseBasePtr_)
      noise->initializeNoise();

    ///acquire K rollouts(With same noise)
    for (int trajID = 0; trajID < numofjunct_ * K_; trajID++)
      rolloutstartState.col(trajID) = rollouts[trajID].stateTraj.back();

    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    Dtype loss = acquisitor_->acquire(task_, policy_, noiseBasePtr_, rollouts, rolloutstartState, timeLimit, true);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
//    Utils::timer->stopTimer("Rollout Trajectory Acquisition");
    ///merge trajectories into one vector
    traj_.reserve(traj_.size() + rollouts.size());
    traj_.insert(traj_.end(), rollouts.begin(), rollouts.end());
    stepsTaken = int(acquisitor_->stepsTaken());
    numOfTra_ = int(traj_.size());
    DataN = 0;
    for (auto &tra : traj_) DataN += tra.size() - 1;

    stateBat_.resize(StateDim, DataN);
    actionBat_.resize(ActionDim, DataN);
    actionNoiseBat_.resize(ActionDim, DataN);
    valueBat_.resize(DataN);
    costBat_.resize(DataN);
    termValueBat_.resize(numOfTra_);
    termValueBatOld_.resize(numOfTra_);
    termStateBat_.resize(StateDim, numOfTra_);

    int colID = 0;
    for (int traID = 0; traID < traj_.size(); traID++) {
      for (int timeID = 0; timeID < traj_[traID].size() - 1; timeID++) {
        stateBat_.col(colID) = traj_[traID].stateTraj[timeID];
        actionBat_.col(colID) = traj_[traID].actionTraj[timeID];
        costBat_(colID) = traj_[traID].costTraj[timeID];
        actionNoiseBat_.col(colID++) = traj_[traID].actionNoiseTraj[timeID];
      }
      termStateBat_.col(traID) = traj_[traID].stateTraj.back();
    }
    // update terimnal value
    vfunction_->forward(termStateBat_, termValueBat_);
    for (colID = 0; colID < traj_.size(); colID++)
      if (traj_[colID].termType == TerminationType::timeout)
        traj_[colID].updateValueTrajWithNewTermValue(termValueBat_(colID), task_[0]->discountFtr());
    Utils::timer->stopTimer("Simulation");
  }

  void VFupdate() {
    ValueBatch valuePrev(DataN), valueTest(DataN);
    Dtype loss;
    vfunction_->forward(stateBat_, valuePrev);
    mixfrac = 0.1;
    Utils::timer->startTimer("Vfunction update");
    int colID = 0;
    for (auto &tra : traj_)
      for (int timeID = 0; timeID < tra.size() - 1; timeID++)
        valueBat_(colID++) = tra.valueTraj[timeID];

    valueBat_ = valueBat_ * mixfrac + valuePrev * (1 - mixfrac);

    for (int i = 0; i < 25; i++)
      loss = vfunction_->performOneSolverIter(stateBat_, valueBat_);

    Utils::timer->stopTimer("Vfunction update");
    LOG(INFO) << "value function loss : " << loss;
  }
  void PolicyUpdater() {
    Utils::timer->startTimer("policy Training");

    /// Update Advantage
    advantage_.resize(DataN);
    bellmanErr_.resize(DataN);

    int dataID = 0;
    for (auto &tra : traj_) {
      ValueBatch advTra = tra.getGAE(vfunction_, discFactor, lambda_, termCost);
      advantage_.block(0, dataID, 1, advTra.cols()) = advTra;
      bellmanErr_.block(0, dataID, 1, advTra.cols()) = tra.bellmanErr;
      dataID += advTra.cols();
    }
    rai::Math::MathFunc::normalize(advantage_);

    /// Update Policy
    Parameter policy_grad = Parameter::Zero(parameter_.rows());
    Parameter Nat_grad = Parameter::Zero(parameter_.rows());
    Parameter fullstep = Parameter::Zero(parameter_.rows());

    policy_->getStdev(stdev_o);
    policy_->getLP(parameter_);

    policy_->getpg(stateBat_, actionBat_, actionNoiseBat_, advantage_, stdev_o, policy_grad);

//    std::function<void(Eigen::Matrix<Dtype, -1, 1> &, Eigen::Matrix<Dtype, -1, 1> &)>
//        fcn = std::bind(&SPG_gae::getFVP, this, std::placeholders::_1, std::placeholders::_2);
//
//    Utils::timer->startTimer("Conjugate gradient");
//    Dtype CGerror = conjugateGradient<Dtype>(fcn, policy_grad, 100, Dtype(1e-11), Nat_grad);
//    Utils::timer->stopTimer("Conjugate gradient");
//
////    LOG(INFO) << "conjugate grad error :" << CGerror;
//    LOG(INFO) << "Grad1    :" << policy_grad.norm();
//    LOG(INFO) << "Grad2    :" << Nat_grad.norm();
//    LOG(INFO) << "Graddiff :" << (Nat_grad - policy_grad).norm()/policy_grad.norm() * 100;
//
//    LOG_IF(FATAL, isnan(Nat_grad.norm())) << "Grad is nan" << std::endl
//                                          << Nat_grad.transpose();
//
//    Dtype beta = std::sqrt(2 * klD_threshold / Nat_grad.dot(policy_grad));
    LOG(INFO) << "std dev: " << stdev_o;

    // TODO: Do something with grad

//    policy_->setLP(parameter_);
    for (int i = 0; i < 3; i++)
      policy_->trainUsingGrad(policy_grad, 3e-3);
//    policy_->trainUsingGrad(Nat_grad,1e-2);
//    policy_->trainUsingGrad(Nat_grad,beta);


    updateVar();/// save stdev & Update Noise Covariance
    Utils::timer->stopTimer("policy Training");

  }

  void getFVP(Eigen::Matrix<Dtype, -1, 1> &gradient, Eigen::Matrix<Dtype, -1, 1> &FVP) {
    policy_->getfvp(stateBat_, actionBat_, actionNoiseBat_, advantage_, stdev_o, gradient, FVP); // TODO : test

    FVP += cg_damping * gradient;
  }

  void updateVar() {
    Action temp;
    policy_->getStdev(stdev_o);
    temp = stdev_o;
    temp = temp.array().square(); //var
    policycov = temp.asDiagonal();
    for (auto &noise : noise_)
      noise->updateCovariance(policycov);
  }

  void sampleBatchOfInitial(StateBatch &initial) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_ *> task_;
  std::vector<Noise_ *> noise_;
  std::vector<Noise::Noise<Dtype, ActionDim> *> noiseBasePtr_;
  FuncApprox::ValueFunction<Dtype, StateDim> *vfunction_;
  Policy_ *policy_;
  Acquisitor_ *acquisitor_;
  Dtype lambda_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfTra_;
  int stepsTaken;
  int DataN;
  int numofjunct_;
  int K_;

  Dtype mixfrac;
  Dtype klD_threshold;
  Dtype cg_damping;

  Dtype termCost;
  Dtype discFactor;
  Dtype dt;
  double timeLimit;

  /////////////////////////// batches
  StateBatch stateBat_, termStateBat_;
  ValueBatch valueBat_, termValueBat_, termValueBatOld_, costBat_;
  ActionBatch actionBat_, actionNoiseBat_;
  ValueBatch advantage_, bellmanErr_;

  /////////////////////////// trajectories //////////////////////
  std::vector<Trajectory_> testTraj_;
  std::vector<Trajectory_> traj_;

  /////////////////////////// Policy parameter
  VectorXD parameter_;
  Action stdev_o;
  Covariance policycov;

  /////////////////////////// plotting
  int iterNumber_ = 0;

  /////////////////////////// random number generator
  RandomNumberGenerator<Dtype> rn_;

  ///////////////////////////testing
  unsigned testingTrajN_;

  /////////////////////////// visualization
  int vis_lv_ = 0;
};

}
}
#endif //RAI_SPG_GAE_HPP

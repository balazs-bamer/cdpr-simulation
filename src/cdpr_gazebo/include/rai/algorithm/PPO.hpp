//
// Created by joonho on 15.05.17.
//

#ifndef RAI_PPO_HPP
#define RAI_PPO_HPP

#include <iostream>
#include "glog/logging.h"

#include <Eigen/Core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"
#include "rai/function/common/StochasticPolicy.hpp"
#include "rai/common/VectorHelper.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>

// common
#include "raiCommon/enumeration.hpp"
#include "rai/RAI_core"
#include "common/PerformanceTester.hpp"
#include <rai/algorithm/common/LearningData.hpp>
#include "rai/tasks/common/Task.hpp"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class PPO {

 public:
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, 1> Parameter;
  typedef rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> Dataset;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::StochasticPolicy<Dtype, StateDim, ActionDim>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;

  PPO(std::vector<Task_ *> &tasks,
      ValueFunc_ *vfunction,
      Policy_ *policy,
      std::vector<Noise_ *> &noises,
      Acquisitor_ *acquisitor,
      Dtype lambda,
      int numOfBranchPerJunction,
      int numofJunctions,
      unsigned testingTrajN,
      int n_epoch = 5,
      int n_minibatch = 0,
      bool KL_adapt = true,
      Dtype clipRangeDecay = 1, //no decay
      Dtype noiseCov = 1,
      Dtype KLThres = 0.01,
      Dtype entCoeff = 0.01,
      Dtype clipCoeff = 0.2,
      Dtype KL_coeff = 1) :
      task_(tasks),
      vfunction_(vfunction),
      policy_(policy),
      noise_(noises),
      acquisitor_(acquisitor),
      lambda_(lambda),
      numOfBranchPerJunct_(numOfBranchPerJunction),
      numOfJunct_(numofJunctions),
      testingTrajN_(testingTrajN),
      KL_adapt_(KL_adapt),
      n_epoch_(n_epoch),
      n_minibatch_(n_minibatch),
      covIn(noiseCov),
      clipRangeDecay_(clipRangeDecay),
      KLThres_(KLThres),
      KLCoeff_(KL_coeff),
      clipCoeff_(clipCoeff),
      entCoeff_(entCoeff), Dataset_(true) {
    updateN = 0;
    ///Construct Dataset
    ///Additional valueTensor for Trustregion update
    Tensor<Dtype, 2> valuePred("predictedValue");
    Dataset_.append(valuePred);

    Utils::logger->addVariableToLog(2, "klD", "");
    Utils::logger->addVariableToLog(2, "Stdev", "");
    Utils::logger->addVariableToLog(2, "gradnorm", "");

    parameter_.setZero(policy_->getLPSize());
    policy_->getLP(parameter_);
    algoParams.resize(4);
    algoParams << KLCoeff_, entCoeff_, clipCoeff_, clipRangeDecay;
    policy_->setParams(algoParams);
    vfunction_->setClipRate(clipCoeff_);
    vfunction_->setClipRangeDecay(clipRangeDecay);
    ///update input stdev
    stdev_o.setOnes();
    stdev_o *= std::sqrt(covIn);
    policy_->setStdev(stdev_o);
    updatePolicyVar();

    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_.push_back(noise_[i]);
  };
  ~PPO() { delete Dataset_.miniBatch; };
  void runOneLoop(int numOfSteps) {
    iterNumber_++;
    tester_.testPerformance(task_,
                            noiseBasePtr_,
                            policy_,
                            task_[0]->timeLimit(),
                            testingTrajN_,
                            acquisitor_->stepsTaken(),
                            vis_lv_,
                            std::to_string(iterNumber_));
    acquisitor_->acquireVineTrajForNTimeSteps(task_,
                                              noiseBasePtr_,
                                              policy_,
                                              numOfSteps,
                                              numOfJunct_,
                                              numOfBranchPerJunct_,
                                              vfunction_,
                                              vis_lv_);
    Utils::timer->startTimer("data processing");
    Dataset_.appendTrajsWithAdvantage(acquisitor_->traj, task_[0], false, vfunction_, lambda_, true);
    Utils::timer->stopTimer("data processing");
    PPOUpdater();

    ///Logging
    LOG(INFO) << "KL divergence = " << KL_;
    if (KL_adapt_) LOG(INFO) << "KL coefficient = " << KLCoeff_;
    Utils::logger->appendData("Stdev", acquisitor_->stepsTaken(), policyGradNorm_);
    Utils::logger->appendData("klD", acquisitor_->stepsTaken(), KL_);
  }

  void runOneLoop(std::vector<Trajectory_>& traj){
    Utils::timer->startTimer("data processing");
    Dataset_.appendTrajsWithAdvantage(traj, task_[0], false, vfunction_, lambda_, true);
    Utils::timer->stopTimer("data processing");
    PPOUpdater();

    ///Logging
    LOG(INFO) << "KL divergence = " << KL_;
    if (KL_adapt_) LOG(INFO) << "KL coefficient = " << KLCoeff_;
    Utils::logger->appendData("Stdev", 0, policyGradNorm_);
    Utils::logger->appendData("klD", 0, KL_);
  }


  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }
 private:
  void PPOUpdater() {
    Utils::timer->startTimer("policy Training");

    /// Update Policy & Value
    Parameter policy_grad = Parameter::Zero(policy_->getLPSize());
    Dtype loss;
    /// Append predicted value to Dataset_ for trust region update
    Dataset_.extraTensor2D[0].resize(Dataset_.maxLen, Dataset_.batchNum);
    vfunction_->forward(Dataset_.states, Dataset_.extraTensor2D[0]);
    policy_->getStdev(stdev_t);
    for (int i = 0; i < n_epoch_; i++) {
      while (Dataset_.iterateBatch(n_minibatch_)) {
        Utils::timer->startTimer("Vfunction update");
        loss = vfunction_->performOneSolverIter_trustregion(Dataset_.miniBatch->states,
                                                              Dataset_.miniBatch->values,
                                                              Dataset_.miniBatch->extraTensor2D[0]);
        Utils::timer->stopTimer("Vfunction update");
        policy_->getStdev(stdev_o);
        LOG_IF(FATAL, isnan(stdev_o.norm())) << "stdev is nan!" << stdev_o.transpose();
        Utils::timer->startTimer("Gradient computation");
        if (KL_adapt_) policy_->PPOpg_kladapt(Dataset_.miniBatch, stdev_o, policy_grad);
        else policy_->PPOpg(Dataset_.miniBatch, stdev_o, policy_grad);
        Utils::timer->stopTimer("Gradient computation");
        LOG_IF(FATAL, isnan(policy_grad.norm())) << "policy_grad is nan!" << policy_grad.transpose();
        Utils::logger->appendData("gradnorm", updateN++, policy_grad.norm());
        Utils::timer->startTimer("SGD");
        policy_->trainUsingGrad(policy_grad);
        Utils::timer->stopTimer("SGD");

        KL_ = policy_->PPOgetkl(Dataset_.miniBatch, stdev_t);
        LOG_IF(FATAL, isnan(KL_)) << "KL is nan!" << KL_;
      }
      if (KL_adapt_) {
        if (KL_ > KLThres_ * 1.5) KLCoeff_ *= 2;
        if (KL_ < KLThres_ / 1.5) KLCoeff_ *= 0.5;
        algoParams[0] = KLCoeff_;
        policy_->setParams(algoParams);
      }
    }
    updatePolicyVar();/// save stdev & Update Noise Covariance
    Utils::timer->stopTimer("policy Training");
    policyGradNorm_ = policy_grad.norm();
  }

  void updatePolicyVar() {
    Action temp;
    policy_->getStdev(stdev_o);
    temp = stdev_o;
    temp = temp.array().square(); //var
    policycov = temp.asDiagonal();
    for (auto &noise : noise_)
      noise->updateCovariance(policycov);
  }

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_ *> task_;
  std::vector<Noise_ *> noise_;
  std::vector<Noise::Noise<Dtype, ActionDim> *> noiseBasePtr_;
  ValueFunc_ *vfunction_;
  Policy_ *policy_;
  Acquisitor_ *acquisitor_;
  Dtype lambda_;
  PerformanceTester<Dtype, StateDim, ActionDim> tester_;
  Dataset Dataset_;
  Dtype KL_ = 0;
  Dtype policyGradNorm_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfJunct_;
  int numOfBranchPerJunct_;
  int n_epoch_;
  int n_minibatch_;
  Dtype covIn;
  Dtype clipCoeff_;
  Dtype entCoeff_;
  Dtype KLCoeff_;
  Dtype KLThres_;
  Dtype clipRangeDecay_;
  double timeLimit;
  bool KL_adapt_;
  int updateN;

  /////////////////////////// Policy parameter
  VectorXD parameter_;
  VectorXD algoParams;
  Action stdev_o;
  Action stdev_t;
  Covariance policycov;
  /////////////////////////// plotting
  int iterNumber_ = 0;
  ///////////////////////////testing
  unsigned testingTrajN_;
  /////////////////////////// visualization
  int vis_lv_ = 0;
};

}
}
#endif //RAI_PPO_HPP

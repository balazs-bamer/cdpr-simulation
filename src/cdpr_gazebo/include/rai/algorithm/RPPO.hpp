//
// Created by joonho on 12/1/17.
//

#ifndef RAI_RPPO_HPP
#define RAI_RPPO_HPP

#include <iostream>
#include "glog/logging.h"

#include <Eigen/Core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>

// Neural network
//function approximations
#include "rai/function/tensorflow/RecurrentStochasticPolicyValue_TensorFlow.hpp"
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
class RPPO {

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
  using PolicyValue_ = FuncApprox::RecurrentStochasticPolicyValue_Tensorflow<Dtype, StateDim, ActionDim>;

  RPPO(std::vector<Task_ *> &tasks,
       PolicyValue_ *policy,
       std::vector<Noise_ *> &noises,
       Acquisitor_ *acquisitor,
       Dtype lambda,
       int numOfBranchPerJunction,
       int numofJunctions,
       unsigned testingTrajN,
       int n_epoch = 30,
       int n_minibatch = 0,
       int segLen = 0,
       int stride = 1,
       bool keepHiddenState = true,
       Dtype clipRangeDecay = 0.99,
       Dtype noiseCov = 1,
       Dtype vCoeff = 0.5,
       Dtype entCoeff = 0.01,
       Dtype clipCoeff = 0.2) :
      task_(tasks),
      policy_(policy),
      vfunction_(policy),
      noise_(noises),
      acquisitor_(acquisitor),
      lambda_(lambda),
      numOfBranchPerJunct_(numOfBranchPerJunction),
      numOfJunct_(numofJunctions),
      testingTrajN_(testingTrajN),
      n_epoch_(n_epoch),
      n_minibatch_(n_minibatch),
      segLen_(segLen),
      stride_(stride),
      stateFull_(keepHiddenState),
      covIn_(noiseCov),
      clipRangeDecay_(clipRangeDecay),
      clipCoeff_(clipCoeff),
      entCoeff_(entCoeff), vCoeff_(vCoeff), Dataset_(true, true) {

    ///Construct Dataset
    ///Additional valueTensor for Trustregion update
    Tensor<Dtype, 2> valuePred("predictedValue");
    Dataset_.append(valuePred);

    Utils::logger->addVariableToLog(2, "klD", "");
    Utils::logger->addVariableToLog(2, "Stdev", "");
    Utils::logger->addVariableToLog(2, "gradnorm", "");
    updateN = 0;

    parameter_.setZero(policy_->getLPSize());
    policy_->getLP(parameter_);
    algoParams.resize(4);
    algoParams << vCoeff_, entCoeff_, clipCoeff_, clipRangeDecay_;
    policy_->setParams(algoParams);

    timeLimit = task_[0]->timeLimit();
    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_.push_back(noise_[i]);

    ///update input stdev
    stdev_o.setOnes();
    stdev_o *= std::sqrt(covIn_);
    policy_->setStdev(stdev_o);
    updatePolicyVar();
  };

  ~RPPO() { delete Dataset_.miniBatch; };

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
    PPOUpdater();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void PPOUpdater() {
    Utils::timer->startTimer("policy Training");
    Utils::timer->startTimer("Data Processing");
    Dataset_.appendTrajsWithAdvantage(acquisitor_->traj, task_[0], true, vfunction_, lambda_,true);
    Utils::timer->stopTimer("Data Processing");

    /// Update Policy & Value
    Parameter policy_grad = Parameter::Zero(policy_->getLPSize());
    Dtype KL = 0;
    Dtype loss;

    Utils::timer->startTimer("Data Processing");
    if(segLen_!=0) Dataset_.divideSequences(segLen_, stride_, stateFull_);
    Utils::timer->stopTimer("Data Processing");

    vfunction_->forward(Dataset_.states, Dataset_.extraTensor2D[0], Dataset_.hiddenStates);

    policy_->getStdev(stdev_t);
    for (int i = 0; i < n_epoch_; i++) {
      while (Dataset_.iterateBatch(n_minibatch_)) {

        policy_->getStdev(stdev_o);
        LOG_IF(FATAL, isnan(stdev_o.norm())) << "stdev is nan!" << stdev_o.transpose();

        Utils::timer->startTimer("Gradient computation");
        policy_->PPOpg(Dataset_.miniBatch, Dataset_.miniBatch->extraTensor2D[0], stdev_o, policy_grad);
        Utils::timer->stopTimer("Gradient computation");

        LOG_IF(FATAL, isnan(policy_grad.norm())) << "policy_grad is nan!" << policy_grad.transpose();
        Utils::logger->appendData("gradnorm", updateN++, policy_grad.norm());

        Utils::timer->startTimer("SGD");
        policy_->trainUsingGrad(policy_grad);
        Utils::timer->stopTimer("SGD");

        KL = policy_->PPOgetkl(Dataset_.miniBatch, stdev_t);
        LOG_IF(FATAL, isnan(KL)) << "KL is nan!" << KL;
      }
    }
    updatePolicyVar();/// save stdev & Update Noise Covariance
    Utils::timer->stopTimer("policy Training");

///Logging
    LOG(INFO) << "KL divergence = " << KL;
    Utils::logger->appendData("Stdev", acquisitor_->stepsTaken(), policy_grad.norm());
    Utils::logger->appendData("klD", acquisitor_->stepsTaken(), KL);
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
  Policy_ *policy_;
  ValueFunc_ *vfunction_;
  Acquisitor_ *acquisitor_;
  Dtype lambda_;
  PerformanceTester<Dtype, StateDim, ActionDim> tester_;
  Dataset Dataset_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfJunct_;
  int numOfBranchPerJunct_;
  int n_epoch_;
  int n_minibatch_;
  int stride_;
  int segLen_;
  Dtype covIn_;
  Dtype clipCoeff_;
  Dtype entCoeff_;
  Dtype vCoeff_;
  Dtype clipRangeDecay_;
  double timeLimit;
  int updateN;
  bool stateFull_;

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
#endif //RAI_RPPO_HPP

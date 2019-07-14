//
// Created by jhwangbo on 12.09.16.
//

#ifndef RAI_AGSPARSE_HPP
#define RAI_AGSPARSE_HPP

#include <iostream>

#include "rai/experienceAcquisitor/AcquisitorCommonFunc.hpp"
#include "rai/tasks/common/Task.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <boost/bind.hpp>
#include <math.h>
#include "rai/RAI_core"
#include <vector>
#include <stdlib.h>
#include "rai/common/VectorHelper.hpp"

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// Acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor.hpp"

// common
#include "raiCommon/enumeration.hpp"
#include "rai/RAI_core"
#include "common/PerformanceTester.hpp"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class AG_tree {

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
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> FimInActionSapce;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;

  AG_tree(std::vector<Task_*> &task,
          FuncApprox::ValueFunction<Dtype, StateDim> *vfunction,
          FuncApprox::DeterministicPolicy<Dtype, StateDim, ActionDim> *policy,
          std::vector<Noise_*> &noise,
          Acquisitor_* acquisitor,
          int numOfInitialTra,
          int numOfBranches,
          int noiseDepth,
          double initialTrajTailTime,
          double branchTrajLength,
          Dtype learningRate = 300.0,
          int testingTrajN = 1) :
      task_(task),
      vfunction_(vfunction),
      policy_(policy),
      numOfBranches_(numOfBranches),
      noise_(noise),
      acquisitor_(acquisitor),
      numOfInitialTra_(numOfInitialTra),
      initialTraj_(numOfInitialTra),
      junctionTraj_(numOfBranches),
      testTrajectory_(testingTrajN),
      branchTraj_(noiseDepth, std::vector<Trajectory_>(numOfBranches)),
      noiseDepth_(noiseDepth),
      initialTrajTailTime_(initialTrajTailTime),
      branchTrajTime_(branchTrajLength),
      learningRate_(learningRate),
      testingTrajN_(testingTrajN),
      stateBatchPtrain_("state"),
      StateBatchVtrain_("state"),
      valueBatchVtrain_("targetValue")
  {
    parameter_.setZero(policy_->getAPSize());
    policy_->getAP(parameter_);
    jaco_.resize(ActionDim, policy_->getAPSize());
    fimCholesky_.resize(ActionDim, policy_->getAPSize());
    Utils::logger->addVariableToLog(2, "Nominal performance", "");
    noNoiseRaw_.resize(task_.size());
    noiseBasePtr_.resize(task_.size());
    noNoise_.resize(task_.size());

    for (int i=0; i<task_.size(); i++)
      noNoise_[i] = &noNoiseRaw_[i];

    for (int i=0; i<task_.size(); i++)
      noiseBasePtr_[i] = noise_[i];
  }

  ~AG_tree() {}

  void runOneLoop() {
    interationNumber_++;
    policy_->getAP(parameter_);
    Dtype dicFtr = task_[0]->discountFtr();
    Dtype timeLimit = task_[0]->timeLimit();
    Dtype dt = task_[0]->dt();
    /// clearout trajectories
    for (auto &tra : initialTraj_) tra.clear();
    for (auto &tra : junctionTraj_) tra.clear();
    for (auto &set : branchTraj_)
      for (auto &tra : set) tra.clear();
    for (auto &tra : testTrajectory_) tra.clear();

    ///////////////////////// testing (not part of the algorithm) /////////////////////////
    iterNumber_++;
    tester_.testPerformance(task_,
                            noiseBasePtr_,
                            policy_,
                            task_[0]->timeLimit(),
                            testingTrajN_,
                            acquisitor_->stepsTaken(),
                            vis_lv_,
                            std::to_string(iterNumber_));

    ///////////////////////// stage 1: simulation //////////////////
    Utils::timer->startTimer("simulation");
    std::vector<std::vector<Dtype> > valueJunction(noiseDepth_ + 1, std::vector<Dtype>(numOfBranches_));
    std::vector<State> advTuple_state;
    std::vector<Dtype> advTuple_advantage;
    std::vector<Dtype> advTuple_importance;
    std::vector<Dtype> advTuple_MD2;
    std::vector<Action> advTuple_actionNoise;
    std::vector<Action> advTuple_gradient;

    /// run initial Trajectories
    StateBatch startStateOrg(StateDim, numOfInitialTra_);
    sampleRandomBathOfInitial(startStateOrg);

    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquire(task_, policy_, noNoise_, initialTraj_, startStateOrg, timeLimit, true);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    LOG(INFO) << "initial trajectories are computed";

    /// update terminal value and value trajectory of the initial trajectories
    rai::Tensor<Dtype,2> terminalValueOrg({1, numOfInitialTra_}, "value"), terminalValueBra({1, numOfBranches_}, "value");
    StateBatch terminalStateOrg(StateDim, numOfInitialTra_), terminalStateBra(StateDim, numOfBranches_);
    rai::Tensor<Dtype,3> terminalStateOrgT({StateDim, 1, numOfInitialTra_},"state");
    rai::Tensor<Dtype,3> terminalStateBraT({StateDim, 1, numOfBranches_},"state");

    rai::Op::VectorHelper::collectTerminalStates(initialTraj_, terminalStateOrg);

    terminalStateOrgT.copyDataFrom(terminalStateOrg);
    vfunction_->forward(terminalStateOrgT, terminalValueOrg);

    for (int trajID = 0; trajID < numOfInitialTra_; trajID++)
      if (initialTraj_[trajID].termType == TerminationType::timeout)
        initialTraj_[trajID].updateValueTrajWithNewTermValue(terminalValueOrg[trajID], dicFtr);

    /// sample random starting points along initial trajectories and run episodes
    StateBatch startStateJunct(StateDim, numOfBranches_);
    std::vector<std::pair<int, int> > indx;
    rai::Op::VectorHelper::sampleRandomStates(initialTraj_, startStateJunct, int(initialTrajTailTime_ / dt), indx);

    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquire(task_, policy_, noiseBasePtr_, junctionTraj_, startStateJunct, dt * noiseDepth_, true);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    LOG(INFO) << "junctions are computed";

    for (int trajID = 0; trajID < numOfBranches_; trajID++)
      valueJunction[0][trajID] = initialTraj_[indx[trajID].first].valueTraj[indx[trajID].second];

    for (int depthID = 1; depthID < noiseDepth_ + 1; depthID++) {
      StateBatch nthState(StateDim, numOfBranches_);
      for (int i = 0; i < junctionTraj_.size(); i++)
        nthState.col(i) = junctionTraj_[i].stateTraj[depthID];
      acquisitor_->acquire(task_, policy_, noNoise_, branchTraj_[depthID-1], nthState, timeLimit, true);
      rai::Op::VectorHelper::collectTerminalStates(branchTraj_[depthID-1], terminalStateBra);
      terminalStateBraT.copyDataFrom(terminalStateBra);

      vfunction_->forward(terminalStateBraT, terminalValueBra);
      for (int trajID = 0; trajID < numOfBranches_; trajID++) {
        branchTraj_[depthID - 1][trajID].updateValueTrajWithNewTermValue(0.0, dicFtr);
        valueJunction[depthID][trajID] = branchTraj_[depthID - 1][trajID].valueTraj[0];
        advTuple_state.push_back(junctionTraj_[trajID].stateTraj[depthID - 1]);
        advTuple_actionNoise.push_back(junctionTraj_[trajID].actionNoiseTraj[depthID - 1]);
        advTuple_advantage.push_back(valueJunction[depthID][trajID] * dicFtr
                                         + junctionTraj_[trajID].costTraj[depthID - 1]
                                         - valueJunction[depthID - 1][trajID]);
        advTuple_gradient.push_back(advTuple_actionNoise.back() / (advTuple_actionNoise.back().norm())
                                        * advTuple_advantage.back());
      }
    }
    LOG(INFO) << "branch trajectories are computed";

    rai::Math::MathFunc::normalize(advTuple_advantage);
    Utils::timer->stopTimer("simulation");

    ///////////////////////// stage 2: vfunction train //////////////////
    LOG(INFO) << "value function training";
    Utils::timer->startTimer("vfunction Train");
    StateBatchVtrain_.resize(StateDim,1, numOfBranches_ + advTuple_gradient.size());
    valueBatchVtrain_.resize(1, numOfBranches_ + advTuple_gradient.size());
    StateBatchVtrain_.setZero();
    valueBatchVtrain_.setZero();
    int batIdx = 0;

    for (int trajID = 0; trajID < numOfBranches_; trajID++)
      for (int depthID = 0; depthID < noiseDepth_ + 1 ; depthID++) {
        StateBatchVtrain_.batch(batIdx) = junctionTraj_[trajID].stateTraj[depthID];
        valueBatchVtrain_[batIdx++] = valueJunction[depthID][trajID];
      }

    for (int i = 0; i < 300; i++) {
      Dtype loss = vfunction_->performOneSolverIter(StateBatchVtrain_, valueBatchVtrain_);
      LOG_IF(INFO, i % 50 == 0) << "value function learning loss: " << loss;
      if(loss < 0.0001) break;
    }
    Utils::timer->stopTimer("vfunction Train");

    ///////////////////////// stage 3: Policy train //////////////////
    Utils::timer->startTimer("policy Training");
    Dtype terminationCost = task_[0]->termValue();
    Dtype discountFactor = task_[0]->discountFtr();

    int dataLength = advTuple_gradient.size();
    int dataUse = dataLength;
    VectorXD paramUpdate = VectorXD::Zero(policy_->getLPSize());
    VectorXD newParam = VectorXD::Zero(policy_->getLPSize());

    /// forward policy in a batch for speed
    costWRTAction_.resize(ActionDim, dataUse);

    stateBatchPtrain_.resize(StateDim, 1, dataUse);
    actionBatchPtrain_.resize(ActionDim, 1, dataUse);
    for (int i = 0; i < dataUse; i++)
      stateBatchPtrain_.batch(i) = advTuple_state[i];
    policy_->forward(stateBatchPtrain_, actionBatchPtrain_);

    cholInv(noise_[0]->getCovariance(), covInv_);
    for(auto &actionNoise: advTuple_actionNoise) {
      advTuple_MD2.push_back((actionNoise.transpose() * (covInv_ * actionNoise)).sum());
      advTuple_importance.push_back(exp(-0.5 * advTuple_MD2.back()));
    }

    //// for plotting
    stateAdvantage_.resize(StateDim, advTuple_gradient.size());
    gradAdvantage_.resize(ActionDim, advTuple_gradient.size());

    for (int i = 0; i < advTuple_gradient.size(); i++) {
      stateAdvantage_.col(i) = advTuple_state[i];
      gradAdvantage_.col(i) = -advTuple_gradient[i];
    }

    for (int dataID = 0; dataID < dataUse; dataID++) {
      State state = stateBatchPtrain_.batch(dataID);
      JacobianCostResAct jacobianQwrtAction = -advTuple_gradient[dataID];

      /// take negative for reducing cost
      costWRTAction_.col(dataID) = jacobianQwrtAction.transpose();
      Utils::timer->startTimer("JacobianOutputWRT param");
      policy_->getJacobianAction_WRT_LP(state, jaco_);
      Utils::timer->stopTimer("JacobianOutputWRT param");
      VectorXD jacobianQwrtParam = jacobianQwrtAction * jaco_;
      Covariance noise_cov = noise_[0]->getCovariance();
      fimInActionSpace_ = noise_cov.inverse();

      Utils::timer->startTimer("Chole and SVD");
      Eigen::LLT<FimInActionSapce> chole(fimInActionSpace_); // compute the Cholesky decomposition of A
      fimInActionSpaceCholesky_ = chole.matrixL();
      fimCholesky_ = fimInActionSpaceCholesky_.transpose() * jaco_;
      Eigen::JacobiSVD<MatrixXD> svd(fimCholesky_, Eigen::ComputeThinU | Eigen::ComputeThinV);
      MatrixXD singluarValues = svd.singularValues();
      MatrixXD vMatrix = svd.matrixV();
      FimInActionSapce
          signularValueInverseSquaredMatrix = singluarValues.cwiseInverse().array().square().matrix().asDiagonal();
      Utils::timer->stopTimer("Chole and SVD");
      VectorXD naturalGradientDirection(policy_->getLPSize());
      naturalGradientDirection =
          vMatrix * (signularValueInverseSquaredMatrix * (vMatrix.transpose() * jacobianQwrtParam));

//      paramUpdate += learningRate_ * naturalGradientDirection / dataUse;

      Dtype learningRate = 2300.0;
      Dtype beta = sqrt(Dtype(2) * 7000.0 / naturalGradientDirection.dot(jacobianQwrtParam));
      learningRate = std::min(learningRate, beta);
      paramUpdate += learningRate * naturalGradientDirection / dataUse;

    }

    newParam = parameter_ + paramUpdate;
    policy_->setAP(newParam);
    parameter_ = newParam;
    Utils::timer->stopTimer("policy Training");
  }

  void sampleRandomBathOfInitial(StateBatch& initial) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  ActionBatch &getGradDir() { return gradAdvantage_; }
  StateBatch &getStateBatch() { return stateAdvantage_; }
  StateBatch &getValueTrainStateBatch() { return StateBatchVtrain_; }
  ValueBatch &getValueTrainValueBatch() { return valueBatchVtrain_; }
  std::vector<std::vector<Trajectory_> > getBranchTraj() { return branchTraj_; }
  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_*> task_;
  FuncApprox::ValueFunction<Dtype, StateDim> *vfunction_;
  FuncApprox::DeterministicPolicy<Dtype, StateDim, ActionDim> *policy_;
  Acquisitor_* acquisitor_;
  std::vector<Noise_*> noise_;
  std::vector<Noise::Noise<Dtype, ActionDim>* > noNoise_;
  std::vector<Noise::NoNoise<Dtype, ActionDim> > noNoiseRaw_;
  std::vector<Noise::Noise<Dtype, ActionDim>* > noiseBasePtr_;
  Dtype learningRate_;
  int iterNumber_ = 0;
  PerformanceTester<Dtype, StateDim, ActionDim> tester_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfInitialTra_ = 1;
  int numOfBranches_;
  int noiseDepth_ = 1;
  double initialTrajTailTime_, branchTrajTime_;

  /////////////////////////// trajectories //////////////////////
  std::vector<Trajectory_> initialTraj_, junctionTraj_;
  std::vector<std::vector<Trajectory_> > branchTraj_;
  std::vector<Trajectory_> testTrajectory_;

  /////////////////////////// FIM related variables
  FimInActionSapce fimInActionSpace_, fimInActionSpaceCholesky_;
  JacobianActResParam jaco_, fimCholesky_;
  Dtype klD_threshold = 0.1;
  Covariance covInv_;
  int testingTrajN_;

  /////////////////////////// Policy parameter
  VectorXD parameter_;

  /////////////////////////// plotting
  int interationNumber_ = 0;
  rai::Tensor<Dtype,2> costWRTAction_;
  rai::Tensor<Dtype,3> stateBatchPtrain_;
  rai::Tensor<Dtype,3> actionBatchPtrain_;
  rai::Tensor<Dtype,2> stateAdvantage_;
  rai::Tensor<Dtype,2> gradAdvantage_;

  /////////////////////////// qfunction training
  rai::Tensor<Dtype,3> StateBatchVtrain_;
  rai::Tensor<Dtype,2> valueBatchVtrain_;

  /////////////////////////// random number generator
  RandomNumberGenerator<Dtype> rn_;

  /////////////////////////// visualization
  int vis_lv_ = 0;

};

}
}

#endif //RAI_AGSPARSE_HPP

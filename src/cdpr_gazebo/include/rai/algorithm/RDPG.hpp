//
// Created by joonho on 11/21/17.
//

#ifndef RAI_RDPG_HPP
#define RAI_RDPG_HPP

#include <iostream>
#include "glog/logging.h"

#include "rai/tasks/common/Task.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/NoNoise.hpp>
#include "rai/RAI_core"
#include <vector>

#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"
#include "rai/function/common/StochasticPolicy.hpp"
#include "rai/common/VectorHelper.hpp"

// memory
#include "rai/memory/Trajectory.hpp"
#include "rai/memory/ReplayMemoryHistory.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <rai/algorithm/common/LearningData.hpp>
#include <rai/function/tensorflow/RecurrentQfunction_TensorFlow.hpp>
#include <rai/function/tensorflow/RecurrentDeterministicPolicy_Tensorflow.hpp>

// common
#include "raiCommon/enumeration.hpp"
#include "common/PerformanceTester.hpp"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class RDPG {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, 1> Parameter;
  typedef rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> Dataset;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;

  using Qfunction_ = FuncApprox::RecurrentQfunction_TensorFlow<Dtype, StateDim, ActionDim>;
  using Policy_ = FuncApprox::RecurrentDeterministicPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_Sequential<Dtype, StateDim, ActionDim>;
  using ReplayMemory_ = rai::Memory::ReplayMemoryHistory<Dtype, StateDim, ActionDim>;

  RDPG(std::vector<Task_ *> &task,
       Qfunction_ *qfunction,
       Qfunction_ *qfunction_target,
       Policy_ *policy,
       Policy_ *policy_target,
       std::vector<Noise_ *> &noise,
       Acquisitor_ *acquisitor,
       ReplayMemory_ *memory,
       unsigned testingTrajN,
       unsigned n_epoch,
       unsigned n_newEpisodesPerIter,
       unsigned batchSize,
       int segLen = 0,
       int stride = 1,
       Dtype tau = 1e-3):
      qfunction_(qfunction),
      qfunction_target_(qfunction_target),
      policy_(policy),
      policy_target_(policy_target),
      noise_(noise),
      acquisitor_(acquisitor),
      memory(memory),
      n_epoch_(n_epoch),
      n_newEpisodesPerIter_(n_newEpisodesPerIter),
      segLen_(segLen),
      stride_(stride),
      batSize_(batchSize),
      tau_(tau),
      testingTrajN_(testingTrajN),
      task_(task),
      Dataset_(false,true){

    Utils::logger->addVariableToLog(2, "gradnorm", "");
    Utils::logger->addVariableToLog(2, "Qloss", "");

    ///Construct Dataset
    timeLimit = task_[0]->timeLimit();

    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_.push_back(noise_[i]);

    policy_->copyAPFrom(policy_target_);
    qfunction_->copyAPFrom(qfunction_target_);
    Dataset_.useValue = true;
  };

  ~RDPG(){};

  void initiallyFillTheMemory() {
    LOG(INFO) << "FillingMemory" ;
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquireNEpisodes(task_, noiseBasePtr_, policy_, memory->getCapacity());
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();

    timer->startTimer("SavingHistory");
    memory->SaveHistory(acquisitor_->traj);
    timer->stopTimer("SavingHistory");
    LOG(INFO) << "Done" ;
  }

  void learnForNepisodes(int numOfEpisodes) {
    iterNumber_++;

    //////////////// testing (not part of the algorithm) ////////////////////
    Utils::timer->disable();
    tester_.testPerformance(task_,
                            noise_,
                            policy_,
                            task_[0]->timeLimit(),
                            testingTrajN_,
                            acquisitor_->stepsTaken(),
                            vis_lv_,
                            std::to_string(iterNumber_));

    /// reset all for learning
    for (auto &task : task_)
      task->setToInitialState();
    for (auto &noise : noise_)
      noise->initializeNoise();

    Utils::timer->enable();
    /////////////////////////////////////////////////////////////////////////
    if(numOfEpisodes > n_newEpisodesPerIter_) numOfEpisodes = n_newEpisodesPerIter_;

    for (unsigned i = 0; i < numOfEpisodes/n_newEpisodesPerIter_; i++)
      learnForOneCycle();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void learnForOneCycle() {
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquireNEpisodes(task_, noiseBasePtr_, policy_, n_newEpisodesPerIter_);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();

    timer->startTimer("SavingHistory");
    memory->SaveHistory(acquisitor_->traj);
    timer->stopTimer("SavingHistory");

    Utils::timer->startTimer("Qfunction and Policy update");
    for (unsigned i = 0; i <  n_epoch_; i++)
      updateQfunctionAndPolicy();
    Utils::timer->stopTimer("Qfunction and Policy update");
  }

  void updateQfunctionAndPolicy() {
    Dtype termValue = task_[0]->termValue();
    Dtype disFtr = task_[0]->discountFtr();

    timer->startTimer("SamplingHistory");
    memory->sampleRandomHistory(Dataset_, batSize_);
    timer->stopTimer("SamplingHistory");
    Utils::timer->startTimer("Data Processing");
    ///Target
    Tensor<Dtype, 2> value_;
    Tensor<Dtype, 3> action_target({ActionDim, Dataset_.maxLen, batSize_},"sampledAction");

    value_.resize(Dataset_.maxLen, batSize_);
    policy_target_->forward(Dataset_.states, action_target);
    qfunction_target_->forward(Dataset_.states, action_target, value_);
    for (unsigned batchID = 0; batchID < batSize_; batchID++) {
      if (TerminationType(Dataset_.termtypes[batchID]) == TerminationType::terminalState)
        value_.eMat()(Dataset_.lengths[batchID] - 1, batchID) = termValue; ///value for last state
    }

    for (unsigned batchID = 0; batchID < batSize_; batchID++){
      for(unsigned timeID = 0; timeID< Dataset_.lengths[batchID] - 1 ; timeID++)
        Dataset_.values.eMat()(timeID,batchID) = Dataset_.costs.eMat()(timeID,batchID)  + disFtr * value_.eMat()(timeID+1,batchID) ;
      Dataset_.lengths[batchID] -=1;
      /// Ignore the last sample
      /// (i.e. we have no info about the transition after the last state (s_n, a_n, r_n) -> (s_{n+1}))
    }

    if(segLen_!=0) Dataset_.divideSequences(segLen_, stride_, true);
    Utils::timer->stopTimer("Data Processing");
//
//    std::cout << "??" << std::endl;
//    std::cout << Dataset_.values.col(0).transpose()<< std::endl;
//    std::cout << Dataset_.values.col(1).transpose()<< std::endl;
//
//    std::cout << Dataset_.values.col(Dataset_.batchNum-1).transpose()<< std::endl;
//    std::cout << Dataset_.values.rows()<< std::endl;
//    std::cout << Dataset_.lengths[0]<< std::endl;

    Dtype Qloss;
    Utils::timer->startTimer("Qfunction update");
    Qloss = qfunction_->performOneSolverIter(&Dataset_, Dataset_.values);
    Utils::timer->stopTimer("Qfunction update");

    Utils::logger->appendData("Qloss", qfunction_->getGlobalStep(), Qloss);
    Dtype gradnorm;
    Utils::timer->startTimer("Policy update");
    gradnorm = policy_->backwardUsingCritic(qfunction_, &Dataset_);
    Utils::timer->stopTimer("Policy update");
    Utils::logger->appendData("gradnorm", policy_->getGlobalStep(), gradnorm);

    Utils::timer->startTimer("Target update");
    qfunction_target_->interpolateAPWith(qfunction_, tau_);
    policy_target_->interpolateAPWith(policy_, tau_);
    Utils::timer->stopTimer("Target update");
  }

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_ *> task_;
  std::vector<Noise_ *> noise_;
  std::vector<Noise::Noise<Dtype, ActionDim> *> noiseBasePtr_;
  Qfunction_ *qfunction_, *qfunction_target_;

  Policy_ *policy_, *policy_target_;
  Acquisitor_ *acquisitor_;
  ReplayMemory_ *memory;
  PerformanceTester<Dtype, StateDim, ActionDim> tester_;
  Dataset Dataset_;

  /////////////////////////// Algorithmic parameter ///////////////////
  double timeLimit;
  Dtype tau_;
  unsigned batSize_;
  unsigned n_epoch_;
  unsigned n_newEpisodesPerIter_;
  int stride_;
  int segLen_;

  /////////////////////////// plotting
  int iterNumber_ = 0;

  ///////////////////////////testing
  unsigned testingTrajN_;

  /////////////////////////// visualization
  int vis_lv_ = 0;
};

}
}
#endif //RAI_RDPG_HPP
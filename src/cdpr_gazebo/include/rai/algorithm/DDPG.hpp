//
// Created by jhwangbo on 12.09.16.
//

#ifndef RAI_TRUSTREGIONPOLICYOPTIMIZATION_HPP
#define RAI_TRUSTREGIONPOLICYOPTIMIZATION_HPP

#include <iostream>


#include "rai/tasks/common/Task.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include "rai/memory/Trajectory.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/OrnsteinUhlenbeckNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Jacobi>
#include <Eigen/Cholesky>
#include <boost/bind.hpp>
#include <rai/RAI_core>


// acquisitor
#include "rai/experienceAcquisitor/ExperienceTupleAcquisitor.hpp"
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <rai/algorithm/common/PerformanceTester.hpp>

// Neural network
//function approximations
#include "rai/function/common/DeterministicPolicy.hpp"
#include "rai/function/common/Qfunction.hpp"

// common
#include "raiCommon/enumeration.hpp"
#include "rai/RAI_core"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class DDPG {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ReplayMemory_ = rai::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Qfunction_ = FuncApprox::Qfunction<Dtype, StateDim, ActionDim>;
  using Policy_ = FuncApprox::DeterministicPolicy<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Trajectory = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::ExperienceTupleAcquisitor<Dtype, StateDim, ActionDim>;

  DDPG(std::vector<Task_ *> &task,
       Qfunction_ *qfunction,
       Qfunction_ *qfunction_target,
       Policy_ *policy,
       Policy_ *policy_target,
       std::vector<Noise_ *> &noise,
       Acquisitor_ *acquisitor,
       ReplayMemory_ *memory,
       unsigned n_epoch,
       unsigned n_newSamplePerIter,
       unsigned batchSize,
       unsigned testingTrajN,  // how many trajectories to use to test the policy
       Dtype tau = 1e-3) :
      qfunction_(qfunction),
      qfunction_target_(qfunction_target),
      policy_(policy),
      policy_target_(policy_target),
      noise_(noise),
      acquisitor_(acquisitor),
      memorySARS_(memory),
      batSize_(batchSize),
      n_epoch_(n_epoch),
      n_newSamplePerIter_(n_newSamplePerIter),
      tau_(tau),
      testingTrajN_(testingTrajN),
      task_(task) {

    for (auto &task : task_)
      task->setToInitialState();

    policy_->copyAPFrom(policy_target_);
    qfunction_->copyAPFrom(qfunction_target_);
  };

  ~DDPG() {};

  void initiallyFillTheMemory() {
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    Utils::timer->startTimer("Simulation");
    acquisitor_->acquire(task_, policy_, noise_, memorySARS_, batSize_ * 40);
    Utils::timer->stopTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
  }

  void learnForNSteps(int numOfSteps) {
    iterNumber_++;

    //////////////// testing (not part of the algorithm) ////////////////////
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
    /////////////////////////////////////////////////////////////////////////
    for (unsigned i = 0; i < numOfSteps / n_newSamplePerIter_; i++)
        learnForOneCycle();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void learnForOneCycle() {
    Utils::timer->startTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquire(task_, policy_, noise_, memorySARS_, n_newSamplePerIter_);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    Utils::timer->stopTimer("Simulation");
    Utils::timer->startTimer("Qfunction and Policy update");
    for (unsigned i = 0; i <  n_epoch_; i++)
      updateQfunctionAndPolicy();
    Utils::timer->stopTimer("Qfunction and Policy update");
  }

  void updateQfunctionAndPolicy() {
    ///RAI convention : dim[-1] = batchNum, dim[-2] = sequence length.
    rai::Tensor<Dtype,3> state_t({StateDim, 1, batSize_}, "state");
    rai::Tensor<Dtype,3> state_tp1({StateDim, 1, batSize_}, "state");
    rai::Tensor<Dtype,3> action_t({ActionDim, 1, batSize_}, "sampledAction");
    rai::Tensor<Dtype,3> action_tp1({ActionDim, 1, batSize_}, "sampledAction");
    rai::Tensor<Dtype,2> value_t({1, batSize_}, "targetQValue");
    rai::Tensor<Dtype,2> value_tp1({1, batSize_}, "targetQValue");
    rai::Tensor<Dtype,2> cost_t({1, batSize_}, "costs");
    rai::Tensor<Dtype,1> termType({batSize_}, "termtypes");

    Dtype termValue = task_[0]->termValue();
    Dtype disFtr = task_[0]->discountFtr();

    memorySARS_->sampleRandomBatch(state_t, action_t, cost_t, state_tp1, termType);

    ///// DDPG
    Utils::timer->startTimer("Qfunction update");
    policy_target_->forward(state_tp1, action_tp1);
    qfunction_target_->forward(state_tp1, action_tp1, value_tp1);
    for (unsigned tupleID = 0; tupleID < batSize_; tupleID++)
      if (TerminationType(termType[tupleID]) == TerminationType::terminalState)
        value_tp1[tupleID] = termValue;

    for (unsigned tupleID = 0; tupleID < batSize_; tupleID++)
      value_t[tupleID] = cost_t[tupleID] + disFtr * value_tp1[tupleID];
    qfunction_->performOneSolverIter(state_t, action_t, value_t);
    Utils::timer->stopTimer("Qfunction update");

    Utils::timer->startTimer("Policy update");
    policy_->backwardUsingCritic(qfunction_, state_t);
    Utils::timer->stopTimer("Policy update");

    Utils::timer->startTimer("Target update");
    qfunction_target_->interpolateAPWith(qfunction_, tau_);
    policy_target_->interpolateAPWith(policy_, tau_);
    Utils::timer->stopTimer("Target update");
  }

  /////////////////////////// Core ///////////////////////////////
  std::vector<Task_ *> task_;
  Qfunction_ *qfunction_, *qfunction_target_;
  Policy_ *policy_, *policy_target_;
  Acquisitor_ *acquisitor_;
  ReplayMemory_ *memorySARS_;
  std::vector<Noise_ *> noise_;
  PerformanceTester<Dtype, StateDim, ActionDim> tester_;
  unsigned n_epoch_;
  unsigned n_newSamplePerIter_;
  unsigned batSize_;
  Dtype tau_;

  /////////////////////////// visualization
  int vis_lv_ = 0;

  /////////////////////////// testing
  unsigned testingTrajN_;
  int iterNumber_ = 0;
};

}
}

#endif //RAI_TRUSTREGIONPOLICYOPTIMIZATION_HPP

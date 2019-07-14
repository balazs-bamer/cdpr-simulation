//
// Created by jhwangbo on 13.04.17.
//

#ifndef RAI_BELLMANTUPLES_HPP
#define RAI_BELLMANTUPLES_HPP

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "enumeration.hpp"
#include "Trajectory.hpp"
#include "math/RandomNumberGenerator.hpp"


namespace rai {
namespace Memory {

template<typename Dtype, int stateDim, int actionDim>
class BellmanTupleSet {

  typedef Eigen::Matrix<Dtype, stateDim, -1> StateBatch;
  typedef Eigen::Matrix<Dtype, actionDim, -1> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, -1> CostBatch;
  typedef Trajectory<Dtype, stateDim, actionDim> Trajectory_;
  typedef FuncApprox::ValueFunction<Dtype, stateDim> Vfunction_;

  class BellmanTuple {
   public:
    BellmanTuple(unsigned trajId,
                 unsigned startTimeId,
                 unsigned endTimeId,
                 Dtype discCosts,
                 Dtype cumulDiscFctr,
                 bool terminated) :
        trajId_(trajId),
        startTimeId_(startTimeId),
        endTimeId_(endTimeId),
        discCosts_(discCosts),
        cumulDiscFctr_(cumulDiscFctr),
        terminated_(terminated) {}
    unsigned trajId_, startTimeId_, endTimeId_;
    Dtype discCosts_;
    Dtype cumulDiscFctr_;
    bool terminated_;
  };

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void sampleRandBellTuples(StateBatch &state_t_batch,
                            StateBatch &state_tk_batch,
                            ActionBatch &action_t_batch,
                            CostBatch &discCost_batch,
                            CostBatch &dictFctr_batch,
                            std::vector<bool> &terminalState,
                            std::vector<unsigned> &idx) {
    unsigned batchSize = state_t_batch.cols();
    unsigned size = tuples_.size();
    LOG_IF(FATAL, tuples_.size() < state_t_batch.cols() * 1.2) <<
                                                               "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    idx = rn_.getNrandomSubsetIdx(tuples_.size(), state_t_batch.cols());

    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      unsigned j = idx[i];
      state_t_batch.col(i) = traj_[tuples_[j].trajId_].stateTraj[tuples_[j].startTimeId_];
      state_tk_batch.col(i) = traj_[tuples_[j].trajId_].stateTraj[tuples_[j].endTimeId_];
      action_t_batch.col(i) = traj_[tuples_[j].trajId_].actionTraj[tuples_[j].startTimeId_];
      discCost_batch[i] = tuples_[j].discCosts_;
      dictFctr_batch[i] = tuples_[j].cumulDiscFctr_;
      terminalState[i] = tuples_[j].terminated_;
    }
  }

  void sampleRandStates(StateBatch &state_t_batch) {

    LOG_IF(FATAL, tuples_.size() < state_t_batch.cols() * 1.2) <<
                                                               "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    std::vector<unsigned> idx = rn_.getNrandomSubsetIdx(tuples_.size(), state_t_batch.cols());
    ///// saving memory to the batch
    for (unsigned i = 0; i < state_t_batch.cols(); i++)
      state_t_batch.col(i) = traj_[tuples_[idx[i]].trajId_].stateTraj[tuples_[idx[i]].startTimeId_];
  }

  /// nu=1 means only the transition tuples
  void appendTraj(std::vector<Trajectory_> &traj, unsigned nu, Dtype discFtr) {
    unsigned traId = traj_.size();
    for (auto &tra: traj) {
      traj_.push_back(tra);
      for (int i = 0; i < tra.size() - 2; i++) {
        for (int j = 1; j < nu + 1 && i + j < tra.size() - 1; j++) {
          Dtype cumDiscF = pow(discFtr, j);
          Dtype cumCost = Dtype(0);
          for (int k = j - 1; k > -1; k--)
            cumCost = tra.costTraj[i + k] + cumCost * discFtr;

          bool terminated = i + j == tra.size() - 1 && tra.termType == TerminationType::terminalState;
          tuples_.push_back(BellmanTuple(traId, i, i + j, cumCost, cumDiscF, terminated));
        }
      }
      traId++;
    }
  }

  unsigned tupleSize() {
    return tuples_.size();
  }

  void popTuples(std::vector<unsigned> &idx) {
    std::vector<BellmanTuple> tuples_copy = tuples_;
    tuples_.clear();
    bool keep[tuples_copy.size()];

    for (int i = 0; i < tuples_copy.size(); i++)
      keep[i] = true;

    for (int i = 0; i < idx.size(); i++)
      keep[idx[i]] = false;

    for (int i = 0; i < tuples_copy.size(); i++)
      if (keep[i]) tuples_.push_back(tuples_copy[i]);
  }

 private:
  std::vector<Trajectory_> traj_;
  std::vector<BellmanTuple> tuples_;
  RandomNumberGenerator <Dtype> rn_;

};

}
}
#endif //RAI_BELLMANTUPLES_HPP

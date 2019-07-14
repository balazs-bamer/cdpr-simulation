/*
 * ReplayMemory.hpp
 *
 *  Created on: Mar 28, 2016
 *      Author: jemin
 */

#ifndef ReplayMemorySARS_HPP_
#define ReplayMemorySARS_HPP_

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mutex>
#include <algorithm>
#include "raiCommon/utils/RandomNumberGenerator.hpp"
#include "glog/logging.h"
#include "raiCommon/enumeration.hpp"

namespace rai {
namespace Memory {

template<typename Dtype, int stateDimension, int actionDimension>
class ReplayMemorySARS {
  typedef Eigen::Matrix<Dtype, stateDimension, 1> State;
  typedef Eigen::Matrix<Dtype, actionDimension, 1> Action;

  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

 public:

  ReplayMemorySARS(unsigned capacity) :
      size_(0), memoryIdx_(0) {
    state_t_ = new Tensor3D({stateDimension, 1, capacity}, "state");
    state_tp1_ = new Tensor3D({stateDimension, 1, capacity}, "state");
    action_t_ = new Tensor3D({actionDimension, 1, capacity}, "action");
    cost_ = new Tensor2D({1, capacity}, "costs");
    terminationFlag_ = new Tensor1D({capacity}, "termtypes");
    capacity_ = capacity;
  }

  ~ReplayMemorySARS() {
    delete state_t_;
    delete state_tp1_;
    delete action_t_;
    delete cost_;
    delete terminationFlag_;
  }

  inline void saveAnExperienceTuple(State &state_t,
                                    Action &action_t,
                                    Dtype cost,
                                    State &state_tp1,
                                    TerminationType termType) {
    std::lock_guard<std::mutex> lockModel(memoryMutex_);
    state_t_->batch(memoryIdx_) = state_t;
    state_tp1_->batch(memoryIdx_) = state_tp1;
    action_t_->batch(memoryIdx_) = action_t;
    (*cost_)[memoryIdx_] = cost;
    (*terminationFlag_)[memoryIdx_] = Dtype(termType);
    memoryIdx_ = (memoryIdx_ + 1) % capacity_;
    size_++;
    size_ = std::min(size_, capacity_);
  }

  inline void sampleRandomBatch(Tensor3D &state_t_batch,
                                Tensor3D &action_t_batch,
                                Tensor2D &cost_batch,
                                Tensor3D &state_tp1_batch,
                                Tensor1D &terminationFlag_tp1_batch) {
    int batchSize = state_t_batch.batches();
    LOG_IF(FATAL, size_ < batchSize * 1.2) <<
                                           "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (unsigned i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, size_ - 1);
      for (unsigned j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }
    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      state_t_batch.batch(i) = state_t_->batch(memoryIdx[i]);
      state_tp1_batch.batch(i) = state_tp1_->batch(memoryIdx[i]);
      action_t_batch.batch(i) = action_t_->batch(memoryIdx[i]);
      cost_batch[i] = cost_->at(memoryIdx[i]);
      terminationFlag_tp1_batch[i] = terminationFlag_->at(memoryIdx[i]);
    }
  }

  inline void sampleRandomBatch(ReplayMemorySARS &batchMemory) {
    unsigned batchSize = batchMemory.getSize();
    LOG_IF(FATAL, size_ < batchSize) <<
                                     "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (unsigned i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, size_ - 1);
      for (unsigned j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      State state = state_t_->batch(memoryIdx[i]);
      State state_tp1 = state_tp1_->batch(memoryIdx[i]);
      Action action = action_t_->batch(memoryIdx[i]);
      Dtype cost = cost_->at(memoryIdx[i]);
      TerminationType termination = TerminationType(terminationFlag_->at(memoryIdx[i]));
      batchMemory.saveAnExperienceTuple(state, action, cost, state_tp1, termination);
    }
  }
  inline void sampleRandomStateBatch(Tensor2D &state_t_batch) {
    unsigned batchSize = state_t_batch.cols();
    LOG_IF(FATAL, size_ < batchSize) <<
                                     "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (unsigned i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, size_ - 1);
      for (unsigned j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      State state = state_t_->col(memoryIdx[i]);
      state_t_batch.col(i) = state;
    }
  }

  inline void clearTheMemoryAndSetNewBatchSize(int newMemorySize) {
    std::lock_guard<std::mutex> lockModel(memoryMutex_);
    delete state_t_;
    delete state_tp1_;
    delete action_t_;
    delete cost_;
    delete terminationFlag_;

    state_t_ = new Tensor3D({stateDimension, 1, newMemorySize}, "state");
    state_tp1_ = new Tensor3D({stateDimension, 1, newMemorySize}, "state");
    action_t_ = new Tensor3D({actionDimension, 1, newMemorySize}, "sampledAction");
    cost_ = new Tensor2D({1, newMemorySize}, "costs");
    terminationFlag_ = new Tensor1D({newMemorySize}, "termtypes");

    size_ = 0;
    memoryIdx_ = 0;
    capacity_ = newMemorySize;
  }

  inline void saveAnExperienceTupleWithSparcification_DiagonalMetric(State &state_t,
                                                                     Action &action_t,
                                                                     Dtype cost,
                                                                     State &state_tp1,
                                                                     TerminationType termType,
                                                                     State &stateMetricInverse,
                                                                     Action &actionMetricInverse,
                                                                     Dtype threshold) {
    bool saved = true;
    for (unsigned memoryID = 0; memoryID < size_; memoryID++) {
      auto diff_state = state_t_->batch(memoryID) - state_t;
      auto diff_action = action_t_->batch(memoryID) - action_t;
      auto dist = sqrt(diff_state.cwiseProduct(diff_state).dot(stateMetricInverse) +
          diff_action.cwiseProduct(diff_action).dot(actionMetricInverse));

      if (dist < threshold) {
        saved = false;
        break;
      }
    }

    if (saved)
      saveAnExperienceTuple(state_t, action_t, cost, state_tp1, termType);
  }

  Dtype getDist2ClosestSample(State &state_t,
                              Action &action_t,
                              State &stateMetricInverse,
                              Action &actionMetricInverse) {
    Dtype dist, closest_dist = 1e99;
    for (unsigned memoryID = 0; memoryID < size_; memoryID++) {
      dist = sqrt((state_t_->batch(memoryID) - state_t).squaredNorm()
                      + (action_t_->batch(memoryID) - action_t).squaredNorm());
      if (dist < closest_dist)
        closest_dist = dist;
    }
    return closest_dist;
  }

  inline Tensor3D *getState_t() {
    return state_t_;
  }

  inline Tensor3D *getState_tp1() {
    return state_tp1_;
  }

  inline Tensor3D *getAction_t() {
    return action_t_;
  }

  inline Tensor2D *getCost_() {
    return cost_;
  }

  inline Tensor1D *getTeminationFlag() {
    return terminationFlag_;
  }

  unsigned getCapacity() {
    return capacity_;
  }

  unsigned getSize() {
    return size_;
  }

  void printOutMemory() {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------Replay memory printout" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "state_t_" << std::endl << state_t_->batchBlock(0, size_) << std::endl;
    std::cout << "action_t_" << std::endl << action_t_->batchBlock(0, size_) << std::endl;
    std::cout << "cost_" << std::endl << cost_->eMat().leftCols(size_) << std::endl;
    std::cout << "terminationFlag_" << std::endl << terminationFlag_->block(0, size_) << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
  }

 private:
  Tensor3D *state_t_;
  Tensor3D *state_tp1_;
  Tensor3D *action_t_;
  Tensor2D *cost_;
  Tensor1D *terminationFlag_;

  unsigned size_;
  unsigned memoryIdx_;
  unsigned capacity_;

 private:
  static std::mutex memoryMutex_;
  RandomNumberGenerator <Dtype> rn_;
};
}
}

template<typename Dtype, int stateDimension, int actionDimension>
std::mutex rai::Memory::ReplayMemorySARS<Dtype, stateDimension, actionDimension>::memoryMutex_;

#endif /* ReplayMemorySARS_HPP_ */

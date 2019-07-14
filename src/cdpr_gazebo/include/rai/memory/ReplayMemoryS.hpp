/*
 * ReplayMemoryS.hpp
 *
 *  Created on: Mar 28, 2016
 *      Author: jemin
 */

#ifndef ReplayMemoryS_HPP_
#define ReplayMemoryS_HPP_

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

template<typename Dtype, int stateDimension>
class ReplayMemoryS {

  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> MatrixXD;
  typedef Eigen::Matrix<Dtype, stateDimension, 1> State;

  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

 public:

  ReplayMemoryS(int memorySize) :
      numberOfStoredTuples_(0), memoryIdx_(0) {
    state_t_ = new Tensor3D({stateDimension, 1, memorySize}, "state");
    memorySize_ = memorySize;
  }

  ~ReplayMemoryS() {
    delete state_t_;
  }

  void saveState(State &state_t) {
    state_t_->batch(memoryIdx_) = state_t;
    memoryIdx_ = (memoryIdx_ + 1) % memorySize_;
    numberOfStoredTuples_++;
    numberOfStoredTuples_ = std::min(numberOfStoredTuples_, memorySize_);
  }

  void sampleRandomBatch(Tensor3D &state_t_batch) {
    int batchSize = state_t_batch.batches();
    LOG_IF(FATAL, numberOfStoredTuples_ < batchSize * 1.2) <<
                                                           "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned int memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (int i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, numberOfStoredTuples_ - 1);
      for (int j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (int i = 0; i < batchSize; i++) {
      state_t_batch.batch(i) = state_t_->batch(memoryIdx[i]);
    }
  }

  void sampleRandomBatch(ReplayMemoryS &batchMemory) {

    int batchSize = batchMemory.getMemorySize();
    LOG_IF(FATAL, numberOfStoredTuples_ < batchSize) <<
                                                     "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned int memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (int i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, numberOfStoredTuples_ - 1);
      for (int j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (int i = 0; i < batchSize; i++) {
      State state = state_t_->batch(memoryIdx[i]);
      batchMemory.saveState(state);
    }
  }

  void clearTheMemoryAndSetNewBatchSize(int newMemorySize) {
    delete state_t_;
    state_t_ = new Tensor3D({stateDimension, 1, newMemorySize}, "state");
    numberOfStoredTuples_ = 0;
    memoryIdx_ = 0;
    memorySize_ = newMemorySize;
  }

  void saveStateWithSparcification_DiagonalMetric(State &state_t,
                                                  State stateMetricInverse,
                                                  Dtype threshold) {
    bool saved = true;
    for (int memoryID = 0; memoryID < numberOfStoredTuples_; memoryID++) {
      auto diff_state = state_t_->batch(memoryID) - state_t;
      auto dist = sqrt(diff_state.cwiseProduct(diff_state).dot(stateMetricInverse));

      if (dist < threshold) {
        saved = false;
        break;
      }
    }

    if (saved)
      saveState(state_t);
  }

  inline Tensor3D *getState_t() {
    return state_t_;
  }

  int getMemorySize() {
    return memorySize_;
  }

  int getNumberOfStates() {
    return numberOfStoredTuples_;
  }

 private:
  Tensor3D *state_t_;
  int numberOfStoredTuples_;
  int memoryIdx_;
  int memorySize_;

 private:
  static std::mutex memoryMutex_;
  RandomNumberGenerator<Dtype> rn_;
};
}
}

template<typename Dtype, int stateDimension>
std::mutex rai::Memory::ReplayMemoryS<Dtype, stateDimension>::memoryMutex_;

#endif /* ReplayMemorySARS_HPP_ */

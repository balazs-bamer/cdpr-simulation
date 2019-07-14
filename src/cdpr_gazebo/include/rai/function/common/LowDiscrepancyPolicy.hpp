//
// Created by jhwangbo on 15.01.17.
//

#ifndef RAI_LOWDISCREPANCYPOLICY_HPP
#define RAI_LOWDISCREPANCYPOLICY_HPP
#include "Policy.hpp"

#include "rai/noiseModel/Noise.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include <algorithm>
#include <flann/flann.hpp>
#include <raiCommon/utils/rai_timer/RAI_timer_ToInclude.hpp>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class LowDiscrepancyPolicy : public Policy<Dtype, stateDim, actionDim> {

 public:

  using Policy_ = Policy<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::Noise<Dtype, actionDim>;
  using ReplayMemory_ = Memory::ReplayMemorySARS<Dtype, stateDim, actionDim>;
  using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
  using StateAction = Eigen::Matrix<Dtype, stateDim + actionDim, 1>;
  using StateActionBatch = Eigen::Matrix<Dtype, stateDim + actionDim, -1>;

  typedef typename Policy_::State State;
  typedef typename Policy_::StateBatch StateBatch;
  typedef typename Policy_::Action Action;
  typedef typename Policy_::ActionBatch ActionBatch;
  typedef typename Policy_::Gradient Gradient;
  typedef typename Policy_::Jacobian Jacobian;

  LowDiscrepancyPolicy(Policy_ *basePolicy,
                       Noise_ *noise,
                       ReplayMemory_ *memory,
                       State &stateMetricInverse,
                       Action &actionMetricInverse,
                       Dtype threshold,
                       int buildTreeAt = 1000)
      :
      basePolicy_(basePolicy), noise_(noise), memory_(memory),
      stateMetInv_(stateMetricInverse), actionMetInv_(actionMetricInverse),
      threshold_(threshold), buildTreeAt_(buildTreeAt), treeBuilt_(false) {}

  virtual ~LowDiscrepancyPolicy() {
    delete kdTree;
  }

  void forward(State &input, Action &output) {
    if (memory_->getSize() > buildTreeAt_) {
      forwardUsingTree(input, output);
    } else {
      forwardUsingBruteforce(input, output);
    }
  }

  void forward(StateBatch &input, ActionBatch &output) {
    if (memory_->getSize() > buildTreeAt_) {
      forwardUsingTree(input, output);
    } else {
      forwardUsingBruteforce(input, output);
    }
  }

 private:

  void forwardUsingTree(State &input, Action &output) {
    if (!treeBuilt_) buildTree();
    StateAction stateAction;
    Dtype dist;
    int ind;
    flann::Matrix<Dtype> distMat(&dist, 1, 1);
    flann::Matrix<int> indices(&ind, 1, 1);
    stateAction.topRows(stateDim) = input;
    Action noiselessOutput, noisyOutput;
    basePolicy_->forward(input, noiselessOutput);
    Dtype maximumDist = Dtype(0);
    flann::Matrix<Dtype> queryMat(stateAction.data(), 1, stateDim + actionDim);
//    output = noisyOutput;

    for (int i = 0; i < int(sqrt(actionDim)) * 5; i++) {
      noisyOutput = noiselessOutput + noise_->sampleNoise();
      stateAction.bottomRows(actionDim) = noisyOutput;
//      Utils::timer->startTimer("checking distance");
      kdTree->knnSearch(queryMat, indices, distMat, 1, flann::SearchParams(128));
//      dist = (memory_->getState_t().col(ind) - input).squaredNorm() +
//          (memory_->getAction_t().col(ind) - noisyOutput).squaredNorm();
//      dist = memory_->getDist2ClosestSample(input,
//                                            noisyOutput,
//                                            stateMetInv_,
//                                            actionMetInv_);
//      Utils::timer->stopTimer("checking distance");
      if (distMat.ptr()[0] > threshold_) {
        output = noisyOutput;
        break;
      }
      if (distMat.ptr()[0] > maximumDist) {
        output = noisyOutput;
        maximumDist = dist;
      }
    }
    stateAction.bottomRows(actionDim) = output;
    stateActionMem_.col(storedTupleN_++) = stateAction;
    flann::Matrix<Dtype> addData(stateActionMem_.data()+ (stateDim + actionDim) * (storedTupleN_-1), 1, stateDim + actionDim);
    kdTree->addPoints(addData);
  }

  void forwardUsingTree(StateBatch &input, ActionBatch &output) {
    if (!treeBuilt_) buildTree();
    int size = input.cols();
    StateAction stateAction;
    Dtype dist;
    int ind;
    flann::Matrix<Dtype> queryMat(stateAction.data(), 1, stateDim + actionDim);
    flann::Matrix<Dtype> distMat(&dist, 1, 1);
    flann::Matrix<int> indices(&ind, 1, 1);

    ActionBatch noiselessOutput;
    Action noisyOutput;
    State state;
    Utils::timer->startTimer("network forward");
    basePolicy_->forward(input, noiselessOutput);
    Utils::timer->stopTimer("network forward");
    for (int idx = 0; idx < input.cols(); idx++) {
      stateAction.topRows(stateDim) = input.col(idx);
      Dtype maximumDist = Dtype(0);
      std::cout << "noiselessOutput.col(idx) " << noiselessOutput.col(idx).transpose() << std::endl;
      for (int i = 0; i < int(sqrt(actionDim)) * 5; i++) {
        noisyOutput = noiselessOutput.col(idx) + noise_->sampleNoise();
        stateAction.bottomRows(actionDim) = noisyOutput;
        kdTree->knnSearch(queryMat, indices, distMat, 1, flann::SearchParams(-1));
        std::cout << "dist " << distMat.ptr()[0] << std::endl;

        if (dist > threshold_) {
          output.col(idx) = noisyOutput;
          break;
        }
        if (dist > maximumDist) {
          output.col(idx) = noisyOutput;
          maximumDist = dist;
        }
      }
    }

    StateActionBatch dataToTree(stateDim + actionDim, size);
    dataToTree.topRows(stateDim) = input;
    dataToTree.bottomRows(actionDim) = output;
    flann::Matrix<Dtype> moreData(dataToTree.data(), size, stateDim + actionDim);
    kdTree->addPoints(moreData);
  }

  void forwardUsingBruteforce(State &input, Action &output) {
    Action noiselessOutput, noisyOutput;
    basePolicy_->forward(input, noiselessOutput);
    Dtype maximumDist = Dtype(0);
    for (int i = 0; i < int(sqrt(actionDim)) * 5; i++) {
      noisyOutput = noiselessOutput + noise_->sampleNoise();
      Dtype dist = memory_->getDist2ClosestSample(input,
                                                  noisyOutput,
                                                  stateMetInv_,
                                                  actionMetInv_);
      if (dist > threshold_) {
        output = noisyOutput;
        break;
      }
      if (dist > maximumDist) {
        output = noisyOutput;
        maximumDist = dist;
      }
    }
  }

  void forwardUsingBruteforce(StateBatch &input, ActionBatch &output) {
    ActionBatch noiselessOutput;
    Action noisyOutput;
    State state;
    Utils::timer->startTimer("network forward");
    basePolicy_->forward(input, noiselessOutput);
    Utils::timer->stopTimer("network forward");
    for (int idx = 0; idx < input.cols(); idx++) {
      state = input.col(idx);
      Dtype maximumDist = Dtype(0);
      for (int i = 0; i < int(sqrt(actionDim)) * 5; i++) {
        noisyOutput = noiselessOutput.col(idx) + noise_->sampleNoise();
        Utils::timer->startTimer("checking distance");
        Dtype dist = memory_->getDist2ClosestSample(state,
                                                    noisyOutput,
                                                    stateMetInv_,
                                                    actionMetInv_);
        Utils::timer->stopTimer("checking distance");
        if (dist > threshold_) {
          output.col(idx) = noisyOutput;
          break;
        }
        if (dist > maximumDist) {
          output.col(idx) = noisyOutput;
          maximumDist = dist;
        }
      }
    }
  }

  void buildTree() {
    stateActionMem_.resize(stateDim + actionDim, memory_->getCapacity());
    stateActionMem_.topLeftCorner(stateDim, memory_->getSize()) = memory_->getState_t();
    stateActionMem_.bottomLeftCorner(actionDim, memory_->getSize()) = memory_->getAction_t();
    flann::Matrix<Dtype> dataset(stateActionMem_.data(), memory_->getSize(), stateDim + actionDim);
    kdTree = new flann::Index<flann::L2<Dtype> >(dataset, flann::KDTreeIndexParams(1));
    LOG(INFO) << "building Tree index for replay memory of size "<<kdTree->size();
    storedTupleN_ = kdTree->size();
    kdTree->buildIndex();
    treeBuilt_ = true;
  }

  Policy_ *basePolicy_;
  Noise_ *noise_;
  ReplayMemory_ *memory_;
  State stateMetInv_;
  Action actionMetInv_;
  Dtype threshold_;
  int buildTreeAt_;
  bool treeBuilt_ = false;
  flann::Index<flann::L2<Dtype> > *kdTree;
  MatrixXD stateActionMem_;
  int storedTupleN_ = 0;
};

}
} // namespaces

#endif //RAI_LOWDISCREPANCYPOLICY_HPP

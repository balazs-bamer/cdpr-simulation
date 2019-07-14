//
// Created by jhwangbo on 08/08/17.
// This class's old data is deleted when you acquire new data
//

#ifndef RAI_LearningData_HPP
#define RAI_LearningData_HPP

#include <rai/memory/Trajectory.hpp>
#include <rai/RAI_core>
#include <rai/common/VectorHelper.hpp>
#include "rai/tasks/common/Task.hpp"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class LearningData {
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;

 public:
  LearningData()
      : maxLen(0),
        batchNum(0),
        batchID(0),
        dataN(0),
        miniBatch(nullptr),
        extraTensor1D(0),
        extraTensor2D(0),
        extraTensor3D(0),
        hDim(-1), isRecurrent(false) {
    states = "state";
    actions = "sampledAction";
    actionNoises = "actionNoise";
//    hiddenStates = "h_init";

    costs = "costs";
    values = "targetValue";
    advantages = "advantage";
//    stdevs = "stdevs";

    lengths = "length";
    termtypes = "termtypes";
  };
  LearningData(bool useminibatch, bool recurrent = false)
      : maxLen(0),
        batchNum(0),
        batchID(0),
        dataN(0),
        miniBatch(nullptr),
        extraTensor1D(0),
        extraTensor2D(0),
        extraTensor3D(0),
        hDim(-1), isRecurrent(recurrent) {
    if(useminibatch) miniBatch = new LearningData<Dtype, StateDim, ActionDim>;
    if(isRecurrent && useminibatch) miniBatch->isRecurrent = true;

    states = "state";
    actions = "sampledAction";
    actionNoises = "actionNoise";
//    hiddenStates = "h_init";

    costs = "costs";
    values = "targetValue";
    advantages = "advantage";
//    stdevs = "stdevs";

    lengths = "length";
    termtypes = "termtypes";
  };


  ///method for additional data
  void append(Tensor1D &newData) {
    if (newData.size() == -1) newData.resize(0); //rai Tensor is not initialized
    extraTensor1D.push_back(newData);
    if (miniBatch) miniBatch->extraTensor1D.push_back(newData);
  }
  void append(Tensor2D &newData) {
    if (newData.size() == -1) newData.resize(0, 0); //rai Tensor is not initialized
    extraTensor2D.push_back(newData);
    if (miniBatch) miniBatch->extraTensor2D.push_back(newData);
  }
  void append(Tensor3D &newData) {
    if (newData.size() == -1) newData.resize(0, 0, 0); //rai Tensor is not initialized
    extraTensor3D.push_back(newData);
    if (miniBatch) miniBatch->extraTensor3D.push_back(newData);
  }
  void appendTrajs(std::vector<Trajectory_> &traj, Task_ *task, bool isRecurrent_ = false, ValueFunc_ *vf = nullptr) {
    LOG_IF(FATAL, traj.size() == 0) << "No Data to save. call acquire~~() first";
    Trajectory_ test;
    test.hiddenStateTraj;
    test.discountFct_;
    dataN = 0;
    maxLen = 0;
    if (vf) {
      if (!useValue) {
        useValue = true;
        if (miniBatch) miniBatch->useValue = true;
      }
    }
    if (isRecurrent_) {
      LOG_IF(FATAL, traj[0].hiddenStateTraj.size() == 0) << "hiddenStateTraj is empty";
      hDim = traj[0].hiddenStateTraj[0].rows();
      if (miniBatch) miniBatch->hDim = hDim;
      isRecurrent = true;
      if (miniBatch) miniBatch->isRecurrent = true;
    }

    for (auto &tra : traj) dataN += tra.size() - 1;
    if (isRecurrent_) {
      /////Zero padding tensor//////////////////
      for (auto &tra : traj)
        if (maxLen < tra.stateTraj.size() - 1) maxLen = int(tra.stateTraj.size()) - 1;

      batchNum = int(traj.size());
      resize(maxLen, batchNum);
      setZero();

      for (int i = 0; i < batchNum; i++) {
        states.partiallyFillBatch(i, traj[i].stateTraj, 1);
        actions.partiallyFillBatch(i, traj[i].actionTraj, 1);
        actionNoises.partiallyFillBatch(i, traj[i].actionNoiseTraj, 1);
        for (int timeID = 0; timeID < traj[i].size() - 1; timeID++) {
          costs.eMat()(timeID, i) = traj[i].costTraj[timeID];
        }
        hiddenStates.partiallyFillBatch(i, traj[i].hiddenStateTraj, 1);
        lengths[i] = traj[i].stateTraj.size() - 1;
        termtypes[i] = Dtype(traj[i].termType);
      }

    } else {
      resize(1, dataN);
      setZero();

      int pos = 0;
      for (int traID = 0; traID < traj.size(); traID++) {
        for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++) {
          states.batch(pos) = traj[traID].stateTraj[timeID];
          actions.batch(pos) = traj[traID].actionTraj[timeID];
          actionNoises.batch(pos) = traj[traID].actionNoiseTraj[timeID];
          costs[pos] = traj[traID].costTraj[timeID];
          pos++;
        }
      }
    }

    // update terimnal value
    if (vf) {
      Tensor2D termValues;
      Tensor3D termStates("state");

      termValues.resize(1, traj.size());
      termStates.resize(StateDim, 1, traj.size());

      ///update value traj
      for (int traID = 0; traID < traj.size(); traID++)
        termStates.batch(traID) = traj[traID].stateTraj.back();
      vf->forward(termStates, termValues);
      for (int traID = 0; traID < traj.size(); traID++)
        if (traj[traID].termType == TerminationType::timeout) {
          traj[traID].updateValueTrajWithNewTermValue(termValues[traID], task->discountFtr());
        }

      int colID = 0;
      if (isRecurrent_) {
        for (int traID = 0; traID < traj.size(); traID++)
          for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++)
            values.eMat()(timeID, traID) = traj[traID].valueTraj[timeID];
      } else {
        for (int traID = 0; traID < traj.size(); traID++)
          for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++)
            values.eMat()(0, colID++) = traj[traID].valueTraj[timeID];
      }

    }
  }
  void appendTrajsWithAdvantage(std::vector<Trajectory_> &traj, Task_ *task, bool isRecurrent_,
                                ValueFunc_ *vf, Dtype lambda = 0.97, bool normalizeAdv = true) {
    LOG_IF(FATAL, !vf) << "value function is necessary for this method";
    appendTrajs(traj, task, isRecurrent_, vf);
    computeAdvantage(traj, task, vf, lambda, normalizeAdv);
  }

  bool iterateBatch(int minibatchNum = 0) {
    // fill minibatch with data

    minibatchNum = std::max(minibatchNum, 1);
    int cur_batch_size = batchNum / minibatchNum;
    if (cur_batch_size >= batchNum - batchID) {
      cur_batch_size = batchNum - batchID;
    }
//    LOG(INFO) << "batchsize:" << cur_batch_size;

    if (batchID >= batchNum) {
      batchID = 0;
      return false;
    }

//    Utils::timer->startTimer("fillminiBatch");
    fillminiBatch(cur_batch_size);
//    Utils::timer->stopTimer("fillminiBatch");

    batchID += cur_batch_size;
    return true;
  }

  void divideSequences(int segLen, int stride = 1, bool keepHiddenState = true) {
    ///Implementation of truncated BPTT
    ///k1: stride
    ///k2: len

    LOG_IF(FATAL, segLen > maxLen) << "segLen should be smaller than this->maxLen";

    ///copyData
    Tensor3D states_t(states);
    Tensor3D actions_t(actions);
    Tensor3D actionNoises_t(actionNoises);
    Tensor3D hiddenStates_t(hiddenStates);

    Tensor2D values_t(values);
    Tensor2D advantages_t(advantages);
    Tensor1D lengths_t(lengths);
    Tensor1D termtypes_t(termtypes);

    int batchNum_t = batchNum;

    //vectors for additional data
    std::vector<rai::Tensor<Dtype, 3>> extraTensor3D_t;
    std::vector<rai::Tensor<Dtype, 2>> extraTensor2D_t;
    std::vector<rai::Tensor<Dtype, 1>> extraTensor1D_t;
    for (int i = 0; i < extraTensor1D.size(); i++)
      extraTensor1D_t.push_back(extraTensor1D[i]);
    for (int i = 0; i < extraTensor2D.size(); i++)
      extraTensor2D_t.push_back(extraTensor2D[i]);
    for (int i = 0; i < extraTensor3D.size(); i++)
      extraTensor3D_t.push_back(extraTensor3D[i]);

    int expandedBatchNum = 0;
    int segNum[batchNum];
    for (int i = 0; i < batchNum; i++) {
      segNum[i] = std::ceil((lengths[i] - segLen) / stride) + 1;
      expandedBatchNum += segNum[i];
    }

    resize(segLen, expandedBatchNum);

    hiddenStates.setZero();
    int segID = 0;
    int position = 0;
    for (int i = 0; i < batchNum_t; i++) {

      for (int j = 0; j < segNum[i]; j++) {
        position = stride * j;
        if (position > lengths_t[i] - segLen)
          position = std::max(0, (int) lengths_t[i] - segLen); //to make last segment full
        states.batch(segID) = states_t.batch(i).block(0, position, StateDim, segLen);
        actions.batch(segID) = actions_t.batch(i).block(0, position, ActionDim, segLen);
        actionNoises.batch(segID) = actionNoises_t.batch(i).block(0, position, ActionDim, segLen);

        if (useValue) values.col(segID) = values_t.block(position, i, segLen, 1);
        if (useAdvantage) advantages.col(segID) = advantages_t.block(position, i, segLen, 1);

        lengths[segID] = std::min(segLen, (int) lengths_t[i]);
        termtypes[segID] = termtypes_t[i];

        for (int k = 0; k < extraTensor1D.size(); k++)
          extraTensor1D[k][segID] = extraTensor1D_t[k][i];
        for (int k = 0; k < extraTensor2D.size(); k++)
          extraTensor2D[k].col(segID) = extraTensor2D_t[k].block(position, i, segLen, 1);
        for (int k = 0; k < extraTensor3D.size(); k++)
          extraTensor3D[k].batch(segID) =
              extraTensor3D_t[k].batch(i).block(0, position, extraTensor3D_t[k].dim(0), segLen);

        if (keepHiddenState) {
          /// We only use the first column.
          hiddenStates.batch(segID).col(0) = hiddenStates_t.batch(i).col(position);
//          hiddenStates.batch(segID) =  hiddenStates_t.batch(segID).block(0,stride*j,hDim,segLen);
        }
        segID++;
      }
    }

  }

  void computeAdvantage(std::vector<Trajectory_> &traj,
                        Task_ *task,
                        ValueFunc_ *vf,
                        Dtype lambda,
                        bool normalize = true) {
    int batchID = 0;
    int dataID = 0;
    Dtype sum = 0;
    Eigen::Matrix<Dtype, 1, -1> temp(1, dataN);

    useAdvantage = true;
    if (miniBatch) miniBatch->useAdvantage = true;
    advantages.resize(maxLen, batchNum);

    Utils::timer->startTimer("GAE");
    for (auto &tra : traj) {
      ///compute advantage for each trajectory
      Tensor1D advs = tra.getGAE(vf, task->discountFtr(), lambda, task->termValue());
      std::memcpy(temp.data()+dataID,advs.data(),sizeof(Dtype) * advs.size());
      dataID += advs.size();
    }

    if (normalize) rai::Math::MathFunc::normalize(temp);

    if (isRecurrent) {
      dataID = 0;
      advantages.setZero();
      for (auto &tra : traj) {
        advantages.block(0, batchID, tra.size() - 1, 1) = temp.block(0, dataID, 1, tra.size() - 1).transpose();
        dataID += tra.size() - 1;
        batchID++;
      }
    } else {
      advantages.copyDataFrom(temp);
    }
    Utils::timer->stopTimer("GAE");
  }
  void resize(int hdim, int maxlen, int batches) {
    ///For recurrent functions.
    ///Keep first dimension.
    this->hDim = hdim;

    maxLen = maxlen;
    batchNum = batches;

    states.resize(StateDim, maxlen, batches);
    actions.resize(ActionDim, maxlen, batches);
    actionNoises.resize(ActionDim, maxlen, batches);

    costs.resize(maxlen, batches);
    if (useValue) values.resize(maxlen, batches);
    if (useAdvantage) advantages.resize(maxlen, batches);

    if (isRecurrent) lengths.resize(batches);
    if (isRecurrent) hiddenStates.resize(hdim, maxlen, batches);
    termtypes.resize(batches);

    for (int k = 0; k < extraTensor1D.size(); k++)
      extraTensor1D[k].resize(batches);
    for (int k = 0; k < extraTensor2D.size(); k++)
      extraTensor2D[k].resize(maxlen, batches);
    for (int k = 0; k < extraTensor3D.size(); k++)
      extraTensor3D[k].resize(extraTensor3D[k].dim(0), maxlen, batches);
  }

  void resize(int maxlen, int batches) {
    ///Keep first dimension.

    LOG_IF(FATAL, isRecurrent && hDim == -1) << "you should set this->hDim first or call resize(hdim, maxlen, batches)";

    maxLen = maxlen;
    batchNum = batches;

    states.resize(StateDim, maxlen, batches);
    actions.resize(ActionDim, maxlen, batches);
    actionNoises.resize(ActionDim, maxlen, batches);

    costs.resize(maxlen, batches);
    if (useValue) values.resize(maxlen, batches);
    if (useAdvantage) advantages.resize(maxlen, batches);

    if (isRecurrent) lengths.resize(batches);
    if (isRecurrent) hiddenStates.resize(hDim, maxlen, batches);
    termtypes.resize(batches);

    for (int k = 0; k < extraTensor1D.size(); k++)
      extraTensor1D[k].resize(batches);
    for (int k = 0; k < extraTensor2D.size(); k++)
      extraTensor2D[k].resize(maxlen, batches);
    for (int k = 0; k < extraTensor3D.size(); k++)
      extraTensor3D[k].resize(extraTensor3D[k].dim(0), maxlen, batches);
  }

  void setZero() {
    states.setZero();
    actions.setZero();
    actionNoises.setZero();

    costs.setZero();
    if (useValue) values.setZero();
    if (useAdvantage) advantages.setZero();
    if (isRecurrent) lengths.setZero();
    termtypes.setZero();

    for (auto &ten3D: extraTensor3D) ten3D.setZero();
    for (auto &ten2D: extraTensor2D) ten2D.setZero();
    for (auto &ten1D: extraTensor1D) ten1D.setZero();
  }

 private:
  void fillminiBatch(int batchSize = 0) {

    if (miniBatch->batchNum != batchSize || miniBatch->maxLen != maxLen) {
      miniBatch->resize(maxLen, batchSize);
    }

    miniBatch->states = states.batchBlock(batchID, batchSize);
    miniBatch->actions = actions.batchBlock(batchID, batchSize);
    miniBatch->actionNoises = actionNoises.batchBlock(batchID, batchSize);

    miniBatch->costs = costs.block(0, batchID, maxLen, batchSize);
    if (useValue) miniBatch->values = values.block(0, batchID, maxLen, batchSize);
    if (useAdvantage) miniBatch->advantages = advantages.block(0, batchID, maxLen, batchSize);

    if (isRecurrent) miniBatch->lengths = lengths.block(batchID, batchSize);
    if (isRecurrent) miniBatch->hiddenStates = hiddenStates.batchBlock(batchID, batchSize);

    miniBatch->termtypes = termtypes.block(batchID, batchSize);

    for (int i = 0; i < extraTensor3D.size(); i++) {
      if (miniBatch->extraTensor3D[i].batches() != batchSize
          || miniBatch->extraTensor3D[i].rows() != extraTensor3D[i].rows()
          || miniBatch->extraTensor3D[i].cols() != extraTensor3D[i].cols())
        miniBatch->extraTensor3D[i].resize(extraTensor3D[i].rows(), extraTensor3D[i].cols(), batchSize);
      miniBatch->extraTensor3D[i] = extraTensor3D[i].batchBlock(batchID, batchSize);
    }
    for (int i = 0; i < extraTensor2D.size(); i++) {
      if (miniBatch->extraTensor2D[i].cols() != batchSize
          || miniBatch->extraTensor2D[i].rows() != extraTensor2D[i].rows())
        miniBatch->extraTensor2D[i].resize(extraTensor2D[i].rows(), batchSize);

      miniBatch->extraTensor2D[i] = extraTensor2D[i].block(0, batchID, extraTensor2D[i].rows(), batchSize);
    }
    for (int i = 0; i < extraTensor1D.size(); i++) {
      if (miniBatch->extraTensor1D[i].dim(0) != batchSize != batchSize)
        miniBatch->extraTensor1D[i].resize(batchSize);
      miniBatch->extraTensor1D[i] = extraTensor1D[i].block(batchID, batchSize);
    }
  }

 public:
  int maxLen;
  int batchNum;
  int batchID;
  int hDim;
  int dataN;

  bool useValue = false;
  bool useAdvantage = false;
  bool isRecurrent;

  Tensor3D states;
  Tensor3D actions;
  Tensor3D actionNoises;
  Tensor3D hiddenStates;
  Tensor2D costs;
  Tensor2D values;
  Tensor2D advantages;
  Tensor1D lengths;
  Tensor1D termtypes;

  //vectors for additional data
  std::vector<rai::Tensor<Dtype, 3>> extraTensor3D;
  std::vector<rai::Tensor<Dtype, 2>> extraTensor2D;
  std::vector<rai::Tensor<Dtype, 1>> extraTensor1D;

  LearningData<Dtype, StateDim, ActionDim> *miniBatch;
};
}
}

#endif //RAI_LearningData_HPP

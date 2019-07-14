//
// Created by joonho on 11/21/17.
//

#ifndef RAI_RECURRENTDETERMINISTICPOLICY_HPP
#define RAI_RECURRENTDETERMINISTICPOLICY_HPP

#include "rai/function/common/DeterministicPolicy.hpp"
#include "common/RecurrentParametrizedFunction_TensorFlow.hpp"
#include "RecurrentQfunction_TensorFlow.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class RecurrentDeterministicPolicy_TensorFlow : public virtual DeterministicPolicy<Dtype, stateDim, actionDim>,
                                                public virtual RecurrentParameterizedFunction_TensorFlow<Dtype,
                                                                                                stateDim,
                                                                                                actionDim> {

 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = RecurrentParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_tensorflow = RecurrentQfunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;

  using Pfunction_tensorflow::h;
  using Pfunction_tensorflow::hdim;

  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;

  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;
  typedef typename PolicyBase::Dataset Dataset;

  RecurrentDeterministicPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentDeterministicPolicy_TensorFlow(std::string computeMode,
                                          std::string graphName,
                                          std::string graphParam,
                                          Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(
          "RecurrentDeterministicPolicy", computeMode, graphName, graphParam, learningRate) {
  }
  virtual void forward(Tensor3D &states, Tensor3D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()},  states.dim(1), "length");
    if (h.cols() != states.batches()) {
      h.resize(hdim, states.batches());
      h.setZero();
    }
    this->tf_->run({states, h, len}, {"action", "h_state"}, {}, vectorOfOutputs);
    h.copyDataFrom(vectorOfOutputs[1]);
    actions.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual void forward(Tensor3D &states, Tensor3D &actions, Tensor2D &hiddenStates) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()},  states.dim(1), "length");

    this->tf_->run({states, hiddenStates, len}, {"action"}, {}, vectorOfOutputs);
    actions.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual Dtype performOneSolverIter(Dataset *minibatch, Tensor3D &actions) {
    std::vector<MatrixXD> vectorOfOutputs;
    actions = "targetAction";

    if (h.cols() != minibatch->batchNum) h.resize(hdim, minibatch->batchNum);
    h = minibatch->hiddenStates.col(0);

    this->tf_->run({minibatch->states, minibatch->lengths, actions, h},
                   {"trainUsingTargetAction/loss"},
                   {"trainUsingTargetAction/solver"}, vectorOfOutputs);

    return vectorOfOutputs[0](0);
  };

  Dtype backwardUsingCritic(Qfunction_tensorflow *qFunction, Dataset *minibatch) {
    std::vector<MatrixXD> dummy;
    Tensor3D gradients("trainUsingCritic/gradientFromCritic");
    Tensor2D hiddenState({hdim, minibatch->batchNum}, "h_init");
    hiddenState = minibatch->hiddenStates.col(0);

    auto pQfunction = dynamic_cast<Qfunction_tensorflow const *>(qFunction);
    LOG_IF(FATAL, pQfunction == nullptr) << "You are mixing two different library types" << std::endl;
    forward(minibatch->states, minibatch->actions, hiddenState);
    Dtype averageQ = pQfunction->getGradient_AvgOf_Q_wrt_action(minibatch, gradients);

    this->tf_->run({minibatch->states, minibatch->lengths, hiddenState, gradients}, {"trainUsingCritic/gradnorm"},
                   {"trainUsingCritic/applyGradients"}, dummy);
    return dummy[0](0);
  }
};
} // namespace FuncApprox
} // namespace rai


#endif //RAI_RECURRENTDETERMINISTICPOLICY_HPP

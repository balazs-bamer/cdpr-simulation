#include "rai/function/common/Qfunction.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class Qfunction_TensorFlow : public virtual ParameterizedFunction_TensorFlow<Dtype,
                                                                             stateDim + actionDim, 1>,
                             public virtual Qfunction<Dtype, stateDim, actionDim> {

 public:
  using QfunctionBase = Qfunction<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim + actionDim, 1>;
  typedef typename QfunctionBase::State State;
  typedef typename QfunctionBase::Action Action;
  typedef typename QfunctionBase::Jacobian Jacobian;
  typedef typename QfunctionBase::Value Value;
  typedef typename QfunctionBase::Dataset Dataset;

  typedef typename QfunctionBase::Tensor1D Tensor1D;
  typedef typename QfunctionBase::Tensor2D Tensor2D;
  typedef typename QfunctionBase::Tensor3D Tensor3D;

  Qfunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  Qfunction_TensorFlow(std::string computeMode,
                       std::string graphName,
                       std::string graphParam,
                       Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "Qfunction", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void forward(Tensor2D &states,Tensor2D &actions, Tensor2D &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states, actions}, {"QValue"}, vectorOfOutputs);
    values.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual void forward(Tensor3D &states,Tensor3D &actions, Tensor2D &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states, actions}, {"QValue"}, vectorOfOutputs);
    values.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor3D &actions, Tensor2D &values) {
    std::vector<MatrixXD> loss;
    this->tf_->run({states,
                    actions,
                    values},
                    {"trainUsingTargetQValue/loss"},
                   {"trainUsingTargetQValue/solver"}, loss);
    return loss[0](0);
  }

  virtual Dtype performOneSolverIter_infimum(Tensor3D &states, Tensor3D &actions, Tensor2D &values, Dtype linSlope) {
    std::vector<MatrixXD> loss;
    Tensor1D slope({1}, linSlope, "trainUsingTargetQValue_infimum/learningRate");

    this->tf_->run({states,
                    actions,
                    values,
                    slope}, {"trainUsingTargetQValue_infimum/loss"},
                   {"trainUsingTargetQValue_infimum/solver"}, loss);

    return loss[0](0);
  }

  virtual Dtype getGradient_AvgOf_Q_wrt_action(Tensor3D &states, Tensor3D &actions,
                                       Tensor3D &gradients) const {
    gradients.resize(actionDim, actions.cols(), actions.batches());
    std::vector<MatrixXD> outputs;
    this->tf_->run({states,
                    actions},
                   {"gradient_AvgOf_Q_wrt_action", "average_Q_value"}, {}, outputs);
    gradients.copyDataFrom(outputs[0]);
    return outputs[1](0);
  }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};
}
}
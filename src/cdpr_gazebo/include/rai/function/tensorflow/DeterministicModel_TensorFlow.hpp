//
// Created by jhwangbo on 01/09/17.
//

#ifndef RAI_DETERMINISTICMODEL_TENSORFLOW_HPP
#define RAI_DETERMINISTICMODEL_TENSORFLOW_HPP

#include "rai/function/common/DeterministicModel.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int inputDim, int outputDim>
class DeterministicModel_TensorFlow : public virtual DeterministicModel<Dtype, inputDim, outputDim>,
                                      public virtual ParameterizedFunction_TensorFlow<Dtype, inputDim, outputDim> {

  using ModelBase = DeterministicModel<Dtype, inputDim, outputDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, inputDim, outputDim>;

  typedef typename ModelBase::Input Input;
  typedef typename ModelBase::InputBatch InputBatch;
  typedef typename ModelBase::Output Output;
  typedef typename ModelBase::OutputBatch OutputBatch;
  typedef typename Pfunction_tensorflow::Tensor1D Tensor1D;
  typedef typename Pfunction_tensorflow::Tensor2D Tensor2D;
  typedef typename Pfunction_tensorflow::Tensor3D Tensor3D;

 public:

  DeterministicModel_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  DeterministicModel_TensorFlow(std::string computeMode,
                                std::string graphName,
                                std::string graphParam,
                                Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "DeterministicModel", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void forward(Input &input, Output &output) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->forward({{"input", input}},
                       {"output"}, vectorOfOutputs);
    output = vectorOfOutputs[0];
  }

  virtual void forward(InputBatch &inputs, OutputBatch &outputs) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->forward({{"input", inputs}},
                       {"output"}, vectorOfOutputs);
    outputs = vectorOfOutputs[0];
  }

  virtual Dtype performOneSolverIter(InputBatch &inputs, OutputBatch &outputs) {
    std::vector<MatrixXD> loss, dummy;
    this->tf_->run({{"input", inputs},
                    {"targetOutput", outputs}}, {"squareLoss/loss"},
                   {"squareLoss/solver"}, loss);
    return loss[0](0);
  }

  virtual void getJacobian(Tensor2D &input, Tensor3D &jaco) {
    std::vector<tensorflow::Tensor> temp;
    input.setName("input");
    this->tf_->run({input}, {"jac_output_wrt_input"}, {}, temp);
    jaco.resize({temp[0].dim_size(2), temp[0].dim_size(1), temp[0].dim_size(0)});
    std::memcpy(jaco.data(), temp[0].template flat<Dtype>().data(), sizeof(Dtype) * jaco.size());
  }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};

}
}

#endif //RAI_DETERMINISTICMODEL_TENSORFLOW_HPP

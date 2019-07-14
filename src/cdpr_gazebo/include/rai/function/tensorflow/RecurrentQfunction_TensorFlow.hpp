  #include "rai/function/common/Qfunction.hpp"
#include "common/RecurrentParametrizedFunction_TensorFlow.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class RecurrentQfunction_TensorFlow : public virtual RecurrentParameterizedFunction_TensorFlow<Dtype,
                                                                                      stateDim + actionDim, 1>,
                                      public virtual Qfunction<Dtype, stateDim, actionDim> {

 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  using QfunctionBase = Qfunction<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = RecurrentParameterizedFunction_TensorFlow<Dtype, stateDim + actionDim, 1>;

  using Pfunction_tensorflow::h;
  using Pfunction_tensorflow::hdim;

  typedef typename QfunctionBase::State State;

  typedef typename QfunctionBase::Action Action;
  typedef typename QfunctionBase::Jacobian Jacobian;
  typedef typename QfunctionBase::Value Value;
  typedef typename QfunctionBase::Tensor1D Tensor1D;
  typedef typename QfunctionBase::Tensor2D Tensor2D;
  typedef typename QfunctionBase::Tensor3D Tensor3D;
  typedef typename QfunctionBase::Dataset Dataset;

  RecurrentQfunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentQfunction_TensorFlow(std::string computeMode,
                                std::string graphName,
                                std::string graphParam,
                                Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(
          "RecurrentQfunction", computeMode, graphName, graphParam, learningRate) {
  }
  virtual void forward(Tensor3D &states, Tensor3D &actions, Tensor2D &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()}, states.dim(1), "length");
    Tensor2D hiddenState({hdim, states.batches()},0, "h_init");

    this->tf_->run({states, actions, hiddenState, len}, {"QValue", "h_state"}, {}, vectorOfOutputs);
    values.copyDataFrom(vectorOfOutputs[0]);
  }
  virtual void forward(Tensor3D &states, Tensor3D &actions, Tensor2D &values, Tensor3D &hiddenStates) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()}, states.dim(1), "length");
    Tensor2D hiddenState({hdim, states.batches()}, "h_init");
    hiddenState = hiddenStates.col(0);

    this->tf_->run({states, actions, hiddenState, len}, {"QValue", "h_state"}, {}, vectorOfOutputs);
    values.copyDataFrom(vectorOfOutputs[0]);
  }
  virtual void test(Tensor3D &states, Tensor3D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()}, states.dim(1), "length");

    if (h.cols() != states.batches()) {
      h.resize(hdim, states.batches());
    }
      h.setZero();
//    }

    Eigen::Matrix<Dtype,-1,1> test;
    this->tf_->run({states, actions, h, len}, {"test"}, {}, vectorOfOutputs);
    test.resize(vectorOfOutputs[0].template flat<Dtype>().size());
    std::memcpy(test.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * test.size());
    LOG(INFO) << test.transpose();
  }

  virtual Dtype performOneSolverIter( Tensor3D &states,  Tensor3D &actions, Tensor1D &lengths,Tensor3D &values){
    std::vector<MatrixXD> vectorOfOutputs;
    values = "targetQValue";

    if(h.cols()!= states.batches()) h.resize(hdim, states.batches());
    h.setZero();

    this->tf_->run({states, actions, lengths, values, h},
                   {"trainUsingTargetQValue/loss"},
                   {"trainUsingTargetQValue/solver"}, vectorOfOutputs);

    return vectorOfOutputs[0](0);
  };

  virtual Dtype performOneSolverIter(Dataset *minibatch, Tensor2D &values){
    std::vector<MatrixXD> vectorOfOutputs;
    values = "targetQValue";
    Tensor2D hiddenState({hdim, minibatch->batchNum}, "h_init");
    hiddenState = minibatch->hiddenStates.col(0);

    this->tf_->run({minibatch->states, minibatch->actions, minibatch->lengths, values, hiddenState},
                    {"trainUsingTargetQValue/loss"},
                   {"trainUsingTargetQValue/solver"}, vectorOfOutputs);
//   LOG(INFO) << vectorOfOutputs[1];
//    LOG(INFO) << vectorOfOutputs[1].rows();
//    LOG(INFO) << vectorOfOutputs[1].cols();
//    LOG(INFO)<< minibatch->lengths[0] << ", "<< minibatch->lengths[1];

    return vectorOfOutputs[0](0);
  };


  virtual Dtype test(Dataset *minibatch, Tensor2D &values){
    std::vector<MatrixXD> vectorOfOutputs;
    values = "targetQValue";
    if(h.cols()!= minibatch->batchNum) h.resize(hdim, minibatch->batchNum);
    h.setZero();

    Eigen::Matrix<Dtype,-1,1> test;
    this->tf_->run({minibatch->states, minibatch->actions, minibatch->lengths, values, h},
                   {"test"},
                   {}, vectorOfOutputs);

    test.resize(vectorOfOutputs[0].size());
    std::memcpy(test.data(), vectorOfOutputs[0].data(), sizeof(Dtype) * test.size());
    LOG(INFO) << test.transpose();
    return vectorOfOutputs[0](0);
  };

  Dtype getGradient_AvgOf_Q_wrt_action(Dataset *minibatch, Tensor3D &gradients) const
  {
    gradients.resize(minibatch->actions.dim());
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor2D hiddenState({hdim, minibatch->batchNum}, "h_init");
    hiddenState = minibatch->hiddenStates.col(0);
    this->tf_->run({minibatch->states,
                    minibatch->actions, minibatch->lengths, hiddenState},
                   {"gradient_AvgOf_Q_wrt_action", "average_Q_value"}, {}, vectorOfOutputs);

    gradients.copyDataFrom(vectorOfOutputs[0]);
    return vectorOfOutputs[1].scalar<Dtype>()();
  }
};
}
}
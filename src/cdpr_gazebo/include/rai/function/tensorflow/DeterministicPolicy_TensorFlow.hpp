#include "rai/function/common/DeterministicPolicy.hpp"
#include "rai/function/common/Qfunction.hpp"
#include "Qfunction_TensorFlow.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class DeterministicPolicy_TensorFlow : public virtual DeterministicPolicy<Dtype, stateDim, actionDim>,
                                       public virtual ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim> {

 public:
  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_tensorflow = Qfunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;
  typedef typename PolicyBase::JacoqWRTparam JacoqWRTparam;
  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;

  DeterministicPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  DeterministicPolicy_TensorFlow(std::string computeMode,
                                 std::string graphName,
                                 std::string graphParam,
                                 Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "DeterministicPolicy", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void forward(Tensor2D &states, Tensor2D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"action"}, vectorOfOutputs);
    actions.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual void forward(Tensor3D &states, Tensor3D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"action"}, vectorOfOutputs);
    actions.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual Dtype backwardUsingCritic(Qfunction_ *qFunction, Tensor3D &states) {
    Tensor3D actions("sampledAction");
    actions.resize(actionDim, states.cols(), states.batches());

    forward(states, actions);
    auto pQfunction = dynamic_cast<Qfunction_tensorflow const *>(qFunction);
    LOG_IF(FATAL, pQfunction == nullptr) << "You are mixing two different library types" << std::endl;
    Tensor3D gradients("trainUsingCritic/gradientFromCritic");
    Dtype averageQ = pQfunction->getGradient_AvgOf_Q_wrt_action(states, actions, gradients);

    std::vector<MatrixXD> dummy;
    this->tf_->run({states,
                    gradients}, {"trainUsingCritic/gradnorm"},
                   {"trainUsingCritic/applyGradients"}, dummy);
    return averageQ;
  }

//  virtual Dtype getGradQwrtParam(Qfunction_ *qFunction, Tensor3D &states, JacoqWRTparam &jaco) {
//    ActionBatch actions;
//    forward(states, actions);
//    auto pQfunction = dynamic_cast<Qfunction_tensorflow const *>(qFunction);
//    LOG_IF(FATAL, pQfunction == nullptr) << "You are mixing two different library types" << std::endl;
//    typename Qfunction_tensorflow::JacobianQwrtActionBatch gradients;
//    Dtype averageQ = pQfunction->getGradient_AvgOf_Q_wrt_action(states, actions, gradients);
//    std::vector<MatrixXD> output;
//    std::vector<MatrixXD> dummy;
//    this->tf_->run({{"state", states},
//                    {"gradQwrtParamMethod/gradientFromCritic", gradients}},
//                   {"gradQwrtParamMethod/gradientQwrtParam"},
//                   {"gradQwrtParamMethod/gradientQwrtParam"}, output);
//    jaco = output[0];
//    return averageQ;
//  }

  void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) {
    std::vector<MatrixXD> temp;
    this->tf_->run({{"state", state}}, {"jac_Action_wrt_Param"}, {}, temp);
    jacobian = temp[0];
  }

  void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian) {
    std::vector<MatrixXD> temp;
    this->tf_->run({{"state", state}}, {"jac_Action_wrt_State"}, {}, temp);
    jacobian = temp[0];
  }
 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};
} // namespace FuncApprox
} // namespace rai

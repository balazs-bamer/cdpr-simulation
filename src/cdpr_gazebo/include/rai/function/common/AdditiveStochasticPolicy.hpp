//
// Created by jhwangbo on 26/06/17.
//

#ifndef RAI_ADDITIVEPOLICY_HPP
#define RAI_ADDITIVEPOLICY_HPP

#include "StochasticPolicy.hpp"
#include "rai/function/common/Qfunction.hpp"
namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class AdditiveStochasticPolicy : public virtual StochasticPolicy<Dtype, stateDim, actionDim> {

 public:
  using Policy_ = StochasticPolicy<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  using PolicyBase = StochasticPolicy<Dtype, stateDim, actionDim>;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef typename PolicyBase::Dataset Dataset;

  typedef typename Policy_::State State;
  typedef typename Policy_::StateBatch StateBatch;
  typedef typename Policy_::Action Action;
  typedef typename Policy_::ActionBatch ActionBatch;
  typedef typename Policy_::Gradient Gradient;
  typedef typename Policy_::Jacobian Jacobian;
  typedef typename Policy_::JacobianWRTparam JacobianWRTparam;
  typedef typename Policy_::Jacobian JacobianWRTstate;
  typedef typename Policy_::JacoqWRTparam JacoqWRTparam;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;
  using PfunctionBase = ParameterizedFunction<Dtype, stateDim, actionDim>;
  typedef typename PfunctionBase::Parameter Parameter;

  AdditiveStochasticPolicy(Policy_ *staticPolicy, Policy_ *learnablePolicy) :
      staticPolicy_(staticPolicy), learnablePolicy_(learnablePolicy) {
  }

  ///TRPO
  // singlesample
  virtual void TRPOpg(Dataset &minibatch,
                      Action &Stdev,
                      VectorXD &grad) {
    Tensor3D firstAction;
    firstAction.resize(minibatch.actions.dim());
    staticPolicy_->forward(minibatch.states, firstAction);
    minibatch.actions -= firstAction;
    learnablePolicy_->TRPOpg(minibatch, Stdev, grad);
    minibatch.actions += firstAction;
  }

  virtual Dtype TRPOcg(Dataset &batch,
                       Action &Stdev,
                       VectorXD &grad,
                       VectorXD &getng)  {
    Tensor3D firstAction;
    firstAction.resize(batch.actions.dim());
    staticPolicy_->forward(batch.states, firstAction);
    batch.actions -= firstAction;
    Dtype returnTemp = learnablePolicy_->TRPOcg(batch, Stdev, grad, getng);
    batch.actions += firstAction;
    return returnTemp;
  }


  virtual Dtype TRPOloss(Dataset &batch,
                         Action &Stdev) {
    Tensor3D firstAction;
    firstAction.resize(batch.actions.dim());
    staticPolicy_->forward(batch.states, firstAction);
    batch.actions -= firstAction;
    Dtype returnTemp = learnablePolicy_->TRPOloss(batch, Stdev);
    batch.actions += firstAction;
    return returnTemp;
  }

  ///PPO
  virtual void PPOpg(Dataset *batch,
                     Action &Stdev,
                     VectorXD &grad) {
    Tensor3D firstAction;
    firstAction.resize(batch->actions.dim());
    staticPolicy_->forward(batch->states, firstAction);
    batch->actions -= firstAction;
    learnablePolicy_->PPOpg(batch, Stdev, grad);
    batch->actions += firstAction;
  }

  virtual void PPOpg_kladapt(Dataset *batch,
                             Action &Stdev,
                             VectorXD &grad) {
    Tensor3D firstAction;
    firstAction.resize(batch->actions.dim());
    staticPolicy_->forward(batch->states, firstAction);
    batch->actions -= firstAction;
    learnablePolicy_->PPOpg_kladapt(batch, Stdev, grad);
    batch->actions += firstAction;
  }

  virtual Dtype PPOgetkl(Dataset *batch,
                         Action &Stdev) {
    Tensor3D firstAction;
    firstAction.resize(batch->actions.dim());
    staticPolicy_->forward(batch->states, firstAction);
    batch->actions -= firstAction;
    Dtype returnTemp = learnablePolicy_->PPOgetkl(batch, Stdev);
    batch->actions += firstAction;
    return returnTemp;
  }

  virtual void setStdev(const Action &Stdev) {
    learnablePolicy_->setStdev(Stdev);
  }

  virtual void getStdev(Action &Stdev) {
    learnablePolicy_->getStdev(Stdev);
  }

  virtual void forward(State &state, Action &action) {
    Action firstAction, secondAction;
    staticPolicy_->forward(state, firstAction);
    learnablePolicy_->forward(state, secondAction);
    action = firstAction + secondAction;
  }

  virtual void forward(StateBatch &states, ActionBatch &actions) {
    ActionBatch firstAction(actions.rows(), actions.cols()), secondAction(actions.rows(), actions.cols());
    staticPolicy_->forward(states, firstAction);
    learnablePolicy_->forward(states, secondAction);
    actions = firstAction + secondAction;
  }

  virtual void forward(Tensor3D &states, Tensor3D &actions) {
    Tensor3D firstAction, secondAction;
    firstAction.resize(actions.dim());
    secondAction.resize(actions.dim());
    staticPolicy_->forward(states, firstAction);
    learnablePolicy_->forward(states, secondAction);
    actions.setZero();
    actions += firstAction;
    actions += secondAction;
  }

  virtual void forward(Tensor2D &states, Tensor2D &actions) {
    Tensor2D firstAction, secondAction;
    firstAction.resize(actions.dim());
    secondAction.resize(actions.dim());
    staticPolicy_->forward(states, firstAction);
    learnablePolicy_->forward(states, secondAction);
    actions.setZero();
    actions += firstAction;
    actions += secondAction;
  }

  virtual void trainUsingGrad(const VectorXD &grad) {
    learnablePolicy_->trainUsingGrad(grad);
  }

  virtual void trainUsingGrad(const VectorXD &grad, const Dtype learningrate) {
    learnablePolicy_->trainUsingGrad(grad, learningrate);
  }

  virtual void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) {
    learnablePolicy_->getJacobianAction_WRT_LP(state, jacobian);
  }

  virtual void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian) {
    learnablePolicy_->getJacobianAction_WRT_State(state, jacobian);
  }


  /// parameterized function methods

  virtual int getLPSize() {
    return learnablePolicy_->getLPSize();
  }

  virtual int getAPSize() {
    return learnablePolicy_->getAPSize();
  }

  virtual void getLP(Parameter &param) {
    learnablePolicy_->getLP(param);
  }

  virtual void setLP(Parameter &param) {
    learnablePolicy_->setLP(param);
  }

  virtual void getAP(Parameter &param) {
    learnablePolicy_->getAP(param);
  }

  virtual void setAP(Parameter &param) {
    learnablePolicy_->setAP(param);
  }

  virtual void copyLPFrom(PfunctionBase const *const refFcn) {
    learnablePolicy_->copyLPFrom(refFcn);
  }

  virtual void interpolateLPWith(PfunctionBase const *anotherFcn, Dtype ratio) {
    learnablePolicy_->interpolateLPWith(anotherFcn, ratio);
  }

  virtual void copyAPFrom(PfunctionBase const *const refFcn) {
    learnablePolicy_->copyAPFrom(refFcn);
  }

  virtual void interpolateAPWith(PfunctionBase const *anotherFcn, Dtype ratio) {
    learnablePolicy_->interpolateAPWith(anotherFcn, ratio);
  }

  virtual void setLearningRate(Dtype learningRate) {
    learnablePolicy_->setLearningRate(learningRate);
  }

  virtual Dtype getLearningRate() {
    return learnablePolicy_->getLearningRate();
  }

  virtual void dumpParam(std::string fileName) {
    learnablePolicy_->dumpParam(fileName);
  }


  Policy_ *staticPolicy_;
  Policy_ *learnablePolicy_;
};

}
}

#endif //RAI_ADDITIVEPOLICY_HPP

//
// Created by joonho on 23.03.17.
//

#ifndef RAI_STOCHPOL_TENSORFLOW_HPP
#define RAI_STOCHPOL_TENSORFLOW_HPP

#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include "rai/function/common/StochasticPolicy.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"
namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class StochasticPolicy_TensorFlow : public virtual StochasticPolicy<Dtype, stateDim, actionDim>,
                                    public virtual ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim> {
 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, -1> JacobianWRTparam;

  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using PolicyBase = StochasticPolicy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, actionDim>;

  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;

  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;

  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;
  typedef typename PolicyBase::Dataset Dataset;

  StochasticPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  StochasticPolicy_TensorFlow(std::string computeMode,
                              std::string graphName,
                              std::string graphParam,
                              Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "StochasticPolicy", computeMode, graphName, graphParam, learningRate) {
  }
  virtual ~StochasticPolicy_TensorFlow(){};

  ///TRPO
  //batch
  virtual void TRPOpg(Dataset &batch,
                      Action &Stdev,
                      VectorXD &grad) {
    std::vector<MatrixXD> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");

    this->tf_->run({batch.states,
                    batch.actions,
                    batch.actionNoises,
                    batch.advantages,
                    StdevT},
                   {"Algo/TRPO/Pg"},
                   {},
                   vectorOfOutputs);

    grad = vectorOfOutputs[0];
  }

  virtual Dtype TRPOcg(Dataset &batch,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getng) {
    std::vector<MatrixXD> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D gradT(grad, {grad.rows()}, "tangent");

    this->tf_->run({batch.states,
                    batch.actions,
                    batch.actionNoises,
                    batch.advantages,
                    StdevT,
                    gradT},
                   {"Algo/TRPO/Cg", "Algo/TRPO/Cgerror"}, {}, vectorOfOutputs);
    getng = vectorOfOutputs[0];
    return  vectorOfOutputs[1](0);
  }

  virtual Dtype TRPOloss(Dataset &batch,
                         Action &Stdev) {
    std::vector<MatrixXD> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");

    this->tf_->run({batch.states,
                    batch.actions,
                    batch.actionNoises,
                    batch.advantages,
                    StdevT},
                   {"Algo/TRPO/loss"},
                   {}, vectorOfOutputs);

    return vectorOfOutputs[0](0);
  }

  ///PPO
  virtual void test(Dataset *minibatch,
                     Action &Stdev) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    VectorXD test;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    ///Test function for debugging
    this->tf_->run({minibatch->states,
                    minibatch->actions,
                    minibatch->actionNoises,
                    minibatch->advantages,
                    StdevT},
                   {"test"},
                   {},
                   vectorOfOutputs);
    test.resize(vectorOfOutputs[0].template flat<Dtype>().size());
    std::memcpy(test.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * test.size());
  LOG(INFO) << test.transpose();
  }
  virtual void PPOpg(Tensor3D &states,
                     Tensor3D &actions,
                     Tensor3D &actionNoise,
                     Advantages &advs,
                     Action &Stdev,
                     Tensor1D &len,
                     VectorXD &grad) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D advsT(advs, {advs.cols()}, "advantage");

    this->tf_->run({states,
                    actions,
                    actionNoise,
                    advsT,
                    StdevT},
                   {"Algo/PPO/Pg"},
                   {},
                   vectorOfOutputs);
    std::memcpy(grad.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }

  virtual void PPOpg_kladapt(Tensor3D &states,
                             Tensor3D &action,
                             Tensor3D &actionNoise,
                             Advantages &advs,
                             Action &Stdev,
                             Tensor1D &len,
                             VectorXD &grad) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D advsT(advs, {advs.cols()}, "advantage");

    this->tf_->run({states, action, actionNoise, advsT, StdevT},
                   {"Algo/PPO/Pg2"},
                   {},
                   vectorOfOutputs);

    std::memcpy(grad.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }


  virtual void PPOpg(Dataset *minibatch,
                     Action &Stdev,
                     VectorXD &grad) {
    std::vector<MatrixXD> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");

    this->tf_->run({minibatch->states,
                    minibatch->actions,
                    minibatch->actionNoises,
                    minibatch->advantages,
                    StdevT},
                   {"Algo/PPO/Pg"},
                   {},
                   vectorOfOutputs);
    grad = vectorOfOutputs[0];
  }

  virtual void PPOpg_kladapt(Dataset *minibatch,
                             Action &Stdev,
                             VectorXD &grad) {
    std::vector<MatrixXD> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");

    this->tf_->run({minibatch->states,
                    minibatch->actions,
                    minibatch->actionNoises,
                    minibatch->advantages,
                    StdevT},
                   {"Algo/PPO/Pg2"},
                   {},
                   vectorOfOutputs);
    grad = vectorOfOutputs[0];
  }

  virtual Dtype PPOgetkl(Dataset *minibatch,
                         Action &Stdev) {
    std::vector<MatrixXD> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");

    this->tf_->run({minibatch->states,
                    minibatch->actions,
                    minibatch->actionNoises,
                    StdevT},
                   {"Algo/PPO/kl_mean"},
                   {},
                   vectorOfOutputs);
    return vectorOfOutputs[0](0);
  }

  virtual void setStdev(const Action &Stdev) {
    this->tf_->run({{"Stdev_placeholder", Stdev}}, {}, {"assignStdev"});
  }

  virtual void getStdev(Action &Stdev) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"Stdev_placeholder", Stdev}}, {"getStdev"}, {}, vectorOfOutputs);
    Stdev = vectorOfOutputs[0];
  }
  
  virtual void setParams(const VectorXD params) {
    std::vector<MatrixXD> dummy;
    this->tf_->run({{"PPO_params_placeholder", params}}, {}, {"PPO_param_assign_ops"}, dummy);
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

  void trainUsingGrad(const VectorXD &grad) {
    std::vector<MatrixXD> dummy;
    this->tf_->run({{"trainUsingGrad/Inputgradient", grad}}, {},
                   {"trainUsingGrad/applyGradients"}, dummy);
  }
  virtual void trainUsingGrad(const VectorXD &grad, const Dtype learningrate) {
    std::vector<MatrixXD> dummy;
    this->tf_->run({{"trainUsingGrad/Inputgradient", grad}}, {},
                   {"trainUsingGrad/applyGradients"}, dummy);
  }
  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor3D &actions) {
    std::vector<MatrixXD> loss;
    this->tf_->run({states,
                    actions},
                   {"trainUsingTarget/loss"},
                   {"trainUsingTarget/solver"}, loss);
    return loss[0](0);
  }

};
}//namespace FuncApprox
}//namespace rai

#endif //RAI_STOCHPOL_TENSORFLOW_HPP

//
// Created by jhwangbo on 26/06/17.
//

#ifndef RAI_DETERMINISTICPOLICY_HPP
#define RAI_DETERMINISTICPOLICY_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Qfunction.hpp"
#include "ParameterizedFunction.hpp"
#include "Policy.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class DeterministicPolicy : public virtual Policy <Dtype, stateDim, actionDim> {

 public:

  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;
  typedef typename PolicyBase::JacoqWRTparam JacoqWRTparam;

  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;
  typedef typename PolicyBase::Dataset Dataset;

  DeterministicPolicy(){};
  virtual ~DeterministicPolicy(){};

  virtual Dtype performOneSolverIter(Dataset *minibatch, Tensor3D &actions){
    LOG(FATAL) << "NOT IMPLEMENTED";
    return 0;
  }
  virtual Dtype backwardUsingCritic(Qfunction_ *qFunction, Tensor3D &states){
    LOG(FATAL) << "NOT IMPLEMENTED";
    return 0;
  }
  virtual Dtype getGradQwrtParam(Qfunction_ *qFunction, StateBatch &states, JacoqWRTparam &jaco){
    LOG(FATAL) << "NOT IMPLEMENTED";
    return 0;
  }

  virtual void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian){
    LOG(FATAL) << "NOT IMPLEMENTED";
  };
};

}} // namespaces

#endif //RAI_DETERMINISTICPOLICY_HPP

//
// Created by jemin on 27.07.16.
//

#ifndef RAI_QFUNCTION_HPP
#define RAI_QFUNCTION_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include "ParameterizedFunction.hpp"
#include <rai/algorithm/common/LearningData.hpp>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDimension, int actionDimension>
class Qfunction : public virtual ParameterizedFunction <Dtype, stateDimension + actionDimension, 1>  {

public:

  using FunctionBase = ParameterizedFunction <Dtype, stateDimension + actionDimension, 1>;
  using Dataset = rai::Algorithm::LearningData<Dtype,stateDimension, actionDimension>;

  typedef typename FunctionBase::Input StateAction;
  typedef typename FunctionBase::Output Value;
  typedef typename FunctionBase::Gradient Gradient;
  typedef typename FunctionBase::Jacobian Jacobian;
  typedef typename FunctionBase::Tensor1D Tensor1D;
  typedef typename FunctionBase::Tensor2D Tensor2D;
  typedef typename FunctionBase::Tensor3D Tensor3D;

  typedef Eigen::Matrix<Dtype, stateDimension, 1> State;
  typedef Eigen::Matrix<Dtype, actionDimension, 1> Action;

  Qfunction(){};
  virtual ~Qfunction(){};

  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor3D &actions, Tensor2D &values) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  }

  virtual Dtype performOneSolverIter_infimum(Tensor3D &states, Tensor3D &actions, Tensor2D &values, Dtype linSlope) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  }

  virtual Dtype getGradient_AvgOf_Q_wrt_action(Tensor3D &states, Tensor3D &actions,
                                       Tensor3D &gradients) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  }


};

}} // namespaces

#endif //RAI_QFUNCTION_HPP

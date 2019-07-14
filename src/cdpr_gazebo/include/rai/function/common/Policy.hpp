//
// Created by jemin on 27.07.16.
//

#ifndef RAI_POLICY_HPP
#define RAI_POLICY_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

#include "ParameterizedFunction.hpp"
#include <rai/algorithm/common/LearningData.hpp>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDimension, int actionDimension>
class Policy : public virtual ParameterizedFunction <Dtype, stateDimension, actionDimension> {

public:

  using FunctionBase = ParameterizedFunction <Dtype, stateDimension, actionDimension>;
  using Dataset = rai::Algorithm::LearningData<Dtype,stateDimension, actionDimension>;

  typedef typename FunctionBase::Input State;
  typedef typename FunctionBase::InputBatch StateBatch;
  typedef typename FunctionBase::Output Action;
  typedef typename FunctionBase::OutputBatch ActionBatch;

  typedef typename FunctionBase::Tensor1D Tensor1D;
  typedef typename FunctionBase::Tensor2D Tensor2D;
  typedef typename FunctionBase::Tensor3D Tensor3D;

  typedef typename FunctionBase::Gradient Gradient;
  typedef typename FunctionBase::Jacobian Jacobian;
  typedef typename FunctionBase::JacobianWRTparam JacobianWRTparam;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> JacoqWRTparam;

  Policy(){};
  virtual ~Policy(){};

};

}} // namespaces
#endif //RAI_POLICY_HPP

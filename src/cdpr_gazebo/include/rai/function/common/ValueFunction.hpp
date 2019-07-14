//
// Created by jemin on 27.07.16.
//

#ifndef RAI_VALUE_HPP
#define RAI_VALUE_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDimension>
class ValueFunction : public virtual ParameterizedFunction <Dtype, stateDimension, 1>  {

public:
  using FunctionBase = ParameterizedFunction <Dtype, stateDimension, 1>;

  typedef typename FunctionBase::Input State;
  typedef typename FunctionBase::InputBatch StateBatch;
  typedef typename FunctionBase::Output Value;
  typedef typename FunctionBase::OutputBatch ValueBatch;
  typedef typename FunctionBase::Gradient Gradient;
  typedef typename FunctionBase::Jacobian Jacobian;
  typedef typename FunctionBase::Tensor1D Tensor1D;
  typedef typename FunctionBase::Tensor2D Tensor2D;
  typedef typename FunctionBase::Tensor3D Tensor3D;

  ValueFunction(){};
  virtual ~ValueFunction(){};

  virtual void setClipRate(const Dtype param_in){
    LOG(FATAL) << "NOT IMPLEMENTED";
  }
  virtual void setClipRangeDecay(const Dtype decayRate){
    LOG(FATAL) << "NOT IMPLEMENTED";
  }

  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor2D &values) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor2D &values, Tensor1D &lengths) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter_trustregion(StateBatch &states, ValueBatch &targetOutputs, ValueBatch &old_prediction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter_trustregion(Tensor3D &states, Tensor2D &targetOutputs, Tensor2D &old_prediction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter_trustregion(Tensor3D &states, Tensor2D &targetOutputs, Tensor2D &old_prediction, Tensor1D &lengths) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

};

}} // namespaces

#endif //RAI_VALUE_HPP

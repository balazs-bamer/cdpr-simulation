//
// Created by jemin on 27.07.16.
//

#ifndef RAI_PARAMETERIZEDFUNCTION_HPP
#define RAI_PARAMETERIZEDFUNCTION_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include <glog/logging.h>
#include <boost/type_traits/function_traits.hpp>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>
#include "rai/RAI_Tensor.hpp"

namespace rai {
namespace FuncApprox {

enum class LibraryID {
  notSpecified = 0,
  caffe,
  tensorFlow
};

template<typename Dtype, int inputDimension, int outputDimension>
class ParameterizedFunction {

 public:
  typedef Eigen::Matrix<Dtype, inputDimension, 1> Input;
  typedef Eigen::Matrix<Dtype, inputDimension, Eigen::Dynamic> InputBatch;
  typedef Eigen::Matrix<Dtype, outputDimension, 1> Output;
  typedef Eigen::Matrix<Dtype, outputDimension, Eigen::Dynamic> OutputBatch;
  typedef Eigen::Matrix<Dtype, outputDimension, inputDimension> Jacobian;
  typedef Eigen::Matrix<Dtype, inputDimension, 1> Gradient;
  typedef Eigen::Matrix<Dtype, outputDimension, Eigen::Dynamic> JacobianWRTparam;

  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;
  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>> EigenMat;

  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> Parameter;
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> ParameterGradient;
  using Pfunction = ParameterizedFunction<Dtype, inputDimension, outputDimension>;

  ParameterizedFunction() {};
  virtual ~ParameterizedFunction() {};

  /// must be implemented all by function libraries

  virtual void test(Tensor3D &intputs, Tensor2D &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(Input &input, Output &output) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(InputBatch &intputs, OutputBatch &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(Tensor2D &intputs, Tensor2D &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(Tensor2D &intputs1, Tensor2D &intputs2, Tensor2D &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(Tensor3D &intputs, Tensor2D &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(Tensor3D &intputs, Tensor3D &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };
  virtual void forward(Tensor3D &intputs1, Tensor3D &intputs2, Tensor2D &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(Tensor3D &intputs, Tensor2D &outputs, Tensor3D &hiddenStates) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(Tensor3D &intputs, Tensor3D &outputs, Tensor3D &hiddenStates) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual Dtype performOneSolverIter(InputBatch &states, OutputBatch &targetOutputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor2D &targetOutputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor3D &targetOutputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter_infimum(InputBatch &states, OutputBatch &targetOutputs, Dtype linSlope) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter_huber(InputBatch &states, OutputBatch &targetOutputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };


  virtual void backward(InputBatch &states, OutputBatch &targetOutputs, ParameterGradient &gradient) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void setLearningRate(Dtype LearningRate) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual Dtype getLearningRate() {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void copyStructureFrom(Pfunction const *referenceFunction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void copyAPFrom(Pfunction const *referenceFunction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void copyLPFrom(Pfunction const *referenceFunction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void interpolateLPWith(Pfunction const *anotherFunction, Dtype ratio) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void interpolateAPWith(Pfunction const *anotherFunction, Dtype ratio) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual int getLPSize() { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual int getAPSize() { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void getJacobian(Input &input, Jacobian &jacobian) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void getGradient(Input &input, Gradient &gradient) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void getJacobianOutputWRTparameter(Input &input, JacobianWRTparam &gradient) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void performOneStepTowardsGradient(OutputBatch &diff, InputBatch &input) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void getLP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void getAP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void setLP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void setAP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  /// get error from last batch
  virtual Dtype getLossFromLastBatch() { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void dumpParam(std::string fileName) { LOG(FATAL) << "NOT IMPLEMENTED"; };
  virtual void loadParam(std::string fileName) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  /// recurrent
  virtual bool isRecurrent() { return false; }
  virtual void reset(int n) {}
  virtual void terminate(int n) {}
  virtual int getHiddenStatesize() { return 0; }
  virtual typename EigenMat::ColXpr getHiddenState(int Id){LOG(FATAL) << "NOT IMPLEMENTED"; }
  virtual void getHiddenStates(Tensor2D &h_out){LOG(FATAL) << "NOT IMPLEMENTED"; }

  LibraryID libraryID_ = LibraryID::notSpecified;
  int parameterSize = 0;

};

}
} // namespaces

#endif //RAI_PARAMETERIZEDFUNCTION_HPP

//
// Created by joonho on 12/9/17.
//

#ifndef RAI_RECURRENTPARAMETRIZEDFUNCTION_TENSORFLOW_HPP
#define RAI_RECURRENTPARAMETRIZEDFUNCTION_TENSORFLOW_HPP

#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"
#include <rai/RAI_core>
namespace rai {
namespace FuncApprox {
template<typename Dtype, int inputDimension, int outputDimension>
class RecurrentParameterizedFunction_TensorFlow : public ParameterizedFunction_TensorFlow<Dtype,
                                                                                                   inputDimension,
                                                                                                   outputDimension> {
 public:
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> MatrixXD;
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> testMatrix;

  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, inputDimension, outputDimension>;
  typedef typename Pfunction_tensorflow::Tensor1D Tensor1D;
  typedef typename Pfunction_tensorflow::Tensor2D Tensor2D;
  typedef typename Pfunction_tensorflow::Tensor3D Tensor3D;
  typedef typename Pfunction_tensorflow::EigenMat EigenMat;

  RecurrentParameterizedFunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate), h("h_init") {
    hdim = this->getHiddenStatesize();
    h.resize(hdim, 0);
  }

  RecurrentParameterizedFunction_TensorFlow(std::string functionName,
                                             std::string computeMode,
                                             std::string graphName,
                                             std::string graphParam,
                                             Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(functionName,
                                                             computeMode,
                                                             graphName,
                                                             graphParam,
                                                             learningRate) , h("h_init") {
    hdim = this->getHiddenStatesize();
    h.resize(hdim, 0);
  }
  virtual ~RecurrentParameterizedFunction_TensorFlow(){

  };

  virtual bool isRecurrent() {return true;}

  virtual void reset(int n) {
    //n:index
    if (n >= h.cols())
      h.conservativeResize(hdim, n + 1);
    h.col(n).setZero();
  }

  virtual void terminate(int n) {
    int coldim = h.cols() - 1;
    LOG_IF(FATAL, coldim < 0) << "Initialize Hiddenstates first (Call reset)";
    LOG_IF(FATAL, n > coldim) << "n exceeds batchsize" << n << "vs." << coldim;
    h.removeCol(n);
  }

  virtual int getHiddenStatesize() {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->run({}, {"h_dim"}, {}, vectorOfOutputs);
    return vectorOfOutputs[0].scalar<int>()();
  }

  virtual void getHiddenStates(Tensor2D &h_out){
    h_out = h;
  }

  virtual typename EigenMat::ColXpr getHiddenState(int Id){
    return h.col(Id);
  }

  int hiddenStateDim() { return hdim; }

 protected:
  int hdim = 0;
  Tensor2D h;
};
}
}
#endif //RAI_RECURRENTPARAMETRIZEDFUNCTION_TENSORFLOW_HPP

#pragma once

#include <iomanip>
#include <iostream>
#include <fstream>
#include "rai/function/common/ParameterizedFunction.hpp"
#include "TensorFlowNeuralNetwork.hpp"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <rai/RAI_core>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int inputDimension, int outputDimension>
class ParameterizedFunction_TensorFlow : public virtual ParameterizedFunction<Dtype, inputDimension, outputDimension> {

 public:
  using PfunctionBase = ParameterizedFunction<Dtype, inputDimension, outputDimension>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, inputDimension, outputDimension>;
  typedef typename PfunctionBase::Input Input;
  typedef typename PfunctionBase::InputBatch InputBatch;
  typedef typename PfunctionBase::Tensor1D Tensor1D;
  typedef typename PfunctionBase::Tensor2D Tensor2D;
  typedef typename PfunctionBase::Tensor3D Tensor3D;

  typedef typename PfunctionBase::Output Output;
  typedef typename PfunctionBase::OutputBatch OutputBatch;
  typedef typename PfunctionBase::Gradient Gradient;
  typedef typename PfunctionBase::Jacobian Jacobian;
  typedef typename PfunctionBase::ParameterGradient ParameterGradient;
  typedef typename PfunctionBase::JacobianWRTparam JacobianWRTparam;
  typedef typename PfunctionBase::Parameter Parameter;

  /*
   * When setting nThreads = 0, TensorFlow uses the number of core to determine the number of threads.
   *
   * Setting nThreads = 1 is necessary for determinism: TensorFlow currently is non-deterministic when it uses more than
   * one thread, or is executed on the GPU: Cf. https://github.com/tensorflow/tensorflow/issues/3103
   * In order to get a network that is initialized the same and, you also have to set the TensorFlow random seed when
   * when generating the net (due to the weights being initialized randomly).
   */
  ParameterizedFunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) {
    tf_ = new TensorFlowNeuralNetwork<Dtype>(pathToGraphDefProtobuf, 0);
    this->setLearningRate(learningRate);
  }

  ParameterizedFunction_TensorFlow(std::string functionName,
                                   std::string computeMode,
                                   std::string graphName,
                                   std::string graphParam,
                                   Dtype learningRate = 1e-3) {
    LOG(INFO) << "use pregenerated files if you want to save time creating graph file";
    LOG_IF(FATAL, RAI_LOG_PATH.empty()) << "CALL RAI INIT FIRST";

    std::string shellFilePath;
    shellFilePath = std::string(std::getenv("RAI_ROOT"))
        + "/RAI/include/rai/function/tensorflow/pythonProtobufGenerators/";
    std::string cmnd;
    cmnd = shellFilePath + "run_python_scripts.sh " + shellFilePath + "protobufGenerator.py";

    if (typeid(Dtype) == typeid(double)) cmnd += " 2 ";
    else cmnd += " 1 ";

    cmnd += RAI_LOG_PATH + " " + computeMode + " " + functionName + " " + graphName + " " + graphParam;

    system(cmnd.c_str());
    tf_ = new TensorFlowNeuralNetwork<Dtype>(RAI_LOG_PATH + "/" + functionName + "_" + graphName + ".pb", 0);
    this->setLearningRate(learningRate);
  }

  virtual ~ParameterizedFunction_TensorFlow() {
    delete tf_;
  }

  virtual void forward(Input &input, Output &output) {
    std::vector<MatrixXD> vectorOfOutputs;
    tf_->run({{"input", input},
              {"updateBNparams", notUpdateBN}},
             {"output"}, {}, vectorOfOutputs);
    output = vectorOfOutputs[0];
  }

  virtual void forward(InputBatch &inputs, OutputBatch &outputs) {
    std::vector<MatrixXD> vectorOfOutputs;
    tf_->run({{"input", inputs},
              {"updateBNparams", notUpdateBN}},
             {"output"}, {}, vectorOfOutputs);
    outputs = vectorOfOutputs[0];
  }

  virtual void
  copyStructureFrom(PfunctionBase const *const referenceFunction) {
    Pfunction_tensorflow const *pReferenceFunction = castToThisClass(referenceFunction);
    tf_->setGraphDef(pReferenceFunction->getTensorFlowWrapper()->getGraphDef());
  }

  virtual void getJacobian(Input &input, Jacobian &jacobian) {
    // TODO: implementation
    LOG(FATAL) << "getJacobian: Currently not implemented";
  }

  virtual void getGradient(Input &input, Gradient &gradient) {
    // TODO: implementation
    LOG(FATAL) << "getGradient: Currently not implemented";
  }

  virtual void performOneStepTowardsGradient(OutputBatch &diff, InputBatch &input) {
    // TODO: implementation
    LOG(FATAL) << "performOneStepTowardsGradient: Currently not implemented";
  }

  virtual void backward(InputBatch &states, OutputBatch &targetOutputs, ParameterGradient &gradient) {
    // TODO: implementation
    LOG(FATAL) << "backward: Currently not implemented";
  }

  virtual void getJacobianOutputWRTparameter(Input &input, JacobianWRTparam &gradient) {
    // TODO: implementation
    LOG(FATAL) << "getJacobianOutputWRTparameter: Currently not implemented";
  }

  virtual int getLPSize() {
    return tf_->getLPSize();
  }

  virtual int getAPSize() {
    return tf_->getAPSize();
  }

  virtual void getLP(Parameter &param) {
    tf_->getLP(param);
  }

  virtual void setLP(Parameter &param) {
    tf_->setLP(param);
  }

  virtual void getAP(Parameter &param) {
    tf_->getAP(param);
  }

  virtual void setAP(Parameter &param) {
    tf_->setAP(param);
  }

  virtual void copyLPFrom(PfunctionBase const *const refFcn) {
    Pfunction_tensorflow const *pReferenceFunction = castToThisClass(refFcn);
    tf_->copyLP(pReferenceFunction->getTensorFlowWrapper());
  }

  virtual void interpolateLPWith(PfunctionBase const *anotherFcn, Dtype ratio) {
    Pfunction_tensorflow const *pAnotherFcn = castToThisClass(anotherFcn);
    tf_->interpolateLP(pAnotherFcn->getTensorFlowWrapper(), ratio);
  }

  virtual void copyAPFrom(PfunctionBase const *const refFcn) {
    Pfunction_tensorflow const *pRefFcn = castToThisClass(refFcn);
    tf_->copyAP(pRefFcn->getTensorFlowWrapper());
  }

  virtual void interpolateAPWith(PfunctionBase const *anotherFcn, Dtype ratio) {
    Pfunction_tensorflow const *pAnotherFcn = castToThisClass(anotherFcn);
    tf_->interpolateAP(pAnotherFcn->getTensorFlowWrapper(), ratio);
  }

  virtual TensorFlowNeuralNetwork<Dtype> *getTensorFlowWrapper() {
    return tf_;
  }

  void run(const std::vector<std::pair<std::string, tensorflow::Tensor>> &inputs,
           const std::vector<std::string> &outputTensorNames,
           const std::vector<std::string> &targetNodeNames, std::vector<tensorflow::Tensor> &outputs) {
    tf_->run(inputs, outputTensorNames, targetNodeNames, outputs);
  }

  virtual TensorFlowNeuralNetwork<Dtype> const *getTensorFlowWrapper() const {
    return dynamic_cast<TensorFlowNeuralNetwork<Dtype> const *>(tf_);
  }

  virtual void setLearningRate(Dtype learningRate) {
    Tensor1D lr({1}, learningRate, "trainingOptions/param_assign_placeholder");
    tf_->run({lr}, {}, {"InitLR_assign", "reset_global_step"});
  }

  virtual void setLearningRateDecay(Dtype decayRate, int decaySteps) {
    Tensor1D dr({1}, decayRate, "trainingOptions/param_assign_placeholder");
    rai::Tensor<int, 1> ds({1}, decaySteps, "trainingOptions/param_assign_placeholder_int");
    tf_->run({dr, ds}, {}, {"DecayRateLR_assign", "DecayStepLR_assign"});
  }

  virtual void setMaxGradientNorm(Dtype GradNorm){
    Tensor1D gn({1}, GradNorm, "trainingOptions/param_assign_placeholder");
    tf_->run({gn}, {}, {"max_norm_assign"});
  }

  virtual Dtype getLearningRate() {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    tf_->run({}, {"trainingOptions/LR"}, {}, vectorOfOutputs);
    return vectorOfOutputs[0].scalar<Dtype>()();
  }

  int incrementGlobalStep(){
    std::vector<tensorflow::Tensor> globalStep;
    tf_->run({}, {"global_step"}, {"increment_global_step"}, globalStep);
    return globalStep[0].scalar<int>()();

  }
  int getGlobalStep() {
    return tf_->getGlobalStep();
  }

  virtual void setCheckNumerics(bool checkNumerics) {
    tf_->setCheckNumerics(checkNumerics);
  }

  virtual bool getCheckNumerics() {
    return tf_->getCheckNumerics();
  }

  virtual void dumpParam(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(10, Eigen::DontAlignCols, ", ", "\n");
    VectorXD parameterVector(this->tf_->getAPSize());
    this->tf_->getAP(parameterVector);
    std::ofstream of(fileName.c_str());
    of << parameterVector.transpose().format(CSVFormat) << std::endl;
    of.close();
  }

  virtual void loadParam(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(20, Eigen::DontAlignCols, ", ", "\n");
    VectorXD param(this->tf_->getAPSize());
    std::stringstream parameterFileName;
    std::ifstream indata;
    indata.open(fileName);
    LOG_IF(FATAL, !indata.is_open()) << "Parameter file " << fileName << " could not be opened";
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;
    int paramSize = 0;
    while (std::getline(lineStream, cell, ','))
      param(paramSize++) = std::stof(cell);
    LOG_IF(FATAL, getAPSize() != paramSize) << "Parameter sizes don't match";
    this->tf_->setAP(param);
  }

 protected:

  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;
  using VectorXD = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;

  TensorFlowNeuralNetwork<Dtype> *tf_;

  const MatrixXD notUpdateBN = (MatrixXD(1, 1) << 0).finished();
  const MatrixXD updateBN = (MatrixXD(1, 1) << 1).finished();
  Pfunction_tensorflow *castToThisClass(PfunctionBase *anotherFunction) {
    auto ptr = dynamic_cast<Pfunction_tensorflow * >(anotherFunction);
    LOG_IF(FATAL, ptr == nullptr) << "You are mixing two different library types" << std::endl;
    return ptr;
  }
  Pfunction_tensorflow const *castToThisClass(PfunctionBase const *anotherFunction) {
    auto ptr = dynamic_cast<Pfunction_tensorflow const * >(anotherFunction);
    LOG_IF(FATAL, ptr == nullptr) << "You are mixing two different library types" << std::endl;
    return ptr;
  }
};
} // namespace FuncApprox
} // namespace rai

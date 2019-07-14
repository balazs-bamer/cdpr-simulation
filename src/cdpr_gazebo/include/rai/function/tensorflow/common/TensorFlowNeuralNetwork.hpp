#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <iostream>
#include <typeinfo>
#include "glog/logging.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "rai/common/VectorHelper.hpp"
#include "rai/RAI_Tensor.hpp"

# pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype>
class TensorFlowNeuralNetwork {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using MatrixXD = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorXD = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
  using Tensor3D = Eigen::Tensor<Dtype, 3>;
  TensorFlowNeuralNetwork(tensorflow::GraphDef graphDef, int n_threads = 0,
                          bool logDevicePlacment = false) {
    construct(graphDef, n_threads, logDevicePlacment);
  }

  TensorFlowNeuralNetwork(std::string pathToGraphDefProtobuf, int n_threads = 0,
                          bool logDevicePlacment = false) {
    tensorflow::GraphDef graphDef;
    auto status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), pathToGraphDefProtobuf, &graphDef);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    construct(graphDef, n_threads, logDevicePlacment);
  }

  ~TensorFlowNeuralNetwork() {
    delete session;
  }
  //mat to mat
  inline void run(const std::vector<std::pair<std::string, MatrixXD>> &inputs,
                  const std::vector<std::string> &outputTensorNames,
                  const std::vector<std::string> &targetNodeNames, std::vector<MatrixXD> &outputs) {
    std::vector<std::pair<std::string, tensorflow::Tensor> > namedInputTensorFlowTensors;
    namedEigenMatricesToNamedTFTensors(inputs, namedInputTensorFlowTensors);
    std::vector<tensorflow::Tensor> outputTensorFlowTensors;

    auto status = session->Run(namedInputTensorFlowTensors, outputTensorNames, targetNodeNames,
                               &outputTensorFlowTensors);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    tfTensorsToEigenMatrices(outputTensorFlowTensors, outputs);
  }
  //mat no output (just run a node)
  inline void run(const std::vector<std::pair<std::string, MatrixXD>> &inputs,
                  const std::vector<std::string> &outputTensorNames,
                  const  std::vector<std::string> &targetNodeNames) {

    std::vector<std::pair<std::string, tensorflow::Tensor> > namedInputTensorFlowTensors;
    namedEigenMatricesToNamedTFTensors(inputs, namedInputTensorFlowTensors);
    std::vector<tensorflow::Tensor> outputTensorFlowTensors;

    auto status = session->Run(namedInputTensorFlowTensors, outputTensorNames, targetNodeNames,
                               &outputTensorFlowTensors);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

//  //tensor to tensor
//  inline void run(const std::vector<std::pair<std::string, Tensor3D>> &inputs,
//                  const  std::vector<std::string> &outputTensorNames,
//                  const  std::vector<std::string> &targetNodeNames,  std::vector<Tensor3D> &outputs) {
//    std::vector<std::pair<std::string, tensorflow::Tensor> > namedInputTensorFlowTensors;
//
//    namedEigenTensorsToNamedTFTensors(inputs, namedInputTensorFlowTensors);
//
//    std::vector<tensorflow::Tensor> outputTensorFlowTensors;
//    std::vector<std::string> targetNodeNamesModified = targetNodeNames;
//    auto status = session->Run(namedInputTensorFlowTensors, outputTensorNames, targetNodeNamesModified,
//                               &outputTensorFlowTensors);
//    LOG_IF(FATAL, !status.ok()) << status.ToString();
//
//    tfTensorsToEigenTensors(outputTensorFlowTensors, outputs);
//  }

//  //tensor to tensor
//  inline void run(const std::vector<std::pair<std::string, tensorflow::Tensor>> &inputs,
//                  const std::vector<std::string> &outputTensorNames,
//                  const  std::vector<std::string> &targetNodeNames,  std::vector<Tensor3D> &outputs) {
//
//    std::vector<tensorflow::Tensor> outputTensorFlowTensors;
//    std::vector<std::string> targetNodeNamesModified = targetNodeNames;
//    auto status = session->Run(inputs, outputTensorNames, targetNodeNamesModified, &outputTensorFlowTensors);
//    LOG_IF(FATAL, !status.ok()) << status.ToString();
//    tfTensorsToEigenTensors(outputTensorFlowTensors, outputs);
//  }

  //tensor to tensor
  inline void run(const std::vector<std::pair<std::string, tensorflow::Tensor>> &inputs,
                  const std::vector<std::string> &outputTensorNames,
                  const  std::vector<std::string> &targetNodeNames,  std::vector<tensorflow::Tensor> &outputs) {

    std::vector<tensorflow::Tensor> outputTensorFlowTensors;
    std::vector<std::string> targetNodeNamesModified = targetNodeNames;
    auto status = session->Run(inputs, outputTensorNames, targetNodeNamesModified, &outputTensorFlowTensors);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    outputs = outputTensorFlowTensors;
  }

  //tensor to mat
  inline void run(const std::vector<std::pair<std::string, tensorflow::Tensor>> &inputs,
                  const std::vector<std::string> &outputTensorNames,
                  const  std::vector<std::string> &targetNodeNames, std::vector<MatrixXD> &outputs) {

    std::vector<tensorflow::Tensor> outputTensorFlowTensors;
    std::vector<std::string> targetNodeNamesModified = targetNodeNames;
    auto status = session->Run(inputs, outputTensorNames, targetNodeNamesModified, &outputTensorFlowTensors);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    tfTensorsToEigenMatrices(outputTensorFlowTensors, outputs);
  }

  //tensor no output (just run a node)
  inline void run(const std::vector<std::pair<std::string, tensorflow::Tensor>> &inputs,
                  const std::vector<std::string> &outputTensorNames,
                  const  std::vector<std::string> &targetNodeNames) {

    std::vector<tensorflow::Tensor> outputTensorFlowTensors;
    std::vector<std::string> targetNodeNamesModified = targetNodeNames;
    auto status = session->Run(inputs, outputTensorNames, targetNodeNamesModified, &outputTensorFlowTensors);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void runTargetNodes(const std::vector<std::pair<std::string, MatrixXD>> &inputs,
                      const std::vector<std::string> &targetNodeNames) {
    std::vector<MatrixXD> dummyOutputs;
    run(inputs, {}, targetNodeNames, dummyOutputs);
  }

  void forward(const std::vector<std::pair<std::string, MatrixXD>> &inputs, const std::vector<std::string> &outputNames,
               std::vector<MatrixXD> &outputs) {
    run(inputs, outputNames, {}, outputs);
  }

  void forward(const std::vector<std::pair<std::string, tensorflow::Tensor>> &inputs, const std::vector<std::string> &outputNames,
               std::vector<tensorflow::Tensor> &outputs) {
    run(inputs, outputNames, {}, outputs);
  }

  void forward(const std::vector<std::pair<std::string, Tensor3D>> &inputs, const std::vector<std::string> &outputNames,
               std::vector<Tensor3D> &outputs) {
    run(inputs, outputNames, {}, outputs);
  }

  void copyLP(const TensorFlowNeuralNetwork<Dtype> *otherNetwork) {
    LOG_IF(FATAL, getLPSize() != otherNetwork->getLPSize())
    << "copyWeightsFromOtherNetwork: The two networks need to have the same number of parameters. ("
    << getLPSize() << " vs. " << otherNetwork->getLPSize() << ")";
    tensorflow::Tensor learnableParamOther;
    otherNetwork->getLP(learnableParamOther);
    auto status = session->Run({{"LP_placeholder", learnableParamOther}}, {}, {"assignLP"}, nullptr);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void copyAP(const TensorFlowNeuralNetwork<Dtype> *otherNetwork) {
    LOG_IF(FATAL, getAPSize() != otherNetwork->getAPSize())
    << "copyWeightsFromOtherNetwork: The two networks need to have the same number of parameters. ("
    << getAPSize() << " vs. " << otherNetwork->getAPSize() << ")";
    tensorflow::Tensor allParamOther;
    otherNetwork->getAP(allParamOther);
    auto status = session->Run({{"AP_placeholder", allParamOther}}, {}, {"assignAP"}, nullptr);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void interpolateLP(const TensorFlowNeuralNetwork<Dtype> *otherNetwork, Dtype tau) {
    LOG_IF(FATAL, getLPSize() != otherNetwork->getLPSize())
    << "copyWeightsFromOtherNetwork: The two networks need to have the same number of parameters. ("
    << getLPSize() << " vs. " << otherNetwork->getLPSize() << ")";
    tensorflow::Tensor learnableParamOther;
    otherNetwork->getLP(learnableParamOther);
    tensorflow::Tensor tauTensor(getTensorFlowDataType(), tensorflow::TensorShape({}));
    tauTensor.scalar<Dtype>()() = tau;
    auto status =
        session->Run({{"LP_placeholder", learnableParamOther}, {"tau", tauTensor}}, {}, {"interpolateLP"}, nullptr);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void interpolateAP(const TensorFlowNeuralNetwork<Dtype> *otherNetwork, Dtype tau) {
    LOG_IF(FATAL, getAPSize() != otherNetwork->getAPSize())
    << "copyWeightsFromOtherNetwork: The two networks need to have the same number of parameters. ("
    << getAPSize() << " vs. " << otherNetwork->getAPSize() << ")";
    tensorflow::Tensor allParamOther;
    otherNetwork->getAP(allParamOther);
    tensorflow::Tensor tauTensor(getTensorFlowDataType(), tensorflow::TensorShape({}));
    tauTensor.scalar<Dtype>()() = tau;
    auto status = session->Run({{"AP_placeholder", allParamOther}, {"tau", tauTensor}}, {}, {"interpolateAP"}, nullptr);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void setGraphDef(const tensorflow::GraphDef graphDef) {
    graphDef_ = graphDef;
    auto status = session->Close();
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    status = session->Create(graphDef);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    updateParams();
  }

  tensorflow::GraphDef getGraphDef() const {
    return graphDef_;
  }

  void setCheckNumerics(bool checkNumerics) {
    checkNumerics_ = checkNumerics;
  }

  bool getCheckNumerics() {
    return checkNumerics_;
  }
//////////////////////

  inline static void eigenMatrixToTFTensor(const MatrixXD &matrix, tensorflow::Tensor &tensor) {
    LOG_IF(FATAL, tensor.shape().dims() > 2)
    << "copyEigenMatrixToTensorFlowTensor requires rank 2 tensors (matrices).";
    int rows = std::max(int(tensor.shape().dim_size(0)), 1);
    int cols = std::max(int(tensor.shape().dim_size(1)), 1);
    LOG_IF(FATAL, rows != matrix.cols() || cols != matrix.rows())
    << "dimensions don't match. Eigen matrix and Tensorflow tensor should be transpose of each other Eigen is colmajor and Tensorflow is rowmajor"
    << std::endl
    << "(" << rows << ", " << cols << ") vs (" << matrix.rows() << ", " << matrix.cols() << ")";
    memcpy(tensor.flat<Dtype>().data(), matrix.data(), sizeof(Dtype) * tensor.shape().num_elements());
  }

  inline static void tfTensorToEigenMatrix(const tensorflow::Tensor &tensor, MatrixXD &matrix) {
    LOG_IF(FATAL, tensor.shape().dims() > 2) << "copyTensorFlowTensorToEigenMatrix requires rank 2 tensors (matrices)";
    int rows = std::max(int(tensor.shape().dim_size(0)), 1);
    int cols = std::max(int(tensor.shape().dim_size(1)), 1);
    if (tensor.shape().dims() == 2) matrix.resize(cols, rows);
    else matrix.resize(1, 1);
    memcpy(matrix.data(), tensor.flat<Dtype>().data(), sizeof(Dtype) * tensor.shape().num_elements());
  }

  inline static void EigenTensorTotfTensor(const Tensor3D &Eigtensor, tensorflow::Tensor &tensor) {
    LOG_IF(FATAL, tensor.shape().dims() != 3) << "EigenTensorTotfTensor requires rank 3 tensors.";
    memcpy(tensor.flat<Dtype>().data(), Eigtensor.data(), sizeof(Dtype) * tensor.shape().num_elements());
  }

  inline static void tfTensorToEigenTensor(const tensorflow::Tensor &tensor, Tensor3D &Eigtensor) {
    LOG_IF(FATAL, tensor.shape().dims() > 3)
      << "tfTensorToEigenTensor requires rank <=3. " << " (" << tensor.shape().dims() <<")";
    int d1, d2, d3;
    d1 = std::max(int(tensor.shape().dim_size(0)), 1); //bat
    d2 = std::max(int(tensor.shape().dim_size(1)), 1); //len
    d3 = std::max(int(tensor.shape().dim_size(2)), 1); //dim
    Eigtensor.resize(d3, d2, d1);
    memcpy(Eigtensor.data(), tensor.flat<Dtype>().data(), sizeof(Dtype) * tensor.shape().num_elements());
  }

  static void tfTensorsToEigenMatrices(const std::vector<tensorflow::Tensor> &input, std::vector<MatrixXD> &output) {
    output.clear();
    for (auto &element : input) {
      MatrixXD matrix;
      tfTensorToEigenMatrix(element, matrix);
      output.push_back(matrix);
    }
  }

  static void tfTensorsToEigenTensors(const std::vector<tensorflow::Tensor> &input, std::vector<Tensor3D> &output) {
    output.clear();
    for (auto &element : input) {
      Tensor3D Eigtensor;
      tfTensorToEigenTensor(element, Eigtensor);
      output.push_back(Eigtensor);
    }
  }

  static void namedEigenMatricesToNamedTFTensors(
      const std::vector<std::pair<std::string, MatrixXD>> &input,
      std::vector<std::pair<std::string, tensorflow::Tensor>> &output) {
    output.clear();
    for (auto &element : input) {
      tensorflow::Tensor tensor(getTensorFlowDataType(),
                                tensorflow::TensorShape({element.second.cols(), element.second.rows()}));
      eigenMatrixToTFTensor(element.second, tensor);
      output.push_back(std::pair<std::string, tensorflow::Tensor>(element.first, tensor));
    }
  }
  static void namedEigenTensorsToNamedTFTensors(
      const std::vector<std::pair<std::string, Tensor3D >> &input,
      std::vector<std::pair<std::string, tensorflow::Tensor>> &output) {
    output.clear();
    for (auto &element : input) {
      tensorflow::Tensor tensor(getTensorFlowDataType(),
                                tensorflow::TensorShape({element.second.dimension(2), element.second.dimension(1),
                                                         element.second.dimension(0)}));
      EigenTensorTotfTensor(element.second, tensor);
      output.push_back(std::pair<std::string, tensorflow::Tensor>(element.first, tensor));
    }
  }

///////////////////////////////////////
  void getLP(VectorXD &parameterVector) {
    auto status = this->session->Run({}, {"LP"}, {}, &LPvec_);
    parameterVector.resize(numOfLP_, 1);
    memcpy(parameterVector.data(), LPvec_[0].flat<Dtype>().data(), sizeof(Dtype) * numOfLP_);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void getLP(tensorflow::Tensor &param) const {
    std::vector<tensorflow::Tensor> paramV;
    auto status = this->session->Run({}, {"LP"}, {}, &paramV);
    param = paramV[0];
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void setLP(const VectorXD &param) {
    LOG_IF(FATAL, param.rows() != numOfLP_) << "the param dimensions don't match";
    memcpy(LPvec_[0].flat<Dtype>().data(), param.data(), sizeof(Dtype) * numOfLP_);
    auto status = this->session->Run({{"LP_placeholder", LPvec_[0]}}, {}, {"assignLP"}, nullptr);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void getAP(VectorXD &param) {
    auto status = this->session->Run({}, {"AP"}, {}, &APvec_);
    param.resize(numOfAP_, 1);
    memcpy(param.data(), APvec_[0].flat<Dtype>().data(), sizeof(Dtype) * numOfAP_);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void getAP(tensorflow::Tensor &param) const {
    std::vector<tensorflow::Tensor> paramV;
    auto status = this->session->Run({}, {"AP"}, {}, &paramV);
    param = paramV[0];
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

  void setAP(const VectorXD &param) {
    LOG_IF(FATAL, param.rows() != numOfAP_) << "the param dimensions don't match";
    memcpy(APvec_[0].flat<Dtype>().data(), param.data(), sizeof(Dtype) * numOfAP_);
    auto status = this->session->Run({{"AP_placeholder", APvec_[0]}}, {}, {"assignAP"}, nullptr);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }
  int getGlobalStep(){
    std::vector<tensorflow::Tensor> globalStep;
    auto status = this->session->Run({}, {"global_step"}, {}, &globalStep);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    return globalStep[0].scalar<int>()();
  }

  tensorflow::Session *session;
  std::vector<std::string> namesOfAllParameters;

  static tensorflow::DataType getTensorFlowDataType() {
    if (typeid(Dtype) == typeid(float))
      return tensorflow::DataType::DT_FLOAT;
    else if (typeid(Dtype) == typeid(double))
      return tensorflow::DataType::DT_DOUBLE;
    LOG(FATAL) << "TensorFlowNeuralNetwork is only implemented for float and double";
  }

  int getAPSize() const { return numOfAP_; }
  int getLPSize() const { return numOfLP_; }

 protected:
  tensorflow::GraphDef graphDef_;
  std::vector<std::string> numericsOpNames_;
  bool checkNumerics_;
  std::vector<tensorflow::Tensor> APvec_;
  std::vector<tensorflow::Tensor> LPvec_;
  int numOfAP_;
  int numOfLP_;

  /*
   * Method used in the different constructors
   * @graphDef: tensorflow::GraphDef to be used
   * @n_inter_op_parallelism_threads: The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads.
   * @logDevicePlacement: If true, Upon reading the GraphDef, TensorFlow shows where (which CPU or GPU) a particular op is assigned
   */
  void
  construct(tensorflow::GraphDef graphDef, int n_inter_op_parallelism_threads = 0, bool logDevicePlacment = false) {
    tensorflow::ConfigProto configProto;
    configProto.mutable_gpu_options()->set_allow_growth(true);
    configProto.set_allow_soft_placement(true);
    configProto.set_log_device_placement(logDevicePlacment);
    if (n_inter_op_parallelism_threads > 0)
      configProto.set_inter_op_parallelism_threads(n_inter_op_parallelism_threads);
    tensorflow::SessionOptions sessionOptions;
    sessionOptions.env = tensorflow::Env::Default();
    sessionOptions.config = configProto;
    tensorflow::NewSession(sessionOptions, &session);
    graphDef_ = graphDef;
    // Add the graph to the session
    auto status = session->Create(graphDef);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    // Initialize all variables
    for (int i = 0; i < graphDef_.node_size(); ++i)
      numericsOpNames_.push_back(graphDef_.node(i).name());
    checkNumerics_ = false;
    updateParams();
  }

 private:
  void updateParams() {
    auto status = session->Run({}, {}, {"initializeAllVariables"}, {});
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    std::vector<tensorflow::Tensor> netShape;
    status = session->Run({}, {"numberOfAP", "numberOfLP"}, {}, &netShape);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    numOfAP_ = netShape[0].scalar<int>()();
    numOfLP_ = netShape[1].scalar<int>()();
    APvec_.clear();
    LPvec_.clear();
    status = this->session->Run({}, {"AP"}, {}, &APvec_);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
    status = this->session->Run({}, {"LP"}, {}, &LPvec_);
    LOG_IF(FATAL, !status.ok()) << status.ToString();
  }

};
} // namespace FuncApprox
} // namespace rai




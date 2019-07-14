#include "rai/function/common/ValueFunction.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim>
class ValueFunction_TensorFlow : public virtual ParameterizedFunction_TensorFlow<Dtype, stateDim, 1>,
                                 public virtual ValueFunction<Dtype, stateDim> {

 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;

  using ValueFunctionBase = ValueFunction<Dtype, stateDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, 1>;

  typedef typename ValueFunctionBase::State State;
  typedef typename ValueFunctionBase::StateBatch StateBatch;
  typedef typename ValueFunctionBase::Value Value;
  typedef typename ValueFunctionBase::ValueBatch ValueBatch;
  typedef typename ValueFunctionBase::Gradient Gradient;
  typedef typename ValueFunctionBase::Jacobian Jacobian;

  typedef typename ValueFunctionBase::Tensor1D Tensor1D;
  typedef typename ValueFunctionBase::Tensor2D Tensor2D;
  typedef typename ValueFunctionBase::Tensor3D Tensor3D;

  ValueFunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  ValueFunction_TensorFlow(std::string computeMode,
                           std::string graphName,
                           std::string graphParam,
                           Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "Vfunction", computeMode, graphName, graphParam, learningRate) {
  }

  ~ValueFunction_TensorFlow() {};

  virtual void forward(Tensor2D &states, Tensor2D &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"value"}, vectorOfOutputs);
    values.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual void forward(Tensor3D &states, Tensor2D &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"value"}, vectorOfOutputs);
    values.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor2D &values) {
    std::vector<MatrixXD> loss;
    this->tf_->run({states,
                    values},
                   {"trainUsingTargetValue/loss"},
                   {"trainUsingTargetValue/solver"}, loss);
    return loss[0](0);
  }
  virtual Dtype performOneSolverIter_trustregion(Tensor3D &states, Tensor2D &values, Tensor2D &old_values) {
    std::vector<MatrixXD> loss;
    this->tf_->run({states,
                    values,
                    old_values},
                   {"trainUsingTRValue/loss"},
                   {"trainUsingTRValue/solver"}, loss);
    return loss[0](0);
  }
  virtual Dtype performOneSolverIter_infimum(StateBatch &states, ValueBatch &values, Dtype linSlope) {
    std::vector<MatrixXD> loss, dummy;
    auto slope = Eigen::Matrix<Dtype, 1, 1>::Constant(linSlope);
    this->tf_->run({{"state", states},
                    {"targetValue", values},
                    {"trainUsingTargetValue_inifimum/linSlope", slope},
                    {"updateBNparams", this->notUpdateBN}},
                   {"trainUsingTargetValue_inifimum/loss"},
                   {"trainUsingTargetValue_inifimum/solver"}, loss);
    return loss[0](0);
  }

  virtual void setClipRate(const Dtype param_in){
    std::vector<MatrixXD> dummy;
    VectorXD input(1);
    input << param_in;
    this->tf_->run({{"param_assign_placeholder", input}}, {}, {"clip_param_assign"}, dummy);

  }

  virtual void setClipRangeDecay(const Dtype decayRate){
    std::vector<MatrixXD> dummy;
    VectorXD input(1);
    input << decayRate;
    this->tf_->run({{"param_assign_placeholder", input}}, {}, {"clip_decayrate_assign"}, dummy);

  }

  virtual Dtype test(Tensor3D &states, Tensor2D &values, Tensor2D &old_values, Eigen::Matrix<Dtype,-1,-1> &testout) {
    std::vector<MatrixXD> test;
    this->tf_->run({states,
                    values,
                    old_values},
                   {"test"},
                   {"trainUsingTRValue/solver"}, test);
    testout = test[0];
    return test[0](0);
  }

};
} // namespace FuncApprox
} // namespace rai

//
// Created by jhwangbo on 23.01.17.
//

#include <iostream>
#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>
#include <bazel-tensorflow/external/eigen_archive/Eigen/src/Core/DenseBase.h>

#include "rai/RAI_core"

using std::cout;
using std::endl;
using std::cin;

using rai::FuncApprox::ParameterizedFunction_TensorFlow;
using rai::FuncApprox::Qfunction_TensorFlow;
using rai::FuncApprox::DeterministicPolicy_TensorFlow;
using rai::FuncApprox::ValueFunction_TensorFlow;

using Dtype = double;

using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
using VectorXD = Eigen::Matrix<Dtype, -1, 1>;

double training_mean = 50.0;
double training_variance = 100.0;

using namespace rai;

Dtype sample(double dummy) {
  static std::mt19937 rng;
  static std::normal_distribution<Dtype> nd(training_mean, sqrt(training_variance));
  return nd(rng);
}

int main() {
  RAI_init();

  bool testCopyStructureCopyWeightAndWeightInterpolation =false;
  bool testPolicyTensorFlowTrainCiriticWithSeparateCritic = false;
  bool testJacobianActionWRTState = true;
  bool testJacobianActionWRTParam = false;

  constexpr int StateDim = 5;
  constexpr int ActionDim = 3;

  using Policy_ = Policy_TensorFlow <Dtype, StateDim, ActionDim>;
  using Input = Policy_::Input;
  using Output = Policy_::Output;
  using InputBatch = Policy_::InputBatch;
  using OutputBatch = Policy_::OutputBatch;
  using State = Policy_::State;
  using StateBatch =  Policy_::StateBatch;
  using Action =  Policy_::Action;
  using ActionBatch =  Policy_::ActionBatch;

  Policy_ policy("cpu", "MLP", "tanh 1e-3 2 32 32 1", 0.001);
  Policy_ policy2("cpu", "MLP", "tanh 1e-3 2 32 32 1", 0.001);

  int batchSize = 1;
  StateBatch stateBatch = StateBatch::Random(StateDim, batchSize);
  ActionBatch actionBatch = ActionBatch::Random(ActionDim, batchSize);
  ActionBatch actionBatch2 = ActionBatch::Random(ActionDim, batchSize);

  VectorXD param1, param2;

  policy.getLP(param1);
  policy2.setLP(param1);
  policy2.getLP(param2);

  LOG(INFO) << "LP diff" << (param1-param2).norm()/param1.norm()*100;

  policy.forward(stateBatch,actionBatch);
  policy2.forward(stateBatch,actionBatch2);

  LOG(INFO) << "fwd diff" << (actionBatch-actionBatch2).norm()/actionBatch.norm()*100;



  if (testCopyStructureCopyWeightAndWeightInterpolation)
  {
    cout << "Test: Policy::copyStructureFrom" << endl;

    constexpr int inputDimension = 2;
    constexpr int outputDimension = 2;

    using Policy_TensorFlow = Policy_TensorFlow <Dtype, inputDimension, outputDimension>;
    using Input = Policy_TensorFlow::Input;
    using Output = Policy_TensorFlow::Output;
    using InputBatch = Policy_TensorFlow::InputBatch;
    using OutputBatch = Policy_TensorFlow::OutputBatch;

//    Policy_TensorFlow policy1("generatedPB/policy_2l.pb", 0.001);
//    Policy_TensorFlow policy2("generatedPB/policy_2l.pb", 0.001);

    Policy_TensorFlow policy1("Deterministic_pol","tanh",{32,32,32,32}, 0.001);
    Policy_TensorFlow policy2("Deterministic_pol","tanh",{32,32,32,32}, 0.001);

    policy2.copyStructureFrom(&policy1);

    using Input = Policy_TensorFlow::Input;
    using Output = Policy_TensorFlow::Output;
    using InputBatch = Policy_TensorFlow::InputBatch;
    using OutputBatch = Policy_TensorFlow::OutputBatch;

    int input_dim = 2;
    int output_dim = 2;
    int batchSize = 3072;

    // generate data for testing///
    InputBatch data_x = InputBatch::Random(input_dim, 3072 * 2);
    OutputBatch data_y(output_dim, 3072 * 2);
    OutputBatch data_yNN(output_dim, 3072 * 2);
    OutputBatch data_yNN2(output_dim, 3072 * 2);
    MatrixXD data_yNN2row1(1, 3072 * 2);

    MatrixXD data_x0(1, 3072 * 2);
    MatrixXD data_x1(1, 3072 * 2);
    MatrixXD data_y0(1, 3072 * 2);
    MatrixXD data_y1(1, 3072 * 2);
    MatrixXD data_y0NN(1, 3072 * 2);
    MatrixXD data_y1NN(1, 3072 * 2);

    for (int i = 0; i < 3072 * 2; i++) {
      data_y(0, i) = sin(data_x(0, i) * 2.0);
      data_y(1, i) = cos(data_x(1, i) * 4.0);
    }

    data_x0 = data_x.row(0);
    data_x1 = data_x.row(1);
    data_y0 = data_y.row(0);
    data_y1 = data_y.row(1);

    int nIterations = 5000;

    for (int iteration = 0; iteration < nIterations; ++iteration) {
      Dtype loss = policy1.performOneSolverIter(data_x, data_y);
      if (iteration % 100 == 0)
        cout << iteration <<", loss: "<<loss<< endl;
    }

    policy1.forward(data_x, data_yNN);
    data_y0NN = data_yNN.row(0);
    data_y1NN = data_yNN.row(1);
    Utils::Graph::FigProp3D figure1properties;
    Utils::graph->figure3D(1, figure1properties);
    Utils::graph->append3D_Data(1, data_x0.data(), data_x1.data(), data_y1NN.data(), data_y1NN.cols(), false, Utils::Graph::PlotMethods3D::points, "Output by neural network");
    Utils::graph->append3D_Data(1, data_x0.data(), data_x1.data(), data_y1.data(), data_y1.cols(), false, Utils::Graph::PlotMethods3D::points, "ground Truth");

    VectorXD param1, param2;
    policy1.getAP(param1);
    policy2.getAP(param2);

    cout<<"testing interpolation"<<endl;
    cout<<"from cpp calculation "<<endl<<(param1 * 0.2 + param2 * 0.8).transpose()<<endl;
    policy2.interpolateAPWith(&policy1, 0.2);
    policy2.getAP(param2);
    cout<<"from tensorflow "<<endl<<param2.transpose()<<endl;
    cout << "Press Enter if the two vectors are the same" << endl;
    cin.get();


    policy2.copyAPFrom(&policy1);
    policy2.forward(data_x, data_yNN2);
    data_yNN2row1 = data_yNN2.row(1);
    Utils::graph->append3D_Data(1, data_x0.data(), data_x1.data(), data_yNN2row1.data(), data_yNN2row1.cols(), false, Utils::Graph::PlotMethods3D::points, "Output by another Network");

    Utils::graph->drawFigure(1);

    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  if(testPolicyTensorFlowTrainCiriticWithSeparateCritic)
  {
    cout << "Test: Policy_TensorFlow::trainUsingCritic by first training a critic" << endl;

    constexpr int stateDimension = 2;
    constexpr int actionDimension = 2;

    using Qfunction_TensorFlow_ = Qfunction_TensorFlow<Dtype, stateDimension, actionDimension>;
    using Policy_TensorFlow_ = Policy_TensorFlow<Dtype, stateDimension, actionDimension>;
//    Qfunction_TensorFlow_ qFunction("generatedPB/Qfunction_2l.pb", 1e-3);
//    Qfunction_TensorFlow_ qFunction("Qfunction", "relu",{64,64},5e-3 ,0.001);
    Qfunction_TensorFlow_ qFunction("Qfunction_2l", "2 2 relu 64 64 5e-3" ,0.001);

    using State = Qfunction_TensorFlow_::State;
    using StateBatch = Qfunction_TensorFlow_::StateBatch;
    using Action = Qfunction_TensorFlow_::Action;
    using ActionBatch = Qfunction_TensorFlow_::ActionBatch;
    using Value = Qfunction_TensorFlow_::Value;
    using ValueBatch = Qfunction_TensorFlow_::ValueBatch;

    int nIterations = 50000;
    int batchSize = 1500;
    for (int iteration = 0; iteration < nIterations; ++iteration) {
      StateBatch stateBatch = StateBatch::Random(stateDimension, batchSize);
      ActionBatch actionBatch = ActionBatch::Random(actionDimension, batchSize);
      ValueBatch qValueTargetBatch = ((stateBatch.array()*4).sin() - actionBatch.array()).square().colwise().sum();
      LOG(INFO) << qValueTargetBatch;
      Dtype  loss = qFunction.performOneSolverIter(stateBatch, actionBatch, qValueTargetBatch);
      if (iteration % 1000 == 0) cout << iteration << "loss : "<< loss << endl;
    }

    Utils::Graph::FigProp3D figure1properties;
//    Policy_TensorFlow_ policy("Deterministic_pol","relu",{32,32}, 0.001,5e-5);
    Policy_TensorFlow_ policy("policy_2l","2 2 relu 64 64 1e-3",5e-5);

    StateBatch stateBatchTest = StateBatch::Random(stateDimension, batchSize);
    MatrixXD stateBatchTest0 = stateBatchTest.row(0);
    MatrixXD stateBatchTest1 = stateBatchTest.row(1);
    ActionBatch actionBatchTest;
    MatrixXD actionBatchTest1;
    nIterations = 50000;
    for (int iteration = 0; iteration < nIterations; ++iteration) {
      StateBatch stateBatch = StateBatch::Random(stateDimension, batchSize);
      Dtype averageQ = policy.backwardUsingCritic(&qFunction, stateBatch);
      if (iteration % 100 == 0) {
        cout << iteration <<", average Q: "<< averageQ << endl;
        policy.forward(stateBatchTest, actionBatchTest);
        actionBatchTest1 = actionBatchTest.row(1);
        Utils::graph->figure3D(1, figure1properties);
        Utils::graph->append3D_Data(1, stateBatchTest0.data(), stateBatchTest.data(), actionBatchTest1.data(), batchSize,
                                    false, Utils::Graph::PlotMethods3D::points, "Output by neural network");
        MatrixXD groundTruthSin;
        groundTruthSin = (stateBatchTest1.array()*4).sin();
        Utils::graph->append3D_Data(1, stateBatchTest0.data(), stateBatchTest1.data(), groundTruthSin.data(), batchSize,
                                    false, Utils::Graph::PlotMethods3D::points, "Ground truth");
        Utils::graph->drawFigure(1);
      }
    }
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  if (testJacobianActionWRTParam) {
    constexpr int stateDimension = 2;
    constexpr int actionDimension = 2;
    using Policy_TensorFlow_ = Policy_TensorFlow<Dtype, stateDimension, actionDimension>;
    using State = Policy_TensorFlow_::State;
    using StateBatch = Policy_TensorFlow_::StateBatch;
    using Action = Policy_TensorFlow_::Action;
    using ActionBatch = Policy_TensorFlow_::ActionBatch;
    using Jaco = Policy_TensorFlow_::JacobianWRTparam;

    Policy_TensorFlow_ policy("Deterministic_pol","tanh",{32,32,32,32}, 0.001);

    int paramLength = policy.getAPSize();
    Jaco jaco_jaco(2, paramLength), jaco_num(2, paramLength);
    State input_jaco; Action output_jaco;
    input_jaco << 1.2, 0.9;
    jaco_jaco.setZero();
    policy.getJacobianAction_WRT_LP(input_jaco, jaco_jaco);

    policy.forward(input_jaco, output_jaco);
    VectorXD param(paramLength), paramNew(paramLength);
    Action outputTestjaco;
    policy.getLP(param);

    for (int i = 0; i < paramLength; i++) {
      paramNew = param;
      paramNew(i) += Dtype(1e-7);
      policy.setLP(paramNew);
      policy.forward(input_jaco, outputTestjaco);
      jaco_num.col(i) = (outputTestjaco - output_jaco) / 1e-7;
    }

    cout << "jaco Action wrt param" << endl;
    cout << "jaco from TF is" << endl << jaco_jaco << endl;
    cout << "jaco from numerical is" << endl << jaco_num << endl;
    cout << "the end should be identity, and the rest should be alternating zero columns" << endl;
  }

  if (testJacobianActionWRTState) {
    constexpr int stateDimension = 2;
    constexpr int actionDimension = 2;
    using Policy_TensorFlow_ = Policy_TensorFlow<Dtype, stateDimension, actionDimension>;
    using State = Policy_TensorFlow_::State;
    using StateBatch = Policy_TensorFlow_::StateBatch;
    using Action = Policy_TensorFlow_::Action;
    using ActionBatch = Policy_TensorFlow_::ActionBatch;
    using Jaco = Policy_TensorFlow_::JacobianWRTstate;

    Policy_TensorFlow_ policy("Deterministic_pol","tanh",{32,32,32,32}, 0.001);

    Jaco jaco_jaco, jaco_num;
    State input_jaco; Action output_jaco;
    input_jaco << 1.2, 0.9;
    jaco_jaco.setZero();
    policy.getJacobianAction_WRT_State(input_jaco, jaco_jaco);

    policy.forward(input_jaco, output_jaco);
    Action outputTestjaco;

    for (int i = 0; i < stateDimension; i++) {
      input_jaco(i) += Dtype(1e-7);
      policy.forward(input_jaco, outputTestjaco);
      input_jaco(i) -= Dtype(1e-7);
      jaco_num.col(i) = (outputTestjaco - output_jaco) / 1e-7;
    }

    cout << "jaco from TF is" << endl << jaco_jaco << endl;
    cout << "jaco from numerical is" << endl << jaco_num << endl;
    cout << "the end should be identity, and the rest should be alternating zero columns" << endl;
  }


  cout << "Tests done" << endl;


}
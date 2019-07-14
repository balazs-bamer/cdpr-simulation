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


#include "rai/RAI_core"

using std::cout;
using std::endl;
using std::cin;

using rai::FuncApprox::ParameterizedFunction_TensorFlow;
using rai::FuncApprox::Qfunction_TensorFlow;
using rai::FuncApprox::DeterministicPolicy_TensorFlow;
using rai::FuncApprox::ValueFunction_TensorFlow;

using Dtype = double;

using MatrixXD = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXD = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;

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
  constexpr int stateDimension = 3;
  constexpr int actionDimension = 3;

  bool testCopyStructureCopyWeightAndWeightInterpolation = true;

  using Qfunction_TensorFlow = Qfunction_TensorFlow <Dtype, stateDimension, actionDimension>;
  using State = Qfunction_TensorFlow::State;
  using Action = Qfunction_TensorFlow::Action;
  using Value = Qfunction_TensorFlow::Value;
  using ValueBatch = Qfunction_TensorFlow::ValueBatch;
  using StateBatch = Qfunction_TensorFlow::StateBatch;
  using ActionBatch = Qfunction_TensorFlow::ActionBatch;
  using GradientBatch = Qfunction_TensorFlow::ActionBatch;
  int batchSize = 1500;

  StateBatch stateBatch = StateBatch::Random(stateDimension, batchSize);
  ActionBatch actionBatch = ActionBatch::Random(actionDimension, batchSize);
  ValueBatch qValueTargetBatch = ((stateBatch.array()*4).sin()).square().colwise().sum();
  ValueBatch Qtest,Qtest2;
  Qfunction_TensorFlow Q("cpu", "MLP2", "relu 1e-3 3 3 32 32 1", 0.001);
  Qfunction_TensorFlow Qo("cpu", "MLP2", "relu 1e-3 3 3 32 32 1", 0.001);
  VectorXD param,paramo;

  LOG(INFO) << "param # :" <<   Qo.getLPSize()- Q.getLPSize()<< ", "<< Qo.getAPSize()-Q.getAPSize();

  Q.getAP(param);
  Qo.setAP(param);
  Qo.getAP(paramo);
  LOG(INFO) << "param diff :" <<(param - paramo).norm();

  Q.forward(stateBatch,actionBatch,Qtest);
  Qo.forward(stateBatch,actionBatch,Qtest2);

  LOG(INFO) << "fwd diff :" <<(Qtest - Qtest2).norm()/Qtest2.norm();


  if (testCopyStructureCopyWeightAndWeightInterpolation)
  {
    cout << "Test: Policy::copyStructureFrom" << endl;


    Qfunction_TensorFlow qfunction1("cpu", "MLP2", "relu 5e-3 3 3 2 2 2 2 1", 0.001);
    Qfunction_TensorFlow qfunction2("cpu", "MLP2", "relu 5e-3 3 3 2 2 2 2 1", 0.001);

    qfunction2.copyStructureFrom(&qfunction1);

    int nIterations = 5000;
    int batchSize = 1500;
    for (int iteration = 0; iteration < nIterations; ++iteration) {
      StateBatch stateBatch = StateBatch::Random(stateDimension, batchSize);
      ActionBatch actionBatch = ActionBatch::Random(actionDimension, batchSize);
      ValueBatch qValueTargetBatch = ((stateBatch.array()*4).sin() - actionBatch.array()).square().colwise().sum();
//      ValueBatch qValueTargetBatch = ((stateBatch.array()*4).sin()).square().colwise().sum();
      Dtype loss = qfunction1.performOneSolverIter(stateBatch, actionBatch, qValueTargetBatch);
      if (iteration % 100 == 0) cout << iteration << ", loss = "<<loss<< endl;
    }
    StateBatch stateBatchTest = StateBatch::Random(stateDimension, batchSize);
    ActionBatch actionBatchTest = ActionBatch::Random(actionDimension, batchSize);
    ValueBatch valueBatchTest = ValueBatch::Random(1, batchSize);
    MatrixXD stateBatchTest0 = stateBatchTest.row(0);
    MatrixXD stateBatchTest1 = stateBatchTest.row(1);
    qfunction1.forward(stateBatchTest, actionBatchTest, valueBatchTest);
    Utils::Graph::FigProp3D figure1properties;
    Utils::graph->figure3D(1, figure1properties);
    Utils::graph->append3D_Data(1, stateBatchTest0.data(), stateBatchTest1.data(), valueBatchTest.data(), valueBatchTest.cols(), false, Utils::Graph::PlotMethods3D::points, "Output by neural network");

    VectorXD param1, param2;
    qfunction1.getAP(param1);
    qfunction2.getAP(param2);

    cout<<"testing interpolation"<<endl;
    cout<<"from cpp calculation "<<endl<<(param1 * 0.2 + param2 * 0.8).transpose()<<endl;
    qfunction2.interpolateAPWith(&qfunction1, 0.2);
    qfunction2.getAP(param2);
    cout<<"from tensorflow "<<endl<<param2.transpose()<<endl;
    cout << "Press Enter if the two vectors are the same" << endl;
    cin.get();


    qfunction2.copyAPFrom(&qfunction1);
    qfunction2.forward(stateBatchTest, actionBatchTest, valueBatchTest);
    Utils::graph->append3D_Data(1, stateBatchTest0.data(), stateBatchTest1.data(), valueBatchTest.data(), valueBatchTest.cols(), false, Utils::Graph::PlotMethods3D::points, "Output by another");
    Utils::graph->drawFigure(1);


    StateBatch stateBatch = StateBatch::Random(stateDimension, 3);
    ActionBatch actionBatch = ActionBatch::Random(actionDimension, 3);
    GradientBatch gradientBatch = GradientBatch::Random(actionDimension, 3);
    GradientBatch gradientBatchNUM = GradientBatch::Random(actionDimension, 3);

    ValueBatch originalValue = ValueBatch::Random(1, 3);
    ValueBatch newValue = ValueBatch::Random(1, 3);

    qfunction2.getGradient_AvgOf_Q_wrt_action(stateBatch, actionBatch, gradientBatch);

    qfunction2.forward(stateBatch, actionBatch, originalValue);

    for (int i = 0; i < actionDimension; i++) {
      for (int j = 0; j < 3; j++) {
        actionBatch(i, j) += Dtype(1e-7);
        qfunction2.forward(stateBatch, actionBatch, newValue);
        actionBatch(i, j) -= Dtype(1e-7);
        gradientBatchNUM(i, j) = (newValue - originalValue)(j) / 1e-7;
      }
    }

    cout << "jaco from TF is" << endl << gradientBatch << endl;
    cout << "jaco from numerical is" << endl << gradientBatchNUM / 3 << endl;
    cout << "the two should be identical" << endl;
    cout << "Press Enter to continue if the jacos's are the same" << endl;
    cin.get();
  }


  cout << "Tests done" << endl;


}
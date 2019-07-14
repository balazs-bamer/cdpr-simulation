//
// Created by jhwangbo on 23.01.17.
//

#include <iostream>
#include "rai/function/tensorflow/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>
#include <rai/RAI_core>
#include "math/RandomNumberGenerator.hpp"
#include "rai/RAI_core"

using std::cout;
using std::endl;
using std::cin;

using rai::FuncApprox::ParameterizedFunction_TensorFlow;
using rai::FuncApprox::Qfunction_TensorFlow;
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

  bool testCopyStructureCopyWeightAndWeightInterpolation = true;

  if (testCopyStructureCopyWeightAndWeightInterpolation)
  {
    cout << "Test: Value::copyStructureFrom" << endl;

    constexpr int stateDimension = 2;

    using ValueFunction_TensorFlow = ValueFunction_TensorFlow <Dtype, stateDimension>;
    using State = ValueFunction_TensorFlow::State;
    using Value = ValueFunction_TensorFlow::Value;
    using ValueBatch = ValueFunction_TensorFlow::ValueBatch;
    using StateBatch = ValueFunction_TensorFlow::StateBatch;

    ValueFunction_TensorFlow valueF1("cpu", "MLP", "tanh 1e-3 2 32 32 1", 0.001);
    ValueFunction_TensorFlow valueF2("cpu", "MLP", "tanh 1e-3 2 32 32 1", 0.001);
    valueF2.copyStructureFrom(&valueF1);

    int nIterations = 5000;
    int batchSize = 1500;
    for (int iteration = 0; iteration < nIterations; ++iteration) {
      StateBatch stateBatch = StateBatch::Random(stateDimension, batchSize);
      ValueBatch valueTargetBatch = stateBatch.array().square().colwise().sum();
      Dtype loss = valueF1.performOneSolverIter(stateBatch, valueTargetBatch);
      if (iteration % 100 == 0) cout << iteration << ", loss = "<<loss<< endl;
    }
    StateBatch stateBatchTest = StateBatch::Random(stateDimension, batchSize);
    ValueBatch valueBatchTest = ValueBatch::Random(1, batchSize);
    MatrixXD stateBatchTest0 = stateBatchTest.row(0);
    MatrixXD stateBatchTest1 = stateBatchTest.row(1);
    valueF1.forward(stateBatchTest, valueBatchTest);
    Utils::Graph::FigProp3D figure1properties;
    Utils::graph->figure3D(1, figure1properties);
    Utils::graph->append3D_Data(1, stateBatchTest0.data(), stateBatchTest1.data(), valueBatchTest.data(), valueBatchTest.cols(), false, Utils::Graph::PlotMethods3D::points, "Output by neural network");

    VectorXD param1, param2;
    valueF1.getAP(param1);
    valueF2.getAP(param2);

    cout<<"testing interpolation"<<endl;
    cout<<"from cpp calculation "<<endl<<(param1 * 0.2 + param2 * 0.8).transpose()<<endl;
    valueF2.interpolateAPWith(&valueF1, 0.2);
    valueF2.getAP(param2);
    cout<<"from tensorflow "<<endl<<param2.transpose()<<endl;
    cout << "Press Enter if the two vectors are the same" << endl;
    cin.get();


    valueF2.copyAPFrom(&valueF1);
    valueF2.forward(stateBatchTest, valueBatchTest);
    Utils::graph->append3D_Data(1, stateBatchTest0.data(), stateBatchTest1.data(), valueBatchTest.data(), valueBatchTest.cols(), false, Utils::Graph::PlotMethods3D::points, "Output by another");
    Utils::graph->drawFigure(1);

    cout << "Press Enter to continue if the graphs are the same" << endl;
    cin.get();
  }

  cout << "Tests done" << endl;


}
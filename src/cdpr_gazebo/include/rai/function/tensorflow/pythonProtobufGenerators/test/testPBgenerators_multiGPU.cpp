//
// Created by jhwangbo on 23.01.17.
//
#include "rai/RAI_core"
#include "glog/logging.h"
#include <iostream>
#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"

using std::cout;
using std::endl;
using std::cin;

using rai::FuncApprox::ParameterizedFunction_TensorFlow;
using rai::FuncApprox::Qfunction_TensorFlow;
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
  bool testCalctime = true;

  constexpr int StateDim = 10;
  constexpr int ActionDim = 5;

  using Policy_ = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
  using PolicyBase = rai::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;

  Policy_ policy("gpu,0", "MLP", "tanh 1e-3 10 320 320 5", 0.001);

  int batchSize = 1000;
  StateBatch stateBatch = StateBatch::Random(StateDim, batchSize);
  ActionBatch actionBatch = ActionBatch::Random(ActionDim, batchSize);
  ActionBatch actionBatch2 = ActionBatch::Random(ActionDim, batchSize);
  ValueBatch advs = ValueBatch::Random(1, batchSize);
  VectorXD param1, param2;
  Action stdev;

  policy.getLP(param1);

  if (testCalctime) {
    VectorXD policy_grad = VectorXD::Zero(param1.rows());
    VectorXD FVP = VectorXD::Zero(param1.rows());
    VectorXD fullstep = VectorXD::Zero(param1.rows());

    policy.getLP(param1);
    policy.getStdev(stdev);

    Utils::timer->enable();

      Utils::timer->startTimer("Gradient");
        policy.TRPOpg(stateBatch, actionBatch, actionBatch2, advs, stdev, policy_grad);
      Utils::timer->stopTimer("Gradient");

    Utils::timer->disable();

    rai::Utils::Graph::FigPropPieChart propChart;
    graph->drawPieChartWith_RAI_Timer(1, timer->getTimedItems(), propChart);
    graph->drawFigure(1);
    graph->waitForEnter();

  }

  cout << "Tests done" << endl;

}
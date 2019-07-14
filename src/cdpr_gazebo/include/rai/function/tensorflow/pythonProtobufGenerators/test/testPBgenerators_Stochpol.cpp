//
// Created by joonho on 23.03.17.
//


#include <iostream>
#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/common/Policy.hpp"
#include "raiCommon/math/RAI_math.hpp"

#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>

#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"

using std::cout;
using std::endl;
using std::cin;
const int ActionDim = 2;
const int StateDim = 3;
using Dtype = float;

using NormNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

using PolicyBase = rai::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
using stPolicy = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;

using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
using VectorXD = Eigen::Matrix<Dtype, -1, 1>;
using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
typedef typename PolicyBase::State State;
typedef typename PolicyBase::StateBatch StateBatch;
typedef typename PolicyBase::Action Action;
typedef typename PolicyBase::ActionBatch ActionBatch;
typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;

using namespace rai;
//
//Dtype sample(double dummy) {
//  static std::mt19937 rng;
//  static std::normal_distribution<Dtype> nd(training_mean, sqrt(training_variance));
//  return nd(rng);
//}

int main() {
  RAI_init();
  bool teststdev = false;
  bool testgradient = false;
  const int sampleN = 5;
  stPolicy policy("gpu,0", "MLP", "relu 1e-5 3 32 2", 0.001);


  StateBatch stateBatch = StateBatch::Random(StateDim, sampleN);
  ActionBatch actionBatch;
  ActionBatch actionNoise = ActionBatch::Random(ActionDim, sampleN);
  ActionBatch actionmean;
  ActionBatch actionmeanNew;
  Action Stdev_o, StdevNew;
  State state = State::Ones(StateDim);
  Action action, action2;

  VectorXD parameter, testgrad, temp;
  MatrixXD test;
  Advantages advs = Eigen::Matrix<Dtype, 1, sampleN>::Random(1, sampleN);
  Dtype loss, loss2;

  policy.forward(stateBatch, actionmean);
  std::cout << actionmean <<std::endl;

  actionBatch = actionmean + actionNoise;

  rai::Tensor<Dtype, 3> ActionTensor;
  rai::Tensor<Dtype, 3> ActionNTensor;

  rai::Tensor<Dtype, 3> StateTensor3;
  rai::Tensor<Dtype, 1> len;

//  StateTensor = "state";
  StateTensor3 = "state";
  ActionTensor = "sampledAction";
  ActionNTensor = "actionNoise";
  len = "length";
  StateTensor3.resize(stateBatch.rows(),1,stateBatch.cols());
  StateTensor3.copyDataFrom(stateBatch);

//  StateTensor.resize(stateBatch.rows(),stateBatch.cols());
//  StateTensor = stateBatch;
  ActionTensor.resize(actionBatch.rows(),1,actionBatch.cols());
  ActionNTensor.resize(actionBatch.rows(),1,actionBatch.cols());
  Stdev_o.setConstant(1);
  ActionTensor.setConstant(1);

  Utils::logger->addVariableToLog(2, "Stdev", "");

  for (int i = 0; i<1e3 ; i++) {
    timer->startTimer("dd");
    policy.forward(StateTensor3, ActionTensor);
    policy.getLearningRate();
    policy.PPOpg(StateTensor3,ActionTensor,ActionNTensor,advs,Stdev_o,len,parameter);
    policy.PPOpg_kladapt(StateTensor3,ActionTensor,ActionNTensor,advs,Stdev_o,len,parameter);
    policy.getLP(parameter);
    policy.setPPOparams(0.01, 0.1, 0.2);

    parameter.setZero(policy.getLPSize());
    policy.setStdev(Stdev_o);
    policy.trainUsingGrad(parameter);
    policy.PPOgetkl(StateTensor3, ActionTensor, ActionNTensor, Stdev_o, len);

    Utils::logger->appendData("Stdev", i, Stdev_o.norm());
    timer->stopTimer("dd");
  }
  LOG(INFO) << "policy check";


  /////////////////////// Plotting properties ////////////////////////
  rai::Utils::Graph::FigProp2D figurePropertieskl("N. Steps Taken", "d", "Number of Steps Taken vs d");
  rai::Utils::Graph::FigPropPieChart propChart;


//  policy.forward(state_plot, action_plot);
  graph->figure(2, figurePropertieskl);
  graph->appendData(2, logger->getData("Stdev", 0),
                    logger->getData("Stdev", 1),
                    logger->getDataSize("Stdev"),
                    rai::Utils::Graph::PlotMethods2D::linespoints,
                    "klD",
                    "lw 2 lc 4 pi 1 pt 5 ps 1");
  graph->drawFigure(2);
  graph->waitForEnter();

  graph->drawPieChartWith_RAI_Timer(0, timer->getTimedItems(), propChart);
  std::cout << ActionTensor;
  if (teststdev) {

    NoiseCov Cov;

    Cov = NoiseCov::Ones() * 0.5;
    Cov = Cov.array().abs();
    Action Stdev_n;
    NormNoise Normal(Cov);
    Stdev_n = Normal.getCovariance().diagonal();

    std::cout << "New stdev: " << Stdev_n.transpose() << std::endl;
    policy.getStdev(Stdev_o);
    std::cout << "old stdev: " << Stdev_o.transpose() << std::endl;

    policy.setStdev(Stdev_n);
    policy.getStdev(Stdev_o);
  }

  parameter.setZero(policy.getLPSize());
  testgrad.Random(policy.getLPSize());
  policy.getLP(parameter);
//
//  /////// test gradient
//  if (testgradient) {
//    VectorXD p_n, p_old, rate;
//
//    Dtype Loss = 0;
//    int paramsize;
//    VectorXD pgtest, pgtest2;
//    paramsize = policy.getLPSize();
//    p_n.resize(sampleN);
//    p_old.resize(sampleN);
//    rate.resize(sampleN);
//    policy.getStdev(Stdev_o);
//    pgtest.setZero(policy.getLPSize());
//    pgtest2.setZero(policy.getLPSize());
//
//    Dtype stepsize = 1e-5;
//    VectorXD paramnew(paramsize);
//    Dtype loss2;
//    Dtype Err = 0;
//
//    for (int i = 0; i < sampleN; i++) {
//      Action Temp2 = actionNoise.col(i).array() / Stdev_o.array();
//      p_old[i] =
//          -0.5 * Temp2.transpose() * Temp2 - Stdev_o.array().log().sum();//- 0.5 * std::log(2.0 * M_PI) * ActionDim;
//    }
//
//    policy.TRPOpg(stateBatch, actionBatch, actionNoise, advs, Stdev_o, testgrad);
//    loss = policy.TRPOloss(stateBatch, actionBatch, actionNoise, advs, Stdev_o); // original
//    policy.getLP(parameter);
//
//    Loss = 0;
//    std::cout << Stdev_o << std::endl;
//
//    for (int j = 0; j < paramsize; j++) {
//      paramnew = parameter;
//      paramnew(j) += stepsize;
//      policy.setLP(paramnew);
//
//      loss2 = policy.TRPOloss(stateBatch, actionBatch, actionNoise, advs, Stdev_o);  //perturbed
//
//      policy.forward(stateBatch, actionmeanNew);  //new mean
//      policy.getStdev(StdevNew);
//      ActionBatch Noise_new = actionBatch - actionmeanNew;
//
//      Loss = 0;
//      for (int i = 0; i < sampleN; i++) {
//        Action Temp = Noise_new.col(i).array() / StdevNew.array();
//        p_n[i] =
//            -0.5 * Temp.transpose() * Temp - StdevNew.array().log().sum();//- 0.5 * std::log(2.0 * M_PI) * ActionDim;
//        Loss += std::exp(p_n[i] - p_old[i]) * advs[i];
//      }
//      Loss = Loss / sampleN;
//      Err += std::abs((Loss - loss2) / Loss);
//
//      LOG(INFO) << "Loss " << " " << Loss - loss << ", " << loss2 - loss;
////      LOG(INFO) << " Loss Function Error (%)" << std::abs((Loss - loss2) / Loss) * 100;
//      pgtest(j) = (Loss - loss) / stepsize;
//      pgtest2(j) = (loss2 - loss) / stepsize;
//    }
//
//    LOG(INFO) << pgtest.transpose();
//    LOG(INFO) << pgtest2.transpose();
//    LOG(INFO) << testgrad.transpose();
//
//    LOG(INFO) << " Loss Function Average Error (%) (TF vs. Numeric): " << Err / paramsize * 100;
//    LOG(INFO) << "Grad. error (%) (Numeric) :" << (pgtest - testgrad).norm() / testgrad.norm() * 100;
//    LOG(INFO) << "Grad. error (%) (TFloss) :" << (pgtest2 - testgrad).norm() / testgrad.norm() * 100;
//  }

//  policy.getfvp(stateBatch, actionBatch, actionNoise, advs, Stdev_o,testgrad, fishergrad);
}


//
//logp_n = - 0.5 * tf.reduce_sum(tf.square((OldActionSampled - action_mean) / action_stdev), axis=1) \
//                     - 0.5 * tf.cast(tf.log(2.0 * np.pi),dtype) * action_dim - tf.reduce_sum(tf.log(action_stdev))
//logp_old = - 0.5 * tf.reduce_sum(tf.square((OldActionNoise) / OldStdv), axis=1) \
//                     - 0.5 * tf.cast(tf.log(2.0 * np.pi),dtype) * action_dim - tf.reduce_sum(tf.log(OldStdv))
//
//ratio = tf.exp(logp_n - logp_old, name='rat')
//surr = - tf.reduce_mean(tf.multiply(ratio, advant), name='Surr')
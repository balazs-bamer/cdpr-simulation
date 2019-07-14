#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <csignal>
#include <cstdlib>

#include <glog/logging.h>

#include "rai/tasks/poleBalancing/PoleBalancing.hpp"
#include "rai/noiseModel/OrnsteinUhlenbeckNoise.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Policy_TensorFlow.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include "rai/experienceAcquisitor/ReplayMemoryExperienceAcquisitor.hpp"
#include "algorithm/DeepDeterministicPolicyGradient.hpp"


using std::cout;
using std::endl;

using Dtype = float;

using PoleBalancing = rai::Task::PoleBalancing<Dtype>;
using rai::Task::StateDim;
using rai::Task::ActionDim;
using rai::Task::CommandDim;
using DeepDeterministicPolicyGradient = rai::Algorithm::DeepDeterministicPolicyGradient<Dtype, StateDim, ActionDim, CommandDim>;
using OrnsteinUhlenbeck = rai::Noise::OrnsteinUhlenbeck<Dtype, ActionDim>;
using Policy = rai::FuncApprox::Policy_TensorFlow<Dtype, StateDim, ActionDim>;
using Qfunction_TensorFlow = rai::FuncApprox::Qfunction_TensorFlow<Dtype, StateDim, ActionDim>;
using ReplayMemorySARS = rai::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
using ReplayMemoryExperienceAcquisitor = rai::ExpAcq::ReplayMemoryExperienceAcquisitor<Dtype, StateDim, ActionDim, CommandDim>;


int replayMemorySize = 1000000;
int numberOfTransitionsToBeAcquiredBeforeStartingLearning = 10000;

int batchSize = 64;
double learningRateQfunction = 1e-3;
double learningRatePolicy = 1e-4;
double tau = 1e-3;

int maxStepsPerEpisode = 100;
int nStepsPerEpoch = 1000;
int nEpochs = 150;
double dt = 0.05;
double maxTimePerEpisode = maxStepsPerEpisode*dt;

double theta = 0.15;
double sigma = 0.2;
double discountFactor = 0.99;

int main(int argc, char* argv[]) {
  LOG(INFO) << "Generating all *.pb";

  int error;
  error = system("cd resources; ./run_python_scripts.sh");

  LOG_IF(FATAL, error) << "There was an error with calling run_python_scripts.sh";

  PoleBalancing task;
  task.setDiscountFactor(discountFactor);
  task.setValueAtTerminalState(0);
  task.setControlUpdate_dt(dt);
  task.setTimeLimitPerEpisode(maxTimePerEpisode);
  task.setToPendulumv0Mode();

  OrnsteinUhlenbeck noise(theta, sigma);
  Qfunction_TensorFlow qfunction("resources/Qfunction.pb", learningRateQfunction);
  Qfunction_TensorFlow targetQfunction("resources/Qfunction.pb", learningRateQfunction);
  Policy policy("resources/policy.pb", learningRatePolicy);
  Policy targetPolicy("resources/policy.pb", learningRatePolicy);
  ReplayMemorySARS replayMemory(replayMemorySize);
  ReplayMemoryExperienceAcquisitor replayMemoryExperienceAcquisitor(&task, &noise, &policy, &replayMemory);

  DeepDeterministicPolicyGradient ddpg(&replayMemoryExperienceAcquisitor, &qfunction, &targetQfunction, &policy,
                                       &targetPolicy, numberOfTransitionsToBeAcquiredBeforeStartingLearning, batchSize,
                                       nStepsPerEpoch, tau);

  LOG(INFO) << "Filling Replay Memory" << endl;
  ddpg.fillReplayMemoryBeforeLearning();
  LOG(INFO) << "Replay Memory full" << endl;
  LOG(INFO) << "Parameters: " << endl
            << "Replay Memory size: " << replayMemorySize << endl
            << "Batch size: " << batchSize << endl
            << "Learning Rate Qfunction: " << learningRateQfunction << endl
            << "Learning Rate Policy: " << learningRatePolicy << endl
            << "Tau: " << tau << endl;

  ddpg.runForNEpochs(nEpochs, 1, 5, 10);
}
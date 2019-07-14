//
// Created by jemin on 20.06.16.
//

#ifndef RAI_TRAJECTORY_HPP
#define RAI_TRAJECTORY_HPP

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "raiCommon/enumeration.hpp"
#include "rai/function/common/ValueFunction.hpp"

namespace rai {
namespace Memory {

template<typename Dtype, int stateDim, int actionDim>
class Trajectory {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<Dtype, stateDim, 1> State;
  typedef Eigen::Matrix<Dtype, actionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, -1, 1> HiddenState;

  using Vfunction_ = FuncApprox::ValueFunction<Dtype, stateDim>;

  Trajectory() {}
  ~Trajectory() {}

  //////////////////////////// core methods /////////////////////////////////
  void pushBackHiddenState(HiddenState &hiddenState) {
    hiddenStateTraj.push_back(hiddenState);
  }

  void pushBackTrajectory(State &state,
                          Action &action,
                          Action &actionNoise,
                          Dtype cost) {
    actionTraj.push_back(action);
    stateTraj.push_back(state);
    actionNoiseTraj.push_back(actionNoise);
    costTraj.push_back(cost);
    matrixUpdated = false;
    gaeUpdated = false;
  }

  void terminateTrajectoryAndUpdateValueTraj(TerminationType termTypeArg,
                                             State &terminalState,
                                             Action &terminalAction,
                                             Dtype terminalValue,
                                             Dtype discountFactor) {
    terminalValue_ = terminalValue;
    discountFct_ = discountFactor;
    stateTraj.push_back(terminalState);
    costTraj.push_back(Dtype(0));
    accumCostTraj.resize(costTraj.size());
    actionNoiseTraj.push_back(Action::Zero());
    termType = termTypeArg;

    if (termType == TerminationType::terminalState)
      actionTraj.push_back(Action::Zero());
    else actionTraj.push_back(terminalAction);

    int trajLength = stateTraj.size();
    int trajectoryCounter = trajLength - 1;

    valueTraj.resize(trajLength);
    accumCostTraj.resize(trajLength);

    valueTraj[trajectoryCounter] = terminalValue;
    accumCostTraj[trajectoryCounter] = Dtype(0);
    while (trajectoryCounter-- > 0) {
      valueTraj[trajectoryCounter] = discountFactor
          * valueTraj[trajectoryCounter + 1] + costTraj[trajectoryCounter];
      accumCostTraj[trajectoryCounter] = discountFactor
          * accumCostTraj[trajectoryCounter + 1] + costTraj[trajectoryCounter];
    }

    matrixUpdated = false;
    gaeUpdated = false;
  }

  int size() {
    return int(stateTraj.size());
  }

  void clear() {
    actionTraj.clear();
    stateTraj.clear();
    actionNoiseTraj.clear();
    hiddenStateTraj.clear();
    costTraj.clear();
    accumCostTraj.clear();
    valueTraj.clear();
    termType = TerminationType::not_terminated;
    gaeUpdated = false;
    matrixUpdated = false;
    terminalValue_ = Dtype(0);
  }

  Tensor<Dtype, 3> &getStateMat() {
    updateMatrix();
    return states;
  }

  Tensor<Dtype, 2> &getActionMat() {
    updateMatrix();
    return actions;
  }

  //////////////////////// non-core methods //////////////////////////////////
  void updateValueTrajWithNewTermValue(Dtype terminalValue, Dtype discountFactor) {
    accumCostTraj.resize(size());
    valueTraj.resize(size());
    discountFct_ = discountFactor;

    terminalValue_ = terminalValue;
    Dtype decayedTerminalValue = terminalValue * discountFactor;
    int trajectoryCounter = stateTraj.size() - 1;
    accumCostTraj[trajectoryCounter] = Dtype(0);
    valueTraj[trajectoryCounter--] = terminalValue;

    while (trajectoryCounter > -1) {
      accumCostTraj[trajectoryCounter] = discountFactor
          * accumCostTraj[trajectoryCounter + 1] + costTraj[trajectoryCounter];

      valueTraj[trajectoryCounter] = accumCostTraj[trajectoryCounter] + decayedTerminalValue;
      trajectoryCounter--;

      decayedTerminalValue *= discountFactor;
    }
    gaeUpdated = false;
    matrixUpdated = false;
  }

  Dtype getAverageValue() {
    Dtype sum = Dtype(0);
    for (auto &elem : valueTraj)
      sum += elem;
    return sum / valueTraj.size();
  }

  rai::Tensor<Dtype, 1> &getGAE(Vfunction_ *vfunction,
                                Dtype gamma,
                                Dtype lambda,
                                Dtype terminalCost) {
    if (gaeUpdated) return advantage;
    updateBellmanErr(vfunction, gamma, terminalCost);
    advantage.resize(size() - 1);
    advantage[size() - 2] = bellmanErr[size() - 2];
    Dtype fctr = gamma * lambda;
    for (int timeID = size() - 3; timeID > -1; timeID--) {
      advantage[timeID] = fctr * advantage[timeID + 1] + bellmanErr[timeID];
    }
    gaeUpdated = true;
    return advantage;
  }

  void updateBellmanErr(Vfunction_ *baseline, Dtype discFtr, Dtype termCost) {
    updateMatrix();
    values.resize(1, size());
    bellmanErr.resize(size() - 1);
    baseline->forward(states, values);
    if (termType == TerminationType::terminalState)
      values[size() - 1] = termCost;
    for (int i = 0; i < size() - 1; i++)
      bellmanErr[i] = values[i + 1] * discFtr + costTraj[i] - values[i];
  }

  void printOutTraj() {
    updateMatrix();
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------Trajectory printout, size " << size() << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "stateTrajMat" << std::endl << states << std::endl;
    std::cout << "actionTrajMat" << std::endl << actions << std::endl;
    std::cout << "termType" << std::endl << int(termType) << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
  }

  /// prepend a section of other trajectory to this trajectory
  void prependAsectionOfTraj(Trajectory &otherTraj, unsigned startingId, unsigned endId) {
    std::vector<Action> actionTraj_cpy = actionTraj;
    std::vector<State> stateTraj_cpy = stateTraj;
    std::vector<Action> actionNoiseTraj_cpy = actionNoiseTraj;
    std::vector<Dtype> costTraj_cpy = costTraj;

    unsigned currentSize = size();
    unsigned sectionSize = endId - startingId + 1;
    unsigned newSize = sectionSize + currentSize;

    stateTraj.clear();
    actionTraj.clear();
    actionNoiseTraj.clear();
    costTraj.clear();

    actionTraj.insert(actionTraj.begin(),
                      otherTraj.actionTraj.begin() + startingId,
                      otherTraj.actionTraj.begin() + endId + 1);
    stateTraj.insert(stateTraj.begin(),
                     otherTraj.stateTraj.begin() + startingId,
                     otherTraj.stateTraj.begin() + endId + 1);
    actionNoiseTraj.insert(actionNoiseTraj.begin(),
                           otherTraj.actionNoiseTraj.begin() + startingId,
                           otherTraj.actionNoiseTraj.begin() + endId + 1);
    costTraj.insert(costTraj.begin(), otherTraj.costTraj.begin() + startingId, otherTraj.costTraj.begin() + endId + 1);

    actionTraj.insert(actionTraj.end(), actionTraj_cpy.begin(), actionTraj_cpy.end());
    stateTraj.insert(stateTraj.end(), stateTraj_cpy.begin(), stateTraj_cpy.end());
    actionNoiseTraj.insert(actionNoiseTraj.end(), actionNoiseTraj_cpy.begin(), actionNoiseTraj_cpy.end());
    costTraj.insert(costTraj.end(), costTraj_cpy.begin(), costTraj_cpy.end());

    updateValueTrajWithNewTermValue(terminalValue_, discountFct_);
    matrixUpdated = false;
    gaeUpdated = false;
  }

  std::vector<State> stateTraj;
  std::vector<Action> actionTraj;
  std::vector<Action> actionNoiseTraj;
  std::vector<HiddenState> hiddenStateTraj;
  std::vector<Dtype> costTraj, accumCostTraj;
  std::vector<Dtype> valueTraj;
  TerminationType termType = TerminationType::not_terminated;
  rai::Tensor<Dtype, 1> advantage;
  Dtype terminalValue_ = Dtype(0);
  Dtype discountFct_ = Dtype(0);

 private:

  void updateMatrix() {
    if (matrixUpdated) return;
    states = "state";
    states.resize(stateDim, size(), 1);
    actions.resize(actionDim, size());
    for (int colID = 0; colID < size(); colID++) {
      states.col(0, colID) = stateTraj[colID];
      actions.col(colID) = actionTraj[colID];
    }
    matrixUpdated = true;
  }
  Tensor<Dtype, 3> states;
  Tensor<Dtype, 2> actions, values;
  Tensor<Dtype, 1> bellmanErr;

  bool matrixUpdated = false;
  bool gaeUpdated = false;

};
}
}
#endif //TRAJECTORY_HPP

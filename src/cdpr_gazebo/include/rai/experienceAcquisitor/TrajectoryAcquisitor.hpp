//
// Created by jhwangbo on 3/23/17.
//

#ifndef RAI_TRAJECTORYACQUISITOR_HPP
#define RAI_TRAJECTORYACQUISITOR_HPP
#include <rai/algorithm/common/LearningData.hpp>
#include "Acquisitor.hpp"
#include "rai/memory/Trajectory.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/tasks/common/Task.hpp"
#include "rai/function/common/Policy.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"

namespace rai {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim>
class TrajectoryAcquisitor : public Acquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Trajectory = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using State = Eigen::Matrix<Dtype, StateDim, 1>;
  using Action = Eigen::Matrix<Dtype, ActionDim, 1>;
  using StateBatch = Eigen::Matrix<Dtype, StateDim, -1>;
  using ReplayMemory_ = Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;
  using ValueBatch = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using DataSet = rai::Algorithm::LearningData<Dtype, StateDim, ActionDim>;

 public:
  std::vector<Trajectory> traj;
  TrajectoryAcquisitor(){};
  virtual ~TrajectoryAcquisitor(){};

  virtual Dtype acquire(std::vector<Task_ *> &taskset,
                        Policy_ *policy,
                        std::vector<Noise_ *> &noise,
                        std::vector<Trajectory> &trajectorySet,
                        StateBatch &startingState,
                        double timeLimit,
                        bool countStep,
                        ReplayMemory_ *memory = nullptr) = 0;

  virtual Dtype acquire(std::vector<Task_ *> &taskset,
                        Policy_ *policy,
                        std::vector<Noise_ *> &noise,
                        std::vector<Trajectory> &trajectorySet,
                        double timeLimit,
                        bool countStep,
                        ReplayMemory_ *memory = nullptr) = 0;

  void acquireNEpisodes(std::vector<Task_ *> &task,
                        std::vector<Noise_ *> &noise,
                        Policy_ *policy,
                        int numOfEpisodes,
                        ValueFunc_ *vfunction = nullptr,
                        int vis_lv = 0) {

    Utils::timer->startTimer("Simulation");
    double dt = task[0]->dt();
    double timeLimit = task[0]->timeLimit();
    traj.resize(numOfEpisodes);

    for (auto &tra : traj)
      tra.clear();
    if (vis_lv > 1) task[0]->turnOnVisualization("");
    long double stepsTaken = this->stepsTaken();
    Dtype cost = this->acquire(task,
                               policy,
                               noise,
                               traj,
                               timeLimit,
                               true);
    if (vis_lv > 1) task[0]->turnOffVisualization();

    int stepsInThisLoop = int(this->stepsTaken() - stepsTaken);
    Utils::timer->stopTimer("Simulation");
  }

  void acquireVineTrajForNTimeSteps(std::vector<Task_ *> &task,
                                    std::vector<Noise_ *> &noise,
                                    Policy_ *policy,
                                    int numOfSteps,
                                    int numofjunct,
                                    int numOfBranchPerJunct,
                                    ValueFunc_ *vfunction = nullptr,
                                    int vis_lv = 0,
                                    bool noisifyState = false) {

    Utils::timer->startTimer("Simulation");
    std::vector<Trajectory> trajectories;
    double dt = task[0]->dt();
    double timeLimit = task[0]->timeLimit();

    int numOfTra_ = std::ceil(1.1 * numOfSteps * dt / timeLimit);
    traj.resize(numOfTra_);

    for (auto &tra : traj)
      tra.clear();
    if (vis_lv > 1) task[0]->turnOnVisualization("");
    long double stepsTaken = this->stepsTaken();
    Dtype cost = this->acquire(task,
                               policy,
                               noise,
                               traj,
                               timeLimit,
                               true);
    if (vis_lv > 1) task[0]->turnOffVisualization();

    int stepsInThisLoop = int(this->stepsTaken() - stepsTaken);

    if (numOfSteps > stepsInThisLoop) {
      int stepsneeded = numOfSteps - stepsInThisLoop;
      std::vector<Trajectory> tempTraj_;
      while (true) {
        int numofnewtraj = std::ceil(1.5 * stepsneeded * dt / timeLimit);
        tempTraj_.resize(numofnewtraj);
        for (auto &tra : tempTraj_)
          tra.clear();

        for (auto &noise : noise)
          noise->initializeNoise();

        if (vis_lv > 1) task[0]->turnOnVisualization("");
        this->acquire(task,
                      policy,
                      noise,
                      tempTraj_,
                      timeLimit,
                      true);
        if (vis_lv > 1) task[0]->turnOffVisualization();

        stepsInThisLoop = int(this->stepsTaken() - stepsTaken);
        stepsneeded = numOfSteps - stepsInThisLoop;
        ///merge trajectories
        traj.reserve(traj.size() + tempTraj_.size());
        traj.insert(traj.end(), tempTraj_.begin(), tempTraj_.end());

        if (stepsneeded <= 0) break;
      }
    }
    ///////////////////////////////////////VINE//////////////////////////////
    StateBatch VineStartPosition(StateDim, numofjunct);
    StateBatch rolloutstartState(StateDim, numofjunct * numOfBranchPerJunct);
    trajectories.resize(numofjunct * numOfBranchPerJunct);
    std::vector<std::pair<int, int> > indx;
    rai::Op::VectorHelper::sampleRandomStates(traj, VineStartPosition, int(0.1 * timeLimit / dt), indx);

//    for (int dataID = 0; dataID < numofjunct; dataID++)
//      rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct) =
//          rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct).array().colwise()
//              * VineStartPosition.col(dataID).array();

    for (int junctID = 0; junctID < numofjunct; junctID++)
      for (int branchID = 0; branchID < numOfBranchPerJunct; branchID++ )
      rolloutstartState.col(junctID * numOfBranchPerJunct + branchID) =
          VineStartPosition.col(junctID);


    for (auto &tra : trajectories)
      tra.clear();
    for (auto &noise : noise)
      noise->initializeNoise();

    if(noisifyState) task[0]->noisifyState(rolloutstartState);

    this->acquire(task, policy, noise, trajectories, rolloutstartState, timeLimit, true);

    ///merge trajectories into one vector
    traj.reserve(traj.size() + trajectories.size());
    traj.insert(traj.end(), trajectories.begin(), trajectories.end());
    Utils::timer->stopTimer("Simulation");
  }

  void acquireTrajForNTimeSteps(std::vector<Task_ *> &task,
                                std::vector<Noise_ *> &noise,
                                Policy_ *policy,
                                int numOfSteps,
                                ValueFunc_ *vfunction = nullptr,
                                int vis_lv = 0) {
    acquireVineTrajForNTimeSteps(task, noise, policy, numOfSteps, 0, 0, vfunction, vis_lv);
  }

 private:
  void sampleBatchOfInitial(StateBatch &initial, std::vector<Task_ *> &task) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task[0]->setToInitialState();
      task[0]->getState(state);
      initial.col(trajID) = state;
    }
  }
};

}
}

#endif //RAI_TRAJECTORYACQUISITOR_HPP

//
// Created by jhwangbo on 3/23/17.
//

#ifndef RAI_TRAJECTORYACQUISITOR_SEQUENTIAL_HPP
#define RAI_TRAJECTORYACQUISITOR_SEQUENTIAL_HPP
#include "TrajectoryAcquisitor.hpp"
#include "Acquisitor.hpp"
#include "rai/tasks/common/Task.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include "AcquisitorCommonFunc.hpp"

namespace rai {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim>
class TrajectoryAcquisitor_Sequential : public TrajectoryAcquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using StateBatch = Eigen::Matrix<Dtype, StateDim, -1>;
  using State = Eigen::Matrix<Dtype, StateDim, 1>;
  using Result = Eigen::Matrix<Dtype, 2, 1>;
  using ReplayMemory_ = Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;

 public:
  virtual Dtype acquire(std::vector<Task_ *> &taskset,
                        Policy_ *policy,
                        std::vector<Noise_ *> &noise,
                        std::vector<Trajectory_> &trajectorySet,
                        StateBatch &startingState,
                        double timeLimit,
                        bool countStep,
                        ReplayMemory_ *memory = nullptr) {
    Result stat = Result::Zero();
    for(int i=0; i<trajectorySet.size(); i++) {
      State start_state = startingState.col(i);
      Result statInd = CommonFunc<Dtype, StateDim, ActionDim, 0>::runEpisode(taskset[0],
                                                                   policy,
                                                                   noise[0],
                                                                   trajectorySet[i],
                                                                   start_state,
                                                                   timeLimit,
                                                                   memory);
      stat += statInd;
    }

    if (countStep) this->incrementSteps(stat[1]);
    return stat[0] / trajectorySet.size();
  }

  virtual Dtype acquire(std::vector<Task_ *> &taskset,
                        Policy_ *policy,
                        std::vector<Noise_ *> &noise,
                        std::vector<Trajectory_> &trajectorySet,
                        double timeLimit,
                        bool countStep,
                        ReplayMemory_ *memory = nullptr) {
    Result stat = Result::Zero();
    for(int i=0; i<trajectorySet.size(); i++) {
      Result statInd = CommonFunc<Dtype, StateDim, ActionDim, 0>::runEpisode(taskset[0],
                                                                             policy,
                                                                             noise[0],
                                                                             trajectorySet[i],
                                                                             timeLimit,
                                                                             memory);
      stat += statInd;
    }

    if (countStep) this->incrementSteps(stat[1]);
    return stat[0] / trajectorySet.size();
  }
};

}
}

#endif //RAI_TRAJECTORYACQUISITOR_SEQUENTIAL_HPP

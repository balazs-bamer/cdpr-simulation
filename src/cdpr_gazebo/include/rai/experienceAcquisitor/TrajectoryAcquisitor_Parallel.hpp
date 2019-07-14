//
// Created by joonho on 11/26/17.
//

#ifndef RAI_TRAJECTORYACQUISITOR_BATCH_HPP
#define RAI_TRAJECTORYACQUISITOR_BATCH_HPP

#include "TrajectoryAcquisitor.hpp"
#include "Acquisitor.hpp"
#include "rai/memory/Trajectory.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/tasks/common/Task.hpp"
#include "rai/function/common/Policy.hpp"
#include "AcquisitorCommonFunc.hpp"

namespace rai {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim>
class TrajectoryAcquisitor_Parallel : public TrajectoryAcquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using StateBatch = Eigen::Matrix<Dtype, StateDim, -1>;
  using Result = Eigen::Matrix<Dtype, 2, 1>;
  using ReplayMemory_ = Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;

 public:
  Dtype acquire(std::vector<Task_ *> &taskset,
                Policy_ *policy,
                std::vector<Noise_ *> &noise,
                std::vector<Trajectory_> &trajectorySet,
                StateBatch &startingState,
                double timeLimit,
                bool countStep,
                ReplayMemory_ *memory = nullptr) {
    Result stat;
    stat = CommonFunc<Dtype, StateDim, ActionDim, 0>::runEpisodeInBatchParallel(taskset,
                                                                                policy,
                                                                                noise,
                                                                                trajectorySet,
                                                                                startingState,
                                                                                timeLimit,
                                                                                memory);
    if (countStep)
      this->incrementSteps(stat[1]);
    return stat[0];
  }

  Dtype acquire(std::vector<Task_ *> &taskset,
                Policy_ *policy,
                std::vector<Noise_ *> &noise,
                std::vector<Trajectory_> &trajectorySet,
                double timeLimit,
                bool countStep,
                ReplayMemory_ *memory = nullptr) {
    Result stat;
    stat = CommonFunc<Dtype, StateDim, ActionDim, 0>::runEpisodeInBatchParallel(taskset,
                                                                                policy,
                                                                                noise,
                                                                                trajectorySet,
                                                                                timeLimit,
                                                                                memory);
    if (countStep)
      this->incrementSteps(stat[1]);
    return stat[0];
  }

};

}
}


#endif //RAI_TRAJECTORYACQUISITOR_BATCH_HPP

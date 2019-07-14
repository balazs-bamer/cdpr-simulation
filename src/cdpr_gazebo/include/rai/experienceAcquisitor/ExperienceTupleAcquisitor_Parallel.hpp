//
// Created by jhwangbo on 3/23/17.
// sampling tuples in parallel

#ifndef RAI_EXPERIENCETUPLEACQUISITOR_PARALLEL_HPP
#define RAI_EXPERIENCETUPLEACQUISITOR_PARALLEL_HPP

#include "ExperienceTupleAcquisitor.hpp"
#include "AcquisitorCommonFunc.hpp"

namespace rai {
namespace ExpAcq {

template <typename Dtype, int StateDim, int ActionDim>
class ExperienceTupleAcquisitor_Parallel : public ExperienceTupleAcquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ReplayMemory_ = rai::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;

 public:
  virtual void acquire(std::vector<Task_*> &task,
                       Policy_ *policy,
                       std::vector<Noise_*> &noise,
                       ReplayMemory_ *memory,
                       unsigned stepsToTake) {
    int nIter = stepsToTake / task.size();
    nIter = std::max(1,nIter);
    for(unsigned stepId = 0; stepId < nIter; stepId++)
      CommonFunc<Dtype, StateDim, ActionDim, 0>::takeOneStepInBatch(task, policy, noise, memory);
    this->incrementSteps( nIter * task.size());
  }
};

}
}

#endif //RAI_EXPERIENCETUPLEACQUISITOR_PARALLEL_HPP

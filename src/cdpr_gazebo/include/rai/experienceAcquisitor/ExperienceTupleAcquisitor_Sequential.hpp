//
// Created by jhwangbo on 3/23/17.
//

#ifndef RAI_EXPERIENCETUPLEACQUISITOR_SEQUENTIAL_HPP
#define RAI_EXPERIENCETUPLEACQUISITOR_SEQUENTIAL_HPP

#include "ExperienceTupleAcquisitor.hpp"
#include "AcquisitorCommonFunc.hpp"

namespace rai {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim>
class ExperienceTupleAcquisitor_Sequential : public ExperienceTupleAcquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ReplayMemory_ = rai::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;

 public:
  virtual void acquire(std::vector<Task_ *> &task,
                       Policy_ *policy,
                       std::vector<Noise_ *> &noise,
                       ReplayMemory_ *memory,
                       unsigned stepsToTake) {
    for(int stepId = 0; stepId < stepsToTake; stepId++)
      CommonFunc<Dtype, StateDim, ActionDim, 0>::takeOneStep(task, policy, noise, memory);

    this->incrementSteps(stepsToTake);
  }
};

}
}

#endif //RAI_EXPERIENCETUPLEACQUISITOR_SEQUENTIAL_HPP

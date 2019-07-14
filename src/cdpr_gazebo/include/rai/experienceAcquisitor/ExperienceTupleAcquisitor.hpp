//
// Created by jhwangbo on 3/23/17.
//

#ifndef RAI_EXPERIENCETUPLEACQUISITOR_HPP
#define RAI_EXPERIENCETUPLEACQUISITOR_HPP

#include "Acquisitor.hpp"
#include "rai/tasks/common/Task.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include "rai/function/common/Policy.hpp"

namespace rai {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim>
class ExperienceTupleAcquisitor : public Acquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ReplayMemory_ = rai::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;

 public:
  ExperienceTupleAcquisitor(){};
  virtual ~ExperienceTupleAcquisitor(){};

  virtual void acquire(std::vector<Task_ *> &task,
                       Policy_ *policy,
                       std::vector<Noise_ *> &noise,
                       ReplayMemory_ *memory,
                       unsigned stepsToTake) = 0;

};

}
}

#endif //RAI_EXPERIENCETUPLEACQUISITOR_HPP

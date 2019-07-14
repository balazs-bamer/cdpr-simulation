//
// Created by jhwangbo on 10/08/17.
//

#ifndef RAI_DETERMINISTICMODEL_HPP
#define RAI_DETERMINISTICMODEL_HPP

#include "Model.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class DeterministicModel : public virtual Model<Dtype, stateDim, actionDim> {
 public:

  typedef Eigen::Matrix<Dtype, stateDim, 1> Input;
  typedef Eigen::Matrix<Dtype, stateDim, -1> InputBatch;
  typedef Eigen::Matrix<Dtype, actionDim, 1> Output;
  typedef Eigen::Matrix<Dtype, actionDim, -1> OutputBatch;
  DeterministicModel(){};
  virtual ~DeterministicModel(){};
};

}
}

#endif //RAI_DETERMINISTICMODEL_HPP

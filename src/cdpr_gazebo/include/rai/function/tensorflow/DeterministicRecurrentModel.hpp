//
// Created by jhwangbo on 10/08/17.
//

#ifndef RAI_DETERMINISTICRECURRENTMODEL_HPP
#define RAI_DETERMINISTICRECURRENTMODEL_HPP

#include "DeterministicModel.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class DeterministicRecurrentModel : public virtual Model<Dtype, stateDim, actionDim> {
 public:

  typedef Eigen::Matrix<Dtype, stateDim, 1> State;
  typedef Eigen::Matrix<Dtype, stateDim, -1> StateBatch;
  typedef Eigen::Matrix<Dtype, actionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, actionDim, -1> ActionBatch;
  typedef Eigen::Matrix<Dtype, -1, 1> HiddenState;


  DeterministicRecurrentModel(){}

  ~DeterministicRecurrentModel(){}


  virtual bool isRecurrent() {
    return true;
  }

};

}
}

#endif //RAI_DETERMINISTICRECURRENTMODEL_HPP

//
// Created by jhwangbo on 10/08/17.
//

#ifndef RAI_MODEL_HPP
#define RAI_MODEL_HPP
#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class Model {
 public:
  typedef Eigen::Matrix<Dtype, stateDim, 1> State;
  typedef Eigen::Matrix<Dtype, stateDim, -1> StateBatch;
  typedef Eigen::Matrix<Dtype, actionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, actionDim, -1> ActionBatch;
  Model(){};
  virtual ~Model(){};

};

}
}

#endif //RAI_MODEL_HPP

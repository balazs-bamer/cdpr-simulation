//
// Created by jhwangbo on 3/23/17.
//

#ifndef RAI_ACQUISITOR_HPP
#define RAI_ACQUISITOR_HPP

namespace rai {
namespace ExpAcq {

template <typename Dtype, int StateDim, int ActionDim>
class Acquisitor {

 public:
  Acquisitor(){};
  virtual ~Acquisitor(){};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stepsTaken(){
    return double(stepsTaken_);
  }

  void incrementSteps(unsigned increment){
    stepsTaken_ += increment;
  }

 protected:
  unsigned long stepsTaken_ = 0;

};

}
}

#endif //RAI_ACQUISITOR_HPP

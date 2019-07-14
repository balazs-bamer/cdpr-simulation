//
// Created by jhwangbo on 23.06.16.
//

#ifndef QLEARNING_NOISE_HPP
#define QLEARNING_NOISE_HPP

#include <Eigen/Core>

namespace rai {
namespace Noise {

template<typename Dtype, int noiseVectorDimension>
class Noise {

public:
  Noise(){};
  virtual~Noise(){};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<Dtype, noiseVectorDimension, 1> NoiseVector;
  typedef Eigen::Matrix<Dtype, noiseVectorDimension, noiseVectorDimension> CovarianceMatrix;
  virtual void initializeNoise() { };
  virtual NoiseVector& noisify(NoiseVector& originalVector){ };
  virtual NoiseVector& sampleNoise() = 0;
  virtual void setNoiseLevel(Dtype noiseLevel) {noiseLevel_ = noiseLevel;};
  virtual void scaleNoiseLevel(Dtype scale){noiseLevel_ *= scale;};

protected:
  Dtype noiseLevel_ = 1.0;

};

}
}
#endif //QLEARNING_NOISE_HPP

//
// Created by jhwangbo on 23.06.16.
//

#ifndef QLEARNING_NONOISE_HPP
#define QLEARNING_NONOISE_HPP

#include <Eigen/Core>
#include "Noise.hpp"

namespace rai {
namespace Noise {

template<typename Dtype, int noiseVectorDimension>
class NoNoise : public Noise<Dtype, noiseVectorDimension> {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NoNoise(){};
  ~NoNoise(){};
  typedef Eigen::Matrix<Dtype, noiseVectorDimension, 1> NoiseVector;
  typedef Eigen::Matrix<Dtype, noiseVectorDimension, noiseVectorDimension> CovarianceMatrix;
  void initializeNoise() { };
  NoiseVector& noisify(NoiseVector &originalVector){};
  NoiseVector& sampleNoise(){ return noiseVector; };
  void setNoiseLevel(Dtype noiseLevel) {noiseLevel_ = noiseLevel;};
  void scaleNoiseLevel(Dtype scale){noiseLevel_ *= scale;};

protected:
  Dtype noiseLevel_ = 1.0;
  NoiseVector noiseVector = NoiseVector::Zero();
};

}
}
#endif //QLEARNING_NONOISE_HPP

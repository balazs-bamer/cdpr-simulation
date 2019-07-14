//
// Created by jhwangbo on 27.03.17.
//

/// This method randomly adds noise by the probability is defined by the epsilon


#ifndef RAI_EPSILONNOISE_HPP
#define RAI_EPSILONNOISE_HPP

#include "Noise.hpp"
#include "math/RandomNumberGenerator.hpp"

namespace rai {
namespace Noise {

template<typename Dtype, int noiseVectorDimension>
class EpsilonNoise : public Noise<Dtype, noiseVectorDimension> {
  EpsilonNoise(Noise* baseNoise, float epsilon):
      baseNoise_(baseNoise), epsilon_(epsilon){}

  NoiseVector& sampleNoise(){
    return (rn_.forXPercent(epsilon_) ? baseNoise_->sampleNoise() : zeroVector);
  };


  Noise* baseNoise_;
  float epsilon_;
  rai::RandomNumberGenerator<float> rn_;
  NoiseVector zeroVector = NoiseVector::Zero();


};
}
}
#endif //RAI_EPSILONNOISE_HPP

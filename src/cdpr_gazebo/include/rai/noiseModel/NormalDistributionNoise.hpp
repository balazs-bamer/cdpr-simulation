//
// Created by jhwangbo on 23.06.16.
//

#ifndef QLEARNING_NORMALDISTRIBUTIONNOISE_HPP
#define QLEARNING_NORMALDISTRIBUTIONNOISE_HPP

#include "Noise.hpp"
#include "raiCommon/utils/RandomNumberGenerator.hpp"
#include <Eigen/Cholesky>

namespace rai {
namespace Noise {

template<typename Dtype, int noiseVectorDimension>
class NormalDistributionNoise : public Noise<Dtype, noiseVectorDimension> {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef typename Noise<Dtype, noiseVectorDimension>::NoiseVector NoiseVector;
  typedef Eigen::Matrix<Dtype, noiseVectorDimension, noiseVectorDimension> Covariance;

  NormalDistributionNoise(Covariance cov) {
    initialCov_ = cov;
    cov_ = cov;
    Eigen::LLT<Covariance> lltOfcov(cov);
    chol_ = lltOfcov.matrixL();
  }

  ~NormalDistributionNoise() {};

  virtual NoiseVector &noisify(NoiseVector &originalVector) {
    for (int dimID = 0; dimID < noiseVectorDimension; dimID++)
      noiseVector_(dimID) = rn_.sampleNormal() * this->noiseLevel_;
    originalVector += chol_ * noiseVector_;
    return originalVector;
  }

  virtual NoiseVector& sampleNoise() {
    for (int dimID = 0; dimID < noiseVectorDimension; dimID++)
      noiseVector_(dimID) = rn_.sampleNormal() * this->noiseLevel_;
    noiseVector_ = chol_ * noiseVector_;
    return noiseVector_;
  }

  virtual void updateCovariance(Covariance cov) {
    initialCov_ = cov;
    cov_ = cov;
    Eigen::LLT<Covariance> lltOfcov(cov);
    chol_ = lltOfcov.matrixL();
  }

  virtual Covariance &getCovariance() { return cov_; }

  virtual void scaleNoiseLevel(Dtype scale) {
    cov_ = cov_ * scale;
  };

 private:
  RandomNumberGenerator<Dtype> rn_;
  NoiseVector noiseVector_;
  Covariance cov_;
  Covariance initialCov_;
  Covariance chol_; //// cholesky decomposition of covariance matrix (lower triangular)

};

}
} // namespaces
#endif //QLEARNING_NORMALDISTRIBUTIONNOISE_HPP

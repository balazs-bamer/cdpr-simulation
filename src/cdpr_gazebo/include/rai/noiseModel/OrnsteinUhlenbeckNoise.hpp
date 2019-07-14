//
// Created by jhwangbo on 23.06.16.
//

#ifndef RAI_ORNSTEINUHLENBECKNOISE_HPP
#define RAI_ORNSTEINUHLENBECKNOISE_HPP

#include "rai/noiseModel/NormalDistributionNoise.hpp"
#include "raiCommon/utils/RandomNumberGenerator.hpp"

namespace rai {
namespace Noise {

template<typename Dtype, int dim>
class OrnsteinUhlenbeck : public NormalDistributionNoise<Dtype, dim> {

  /*
   * Implements Ornstein-Uhlenbeck noise which in its general form has the following stochastic differential equation:
   * OU noise has a Gaussian stationary distribution and can replace normal distribution noise
   *
   * dx = theta*(mu - x_t)*dt + sigma*dW_t                        (where W_t denotes the Wiener process)
   *
   * which can be discretized as follows:
   *
   * x_(t+1) = theta*(mu - x_t)*dt + sigma*sqrt(dt)*randn()       (c.f. http://math.stackexchange.com/q/345773)
   *
   */
 public:
  typedef typename NormalDistributionNoise<Dtype, dim>::NoiseVector NoiseVector;
  typedef Eigen::Matrix<Dtype, dim, dim> Covariance;

 public:
  OrnsteinUhlenbeck(Dtype theta, Dtype sigma, Dtype dt = 1)
      : theta_(theta), dt_(dt), NormalDistributionNoise<Dtype, dim>(cov_) {
    sigma_.setConstant(sigma);
    this->setNoiseLevel(1.0);
    initializeNoise();
  }

  OrnsteinUhlenbeck(Dtype theta, NoiseVector stationary_var, Dtype dt = 1)
      : theta_(theta), dt_(dt), NormalDistributionNoise<Dtype, dim>(cov_) {
    updateCovariance(stationary_var.asDiagonal());
    this->setNoiseLevel(1.0);
    initializeNoise();
  }

  void getTheta(Dtype &theta) {
    theta = theta_;
  }

  void getSigma(NoiseVector &sigma) {
    sigma = sigma_;
  }

  void setTheta(Dtype theta) {
    theta_ = theta;
  }

  void setSigma(NoiseVector sigma) {
    sigma_ = sigma;
  }

  void initializeNoise() {
    stationary_std_ = sigma_ / sqrt(2 * theta_);
    stationary_var_ = stationary_std_.cwiseProduct(stationary_std_);

    for (int dimID = 0; dimID < dim; ++dimID)
      noiseState_(dimID) = stationary_std_(dimID) * rn_.sampleNormal();
  }

  NoiseVector &noisify(NoiseVector &originalVector) {
    originalVector += sampleNoise();
    return originalVector;
  }

  NoiseVector &sampleNoise() {
    for (int dimID = 0; dimID < dim; ++dimID)
      noiseState_(dimID) += -theta_ * noiseState_(dimID) * dt_ + sigma_(dimID) * rn_.sampleNormal() * sqrt(dt_);
    return noiseState_;
  }

  void updateCovariance(Covariance cov) {
    stationary_var_ = cov.diagonal();
    stationary_std_ = stationary_var_.cwiseSqrt();
    sigma_ = stationary_std_ * sqrt(2 * theta_);
  }

  Covariance &getCovariance() {
    cov_ = stationary_var_.asDiagonal();
    return cov_;
  }

 private:
  NoiseVector noiseState_ = NoiseVector::Zero();
  RandomNumberGenerator<Dtype> rn_;
  Dtype theta_, dt_;
  NoiseVector sigma_, stationary_var_, stationary_std_;
  Covariance cov_;

};
}
} // namespaces
#endif // RAI_ORNSTEINUHLENBECKNOISE_HPP

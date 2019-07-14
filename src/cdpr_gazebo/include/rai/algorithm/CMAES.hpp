#ifndef CMAES_HPP_
#define CMAES_HPP_

#include <chrono>
#include <math.h>
#include <random>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <cstdlib>

#include <boost/shared_ptr.hpp>
#include <fstream>
#include <chrono>
#include "glog/logging.h"

namespace rai {
namespace Algorithm {
//! CMAES algorithm
/*! CMAES, 2015
 * code written by Jemin Hwangbo
 */


class CMAES {

 protected:
  //  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::VectorXd xmean, initial_theta;
  Eigen::VectorXd StandardDev;

  int NParameter;

  //random Number generator//
  boost::mt19937 rng_;
  boost::normal_distribution<> normal_dist_;
  boost::shared_ptr<boost::variate_generator<boost::mt19937, boost::normal_distribution<> >> gaussian_;
  Eigen::VectorXd weights;
  //  std::default_random_engine generator;
  //  std::normal_distribution<double> distribution;

  double sigma, lambda, mu, mueff, cc, cs, c1, cmu, damps;
  Eigen::VectorXd pc, ps, Dmat, best_policy, randn, arfitness, xold, arindex, cost_history;
  Eigen::MatrixXd Bmat_, artmp, C, invsqrtC, arx;
  Eigen::VectorXd upperbound, lowerbound;

  double eigeneval, chiN, counteval, best_cost, hsig;
  bool readyToSampleAgain;
  int sample_index, cost_index;
  bool optimizationDone;
  int n_rolloutsEvaluated;

 public:
  //// call proper Contruct after this
  CMAES(){};

  /*! Constructor
   * @param initialParameterSet : initial policy parameter vector (either a column or a row vector) from where we start searching
   * @param std : N-dimensional vector (either a column or a row vector) of initial standard deviation on each axes
   */
  template<typename Derived>
  CMAES(const Eigen::MatrixBase<Derived> &initialParameterSet, const Eigen::MatrixBase<Derived> &std, int lambdal=-1):
      optimizationDone(false), sample_index(0), hsig(0.0), n_rolloutsEvaluated(0) {
    /////////////////////////input checking/////////////////////////
    ///////////////////Probably not so interesting//////////////////
    ////////////////////////////////////////////////////////////////
    if (initialParameterSet.cols() == 1)
      initial_theta = initialParameterSet;
    else if (initialParameterSet.rows() == 1)
      initial_theta = initialParameterSet.transpose();
    else
      throw std::runtime_error("CMAES constructor: initial parameter set should either column vector or a row vector!");

    NParameter = initial_theta.rows();

    if (std.cols() == 1)
      StandardDev = std;
    else if (std.rows() == 1)
      StandardDev = std.transpose();
    else
      throw std::runtime_error(
          "CMAES constructor: initial standard deviation should either column vector or a row vector!");

    if (initial_theta.rows() != StandardDev.rows())
      throw std::runtime_error(
          "CMAES constructor: the length of initial standard deviation and the length of the initial policy parameter vector should be the same!");

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////

    xmean.setZero(NParameter);
    //strategic parameters of CMA-ES
    sigma = 1.0;
    if(lambdal==-1)
      lambda = 4.0 + floor(3.0 * log(double(NParameter)));
    else
      lambda = lambdal;
    mu = lambda / 2.0;
    weights.resize(int(mu));

    for (int ii = 0; ii < int(mu); ii++)
      weights(ii) = log(mu + 0.5) - log(ii + 1.0);

    mu = floor(mu);
    weights = weights / weights.sum();
    mueff = pow(weights.sum(), 2.0) / weights.squaredNorm();
    cc = (4.0 + mueff / NParameter) / (NParameter + 4.0 + 2.0 * mueff / NParameter);
    cs = (mueff + 2.0) / (NParameter + mueff + 5.0);
    c1 = 2.0 / (pow(NParameter + 1.3, 2.0) + mueff);
    cmu = std::min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / (pow(NParameter + 2.0, 2.0) + mueff));
    damps = 1.0 + 2.0 * std::max(0.0, sqrt((mueff - 1.0) / (NParameter + 1.0)) - 1.0) + cs;
    pc = Eigen::VectorXd::Zero(NParameter);
    ps = Eigen::VectorXd::Zero(NParameter);
    artmp = Eigen::MatrixXd::Zero(NParameter, int(mu));
    Dmat = Eigen::VectorXd::Ones(NParameter);
    Bmat_ = Eigen::MatrixXd::Identity(NParameter, NParameter);
    C = Eigen::MatrixXd::Identity(NParameter, NParameter);
    invsqrtC = Eigen::MatrixXd::Identity(NParameter, NParameter);
    arx = Eigen::MatrixXd::Zero(NParameter, lambda);
    upperbound = Eigen::VectorXd::Zero(NParameter);
    lowerbound = Eigen::VectorXd::Zero(NParameter);
    upperbound.setConstant(1e100);
    lowerbound.setConstant(-1e100);

    best_policy.setZero(NParameter);
    randn.setZero(NParameter);
    arfitness.setZero(lambda);
    xold.setZero(NParameter);
    arindex.setZero(lambda);

    cost_history = Eigen::MatrixXd::Zero(20000, 1);
    eigeneval = 0.0;
    chiN = pow(NParameter, 0.5) * (1.0 - 1.0 / (4.0 * NParameter) + 1.0 / (21.0 * NParameter * NParameter));
    counteval = 0;
    best_cost = 1e100;
    readyToSampleAgain = true;
    cost_index = 0;

    /// initialize the random number generator
    srand(time(NULL));
    rng_.seed(rand());
    gaussian_.reset(new boost::variate_generator<boost::mt19937, boost::normal_distribution<> >(rng_, normal_dist_));
    //    distribution = std::normal_distribution<double>(0.0,1.0);
    //    generator = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
  }

  ~CMAES() {
    //    std::cout<<"bmat"<<std::endl<<Bmat_<<std::endl;
    //    std::cout<<"arx"<<std::endl<<arx<<std::endl;
  }

 public:

  /*! getNextTheta2Evaluate
   * @param nexTheta2Evaluate : sample the next policy (either a column vector or a row vector)
   */
  template<typename Derived>
  void getNextTheta2Evaluate(Eigen::MatrixBase<Derived> &nexTheta2Evaluate) {

    bool isItColumnVector;
    n_rolloutsEvaluated++;

    if (nexTheta2Evaluate.cols() == 1)
      isItColumnVector = true;
    else if (nexTheta2Evaluate.rows() == 1)
      isItColumnVector = false;
    else
      throw std::runtime_error(
          "CMAES getNextTheta2Evaluate: new policy container should either column vector or a row vector!");

    if (readyToSampleAgain) {
      for (int k = 0; k < lambda; k++) {

        for (int i = 0; i < NParameter; i++)
          randn(i) = (*gaussian_)();

        arx.col(k) = xmean + sigma * Bmat_ * (Dmat.cwiseProduct(randn));
        readyToSampleAgain = false;
      }

      if (isItColumnVector)
        nexTheta2Evaluate.block(0, 0, NParameter, 1) = (arx.col(0).cwiseProduct(StandardDev) + initial_theta);
      else
        nexTheta2Evaluate.block(0, 0, 1, NParameter) =
            (arx.col(0).cwiseProduct(StandardDev) + initial_theta).transpose();

      sample_index = 1;
    } else if (lambda == sample_index) {
      throw std::runtime_error(
          "CMAES getNextTheta2Evaluate: Please input all the cost before asking for more policy :) ");
    } else {
      if (isItColumnVector)
        nexTheta2Evaluate.block(0, 0, NParameter, 1) =
            (arx.col(sample_index).cwiseProduct(StandardDev) + initial_theta);
      else
        nexTheta2Evaluate.block(0, 0, 1, NParameter) =
            (arx.col(sample_index).cwiseProduct(StandardDev) + initial_theta).transpose();

      sample_index++;
    }
  }

  /*! setTheCostFromTheLastTheta
   * @param cost : cost from the previous theta you got
   */
  void setTheCostFromTheLastTheta(double cost) {
    cost_history(n_rolloutsEvaluated - 1) = cost;
    if (!readyToSampleAgain) {
      arfitness(cost_index) = cost;
      cost_index++;
    }

    if (cost < best_cost) {
      best_cost = cost;
      best_policy = arx.col(cost_index - 1);
    }

    if (cost_index == lambda) {
      cost_index = 0;
      readyToSampleAgain = true;
      updateCovariance();
    }
  }

  ///// donot call this method if you are using CMAES. This is only for adapting covariance only
  template<typename Derived, typename Dtype>
  void useCovarianceUpdate(std::vector<Eigen::Matrix<Derived, 4, 1> > &samples, std::vector<Dtype> &values) {
    LOG_IF(FATAL, samples.size() != arx.cols())<<"sample number don't match";
    for(int i=0; i< samples.size(); i++){
      arx.col(i) = samples[i].cwiseQuotient(StandardDev);
      arfitness(i) = values[i];
    }
    updateCovariance();
  }

  template<typename Derived>
  void getCovariance(Eigen::MatrixBase<Derived> &Cov) {
    Cov = sigma * sigma * StandardDev.asDiagonal() * C * StandardDev.asDiagonal();
  }

  template<typename Derived>
  void setBounds(Eigen::MatrixBase<Derived> &upper, Eigen::MatrixBase<Derived> &lower) {

    if (upper.rows() == NParameter)
      upperbound = (upper - initial_theta).cwiseQuotient(StandardDev);
    else if (upper.cols() == NParameter)
      upperbound = (upper.transpose() - initial_theta).cwiseQuotient(StandardDev);
    else
      throw std::runtime_error("CMAES setBounds: Please input a vector (either a row vector or a column vector)");

    if (lower.rows() == NParameter)
      lowerbound = (lower - initial_theta).cwiseQuotient(StandardDev);
    else if (lower.cols() == NParameter)
      lowerbound = (lower.transpose() - initial_theta).cwiseQuotient(StandardDev);
    else
      throw std::runtime_error("CMAES setBounds: Please input a vector (either a row vector or a column vector)");
  }

  bool isOptimizationDone() {
    return optimizationDone;
  }

  int getRolloutNumber() {
    return n_rolloutsEvaluated;
  }

  double getBestCost() {
    return best_cost;
  }

  double getSigma() {
    return sigma;
  }

  Eigen::VectorXd getCostHistory() {
    return cost_history.block(0, 0, n_rolloutsEvaluated - 1, 1);
  }

  template<typename Derived>
  void getBestEverSeenPolicy(Eigen::MatrixBase<Derived> &bestPolicy) {

    if (bestPolicy.cols() == NParameter)
      bestPolicy = best_policy.cwiseProduct(StandardDev) + initial_theta;
    else if (bestPolicy.rows() == NParameter)
      bestPolicy = (best_policy.cwiseProduct(StandardDev) + initial_theta).transpose();
    else
      throw std::runtime_error(
          "CMAES getBestEverSeenPolicy: Please input a vector (either a row vector or a column vector)");
  }

  template<typename Derived>
  void getEstimatedOptimalPolicy(Eigen::MatrixBase<Derived> &bestPolicy) {

    if (bestPolicy.rows() == NParameter)
      bestPolicy = xmean.cwiseProduct(StandardDev) + initial_theta;
    else if (bestPolicy.cols() == NParameter)
      bestPolicy = (xmean.cwiseProduct(StandardDev) + initial_theta).transpose();
    else
      throw std::runtime_error(
          "CMAES getEstimatedOptimalPolicy: Please input a vector (either a row vector or a column vector)");
  }

  void Sort(Eigen::MatrixXd &policy, Eigen::VectorXd &num, Eigen::VectorXd &index) {
    int i, j, flag = 1;    // set flag to 1 to start first pass
    double temp;             // holding variable
    int numLength = num.rows();
    Eigen::VectorXd tempV(numLength);
    for (int w = 0; w < numLength; w++)
      index(w) = w;
    for (i = 1; (i <= numLength) && flag; i++) {
      flag = 0;
      for (j = 0; j < (numLength - 1); j++) {
        if (num(j + 1) < num(j))      // ascending order simply changes to <
        {
          temp = num(j);             // swap elements
          num(j) = num(j + 1);
          num(j + 1) = temp;
          flag = 1;               // indicates that a swap occurred.
          temp = index(j);
          index(j) = index(j + 1);
          index(j + 1) = temp;
          tempV = policy.col(j);
          policy.col(j) = policy.col(j + 1);
          policy.col(j + 1) = tempV;

        }
      }
    }
    return;   //arrays are passed to functions by address; nothing is returned
  }

 private:

  void updateCovariance() {
    counteval = counteval + lambda;
    for (int k = 0; k < lambda; k++)
      arindex(k) = k;

    Sort(arx, arfitness, arindex);
    xold = xmean;
    xmean = arx.block(0, 0, NParameter, mu) * weights;

    ps = (1.0 - cs) * ps + sqrt(cs * (2.0 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
    hsig = ps.squaredNorm() / (1.0 - pow((1.0 - cs), (2.0 * counteval / lambda))) / NParameter
        < 2.0 + 4.0 / (NParameter + 1.0);
    pc = (1 - cc) * pc + hsig * sqrt(cc * (2.0 - cc) * mueff) * (xmean - xold) / sigma;

    for (int i = 0; i < mu; i++)
      artmp.col(i) = (arx.col(i) - xold) / sigma;

    C = (1.0 - c1 - cmu) * C + c1 * (pc * pc.transpose() + (1.0 - hsig) * cc * (2.0 - cc) * C)
        + cmu * artmp * weights.asDiagonal() * artmp.transpose();
    sigma = sigma * exp((cs / damps) * (ps.norm() / chiN - 1.0));

    if (counteval - eigeneval > lambda / (c1 + cmu) / NParameter / 10.0) {
      eigeneval = counteval;

      ////enforcing symmetry
      for (int k = 0; k < NParameter - 1; k++)
        for (int i = k + 1; i < NParameter; i++)
          C(k, i) = C(i, k);

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Dmat = svd.singularValues();
      Bmat_ = svd.matrixU();
      Dmat = Dmat.cwiseSqrt();
      invsqrtC = Bmat_ * Dmat.cwiseInverse().asDiagonal() * Bmat_.transpose();

      if (sigma < 1e-10) optimizationDone = true;
    }
  }

};

}
} // end namespace

#endif

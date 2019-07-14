//
// Created by jhwangbo on 17.04.17.
//

#ifndef RAI_SIMPLEMLP_HPP
#define RAI_SIMPLEMLP_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include "iostream"
#include <fstream>
#include <cmath>
#include <rai/noiseModel/NormalDistributionNoise.hpp>

namespace rai {

namespace FuncApprox {

enum class ActivationType {
  linear,
  relu,
  tanh,
  softsign
};

template<typename Dtype, ActivationType activationType>
struct Activation {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {}
};

template<typename Dtype>
struct Activation<Dtype, ActivationType::relu> {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {
    output = output.cwiseMax(0.0);
  }
};

template<typename Dtype>
struct Activation<Dtype, ActivationType::tanh> {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {
    output = output.array().tanh();
  }
};

template<typename Dtype>
struct Activation<Dtype, ActivationType::softsign> {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {
    for (int i = 0; i < output.size(); i++) {
      output[i] = output[i] / (std::abs(output[i]) + 1.0);
    }
  }
};

template<typename Dtype, int StateDim, int ActionDim, ActivationType activationType>
class MLP_fullyconnected {

 public:
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;

  MLP_fullyconnected(std::string fileName, std::vector<int> hiddensizes) :
      cov(Eigen::Matrix<Dtype, ActionDim, ActionDim>::Identity()), noise_(cov) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    layersizes.push_back(StateDim);
    layersizes.reserve(layersizes.size() + hiddensizes.size());
    layersizes.insert(layersizes.end(), hiddensizes.begin(), hiddensizes.end());
    layersizes.push_back(ActionDim);
    ///[input hidden output]

    params.resize(2 * (layersizes.size() - 1));
    Ws.resize(layersizes.size() - 1);
    bs.resize(layersizes.size() - 1);
    lo.resize(layersizes.size());
    Stdev.resize(ActionDim);

    std::stringstream parameterFileName;
    std::ifstream indata;
    indata.open(fileName);
    LOG_IF(FATAL, !indata) << "MLP file does not exists!" << std::endl;
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;


    ///assign parameters
    for (int i = 0; i < params.size(); i++) {
      int paramSize = 0;

      if (i % 2 == 0) ///W resize
      {
        Ws[i / 2].resize(layersizes[i / 2 + 1], layersizes[i / 2]);
        params[i].resize(layersizes[i / 2] * layersizes[i / 2 + 1]);
      }
      if (i % 2 == 1) ///b resize
      {
        bs[(i - 1) / 2].resize(layersizes[(i + 1) / 2]);
        params[i].resize(layersizes[(i + 1) / 2]);
      }

      while (std::getline(lineStream, cell, ',')) { ///Read param
        params[i](paramSize++) = std::stod(cell);
        if (paramSize == params[i].size()) break;
      }
      if (i % 2 == 0) ///W copy
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(Dtype) * Ws[i / 2].size());
      if (i % 2 == 1) ///b copy
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(Dtype) * bs[(i - 1) / 2].size());
    }
    int cnt = 0;

    while (std::getline(lineStream, cell, ',')) {
      Stdev[cnt++] = std::stod(cell);
      if (cnt == ActionDim) break;
    }

    Action temp;
    temp = Stdev;
    temp = temp.array().square(); //var
    noise_.initializeNoise();
    noise_.updateCovariance(temp.asDiagonal());
  }

  inline Action forward(State &state) {

    lo[0] = state;

    for (int cnt = 0; cnt < Ws.size() - 1; cnt++) {
      lo[cnt + 1] = Ws[cnt] * lo[cnt] + bs[cnt];
      activation_.nonlinearity(lo[cnt + 1]);
    }

    lo[lo.size() - 1] = Ws[Ws.size() - 1] * lo[lo.size() - 2] + bs[bs.size() - 1]; /// output layer
    return lo.back();
  }

  Action noisify(Action actionMean) {
    return noise_.noisify(actionMean);
  }

 private:
  std::vector<Eigen::Matrix<Dtype, -1, 1>> params;
  std::vector<Eigen::Matrix<Dtype, -1, -1>> Ws;
  std::vector<Eigen::Matrix<Dtype, -1, 1>> bs;
  std::vector<Eigen::Matrix<Dtype,-1,1>> lo;

  Activation<Dtype, activationType> activation_;
  Action Stdev;

  std::vector<int> layersizes;
  Eigen::Matrix<Dtype, ActionDim, ActionDim> cov;
  Noise_ noise_;
  bool isTanh = false;
};

}

}

#endif //RAI_SIMPLEMLP_HPP

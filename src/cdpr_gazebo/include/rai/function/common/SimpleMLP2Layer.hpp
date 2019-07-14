//
// Created by jhwangbo on 17.04.17.
//

#ifndef RAI_SIMPLEMLP_HPP
#define RAI_SIMPLEMLP_HPP

//
// Created by jhwangbo on 06.02.17.
//

#ifndef RAI_MLP_QUADCONTROL_HPP
#define RAI_MLP_QUADCONTROL_HPP
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include "iostream"
#include <fstream>
#include <cmath>

namespace rai {

template<int StateDim, int ActionDim, int firstLayerSize, int secondLayerSize>
class MLP_QuadControl {

 public:
  MLP_QuadControl(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    Eigen::VectorXd param(StateDim * firstLayerSize + firstLayerSize + firstLayerSize * secondLayerSize + secondLayerSize + secondLayerSize * ActionDim + ActionDim);
    std::stringstream parameterFileName;
    std::ifstream indata;
    indata.open(fileName);
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;
    int paramSize = 0;
    while (std::getline(lineStream, cell, ','))
      param(paramSize++) = std::stof(cell);

    int memoryPosition = 0;

    memcpy(w1.data(), param.data() + memoryPosition, sizeof(double) * w1.size());

    memoryPosition += w1.size();
    memcpy(b1.data(), param.data() + memoryPosition, sizeof(double) * b1.size());

    memoryPosition += b1.size();
    memcpy(w2.data(), param.data() + memoryPosition, sizeof(double) * w2.size());

    memoryPosition += w2.size();
    memcpy(b2.data(), param.data() + memoryPosition, sizeof(double) * b2.size());

    memoryPosition += b2.size();
    memcpy(w3.data(), param.data() + memoryPosition, sizeof(double) * w3.size());

    memoryPosition += w3.size();
    memcpy(b3.data(), param.data() + memoryPosition, sizeof(double) * b3.size());

  }

  //// every quantities are in the world frame
  Eigen::VectorXd forward(Eigen::VectorXd& input) {

    /// for newer versions of Eigen
    output_ = w3 * (w2 * (w1*input +b1).array().tanh().matrix() + b2).array().tanh().matrix() + b3;

    return output_;
  }
 private:

  Eigen::Matrix<double, firstLayerSize, StateDim> w1;
  Eigen::Matrix<double, firstLayerSize, 1> b1;
  Eigen::Matrix<double, secondLayerSize, firstLayerSize> w2;
  Eigen::Matrix<double, secondLayerSize, 1> b2;
  Eigen::Matrix<double, ActionDim, secondLayerSize> w3;
  Eigen::Matrix<double, ActionDim, 1> b3;
  Eigen::MatrixXd output_;

};

}



#endif //RAI_MLP_QUADCONTROL_HPP


#endif //RAI_SIMPLEMLP_HPP

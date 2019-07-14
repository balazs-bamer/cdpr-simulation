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

class MLP_QuadControl {

 public:
  MLP_QuadControl(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    Eigen::VectorXd param(18 * 64 + 64 + 64 * 64 + 64 + 64 * 4 + 4);
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

    transsThrust2GenForce << 0, 0, length_, -length_,
        -length_, length_, 0, 0,
        dragCoeff_, dragCoeff_, -dragCoeff_, -dragCoeff_,
        1, 1, 1, 1;
    transsThrust2GenForceInv = transsThrust2GenForce.inverse();

    positionScale_ = 0.5;
    angVelScale_ = 0.15;
    linVelScale_ = 0.5;

    positionOffset << -0.0126205,  -0.00247822, -0.000159031;
//    positionOffset.setZero();

  }

  //// every quantities are in the world frame
  void forward(Eigen::Vector4d &quat,
               Eigen::Vector3d &position,
               Eigen::Vector3d &angVel,
               Eigen::Vector3d &linVel,
               Eigen::Vector3d &targetPosition,
               Eigen::Vector4d &action) {

    Eigen::Matrix3d R;
    quatToRotMat(quat, R);
    Eigen::Matrix<double, 18, 1> state;
    state << R.col(0), R.col(1), R.col(2), (position-targetPosition+positionOffset) * positionScale_, angVel * angVelScale_, linVel * linVelScale_;
//    std::cout<<"state" <<state.transpose()<<std::endl;
//    std::cout<<"quat" <<quat.transpose()<<std::endl;

//    action = w3 * (w2 * (w1*state +b1).array().tanh().matrix() + b2).array().tanh().matrix() + b3;
    Eigen::Matrix<double, 64, 1> l1o = w1 * state + b1;
    for (int i = 0; i < 64; i++)
      l1o(i) = std::tanh(l1o(i));
    Eigen::Matrix<double, 64, 1> l2o = w2 * l1o + b2;
    for (int i = 0; i < 64; i++)
      l2o(i) = std::tanh(l2o(i));
    action = w3 * l2o + b3;

    double angle = 2.0 * std::acos(quat(0));
    double kp_rot = -0.2, kd_rot = -0.06;

    Eigen::Vector3d B_torque;

    if(angle > 1e-6)
      B_torque = kp_rot * angle * (R.transpose() * quat.tail(3))
        / std::sin(angle) + kd_rot * (R.transpose() * angVel);
    else
      B_torque = kd_rot * (R.transpose() * angVel);
    B_torque(2) = B_torque(2) * 0.15;
    Eigen::Vector4d genForce;
    genForce << B_torque, mass_ * 9.81;
//    std::cout<<"genForce "<<genForce.transpose()<<std::endl;
    Eigen::Vector4d thrust;
    thrust = transsThrust2GenForceInv * genForce + 2.0 * action;
    thrust = thrust.cwiseMax(1e-6);
    Eigen::Vector4d rotorSpeed;
//    rotorSpeed(0) = std::sqrt(thrust(0) / 0.00015677);
//    rotorSpeed(1) = std::sqrt(thrust(1) / 0.000123321);
//    rotorSpeed(2) = std::sqrt(thrust(2) / 0.00019696);
//    rotorSpeed(3) = std::sqrt(thrust(3) / 0.00015677);
    rotorSpeed = (thrust.array() / 8.5486e-6).sqrt();
    action = rotorSpeed;
  }

  inline void quatToRotMat(Eigen::Vector4d &q, Eigen::Matrix3d &R) {
    R << q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3), 2 * q(1) * q(2) - 2 * q(0) * q(3), 2 * q(0) * q(2)
        + 2 * q(1) * q(3),
        2 * q(0) * q(3) + 2 * q(1) * q(2), q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3), 2 * q(2) * q(3)
        - 2 * q(0) * q(1),
        2 * q(1) * q(3) - 2 * q(0) * q(2), 2 * q(0) * q(1) + 2 * q(2) * q(3), q(0) * q(0) - q(1) * q(1) - q(2) * q(2)
        + q(3) * q(3);
  }

 private:

  Eigen::Matrix<double, 64, 18> w1;
  Eigen::Matrix<double, 64, 1> b1;
  Eigen::Matrix<double, 64, 64> w2;
  Eigen::Matrix<double, 64, 1> b2;
  Eigen::Matrix<double, 4, 64> w3;
  Eigen::Matrix<double, 4, 1> b3;

  Eigen::Matrix4d transsThrust2GenForce;
  Eigen::Matrix4d transsThrust2GenForceInv;

  double length_ = 0.17;
  Position comLocation_, positionOffset;
  double dragCoeff_ = 0.016;
  double mass_ = 0.665;
  double orientationScale_, positionScale_, angVelScale_, linVelScale_;

};

}



#endif //RAI_MLP_QUADCONTROL_HPP

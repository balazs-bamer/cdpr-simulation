//
// Created by jhwangbo on 04.02.17.
//

#ifndef RAI_VECTORHELPER_HPP
#define RAI_VECTORHELPER_HPP
#include "rai/memory/Trajectory.hpp"
#include <vector>
#include "raiCommon/math/RAI_math.hpp"
#include "raiCommon/utils/RandomNumberGenerator.hpp"

namespace rai {
namespace Op {

class VectorHelper {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename Dtype, int stateDim, int actionDim, typename Derived>
  static void collectTerminalStates(std::vector<Memory::Trajectory < Dtype, stateDim, actionDim>
  > &trajSet,
  Eigen::MatrixBase<Derived> &output
  ) {
    for (int i = 0; i < trajSet.size(); i++)
      output.col(i) = trajSet[i].stateTraj.back();
  }

  template<typename Dtype, int stateDim, int actionDim, typename Derived>
  static void sampleRandomStates(std::vector<Memory::Trajectory < Dtype, stateDim, actionDim>> &trajSet,
  Eigen::MatrixBase<Derived> &output,
  int tailSteps,
      std::vector<std::pair<int, int> >
  &indx) {
    RandomNumberGenerator <Dtype> rn_;
    int sampleNumber = output.cols();
    for (int i = 0; i < sampleNumber; i++) {
      int branchID = rn_.intRand(0, trajSet.size() - 1);
      int trajLength = trajSet[branchID].stateTraj.size() - tailSteps;
      if (trajLength < 3) trajLength = trajSet[branchID].stateTraj.size();

      int startingPoint = rn_.intRand(0, trajLength - 2);
      output.col(i) = trajSet[branchID].stateTraj[startingPoint];
      indx.push_back(std::pair<int, int>(branchID, startingPoint));
    }
  }

  template<typename Dtype, int stateDim, int actionDim, typename Derived>
  static void collectNthStates(int n, std::vector<Memory::Trajectory < Dtype, stateDim, actionDim>
  > &trajSet,
  Eigen::MatrixBase<Derived> &output
  ) {
    for (int i = 0; i < trajSet.size(); i++)
      output.col(i) = trajSet[i].stateTraj[n];
  }

  template<typename Dtype>
  static Dtype computeAverage(std::vector<Dtype> vec) {
    Dtype sum = Dtype(0);
    for (auto &elem : vec)
      sum += elem;
    return sum / vec.size();
  }

  template<typename Dtype, int Dim>
  static void VectorofMatricestoTensor3D(const std::vector<Eigen::Matrix<Dtype,Dim,-1> > &input, Eigen::Tensor<Dtype,3> &output){

    int len = input[0].cols();
    int batchnum = input.size();

    output.resize(Dim, len, batchnum);
    output.setZero();
    for(int i = 0; i<batchnum;i++) {
      std::memcpy(output.data() + (Dim * len) * i,input[i].data(),sizeof(Dtype)*Dim*len);
    }
  }

  template<typename Dtype, int Dim>
  static void Tensor3DtoVectorofMatrices(const Eigen::Tensor<Dtype,3> &input, std::vector<Eigen::Matrix<Dtype,Dim,-1> > &output){
    output.clear();
    //(dim,maxlen,batch)
    int len = input.dimension(1);
    int batchnum = input.dimension(2);
    Eigen::Matrix<Dtype,Dim,-1> TempBatch;
    Eigen::Tensor<Dtype,3> TempTensor(Dim,len,batchnum);
    TempTensor = input;
    TempBatch.resize(Dim,len);

    for(int i = 0; i<batchnum;i++) {
      TempBatch.setZero();
      std::memcpy(TempBatch.data(),TempTensor.data()+ (Dim * len) * i, sizeof(Dtype)*Dim*len);
      output.push_back(TempBatch);
    }
  }
//  template<typename Dtype>
//  static void Matto3DTensor(const Eigen::Matrix<Dtype,-1,-1> &input, Eigen::Tensor<Dtype,3> &output){
//    //(dim,batch) -> (dim,1,batch)
//    int dim = input.rows();
//    int batch = input.cols();
//    Eigen::Matrix<Dtype,-1,-1> Temp;
//    Temp = input;
//
//    output.resize(dim,1,batch);
//    std::memcpy(output.data(),Temp.data(), sizeof(Dtype)*dim*batch);
//  }
//  template<typename Dtype>
//  static void TensortoMat (const Eigen::Tensor<Dtype,3> &input, Eigen::Matrix<Dtype,-1,-1> &output){
//    //(dim,?,?) -> (dim,?)
//    int dim = input.dimension(0);
//    int dim2 = input.dimension(1);
//    int dim3 = input.dimension(2);
//
////    Eigen::Matrix<Dtype,dim,dim2+dim3> Temp;
//    output.resize(dim,dim2*dim3);
//    std::memcpy(output.data(),input.data(), sizeof(Dtype)*dim*(dim2*dim3));
//
//  }


};
}
}

#endif //RAI_VECTORHELPER_HPP

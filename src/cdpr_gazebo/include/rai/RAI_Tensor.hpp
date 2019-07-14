//
// Created by jhwangbo on 12/08/17.
//

#ifndef RAI_RAI_TENSOR_HPP
#define RAI_RAI_TENSOR_HPP

#include "vector"
#include "string"
#include "tensorflow/core/public/session.h"
#include "Eigen/Core"
#include <algorithm>
#include "glog/logging.h"
#include <boost/utility/enable_if.hpp>
#include <Eigen/StdVector>
/* RAI tensor follows EIGEN Tensor indexing
 * The data is stored in tensorflow tensor
 * It provides an interface to Eigen::Tensor and Eigen::Matrix/Vector
 *
 *
 *
 *
 *
 */

namespace rai {
template<typename Dtype, int NDim>
class TensorBase {

  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>> EigenMat;
  typedef Eigen::TensorMap<Eigen::Tensor<Dtype, NDim>> EigenTensor;

 public:
  // empty constructor. Resize has to be called before use
  TensorBase(const std::string name = "") {
    namedTensor_.first = name;
    setDataType();
  }

  // empty data constructor
  TensorBase(const std::vector<int> dim, const std::string name = "") {
    init(dim, name);
  }

  // constant constructor
  TensorBase(const std::vector<int> dim, const Dtype constant, const std::string name = "") {
    init(dim, name);
    eTensor().setConstant(constant);
  }

  // copy constructor
  TensorBase(const TensorBase<Dtype, NDim>& copy){
    if (copy.size() != -1 ) {
      init(copy.dim(), copy.getName());
      memcpy(data(), copy.data(), size() * sizeof(Dtype));
    }
    else {
      setName(copy.getName());
      setDataType();
    }
  }

///Eigen Tensor constructor is abigous with std::vector<int> constructor ...
//  // copy constructor from Eigen Tensor
//  Tensor(const Eigen::Tensor<Dtype, NDim> &etensor, const std::string name = "") {
//    auto dims = etensor.dimensions();
//    std::vector<int> dim(dims.size());
//    for (int i = 0; i < dims.size(); i++)
//      dim[i] = dims[i];
//    Tensor(dim, name);
//    std::memcpy(data_->flat().data(), etensor.data(), sizeof(Dtype) * etensor.size());
//  }

  // copy construct from Eigen Matrix
  template<int Rows, int Cols>
  TensorBase(const Eigen::Matrix<Dtype, Rows, Cols> &emat, const std::string name = "") {
    LOG_IF(FATAL, NDim != 2) << "Specify the shape";
    std::vector<int> dim = {emat.rows(), emat.cols()};
    init(dim, name);
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), emat.data(), sizeof(Dtype) * emat.size());
  }

  // this constructor is used when the resulting tensor dim is not 2D
  template<int Rows, int Cols>
  TensorBase(const Eigen::Matrix<Dtype, Rows, Cols> &emat, std::vector<int> dim, const std::string name = "") {
    init(dim, name);
    LOG_IF(FATAL, emat.size() != size_) << "size mismatch";
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), emat.data(), sizeof(Dtype) * emat.size());
  }

  virtual ~TensorBase(){};

  ////////////////////////////
  /////// casting methods ////
  ////////////////////////////
  operator std::pair<std::string, tensorflow::Tensor>() {
    return namedTensor_;
  };

  operator tensorflow::Tensor() {
    return namedTensor_.second;
  };

  template<int Rows, int Cols>
  operator Eigen::Matrix<Dtype, Rows, Cols>() {
    LOG_IF(FATAL, dim_.size() > 2) << "This method works upto 2D Tensor";
//    LOG_IF(FATAL, Rows != dim_[0] || Cols != dim_[1]) << "dimension mismatch";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat;
  };

  operator EigenTensor() {
    EigenTensor mat(namedTensor_.second.flat<Dtype>().data(), esizes_);
    return mat;
  }

  //////////////////////////
  /// Eigen Tensor mirror///
  //////////////////////////
  EigenTensor eTensor() {
    return EigenTensor(namedTensor_.second.flat<Dtype>().data(), esizes_);
  }

  void setConstant(const Dtype constant) {
    eTensor().setConstant(constant);
  }

  void setZero() {
    eTensor().setZero();
  }

  void setRandom() {
    eTensor().setRandom();
  }

  Dtype *data() {
    return namedTensor_.second.flat<Dtype>().data();
  }
  const Dtype *data() const {
    return namedTensor_.second.flat<Dtype>().data();
  }
  ////////////////////////////////
  /// tensorflow tensor mirror ///
  ////////////////////////////////
  tensorflow::TensorShape tfShape() {
    return namedTensor_.second.shape();
  }

  const tensorflow::Tensor &tfTensor() const {
    return namedTensor_.second;
  }

  std::vector<tensorflow::Tensor> &output() {
    return vecTens;
  }

  ///////////////////////////////
  ////////// operators //////////
  ///////////////////////////////
  TensorBase<Dtype, NDim>& operator=(const TensorBase<Dtype, NDim> &rTensor) {
    ///copy everything except for the name
    dim_ = rTensor.dim_;
    dim_inv_ = rTensor.dim_inv_;
    size_ = rTensor.size_;
    esizes_ = rTensor.esizes_;
    vecTens = rTensor.vecTens;
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), rTensor.namedTensor_.second.template flat<Dtype>().data(), sizeof(Dtype) * size_);
    return *this;
  }

  TensorBase<Dtype, NDim>& operator=(const tensorflow::Tensor &tfTensor) {
    LOG_IF(FATAL, dim_inv_ != tfTensor.shape()) << "Tensor shape mismatch";
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), tfTensor.flat<Dtype>().data(), sizeof(Dtype) * size_);
    return *this;
  }

  TensorBase<Dtype, NDim>& operator=(const std::string name) {
    setName(name);
    return *this;
  }

  TensorBase<Dtype, NDim>& operator=(const Eigen::Tensor<Dtype, NDim> &eTensor) {
    for (int i = 0; i < NDim; i++)
      LOG_IF(FATAL, dim_[i] != eTensor.dimension(i))
      << "Tensor size mismatch: " << i << "th Dim " << dim_[i] << "vs" << eTensor.dimension(i);
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), eTensor.data(), sizeof(Dtype) * size_);
    return *this;
  }

  TensorBase<Dtype, NDim>& operator-=(const TensorBase<Dtype, NDim> &other) {
    LOG_IF(FATAL, size() != other.size())<<"size mismatch";
    for (int i = 0; i < size(); i++)
      data()[i] -= other.data()[i];
    return *this;
  }

  TensorBase<Dtype, NDim>& operator+=(const TensorBase<Dtype, NDim> &other) {
    LOG_IF(FATAL, size() != other.size())<<"size mismatch";
    for (int i = 0; i < size(); i++)
      data()[i] += other.data()[i];
    return *this;
  }

  friend TensorBase<Dtype, NDim> operator-(const TensorBase<Dtype, NDim> &lhs, const TensorBase<Dtype, NDim> & rhs) {
    TensorBase<Dtype, NDim> result(lhs);
    result -= rhs;
    return result;
  }
  friend TensorBase<Dtype, NDim>&& operator-(TensorBase<Dtype, NDim> &&lhs, const TensorBase<Dtype, NDim> & rhs) {
    return std::move(lhs -= rhs);
  }
  friend TensorBase<Dtype, NDim>&& operator-(const TensorBase<Dtype, NDim> &lhs, TensorBase<Dtype, NDim> && rhs) {
    return std::move(rhs -= lhs);
  }
  friend TensorBase<Dtype, NDim>&& operator-(TensorBase<Dtype, NDim> &&lhs, TensorBase<Dtype, NDim> && rhs) {
    return std::move(rhs -= lhs);
  }

  friend TensorBase<Dtype, NDim> operator+(const TensorBase<Dtype, NDim> &lhs, const TensorBase<Dtype, NDim> & rhs) {
    TensorBase<Dtype, NDim> result(lhs);
    result += rhs;
    return result;
  }
  friend TensorBase<Dtype, NDim>&& operator+(TensorBase<Dtype, NDim> &&lhs, const TensorBase<Dtype, NDim> & rhs) {
    return std::move(lhs += rhs);
  }
  friend TensorBase<Dtype, NDim>&& operator+(const TensorBase<Dtype, NDim> &lhs, TensorBase<Dtype, NDim> && rhs) {
    return std::move(rhs += lhs);
  }
  friend TensorBase<Dtype, NDim>&& operator+(TensorBase<Dtype, NDim> &&lhs, TensorBase<Dtype, NDim> && rhs) {
    return std::move(lhs += rhs);
  }
  // TODO: Matmul, *

  ///scalar operators
  rai::TensorBase<Dtype, NDim>& operator+=(const Dtype rhs) {
    for (int i = 0; i < size(); i++)
      data()[i] += rhs;
    return *this;
  }
  TensorBase<Dtype, NDim>& operator-=(const Dtype rhs) {
    for (int i = 0; i < size(); i++)
      data()[i] -= rhs;
    return *this;
  }
  TensorBase<Dtype, NDim>& operator*=(const Dtype rhs) {
    for (int i = 0; i < size(); i++)
      data()[i] *= rhs;
    return *this;
  }
  TensorBase<Dtype, NDim>& operator/=(const Dtype rhs) {
    for (int i = 0; i < size(); i++)
      data()[i] /= rhs;
    return *this;
  }

  friend TensorBase<Dtype, NDim> operator+(TensorBase<Dtype, NDim> &lhs, Dtype rhs) {
  TensorBase<Dtype, NDim> result(lhs);
    result += rhs;
    return result;
  }
  friend TensorBase<Dtype, NDim> operator+(Dtype lhs, TensorBase<Dtype, NDim> &rhs) {
    return operator+(rhs, lhs);
  }

  friend TensorBase<Dtype, NDim> operator-(TensorBase<Dtype, NDim> &lhs, Dtype rhs) {
    TensorBase<Dtype, NDim> result(lhs);
    result -= rhs;
    return result;
  }
  friend TensorBase<Dtype, NDim> operator-(Dtype lhs, TensorBase<Dtype, NDim> &rhs) {
    return operator-(rhs, lhs);
  }
  friend TensorBase<Dtype, NDim> operator*(TensorBase<Dtype, NDim> &lhs, Dtype rhs) {
    TensorBase<Dtype, NDim> result(lhs);
    result *= rhs;
    return result;
  }
  friend TensorBase<Dtype, NDim> operator*(Dtype lhs, TensorBase<Dtype, NDim> &rhs) {
    return operator*(rhs, lhs);
  }
  friend TensorBase<Dtype, NDim> operator/(TensorBase<Dtype, NDim> &lhs, Dtype rhs) {
    TensorBase<Dtype, NDim> result(lhs);
    result /= rhs;
    return result;
  }

  template<int rows, int cols>
  void copyDataFrom(const Eigen::Matrix<Dtype, rows, cols> &eMat) {
    LOG_IF(FATAL, size_ != eMat.rows() * eMat.cols())
    << "Data size mismatch: " << size_ << "vs" << eMat.rows() * eMat.cols();
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), eMat.data(), sizeof(Dtype) * size_);
  }

  void copyDataFrom(const tensorflow::Tensor &tfTensor) {
    LOG_IF(FATAL, size_ != tfTensor.flat<Dtype>().size())
    << "Data size mismatch: " << size_ << "vs" << tfTensor.flat<Dtype>().size();
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), tfTensor.flat<Dtype>().data(), sizeof(Dtype) * size_);
  }

//  Dtype *operator[](int x) {
//    return vecTens[x].flat<Dtype>().data();
//  };

  Dtype& operator[](int i) {
    return namedTensor_.second.flat<Dtype>().data()[i];
  }

  Dtype& at(int i){
    return namedTensor_.second.flat<Dtype>().data()[i];
  }

  const Dtype& operator[](int i) const {
    return namedTensor_.second.flat<Dtype>().data()[i];
  }

  const Dtype& at(int i) const {
    return namedTensor_.second.flat<Dtype>().data()[i];
  }

  ///////////////////////////////
  /////////// generic ///////////
  ///////////////////////////////
  const std::string &getName() const { return namedTensor_.first; }
  void setName(const std::string &name) { namedTensor_.first = name; }
  const std::vector<int> &dim() const { return dim_; }
  const int dim(int idx) const { return dim_[idx]; }
  int rows() { return dim_[0]; }
  int cols() { return dim_[1]; }
  int batches() { return dim_[2]; }
  const int size() const { return size_; }

  /// you lose all data calling resize
  void resize(const std::vector<int> dim) {
    LOG_IF(FATAL, NDim != dim.size()) << "tensor rank mismatch";
    dim_inv_.Clear();
    size_ = 1;
    dim_ = dim;
    for (int i = dim.size() - 1; i > -1; i--) {
      dim_inv_.AddDim(dim[i]);
      size_ *= dim[i];
    }
    for (int d = 0; d < NDim; d++) {
      esizes_[d] = dim_[d];
    }
    setDataType();
    namedTensor_.second = tensorflow::Tensor(dtype_, dim_inv_);
  }

 protected:

  void init(const std::vector<int> dim, const std::string name = "") {
    LOG_IF(FATAL, dim.size() != NDim)
    << "specified dimension differs from the Dimension in the template parameter";
    namedTensor_.first = name;
    setDataType();
    size_ = 1;
    dim_ = dim;
    for (int i = dim_.size() - 1; i > -1; i--) {
      dim_inv_.AddDim(dim_[i]);
      size_ *= dim_[i];
    }
    namedTensor_ = {name, tensorflow::Tensor(dtype_, dim_inv_)};
    for (int d = 0; d < NDim; d++)
      esizes_[d] = dim_[d];
  }

  void setDataType() {
    if (typeid(Dtype) == typeid(float))
      dtype_ = tensorflow::DataType::DT_FLOAT;
    else if (typeid(Dtype) == typeid(double))
      dtype_ = tensorflow::DataType::DT_DOUBLE;
    else if (typeid(Dtype) == typeid(int))
      dtype_ = tensorflow::DataType::DT_INT32;
  }

  tensorflow::DataType dtype_;
  std::pair<std::string, tensorflow::Tensor> namedTensor_;
  std::vector<int> dim_;
  tensorflow::TensorShape dim_inv_; /// tensorflow dimension
  long int size_=-1;
  Eigen::DSizes<Eigen::DenseIndex, NDim> esizes_;
  std::vector<tensorflow::Tensor> vecTens;
};

/// Tensor methods
template<typename Dtype, int NDim>
class Tensor : public rai::TensorBase<Dtype, NDim> {
  typedef rai::TensorBase<Dtype, NDim> TensorBase;
 public:
  using TensorBase::TensorBase;
  using TensorBase::eTensor;
  using TensorBase::operator =;
  using TensorBase::operator [];
  using TensorBase::operator std::pair<std::string, tensorflow::Tensor>;
  using TensorBase::operator tensorflow::Tensor;

};

/// 1D method
template<typename Dtype>
class Tensor<Dtype, 1> : public rai::TensorBase<Dtype, 1> {

  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, 1>> EigenMat;
  typedef Eigen::TensorMap<Eigen::Tensor<Dtype, 1>, Eigen::Aligned> EigenTensor;
  typedef rai::TensorBase<Dtype, 1> TensorBase;
  using TensorBase::namedTensor_;
  using TensorBase::dim_inv_;
  using TensorBase::dim_;
  using TensorBase::dtype_;

 public:
  using TensorBase::TensorBase;
  using TensorBase::eTensor;
  using TensorBase::resize;
  using TensorBase::operator=;
  using TensorBase::operator[];

  ///////////////////////////////
  ////////// operators //////////
  ///////////////////////////////

  template<int rows, int cols>
  void operator=(const Eigen::Matrix<Dtype, rows, cols> &eMat) {
    LOG_IF(FATAL, dim_[0] != eMat.rows())
    << "matrix size mismatch: " << dim_[0] << "X1" << "vs" << eMat.rows() << "X1";
    std::memcpy(namedTensor_.second.template flat<Dtype>().data(), eMat.data(), sizeof(Dtype) * this->size_);
  }

  void operator=(const Eigen::Matrix<Dtype, -1, -1> &eMat) {
    LOG_IF(FATAL, dim_[0] != eMat.rows())
    << "matrix size mismatch: " << dim_[0] << "X1" << "vs" << eMat.rows() << "X1";
    std::memcpy(namedTensor_.second.template flat<Dtype>().data(), eMat.data(), sizeof(Dtype) * this->size_);
  }

  ////////////////////////////
  /// Eigen Methods mirror ///
  ////////////////////////////
  EigenMat eMat() {
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data(), TensorBase::dim_[0],1);
    return mat;
  }

  EigenMat block(int startIdx, int size)
  {
    LOG_IF(FATAL, startIdx + size > dim_[0]) << "requested segment exceeds Tensor dimension (startIdx+size v.s lastIdx = " << startIdx + size -1  << " v.s. "<<dim_[0] -1 ;
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data() + startIdx, size, 1);
    return mat;
  }

  /// you lose all data calling resize
  void resize(int n) {
    std::vector<int> dim = {n};
    resize(dim);
  }
};

/// 2D method
template<typename Dtype>
class Tensor<Dtype, 2> : public rai::TensorBase<Dtype, 2> {

  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>> EigenMat;
  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> EigenMatStride;

  typedef Eigen::TensorMap<Eigen::Tensor<Dtype, 2>, Eigen::Aligned> EigenTensor;
  typedef rai::TensorBase<Dtype, 2> TensorBase;
  using TensorBase::namedTensor_;
  using TensorBase::dim_inv_;
  using TensorBase::dim_;
  using TensorBase::dtype_;

 public:
  using TensorBase::TensorBase;
  using TensorBase::eTensor;
  using TensorBase::resize;
  using TensorBase::operator=;
  using TensorBase::operator[];

  ///////////////////////////////
  ////////// operators //////////
  ///////////////////////////////

  template<int rows, int cols>
  void operator=(const Eigen::Matrix<Dtype, rows, cols> &eMat) {
    LOG_IF(FATAL, dim_[0] != eMat.rows() || dim_[1] != eMat.cols())
    << "matrix size mismatch: " << dim_[0] << "X" << dim_[1] << "vs" << eMat.rows() << "X"
    << eMat.cols();
    std::memcpy(namedTensor_.second.template flat<Dtype>().data(), eMat.data(), sizeof(Dtype) * this->size_);
  }

  void operator=(const Eigen::Matrix<Dtype, -1, -1> &eMat) {
    LOG_IF(FATAL, dim_[0] != eMat.rows() || dim_[1] != eMat.cols())
    << "matrix size mismatch: " << dim_[0] << "X" << dim_[1] << "vs" << eMat.rows() << "X"
    << eMat.cols();
    std::memcpy(namedTensor_.second.template flat<Dtype>().data(), eMat.data(), sizeof(Dtype) * this->size_);
  }

  ////////////////////////////
  /// Eigen Methods mirror ///
  ////////////////////////////
  typename EigenMat::ColXpr col(int colId) {
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat.col(colId);
  }

  typename EigenMat::RowXpr row(int rowId) {
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat.row(rowId);
  }

  EigenMatStride block(int rowStart, int colStart, int rowDim, int colDim) {
    EigenMatStride mat(namedTensor_.second.template flat<Dtype>().data() + rowStart + dim_[0] * colStart, rowDim, colDim, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(dim_[0], 1));
    return mat;
  }

  EigenMat eMat() {
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat;
  }

  /// you lose all data calling resize
  void resize(int rows, int cols) {
    std::vector<int> dim = {rows, cols};
    resize(dim);
  }


  void conservativeResize(int rows, int cols) {
    if(dim_.size() == 0) resize({rows,cols});
    else if(dim_[0] == rows && dim_[1] == cols) return;

    Eigen::Matrix<Dtype, -1, -1> Temp(dim_[0],dim_[1]);
    Temp = eMat();
    resize({rows,cols});
    Temp.conservativeResize(rows,cols);
    eMat() = Temp;
  }

  void removeCol(int colID) {
    tensorflow::Tensor Temp(dtype_, dim_inv_);
    memcpy(Temp.flat<Dtype>().data(),
           namedTensor_.second.template flat<Dtype>().data(),
           sizeof(Dtype) * namedTensor_.second.template flat<Dtype>().size());

    resize(dim_[0], dim_[1] - 1);

    memcpy(namedTensor_.second.template flat<Dtype>().data(),
           Temp.flat<Dtype>().data(),
           sizeof(Dtype) * dim_[0] * colID);

    if (colID < dim_[1] - 1) {
      memcpy(namedTensor_.second.template flat<Dtype>().data() + colID * dim_[0],
             Temp.flat<Dtype>().data() + (colID + 1) * dim_[0],
             sizeof(Dtype) * dim_[0] * (dim_[1] - colID - 1));
    }
  }
};
/// 3D method
template<typename Dtype>
class Tensor<Dtype, 3> : public rai::TensorBase<Dtype, 3> {

  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>> EigenMat;
  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>> EigenMatStride;

  typedef Eigen::TensorMap<Eigen::Tensor<Dtype, 3>, Eigen::Aligned> EigenTensor;
  typedef rai::TensorBase<Dtype, 3> TensorBase;

  using TensorBase::namedTensor_;
  using TensorBase::dim_inv_;
  using TensorBase::dim_;
  using TensorBase::dtype_;

 public:
  using TensorBase::TensorBase;
  using TensorBase::eTensor;
  using TensorBase::resize;
  using TensorBase::operator=;
  using TensorBase::operator[];

  /// you lose all data calling resize
  void resize(int rows, int cols, int batches) {
    std::vector<int> dim = {rows, cols, batches};
    resize(dim);
  }

  void conservativeResize(int rows, int cols, int batches) {
    if(dim_.size() == 0) resize({rows,cols,batches});
    else if(dim_[0] == rows && dim_[1] == cols && dim_[2] ==batches ) return;
    std::vector<int> dim = {rows, cols, batches};

    if(dim_[1] == cols && dim_[2] == batches){
      Eigen::Matrix<Dtype, -1, -1> Temp(rows , cols * batches);
      int commonRow = std::min(dim_[0], rows);
      Temp.block(0,0,commonRow,cols * batches) = EigenMatStride(namedTensor_.second.template flat<Dtype>().data(), commonRow, cols * batches, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0],1));
      resize(dim);
      memcpy(namedTensor_.second.template flat<Dtype>().data(), Temp.data(),
             sizeof(Dtype) * Temp.size());
    }
    else if(dim_[0] == rows && dim_[2] == batches){
      Eigen::Matrix<Dtype, -1, -1> Temp(rows*cols, batches);
      int commonCol = std::min(dim_[1], cols);
      Temp.block(0,0,rows * commonCol, batches) = EigenMatStride(namedTensor_.second.template flat<Dtype>().data(), rows * commonCol, batches, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0]*dim_[1],1));
      resize(dim);
      memcpy(namedTensor_.second.template flat<Dtype>().data(), Temp.data(),
             sizeof(Dtype) * Temp.size());
    }
    else if(dim_[0] == rows && dim_[1] == cols){
      Eigen::Matrix<Dtype, -1, -1> Temp(rows*cols, batches);
      int commonBat = std::min(dim_[2], batches);
      Temp.block(0,0,rows * cols, commonBat) = EigenMatStride(namedTensor_.second.template flat<Dtype>().data(), rows * cols, commonBat, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0]*dim_[1],1));
      resize(dim);
      memcpy(namedTensor_.second.template flat<Dtype>().data(), Temp.data(),
             sizeof(Dtype) * Temp.size());
    }
    else if(dim_[0] == rows) {
      Eigen::Matrix<Dtype, -1, -1> Temp(rows*cols, batches);
      int commonCol = std::min(dim_[1], cols);
      int commonBat = std::min(dim_[2], batches);
      Temp.block(0,0,rows*commonCol, commonBat) = EigenMatStride(namedTensor_.second.template flat<Dtype>().data(), rows*commonCol, commonBat, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0]*dim_[1],1));
      resize(dim);
      memcpy(namedTensor_.second.template flat<Dtype>().data(), Temp.data(),
             sizeof(Dtype) * Temp.size());
    }
    else if(dim_[1] == cols) {
      Eigen::Matrix<Dtype, -1, -1> Temp(rows, cols*batches);
      int commonRow = std::min(dim_[0], rows);
      int commonBat = std::min(dim_[2], batches);
      Temp.block(0,0,commonRow,cols * commonBat) = EigenMatStride(namedTensor_.second.template flat<Dtype>().data(), commonRow, cols * commonBat, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0],1));
      resize(dim);
      memcpy(namedTensor_.second.template flat<Dtype>().data(), Temp.data(),
             sizeof(Dtype) * Temp.size());
    }
    else if(dim_[2] == batches) {
      int commonRow = std::min(dim_[0], rows);
      int commonCol = std::min(dim_[1], cols);
      Eigen::Matrix<Dtype, -1, -1> Temp(dim_[0]*commonCol, batches);
      Temp = EigenMatStride(namedTensor_.second.template flat<Dtype>().data(), dim_[0]*commonCol, batches, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0]*dim_[1],1));

      Eigen::Matrix<Dtype, -1, -1> Temp2(rows,cols* batches);
      Temp2.block(0,0,commonRow,commonCol* batches) = EigenMatStride(Temp.data(), commonRow,commonCol*batches, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0],1));
      resize(dim);
      memcpy(namedTensor_.second.template flat<Dtype>().data(), Temp2.data(),
             sizeof(Dtype) * Temp2.size());
    }
    else{
      int commonRow = std::min(dim_[0], rows);
      int commonCol = std::min(dim_[1], cols);
      int commonBat = std::min(dim_[2], batches);
      Eigen::Matrix<Dtype, -1, -1> Temp(dim_[0]*commonCol, commonBat);
      Temp = EigenMatStride(namedTensor_.second.template flat<Dtype>().data(), dim_[0]*commonCol, commonBat, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0]*dim_[1],1));

      Eigen::Matrix<Dtype, -1, -1> Temp2(rows,cols*batches);
      Temp2.block(0,0,commonRow,commonCol* commonBat) = EigenMatStride(Temp.data(), commonRow,commonCol*commonBat, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0],1));
      resize(dim);
      memcpy(namedTensor_.second.template flat<Dtype>().data(), Temp2.data(),
             sizeof(Dtype) * Temp2.size());
    }
  }

  void removeBatch(int batchId) {
    tensorflow::Tensor Temp(dtype_, dim_inv_);
    memcpy(Temp.flat<Dtype>().data(),
           namedTensor_.second.template flat<Dtype>().data(),
           sizeof(Dtype) * namedTensor_.second.template flat<Dtype>().size());

    resize(dim_[0], dim_[1], dim_[2] - 1);

    memcpy(namedTensor_.second.template flat<Dtype>().data(),
           Temp.flat<Dtype>().data(),
           sizeof(Dtype) * dim_[0] * dim_[1] * batchId);

    if (batchId < dim_[2] - 1) {
      memcpy(namedTensor_.second.template flat<Dtype>().data() + batchId * dim_[0] * dim_[1],
             Temp.flat<Dtype>().data() + (batchId + 1) * dim_[0] * dim_[1],
             sizeof(Dtype) * dim_[0] * dim_[1] * (dim_[2] - batchId - 1));
    }
  }

  typename EigenMat::ColXpr col(int batchId, int colId) {
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data() + batchId * dim_[0] * dim_[1], dim_[0], dim_[1]);
    return mat.col(colId);
  }

  typename EigenMat::RowXpr row(int batchId, int rowId) {
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data() + batchId * dim_[0] * dim_[1], dim_[0], dim_[1]);
    return mat.row(rowId);
  }

  /// row(rowId) of all the batches)
  /// shape = dim[1] * dim[2]
  /// [ row(0,rowId).transpose(), row(1,rowId).transpose(), ... ]
  EigenMatStride row(int rowId) {
    EigenMatStride mat(namedTensor_.second.template flat<Dtype>().data() + rowId, dim_[1], dim_[2], Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0] * dim_[1],dim_[0]));
    return mat;
  }

  /// col(colId) of all the batches)
  /// shape = dim[0] * dim[2]
  /// [ col(0,colId), col(1,colId), ... ]
  EigenMatStride col(int colId) {
    EigenMatStride mat(namedTensor_.second.template flat<Dtype>().data() + colId * dim_[0], dim_[0], dim_[2],Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(dim_[0] * dim_[1],1));
    return mat;
  }

  EigenMat batch(int batchId) {
    EigenMat mat(namedTensor_.second.template flat<Dtype>().data() + batchId * dim_[0] * dim_[1], dim_[0], dim_[1]);
    return mat;
  }

  EigenTensor batchBlock(int startBatchID, int size){
    LOG_IF(FATAL,startBatchID + size >dim_[2]) << "endBatchID exceeds last batch ID: "<< startBatchID + size-1  << "vs" << dim_[2]-1;
    EigenTensor ten(namedTensor_.second.template flat<Dtype>().data() + startBatchID * dim_[0] * dim_[1], dim_[0], dim_[1],size);
    return ten;
  }

  template<int rows, int cols>
  void partiallyFillBatch(int batchId, Eigen::Matrix<Dtype, rows, cols> &eMat) {
    LOG_IF(FATAL, dim_[0] != rows) << "Column size mismatch ";
    std::memcpy(namedTensor_.second.template flat<Dtype>().data() + batchId * dim_[0] * dim_[1],
                eMat.data(), sizeof(Dtype) * eMat.size());
  }

  template<int rows>
  void partiallyFillBatch(int batchId, std::vector<Eigen::Matrix<Dtype, rows, 1>> &eMatVec, int ignoreLastN = 0) {
    LOG_IF(FATAL, dim_[0] != eMatVec[0].rows()) << "Column size mismatch " <<  dim_[0] << "vs." <<eMatVec[0].rows() ;
    for (int colId = 0; colId < eMatVec.size() - ignoreLastN; colId++)
      batch(batchId).col(colId) = eMatVec[colId];
  }
};


template<typename Dtype, int NDim>
std::ostream &operator<<(std::ostream &os, TensorBase<Dtype, NDim> &m) {
  os << m.eTensor();
  return os;
}

}

#endif //RAI_RAI_TENSOR_HPP

#pragma once

#include <cassert>
#include <memory>
#include <vector>

namespace MappedData {
namespace Linalg {

template <class T>
class Matrix {
 public:
  Matrix(size_t rows, size_t cols);
  Matrix(size_t rows, size_t cols, T value);
  Matrix(size_t rows, size_t cols,
         const std::initializer_list<std::initializer_list<T>>& values);
  Matrix(const Matrix<T>&);
  Matrix(Matrix<T>&&);
  Matrix<T>& operator=(const Matrix<T>&);
  Matrix<T>& operator=(Matrix<T>&&);
  Matrix<T> operator*(const Matrix<T>&);
  ~Matrix();

  size_t rows() const;
  size_t cols() const;

  const T& operator()(size_t row, size_t col) const;
  T& operator()(size_t row, size_t col);

  void print() const;

 private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};

extern template class Matrix<float>;
extern template class Matrix<double>;
extern template class Matrix<long double>;

namespace {

template <class T, class U>
struct OpTraits {
  using mul = decltype(T() * U());
};

}  // namespace

template <class T, class U>
std::enable_if_t<std::is_floating_point_v<typename OpTraits<T, U>::mul>,
                 Matrix<typename OpTraits<T, U>::mul>>
operator*(Matrix<T>&& lhs, Matrix<U>&& rhs) {
  assert(lhs.cols() == rhs.rows());
  Matrix<typename OpTraits<T, U>::mul> result(lhs.rows(), rhs.cols(),
                                              typename OpTraits<T, U>::mul{});
  for (size_t i = 0; i < lhs.rows(); ++i) {
    for (size_t j = 0; j < rhs.cols(); ++j) {
      for (size_t k = 0; k < lhs.cols(); ++k) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

template <class T, class U>
std::enable_if_t<std::is_floating_point_v<typename OpTraits<T, U>::mul>,
                 Matrix<typename OpTraits<T, U>::mul>>
operator*(const Matrix<T>& lhs, Matrix<U>&& rhs) {
  assert(lhs.cols() == rhs.rows());
  Matrix<typename OpTraits<T, U>::mul> result(lhs.rows(), rhs.cols(),
                                              typename OpTraits<T, U>::mul{});
  for (size_t i = 0; i < lhs.rows(); ++i) {
    for (size_t j = 0; j < rhs.cols(); ++j) {
      for (size_t k = 0; k < lhs.cols(); ++k) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

template <class T, class U>
std::enable_if_t<std::is_floating_point_v<typename OpTraits<T, U>::mul>,
                 Matrix<typename OpTraits<T, U>::mul>>
operator*(Matrix<T>&& lhs, const Matrix<U>& rhs) {
  assert(lhs.cols() == rhs.rows());
  Matrix<typename OpTraits<T, U>::mul> result(lhs.rows(), rhs.cols(),
                                              typename OpTraits<T, U>::mul{});
  for (size_t i = 0; i < lhs.rows(); ++i) {
    for (size_t j = 0; j < rhs.cols(); ++j) {
      for (size_t k = 0; k < lhs.cols(); ++k) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

template <class T, class U>
std::enable_if_t<std::is_floating_point_v<typename OpTraits<T, U>::mul>,
                 Matrix<typename OpTraits<T, U>::mul>>
operator*(const Matrix<T>& lhs, const Matrix<U>& rhs) {
  assert(lhs.cols() == rhs.rows());
  Matrix<typename OpTraits<T, U>::mul> result(lhs.rows(), rhs.cols(),
                                              typename OpTraits<T, U>::mul{});
  for (size_t i = 0; i < lhs.rows(); ++i) {
    for (size_t j = 0; j < rhs.cols(); ++j) {
      for (size_t k = 0; k < lhs.cols(); ++k) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

template <class T>
Matrix<T> CholeskyFactorization(const Matrix<T>& input);

extern template Matrix<float> CholeskyFactorization(const Matrix<float>& input);
extern template Matrix<double> CholeskyFactorization(
    const Matrix<double>& input);
extern template Matrix<long double> CholeskyFactorization(
    const Matrix<long double>& input);

}  // namespace Linalg

namespace Random {

template <class T>
class MultivariateNormalDistribution {
 public:
  MultivariateNormalDistribution(const Linalg::Matrix<T>& A,
                                 const std::vector<T>& mean);
  ~MultivariateNormalDistribution();
  Linalg::Matrix<T> operator()(size_t n);

 private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
  // Matrix<T> choleskyFactorization_;
};

extern template class MultivariateNormalDistribution<float>;
extern template class MultivariateNormalDistribution<double>;
extern template class MultivariateNormalDistribution<long double>;

}  // namespace Random
}  // namespace MappedData
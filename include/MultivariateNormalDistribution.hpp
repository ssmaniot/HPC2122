#pragma once

#include <memory>
#include <random>

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
  // MultivariateNormalDistribution(const Matrix<T>& A,
  //                                const std::vector<T>& mean);
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
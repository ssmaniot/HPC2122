#include "MultivariateNormalDistribution.hpp"

#include <omp.h>

#include <cassert>
#include <iostream>

namespace MappedData {
namespace Random {

namespace {

std::mt19937& RNG() {
  static std::random_device rd;
  static std::mt19937 mt_{rd()};
  return mt_;
}

// Function for Cholesky Factorization of Matrix
template <typename T>
Linalg::Matrix<T> CholeskyFactorization(const Linalg::Matrix<T>& input) {
  assert(input.rows() == input.cols());
  size_t n = input.rows();
  Linalg::Matrix<T> result(n, n, T{});
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < i; ++k) {
      T value = input(i, k);
      for (size_t j = 0; j < k; ++j) value -= result(i, j) * result(k, j);
      result(i, k) = value / result(k, k);
    }
    T value = input(i, i);
    for (size_t j = 0; j < i; ++j) value -= result(i, j) * result(i, j);
    result(i, i) = std::sqrt(value);
  }
  return result;
}

}  // namespace

// MND implementation

template <class T>
class MultivariateNormalDistribution<T>::Impl {
 public:
  Impl(const Linalg::Matrix<T>& covariance, const std::vector<T>& mean)
      : dims_{mean.size()},
        cholesky_{CholeskyFactorization(covariance)},
        mean_(std::make_unique<std::vector<T>>(mean.size())) {
    std::copy(mean.begin(), mean.end(), mean_->begin());
  }

  // Return row-wise random sample vectors from underlying distribution
  Linalg::Matrix<T> generateRandomData(size_t sampleSize) {
    Linalg::Matrix<T> result(sampleSize, dims_);
#pragma omp parallel for
    for (size_t n = 0; n < sampleSize; ++n) {
      for (size_t i = 0; i < dims_; ++i) {
        result(n, i) = mean_->operator[](i);
        for (size_t j = 0; j < i; ++j) {
          result(n, i) += cholesky_(i, j) * dist_(RNG());
        }
      }
    }
    return result;
  }

 private:
  size_t dims_;
  Linalg::Matrix<T> cholesky_;
  std::unique_ptr<std::vector<T>> mean_;
  static std::normal_distribution<T> dist_;
};

template <class T>
std::normal_distribution<T> MultivariateNormalDistribution<T>::Impl::dist_ =
    std::normal_distribution<T>{0, 1};

// Code for MultivariateNormalDistribution

// MultivariateNormalDistribution<T>::MultivariateNormalDistribution(
//     const Matrix<T>& A, const std::vector<T>& mean)
//     : pImpl_(std::make_unique<Impl>(A, mean)) {}
template <class T>
MultivariateNormalDistribution<T>::MultivariateNormalDistribution(
    const Linalg::Matrix<T>& covariance, const std::vector<T>& mean)
    : pImpl_{std::make_unique<Impl>(covariance, mean)} {}

template <class T>
MultivariateNormalDistribution<T>::~MultivariateNormalDistribution() {}

// Return row-wise random sample vectors from underlying distribution
template <class T>
Linalg::Matrix<T> MultivariateNormalDistribution<T>::operator()(size_t n) {
  return pImpl_->generateRandomData(n);
}

// Supported template specialization for MultivariateNormalDistribution
template class MultivariateNormalDistribution<float>;
template class MultivariateNormalDistribution<double>;
template class MultivariateNormalDistribution<long double>;

}  // namespace Random
}  // namespace MappedData
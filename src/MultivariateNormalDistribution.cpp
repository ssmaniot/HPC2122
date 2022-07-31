#include "MultivariateNormalDistribution.hpp"

#include <omp.h>

#include <cassert>
#include <iostream>
#include <random>

namespace MappedData {
namespace Random {

namespace {

std::mt19937& RNG() {
  static std::random_device rd;
  static std::mt19937 mt_{rd()};
  return mt_;
}

}  // namespace

// MND implementation

template <class T>
class MultivariateNormalDistribution<T>::Impl {
 public:
  Impl(const Linalg::Matrix<T>& covariance, const Linalg::Matrix<T>& mean)
      : dims_{mean.cols()},
        cholesky_{std::make_unique<Linalg::Matrix<T>>(
            std::move(Linalg::CholeskyFactorization(covariance)))},
        mean_(std::make_unique<Linalg::Matrix<T>>(mean)) {}

  // Return row-wise random sample vectors from underlying distribution
  Linalg::Matrix<T> generateSample(size_t sampleSize) {
    Linalg::Matrix<T> result(sampleSize, dims_);
#pragma omp parallel for
    for (size_t n = 0; n < sampleSize; ++n) {
      Linalg::Matrix<T> u{dims_, 1};
      for (size_t i = 0; i < dims_; ++i) {
        u(i, 0) = dist_(RNG());
      }
      for (size_t i = 0; i < dims_; ++i) {
        result(n, i) = mean_->operator()(0, i);
        for (size_t j = 0; j <= i; ++j) {
          result(n, i) += cholesky_->operator()(i, j) * u(j, 0);
        }
      }
    }
    return result;
  }

 private:
  size_t dims_;
  std::unique_ptr<Linalg::Matrix<T>> cholesky_;
  std::unique_ptr<Linalg::Matrix<T>> mean_;
  static std::normal_distribution<T> dist_;
};

template <class T>
std::normal_distribution<T> MultivariateNormalDistribution<T>::Impl::dist_ =
    std::normal_distribution<T>{0, 1};

// Code for MultivariateNormalDistribution

template <class T>
MultivariateNormalDistribution<T>::MultivariateNormalDistribution(
    const Linalg::Matrix<T>& covariance, const Linalg::Matrix<T>& mean)
    : pImpl_{std::make_unique<Impl>(covariance, mean)} {}

template <class T>
MultivariateNormalDistribution<T>::~MultivariateNormalDistribution() {}

// Return row-wise random sample vectors from underlying distribution
template <class T>
Linalg::Matrix<T> MultivariateNormalDistribution<T>::operator()(size_t n) {
  return pImpl_->generateSample(n);
}

// Supported template specialization for MultivariateNormalDistribution
template class MultivariateNormalDistribution<float>;
template class MultivariateNormalDistribution<double>;
template class MultivariateNormalDistribution<long double>;

}  // namespace Random
}  // namespace MappedData
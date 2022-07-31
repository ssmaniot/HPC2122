#include <cassert>

#include "MultivariateNormalDistribution.hpp"

namespace MappedData {
namespace Linalg {

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

}  // namespace Linalg
}  // namespace MappedData
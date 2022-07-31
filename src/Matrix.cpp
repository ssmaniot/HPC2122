#include <cmath>
#include <iomanip>
#include <iostream>

#include "MultivariateNormalDistribution.hpp"

namespace MappedData {
namespace Linalg {

// Matrix implementation

template <class T>
class Matrix<T>::Impl {
 public:
  Impl(size_t rows, size_t cols)
      : rows_{rows},
        cols_{cols},
        data_{std::make_unique<std::vector<T>>(rows * cols)} {}

  Impl(size_t rows, size_t cols, T value)
      : rows_{rows},
        cols_{cols},
        data_{std::make_unique<std::vector<T>>(rows * cols, value)} {}

  Impl(size_t rows, size_t cols,
       const std::initializer_list<std::initializer_list<T>>& values)
      : rows_{rows},
        cols_{cols},
        data_{std::make_unique<std::vector<T>>(rows * cols)} {
    assert(values.size() == rows_);
    size_t i = 0;
    for (const auto& row : values) {
      assert(row.size() == cols_);
      std::copy(row.begin(), row.end(), &data_->operator[](i));
      i += cols_;
    }
  }

  Impl(const Impl& other)
      : rows_{other.rows_},
        cols_{other.cols_},
        data_{std::make_unique<std::vector<T>>(other.rows_ * other.cols_)} {
    std::copy(other.data_->begin(), other.data_->end(), data_->begin());
  }

  Impl& operator=(const Impl& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = std::make_unique<std::vector<T>>(rows_ * cols_);
    std::copy(other.data_->begin(), other.data_->end(), data_->begin());
    return *this;
  }

  Impl(Impl&& other)
      : rows_{other.rows_}, cols_{other.cols_}, data_{std::move(other.data_)} {}

  Impl& operator=(Impl&& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = std::move(other.data_);
    return *this;
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

  const T& operator()(size_t row, size_t col) const {
    assert(row < rows_);
    assert(col < cols_);
    return data_->operator[](row* cols_ + col);
  }

  T& operator()(size_t row, size_t col) {
    assert(row < rows_);
    assert(col < cols_);
    return data_->operator[](row* cols_ + col);
  }

  using iterator = std::vector<T>::iterator;

  iterator begin() { return data_->begin(); }

  iterator end() { return data_->end(); }

 private:
  size_t rows_;
  size_t cols_;
  std::unique_ptr<std::vector<T>> data_;
};

// Matrix constructors/destructors

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : pImpl_{std::make_unique<Impl>(rows, cols)} {}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, T value)
    : pImpl_{std::make_unique<Impl>(rows, cols, value)} {}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols,
                  const std::initializer_list<std::initializer_list<T>>& values)
    : pImpl_{std::make_unique<Impl>(rows, cols, values)} {}

template <class T>
Matrix<T>::Matrix(const Matrix<T>& other)
    : pImpl_{std::make_unique<Impl>(*other.pImpl_)} {}

template <class T>
Matrix<T>::Matrix(Matrix<T>&& other) : pImpl_{std::move(other.pImpl_)} {}

template <class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
  pImpl_ = std::make_unique<Impl>(*other.pImpl_);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) {
  pImpl_ = std::move(other.pImpl_);
  return *this;
}

template <class T>
Matrix<T>::~Matrix() {}

// Matrix size

template <class T>
size_t Matrix<T>::rows() const {
  return pImpl_->rows();
}

template <class T>
size_t Matrix<T>::cols() const {
  return pImpl_->cols();
}

// Matrix accessors

template <class T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
  return pImpl_->operator()(row, col);
}

template <class T>
T& Matrix<T>::operator()(size_t row, size_t col) {
  return pImpl_->operator()(row, col);
}

// Pretty printer for Matrix

template <class T>
void Matrix<T>::print() const {
  std::cout << std::fixed << std::setprecision(5);
  for (size_t row = 0; row < rows(); ++row) {
    std::cout << '[';
    for (size_t column = 0; column < cols(); ++column) {
      std::cout << std::setw(9) << operator()(row, column);
      if (column < cols() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
  }
}

// Supported template specialization for Matrix

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<long double>;

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

// Supported template specialization for CholeskyFactorization()
template Matrix<float> CholeskyFactorization(const Matrix<float>& input);
template Matrix<double> CholeskyFactorization(const Matrix<double>& input);
template Matrix<long double> CholeskyFactorization(
    const Matrix<long double>& input);

}  // namespace Linalg
}  // namespace MappedData

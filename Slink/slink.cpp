#include <assert.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "CSV.hpp"
#include "MappedData.hpp"

constexpr size_t PRECISION = 2;
constexpr size_t WIDTH = 9;

class DistanceMatrix {
 public:
  DistanceMatrix(size_t n) : m_Data((n + 1) * (n + 1)), m_shape{n} {
    for (size_t i = 0; i < n + 1; ++i) {
      m_Data[i * (n + 1) + n] = std::numeric_limits<double>::max();
      m_Data[n * (n + 1) + i] = std::numeric_limits<double>::max();
    }
  }

  double &operator()(size_t i, size_t j) {
    // assert(i < n + 1 || j < n + 1);
    return m_Data[i * (m_shape + 1) + j];
  }

  const double &operator()(size_t i, size_t j) const {
    assert(i < m_shape + 1 || j < m_shape + 1);
    return m_Data[i * (m_shape + 1) + j];
  }

 private:
  std::vector<double> m_Data;
  size_t m_shape;
};

template <class T>
void print(T arr[], size_t n, std::string name, size_t padding,
           bool off = false) {
  std::cout << std::setw(padding) << name << ": ";
  for (size_t i = 0; i < n; ++i) {
    if (arr[i] == std::numeric_limits<T>::max()) {
      std::cout << std::setw(WIDTH) << "Inf" << ' ';
    } else {
      T val = (off ? arr[i] + T{1} : arr[i]);
      std::cout << std::setw(WIDTH) << val << ' ';
    }
  }
  std::cout << '\n';
}

template <class T>
void printIdx(T arr[], size_t n, std::string name, size_t padding, size_t idx[],
              bool off = false) {
  std::cout << std::setw(padding) << name << ": ";
  for (size_t i = 0; i < n; ++i) {
    if (arr[idx[i]] == std::numeric_limits<T>::max()) {
      std::cout << std::setw(WIDTH) << "Inf" << ' ';
    } else {
      T val = (off ? arr[idx[i]] + T{1} : arr[idx[i]]);
      std::cout << std::setw(WIDTH) << val << ' ';
    }
  }
  std::cout << '\n';
}

// Main program

int main(int argc, char *argv[]) {
  size_t i, j, k, n;

  CSV::CSV doc;

  try {
    doc = CSV::CSV{"data/normal_groups.csv"};
  } catch (const std::exception &e) {
    std::cout << "[EXCEPTION] Error: " << e.what() << '\n';
    return 1;
  }

  std::cout << "Document has " << doc.rows() << " rows and " << doc.cols()
            << " columns.\n";

  return 0;

  MappedData::MappedData<double> data;
  size_t N = 10, P;
  P = 2;

  if (false) {
    try {
      data = MappedData::MappedData<double>("data.bin");
      N = data.length() / P;
    } catch (const std::exception &e) {
      std::cout << "[EXCEPTION] Error: " << e.what() << '\n';
    }

    std::cout << "N = " << N << '\n';

    DistanceMatrix dm{N};
#pragma omp parallel for
    for (i = 0; i < N; ++i) {
      for (j = 0; j < i; ++j) {
        dm(i, j) = 0;
        for (k = 0; k < P; ++k) {
          dm(i, j) += data[i * P + k] * data[k + j * P];
        }
        dm(i, j) = std::sqrt(dm(i, j));
        dm(j, i) = dm(i, j);
      }
    }
  }

  DistanceMatrix dm{N};
  dm(1, 0) = 1.2;
  dm(2, 0) = 5;
  dm(2, 1) = 3.4;
  dm(3, 0) = 5;
  dm(3, 1) = 4.1;
  dm(3, 2) = 2.1;
  dm(4, 0) = 4.2;
  dm(4, 1) = 5;
  dm(4, 2) = 6;
  dm(4, 3) = 11;
  dm(5, 0) = 7;
  dm(5, 1) = 6;
  dm(5, 2) = 6.2;
  dm(5, 3) = 5;
  dm(5, 4) = 1.9;
  dm(6, 0) = 9;
  dm(6, 1) = 4.1;
  dm(6, 2) = 4.6;
  dm(6, 3) = 13;
  dm(6, 4) = 7;
  dm(6, 5) = 7.5;
  dm(7, 0) = 7.6;
  dm(7, 1) = 6.4;
  dm(7, 2) = 9;
  dm(7, 3) = 4.1;
  dm(7, 4) = 9;
  dm(7, 5) = 5.6;
  dm(7, 6) = 3.6;
  dm(8, 0) = 11;
  dm(8, 1) = 5.3;
  dm(8, 2) = 11.3;
  dm(8, 3) = 4.3;
  dm(8, 4) = 5.5;
  dm(8, 5) = 6.3;
  dm(8, 6) = 8;
  dm(8, 7) = 4.9;
  dm(9, 0) = 4.3;
  dm(9, 1) = 4.5;
  dm(9, 2) = 22;
  dm(9, 3) = 5.5;
  dm(9, 4) = 4.3;
  dm(9, 5) = 4.5;
  dm(9, 6) = 10;
  dm(9, 7) = 2.9;
  dm(9, 8) = 1.4;

  for (i = 0; i < N; ++i) {
    for (j = i + 1; j < N; ++j) {
      dm(i, j) = dm(j, i);
    }
  }

  size_t pi[N + 1];
  double lambda[N + 1];
  double M[N];

  pi[0] = 0;
  lambda[0] = std::numeric_limits<double>::max();

#ifdef DEBUG
  std::cout << "Processing n=" << 0 << '\n';
  std::cout << "  Init:\n";
  std::cout << '\n';
#endif

  // Algorithm main loop
  for (n = 1; n < N; ++n) {
#ifdef DEBUG
    std::cout << "Processing n=" << n << '\n';
    std::cout << "  Init:\n";
#endif

    // Initialization
    pi[n] = n;
    lambda[n] = std::numeric_limits<double>::max();
    for (i = 0; i < n; ++i) {
      M[i] = dm(i, n);
    }

#ifdef DEBUG
    std::cout << "  Update:\n";
#endif

    // Update
    for (i = 0; i < n; ++i) {
#ifdef DEBUG
      std::cout << "    lambda[" << i << "] >= M[" << i << "] ? "
                << (lambda[n] >= M[n]) << '\n';
#endif
      if (lambda[i] >= M[i]) {
        M[pi[i]] = std::min(M[pi[i]], lambda[i]);
        lambda[i] = M[i];
        pi[i] = n;
      } else {
        M[pi[i]] = std::min(M[pi[i]], M[i]);
      }
    }

#ifdef DEBUG
    std::cout << "  Update status:\n";
#endif

    // Update status:
    for (i = 0; i < n; ++i) {
#ifdef DEBUG
      std::cout << "    lambda[" << i << "] >= lambda[pi[" << i << "]] ? "
                << (lambda[i] >= lambda[pi[i]]) << '\n';
#endif
      if (lambda[i] >= lambda[pi[i]]) {
        pi[i] = n;
      }
    }
  }

  size_t idx[N];
  std::iota(&idx[0], &idx[N], 0);
  std::sort(&idx[0], &idx[N],
            [&lambda](size_t a, size_t b) { return lambda[a] < lambda[b]; });

  return 0;
}
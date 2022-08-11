#include <assert.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

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

  size_t rows() const { return m_shape; }
  size_t cols() const { return m_shape; }

 private:
  std::vector<double> m_Data;
  size_t m_shape;
};

template <class T>
void print(T arr[], size_t n, std::string name, size_t padding = 8,
           bool off = false) {
  std::cout << std::setw(padding) << name << ": ";
  for (size_t i = 0; i < n; ++i) {
    if (arr[i] == std::numeric_limits<T>::max()) {
      std::cout << std::setw(WIDTH) << "Inf" << ' ';
    } else {
      T val = (off ? *(arr + i) + T{1} : arr[i]);
      std::cout << std::setw(WIDTH) << val << ' ';
    }
  }
  std::cout << '\n';
}

template <class T>
void printIdx(T arr[], size_t n, std::string name, size_t idx[],
              size_t padding = 8, bool off = false) {
  std::cout << std::setw(padding) << name << ": ";
  for (size_t i = 0; i < n; ++i) {
    // std::cout << "Printing index " << idx[i] << '\n';
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

  if (argc < 2 || argc > 3) {
    std::cerr
        << "Error, correct usage: ./slink <input_file> [<group_feature>]\n";
    return 1;
  }
  std::string fileName = argv[1];
  std::string groupFeature{};
  bool groupFeatureSpecified = (argc == 3);
  if (groupFeatureSpecified) {
    groupFeature = argv[2];
  }

  try {
    doc = CSV::CSV{fileName};
  } catch (const std::exception &e) {
    std::cout << "[EXCEPTION] Error: " << e.what() << '\n';
    return 1;
  }

  std::cout << "Document has " << doc.rows() << " rows and " << doc.cols()
            << " columns.\n";

  auto header = doc.getHeader();
  if (groupFeatureSpecified && std::find(std::begin(header), std::end(header),
                                         groupFeature) == std::end(header)) {
    std::cerr << "Error, feature \"" << groupFeature
              << "\" is not in header.\n";
    return 1;
  }
  std::cout << "Header:";
  for (const auto &s : header) {
    std::cout << ' ' << s;
  }
  std::cout << '\n';

  auto dataTypes = doc.getDataTypes();
  size_t N = doc.rows();
  std::vector<size_t> numericIndexes{};
  for (i = 0; i < dataTypes.size(); ++i) {
    if (!(groupFeatureSpecified && header[i] == groupFeature) &&
        dataTypes[i] == "Numeric") {
      numericIndexes.push_back(i);
    }
  }
  size_t P = numericIndexes.size();

  std::cout << "Dataset with n = " << N << " elements, p = " << P << ".\n";
  std::cout << "Variables with \"String\" data type:";
  for (i = 0; i < header.size(); ++i) {
    if (dataTypes[i] == "String") {
      std::cout << ' ' << header[i];
    }
  }
  std::cout << '\n';
  std::cout << "Done.\n";

  // init chrono
  std::cout << "Begin clustering... (size n = " << N << ")\n";
  auto begin = std::chrono::high_resolution_clock::now();

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
    // std::cout << "\rCurrent step: " << n << "/" << N << " ("
    //           << std::ceil(10000. * static_cast<double>(n) / N) / 100.
    //           << "%)  ";
#ifdef DEBUG
    std::cout << "Processing n=" << n << '\n';
    std::cout << "  Init:\n";
#endif

    // Initialization
    pi[n] = n;
    lambda[n] = std::numeric_limits<double>::max();

#pragma omp parallel for
    for (i = 0; i < n; ++i) {
      M[i] = 0;
      for (const auto j : numericIndexes) {
        double ij = doc(i, j).getNumeric();
        double nj = doc(n, j).getNumeric();
        M[i] += (ij - nj) * (ij - nj);
      }
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

  // end timing
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

  // some nice output
  std::cout << "\rDone.                            \n";
  std::cout << "Time       : " << elapsed.count() << "ms.\n";
  std::cout << "Result     :\n";

#ifdef DEBUG
  std::cout << '\n';
  std::cout << "Sorted:\n";
  size_t is[N];
  std::iota(is, is + N, 0);
  size_t resPerLine = 10;

  i = 0;
  do {
    print(is + i, std::min(resPerLine, N - i), "m", 8);
    print(idx + i, std::min(resPerLine, N - i), "i", 8);
    printIdx(pi + i, std::min(resPerLine, N - i), "pi", idx + i);
    printIdx(lambda + i, std::min(resPerLine, N - i), "lambda", idx + i);
    i += resPerLine;
  } while (i < N);
#endif
  if (false) {
    size_t m;
    std::cout << "\ni     : ";
    for (m = 0; m < N; ++m) {
      std::cout << std::setw(2) << m << ' ';
    }
    std::cout << '\n';
    std::cout << "idx   : ";
    for (m = 0; m < N; ++m) {
      std::cout << std::setw(2) << idx[m] << ' ';
    }
    std::cout << '\n';
    std::cout << "pi    : ";
    for (m = 0; m < N; ++m) {
      std::cout << std::setw(2) << pi[idx[m]] << ' ';
    }
    std::cout << '\n';
    std::cout << "lambda: ";
    for (m = 0; m < N; ++m) {
      std::cout << std::setw(2) << lambda[idx[m]] << ' ';
    }
    std::cout << '\n';
  }

  return 0;
}
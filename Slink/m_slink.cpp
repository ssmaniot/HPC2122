#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

#include "CSV.hpp"
#include "xmmintrin.h"

constexpr size_t PRECISION = 2;
constexpr size_t WIDTH = 9;

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

  auto header = doc.getHeader();
  if (groupFeatureSpecified && std::find(std::begin(header), std::end(header),
                                         groupFeature) == std::end(header)) {
    std::cerr << "Error, feature \"" << groupFeature
              << "\" is not in header.\n";
    return 1;
  }

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

  // Saving data to aligned array
  // float *data = static_cast<float *>(std::malloc(N * P * sizeof(float)));
  float *data =
      static_cast<float *>(_mm_malloc(N * P * sizeof(float), sizeof(float *)));
  if (data == nullptr) {
    std::cerr << "Could not allocate memory for data.\n";
    return 1;
  }

#pragma omp parallel for
  for (i = 0; i < N; ++i) {
    for (j = 0; j < P; ++j) {
      data[i * P + j] = doc(i, numericIndexes[j]).getNumeric();
    }
  }

  // size_t pi[N + 1];
  // float lambda[N + 1];
  // float M[N];
  size_t *pi = static_cast<size_t *>(
      _mm_malloc((N + 1) * sizeof(size_t), sizeof(size_t *)));
  float *lambda = static_cast<float *>(
      _mm_malloc((N + 1) * sizeof(float), sizeof(float *)));
  float *M =
      static_cast<float *>(_mm_malloc(N * sizeof(float), sizeof(float *)));

  if ((pi == nullptr) || (lambda == nullptr) || (M == nullptr)) {
    _mm_free(data);
    if (pi) {
      _mm_free(pi);
    }
    if (lambda) {
      _mm_free(lambda);
    }
    if (M) {
      _mm_free(M);
    }
    std::cerr << "Could not allocate memory for slink data structures.\n";
    return 1;
  }

  // init chrono
  std::cout << "Begin clustering... (size n = " << N << ")\n";
  auto begin = std::chrono::high_resolution_clock::now();

  pi[0] = 0;
  lambda[0] = std::numeric_limits<float>::max();
  // Algorithm main loop
  for (n = 1; n < N; ++n) {
    // Initialization
    pi[n] = n;
    lambda[n] = std::numeric_limits<float>::max();

#pragma omp parallel for
    for (i = 0; i < n; ++i) {
      M[i] = 0;
      for (j = 0; j < P; ++j) {
        float ij = data[i * P + j];
        float nj = data[n * P + j];
        M[i] += (ij - nj) * (ij - nj);
      }
      M[i] = std::sqrt(M[i]);
    }

    // Update
    // #pragma omp parallel for
    for (i = 0; i < n; ++i) {
      if (lambda[i] >= M[i]) {
        M[pi[i]] = std::min(M[pi[i]], lambda[i]);
        lambda[i] = M[i];
        pi[i] = n;
      } else {
        M[pi[i]] = std::min(M[pi[i]], M[i]);
      }
    }

// Update status:
#pragma omp parallel for
    for (i = 0; i < n; ++i) {
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

  // Releasing resources
  // std::free(data);
  _mm_free(data);
  _mm_free(pi);
  _mm_free(lambda);
  _mm_free(M);

  return 0;
}
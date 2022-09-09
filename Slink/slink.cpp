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

namespace {

constexpr size_t PRECISION = 2;
constexpr size_t WIDTH = 9;

float *data = nullptr;
size_t *pi = nullptr;
float *lambda = nullptr;
float *M = nullptr;

}  // namespace

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

#ifdef DEBUG_OUT
  std::cout << "File name: \"" << fileName << "\"\n";
#endif

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
  data =
      static_cast<float *>(_mm_malloc(N * P * sizeof(float), sizeof(float *)));
  if (data == nullptr) {
    std::cerr << "Could not allocate memory for data.\n";
    return 1;
  }

  // #pragma omp parallel for
  for (i = 0; i < N; ++i) {
    for (j = 0; j < P; ++j) {
      data[i * P + j] = doc(i, numericIndexes[j]).getNumeric();
    }
  }

  pi = static_cast<size_t *>(
      _mm_malloc((N + 1) * sizeof(size_t), sizeof(size_t *)));
  lambda = static_cast<float *>(
      _mm_malloc((N + 1) * sizeof(float), sizeof(float *)));
  M = static_cast<float *>(_mm_malloc(N * sizeof(float), sizeof(float *)));

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
#ifdef DEBUG
    std::cout << "Loop on item " << n << '\n';
#endif
    pi[n] = n;
    lambda[n] = std::numeric_limits<float>::max();

#pragma omp parallel for num_threads(12)
    for (i = 0; i < n; ++i) {
      M[i] = 0;
      for (j = 0; j < P; ++j) {
        float ij = data[i * P + j];
        float nj = data[n * P + j];
        M[i] += (ij - nj) * (ij - nj);
      }
      M[i] = std::sqrt(M[i]);
#ifdef DEBUG
      std::cout << "    M[" << i << "] = " << M[i] << '\n';
#endif
    }

    // Update
    // #pragma omp parallel for
    for (i = 0; i < n; ++i) {
#ifdef DEBUG
      std::cout << "    lambda[" << i << "](" << std::setw(3)
                << (lambda[i] == std::numeric_limits<float>::max()
                        ? "Inf"
                        : std::to_string(lambda[i]))
                << ") >= M[" << i << "](" << std::setw(3) << M[i] << ") ? "
                << std::to_string(lambda[i] >= M[i]);
#endif
      if (lambda[i] >= M[i]) {
#ifdef DEBUG
        std::cout << " => M[pi[" << i << "]](M[" << pi[i] << "]) = min(M[pi["
                  << i << "]](" << std::setw(3) << M[pi[i]] << "), lambda[" << i
                  << "](" << std::setw(3)
                  << (lambda[i] == std::numeric_limits<float>::max()
                          ? "Inf"
                          : std::to_string(lambda[i]))
                  << ")), lambda[" << i << "] = M[" << i << "](" << std::setw(3)
                  << M[i] << "), pi[" << i << "] = " << n;
#endif
        M[pi[i]] = std::min(M[pi[i]], lambda[i]);
        lambda[i] = M[i];
        pi[i] = n;
      } else {
#ifdef DEBUG
        std::cout << " => M[pi[" << i << "]](M[" << pi[i] << "]) = min(M[pi["
                  << i << "]](" << std::setw(3) << M[pi[i]] << "), M[" << i
                  << "](" << std::setw(3) << M[i] << "))";
#endif
        M[pi[i]] = std::min(M[pi[i]], M[i]);
      }
#ifdef DEBUG
      std::cout << '\n';
#endif
    }

// Update status:
#ifdef DEBUG
    std::cout << "  Update status:\n";
#endif
    for (i = 0; i < n; ++i) {
#ifdef DEBUG
      std::cout << "    lambda[" + std::to_string(i) + "]("
                << (lambda[i] == std::numeric_limits<float>::max()
                        ? "Inf"
                        : std::to_string(lambda[i])) +
                       ") >= lambda[pi["
                << i << "]](" << std::setw(3)
                << (lambda[pi[i]] == std::numeric_limits<float>::max()
                        ? "Inf"
                        : std::to_string(lambda[pi[i]]))
                << ") ? " << std::to_string(lambda[i] >= lambda[pi[i]]);
#endif
      if (lambda[i] >= lambda[pi[i]]) {
#ifdef DEBUG
        std::cout << " => pi[" << i << "] = " << n;
#endif
        pi[i] = n;
      }
#ifdef DEBUG
      std::cout << '\n';
#endif
    }
#ifdef DEBUG
    std::cout << '\n';
#endif
  }

  size_t idx[N];
  std::iota(&idx[0], &idx[N], 0);
  std::sort(&idx[0], &idx[N],
            [](size_t a, size_t b) { return lambda[a] < lambda[b]; });

  // end timing
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

  // some nice output
  std::cout << "Done.\n";
  std::cout << "Time       : " << elapsed.count() << "ms.\n";

#ifdef DEBUG_OUT
  std::cout << "Result     :\n";

  size_t resPerLine = 10;
  for (i = 0; i < N; i += resPerLine) {
    size_t m;
    std::cout << "\nm     : ";
    for (m = i; m < std::min(i + resPerLine, N); ++m) {
      std::cout << std::setw(WIDTH) << m << ' ';
    }
    std::cout << '\n';
    std::cout << "i     : ";
    for (m = i; m < std::min(i + resPerLine, N); ++m) {
      std::cout << std::setw(WIDTH) << idx[m] << ' ';
    }
    std::cout << '\n';
    std::cout << "pi    : ";
    for (m = i; m < std::min(i + resPerLine, N); ++m) {
      std::cout << std::setw(WIDTH) << pi[idx[m]] << ' ';
    }
    std::cout << '\n';
    std::cout << "lambda: ";
    for (m = i; m < std::min(i + resPerLine, N); ++m) {
      if (lambda[idx[m]] == std::numeric_limits<float>::max()) {
        std::cout << std::setw(WIDTH) << "Inf";
      } else {
        std::cout << std::setw(WIDTH) << lambda[idx[m]];
      }
      std::cout << ' ';
    }
    std::cout << '\n';
  }
#endif

  // Releasing resources
  // std::free(data);
  _mm_free(data);
  _mm_free(pi);
  _mm_free(lambda);
  _mm_free(M);

  return 0;
}
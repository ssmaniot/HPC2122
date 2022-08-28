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
  int i, j, k, n;

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

#ifdef DEBUG
  std::cout << "Processing n=" << 0 << '\n';
  std::cout << "  Init:\n";
  std::cout << '\n';
#endif

  // Algorithm main loop 1
  for (n = 1; n < N / 2; ++n) {
#ifdef DEBUG
    std::cout << "Processing n=" << n << '\n';
    std::cout << "  Init:\n";
#endif

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

#ifdef DEBUG
    std::cout << "  Update:\n";
#endif

    // Update
    // #pragma omp parallel for
    for (i = 0; i < n; ++i) {
#ifdef DEBUG
      std::cout << "    lambda[" << i << "](" << lambda[i] << ") >= M[" << i
                << "](" << M[i] << ") ? " << (lambda[n] >= M[n]) << '\n';
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
      std::cout << "    lambda[" << i << "](" << lambda[i] << ") >= lambda[pi["
                << i << "]](" << lambda[pi[i]] << ") ? "
                << (lambda[i] >= lambda[pi[i]]) << '\n';
#endif
      if (lambda[i] >= lambda[pi[i]]) {
        pi[i] = n;
      }
    }
  }

  size_t idx[N];
  std::iota(&idx[0], &idx[N], 0);
  std::sort(&idx[0], &idx[N / 2],
            [](size_t a, size_t b) { return lambda[a] < lambda[b]; });

  pi[N / 2] = N / 2;
  lambda[N / 2] = std::numeric_limits<float>::max();

  // Algorithm main loop 2
  for (n = n / 2 + 1; n < N; ++n) {
    // Initialization
    pi[n] = n;
    lambda[n] = std::numeric_limits<float>::max();

#pragma omp parallel for
    for (i = N / 2; i < n; ++i) {
      M[i] = 0;
      for (j = 0; j < P; ++j) {
        float ij = data[i * P + j];
        float nj = data[n * P + j];
        M[i] += (ij - nj) * (ij - nj);
      }
      M[i] = std::sqrt(M[i]);
    }

// Update
#pragma omp parallel for
    for (i = N / 2; i < n; ++i) {
      if (lambda[i] >= M[i]) {
        M[pi[i]] = std::min(M[pi[i]], lambda[i]);
        lambda[i] = M[i];
        pi[i] = n;
      } else {
        M[pi[i]] = std::min(M[pi[i]], M[i]);
      }
    }

    // Update status:
    for (i = N / 2; i < n; ++i) {
      if (lambda[i] >= lambda[pi[i]]) {
        pi[i] = n;
      }
    }
  }

  std::sort(&idx[N / 2], &idx[N],
            [](size_t a, size_t b) { return lambda[a] < lambda[b]; });

#ifdef DEBUG_OUT
  std::cout << "Two clusters done, result:\n";
  {
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
  }
#endif

  // ! Implementation of merge
  for (i = N / 2 - 1; i >= 0; --i) {
    // Find minimum distance
    for (n = N / 2; n < N; ++n) {
      M[n] = 0;
      for (j = 0; j < P; ++j) {
        float ij = data[i * P + j];
        float nj = data[n * P + j];
        M[n] += (ij - nj) * (ij - nj);
      }
      M[n] = std::sqrt(M[n]);
    }

    size_t minIdx = N / 2;
    for (n = N / 2 + 1; n < N; ++n) {
      if (M[n] < M[minIdx]) {
        minIdx = n;
      }
    }

    std::cout << "lambda[" << i << "](" << lambda[i] << ") >= M[" << minIdx
              << "](" << M[minIdx] << ") ? " << (lambda[i] >= M[minIdx])
              << '\n';
    if (lambda[i] >= M[minIdx]) {
      lambda[i] = M[minIdx];
      pi[i] = minIdx;
    }
  }
  // ! End of implementation

  std::sort(&idx[0], &idx[N],
            [](size_t a, size_t b) { return lambda[a] < lambda[b]; });

  // end timing
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

  // some nice output
  std::cout << "\rDone.                            \n";
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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>

#include "CSV.hpp"
#include "xmmintrin.h"

constexpr size_t PRECISION = 2;
constexpr size_t WIDTH = 9;

size_t N, P;

float *data = nullptr;
size_t *pi = nullptr;
float *lambda = nullptr;
float *M = nullptr;
size_t *idx = nullptr;

namespace {

std::mutex m;

void printMsg(std::string s) { std::cout << s; }

void slink(int lo, int hi) {
  int i, j, n;
  pi[lo] = 0;
  lambda[lo] = std::numeric_limits<float>::max();
  // Algorithm main loop
  for (n = lo + 1; n < hi; ++n) {
    // Initialization
    pi[n] = n;
    lambda[n] = std::numeric_limits<float>::max();

#pragma omp parallel for
    for (i = lo; i < n; ++i) {
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
    for (i = lo; i < n; ++i) {
      if (lambda[i] >= M[i]) {
        M[pi[i]] = std::min(M[pi[i]], lambda[i]);
        lambda[i] = M[i];
        pi[i] = n;
      } else {
        M[pi[i]] = std::min(M[pi[i]], M[i]);
      }
    }

    // Update status:
    for (i = lo; i < n; ++i) {
      if (lambda[i] >= lambda[pi[i]]) {
        pi[i] = n;
      }
    }
  }
  std::sort(&idx[lo], &idx[hi],
            [](size_t a, size_t b) { return lambda[a] < lambda[b]; });

#ifdef DEBUG_OUT
  std::lock_guard<std::mutex> lk{m};
  std::cout << "slink [" << lo << "," << hi << "):\n";

  int resPerLine = 10;
  for (i = lo; i < hi; i += resPerLine) {
    size_t m;
    std::cout << "\nm     : ";
    for (m = i; m < std::min(i + resPerLine, hi); ++m) {
      std::cout << std::setw(WIDTH) << m << ' ';
    }
    std::cout << '\n';
    std::cout << "i     : ";
    for (m = i; m < std::min(i + resPerLine, hi); ++m) {
      std::cout << std::setw(WIDTH) << idx[m] << ' ';
    }
    std::cout << '\n';
    std::cout << "pi    : ";
    for (m = i; m < std::min(i + resPerLine, hi); ++m) {
      std::cout << std::setw(WIDTH) << pi[idx[m]] << ' ';
    }
    std::cout << '\n';
    std::cout << "lambda: ";
    for (m = i; m < std::min(i + resPerLine, hi); ++m) {
      if (lambda[idx[m]] == std::numeric_limits<float>::max()) {
        std::cout << std::setw(WIDTH) << "Inf";
      } else {
        std::cout << std::setw(WIDTH) << lambda[idx[m]];
      }
      std::cout << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
#endif
}

// Use tri2.csv and the drawing from the paper!
// Connect to closest that is not pointing at a closer point
void merge(int lo, int mid, int hi) {
  int i, j, n;
  std::lock_guard<std::mutex> lk{m};

  // WRONG
  for (n = mid + 1; n < hi; ++n) {
    for (i = lo; i < n; ++i) {
      M[i] = 0;
      for (j = 0; j < P; ++j) {
        float ij = data[i * P + j];
        float nj = data[n * P + j];
        M[i] += (ij - nj) * (ij - nj);
      }
      M[i] = std::sqrt(M[i]);
    }

    // Find closest node s.t. his lambda value is
    // greater than his dist from current node
    size_t best = hi;
    for (i = lo; i < hi - 1; ++i) {
      if (M[i] < lambda[i] && M[i] < M[best]) {
        best = i;
      }
    }

    for (i = lo; i < n; ++i) {
      if (lambda[i] >= M[i]) {
        M[pi[i]] = std::min(M[pi[i]], lambda[i]);
        lambda[i] = M[i];
        pi[i] = n;
      } else {
        M[pi[i]] = std::min(M[pi[i]], M[i]);
      }
    }

    for (i = lo; i < n; ++i) {
      if (lambda[i] >= lambda[pi[i]]) {
        pi[i] = n;
      }
    }
  }

  std::sort(&idx[lo], &idx[hi],
            [](size_t a, size_t b) { return lambda[a] < lambda[b]; });
}

void mergeSlink(int lo, int hi) {
  if (hi - lo > 3) {
    int mid = (hi + lo) / 2;
    std::thread left(mergeSlink, lo, mid);
    std::thread right(mergeSlink, mid, hi);
    left.join();
    right.join();
    merge(lo, mid, hi);
  } else {
    slink(lo, hi);
  }
}

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
  N = doc.rows();
  std::vector<size_t> numericIndexes{};
  for (i = 0; i < dataTypes.size(); ++i) {
    if (!(groupFeatureSpecified && header[i] == groupFeature) &&
        dataTypes[i] == "Numeric") {
      numericIndexes.push_back(i);
    }
  }
  P = numericIndexes.size();

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
  idx = static_cast<size_t *>(_mm_malloc(N * sizeof(size_t), sizeof(size_t *)));

  if ((pi == nullptr) || (lambda == nullptr) || (M == nullptr) ||
      (idx == nullptr)) {
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
    if (idx) {
      _mm_free(idx);
    }
    std::cerr << "Could not allocate memory for slink data structures.\n";
    return 1;
  }

  std::iota(&idx[0], &idx[N], 0);

  // init chrono
  std::cout << "Begin clustering... (size n = " << N << ")\n";
  auto begin = std::chrono::high_resolution_clock::now();

  mergeSlink(0, N);

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
  _mm_free(idx);

  return 0;
}
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

#include "CSV.hpp"

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

  std::cout << "Copying data to array...\n";
  double *data = static_cast<double *>(std::malloc(sizeof(double) * N * P));
#pragma omp parallel for
  for (i = 0; i < N; ++i) {
    for (j = 0; j < P; ++j) {
      data[i * P + j] = doc(i, numericIndexes[j]).getNumeric();
    }
  }
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
        double ij = data[i * P + j];
        double nj = data[n * P + j];
        M[i] += (ij - nj) * (ij - nj);
      }
      M[i] = std::sqrt(M[i]);
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

  // Releasing data
  std::free(data);

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
#ifdef DEBUG
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
      std::cout << std::setw(WIDTH)
                << (lambda[idx[m]] == std::numeric_limits<double>::max()
                        ? "Inf"
                        : std::to_string(lambda[idx[m]]))
                << ' ';
    }
    std::cout << '\n';
  }
#endif

  return 0;
}
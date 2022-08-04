#include <assert.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#define ACC(m, i, j) (*((m) + ((i) * ((i)-1) / 2 + (j))))

constexpr size_t PRECISION = 2;
constexpr size_t WIDTH = 9;

struct _Slink {
  double *dataMatrix{nullptr};
  double *distanceMatrix{nullptr};
  size_t n{0};
  size_t p{0};
};

class DistanceMatrix {
 public:
  DistanceMatrix(size_t n) : m_Data((n + 1) * (n + 1)), n{n} {
    for (size_t i = 0; i < n + 1; ++i) {
      m_Data[i * (n + 1) + n] = std::numeric_limits<double>::max();
      m_Data[n * (n + 1) + i] = std::numeric_limits<double>::max();
    }
  }

  double &operator()(size_t i, size_t j) {
    // assert(i < n + 1 || j < n + 1);
    return m_Data[i * (n + 1) + j];
  }

  const double &operator()(size_t i, size_t j) const {
    assert(i < n + 1 || j < n + 1);
    return m_Data[i * (n + 1) + j];
  }

 private:
  std::vector<double> m_Data;
  size_t n;
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

int main(int argc, char *argv[]) {
  size_t i, j, k, n;

  constexpr size_t N = 10;
  constexpr size_t P = 2;
  DistanceMatrix dm{N};
  double m[N * P];

  m[0] = 1;
  m[1] = 1;
  m[2] = 1.5;
  m[3] = 1.5;
  m[4] = 5;
  m[5] = 5;
  m[6] = 3;
  m[7] = 4;
  m[8] = 4;
  m[9] = 4;
  m[10] = 3;
  m[11] = 3.5;

  for (i = 0; i < N; ++i) {
    for (j = 0; j < i; ++j) {
      for (k = 0; k < P; ++k) {
        dm(i, j) =
            (m[i * P + k] - m[j * P + k]) * (m[i * P + k] - m[j * P + k]);
      }
      dm(i, j) = std::sqrt(dm(i, j));
      dm(j, i) = dm(i, j);
    }
  }

  // dm(1, 0) = 1.2;
  // dm(2, 0) = 5;
  // dm(2, 1) = 3.4;
  // dm(3, 0) = 5;
  // dm(3, 1) = 4.1;
  // dm(3, 2) = 2.1;
  // dm(4, 0) = 4.2;
  // dm(4, 1) = 5;
  // dm(4, 2) = 6;
  // dm(4, 3) = 11;
  // dm(5, 0) = 7;
  // dm(5, 1) = 6;
  // dm(5, 2) = 6.2;
  // dm(5, 3) = 5;
  // dm(5, 4) = 1.9;
  // dm(6, 0) = 9;
  // dm(6, 1) = 4.1;
  // dm(6, 2) = 4.6;
  // dm(6, 3) = 13;
  // dm(6, 4) = 7;
  // dm(6, 5) = 7.5;
  // dm(7, 0) = 7.6;
  // dm(7, 1) = 6.4;
  // dm(7, 2) = 9;
  // dm(7, 3) = 4.1;
  // dm(7, 4) = 9;
  // dm(7, 5) = 5.6;
  // dm(7, 6) = 3.6;
  // dm(8, 0) = 11;
  // dm(8, 1) = 5.3;
  // dm(8, 2) = 11.3;
  // dm(8, 3) = 4.3;
  // dm(8, 4) = 5.5;
  // dm(8, 5) = 6.3;
  // dm(8, 6) = 8;
  // dm(8, 7) = 4.9;
  // dm(9, 0) = 4.3;
  // dm(9, 1) = 4.5;
  // dm(9, 2) = 22;
  // dm(9, 3) = 5.5;
  // dm(9, 4) = 4.3;
  // dm(9, 5) = 4.5;
  // dm(9, 6) = 10;
  // dm(9, 7) = 2.9;
  // dm(9, 8) = 1.4;

  // for (i = 0; i < N; ++i) {
  //   for (j = i + 1; j < N; ++j) {
  //     dm(i, j) = dm(j, i);
  //   }
  // }

#ifdef DEBUG
  std::cout << "Distance matrix:\n";
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      std::cout << std::setw(WIDTH) << dm(i, j) << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
#endif

  size_t pi[N + 1];
  double lambda[N + 1];
  double M[N];

  pi[0] = 0;
  lambda[0] = std::numeric_limits<double>::max();

#ifdef DEBUG
  std::cout << "Processing n=" << 0 << '\n';
  std::cout << "  Init:\n";
  print(pi, 1, "pi", 10, true);
  print(lambda, 1, "lambda", 10);
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
    print(pi, n + 1, "pi", 10, true);
    print(lambda, n + 1, "lambda", 10);
    print(M, n, "M", 10);
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
#ifdef DEBUG
    std::cout << "  End of loop status:\n";
    print(pi, n + 1, "pi", 10, true);
    print(lambda, n + 1, "lambda", 10);
    print(M, n, "M", 10);
    std::cout << "  Update:\n";
    std::cout << '\n';
#endif
  }

#ifdef DEBUG
  std::cout << "End result:\n";
  print(pi, n, "pi", 8, true);
  print(lambda, n, "lambda", 8);
#endif

  size_t idx[N + 1];
  std::iota(&idx[0], &idx[N], 0);
  std::sort(&idx[0], &idx[N],
            [&lambda](size_t a, size_t b) { return lambda[a] < lambda[b]; });

#ifdef DEBUG
  std::cout << '\n';
  std::cout << "Sorted:\n";
  size_t is[N];
  std::iota(is, is + N, 1);
  print(is, n, "m", 8);
  print(idx, n, "i", 8, true);
  printIdx(pi, n, "pi", 8, idx, true);
  printIdx(lambda, n, "lambda", 8, idx);
#endif

  return 0;
}
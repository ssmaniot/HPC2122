#include <iomanip>
#include <iostream>
#include <random>

#include "MatrixOperations.hpp"

int main(int argc, char *argv[]) {
  constexpr size_t p = 100;
  constexpr size_t min_n = 1000;
  constexpr size_t max_n = 1500;
  constexpr size_t groups = 5;
  MappedData::Linalg::Matrix<double> m{1, p};
  size_t i, j, n, g;

  std::random_device rd{};
  std::mt19937 mt{rd()};
  std::normal_distribution<double> ndist{};
  std::uniform_int_distribution<size_t> udist{min_n, max_n};

  std::cout << "group,";
  for (i = 0; i < p; ++i) {
    std::cout << 'x' << i;
    if (i < p - 1) {
      std::cout << ',';
    }
  }
  std::cout << '\n';
  for (g = 0; g < groups; ++g) {
    for (i = 0; i < p; ++i) {
      m(0, i) = 3 * ndist(mt);
    }
    n = udist(mt);
    for (i = 0; i < n; ++i) {
      std::cout << g << ',';
      for (j = 0; j < p; ++j) {
        std::cout << (m(0, j) + ndist(mt));
        if (j < p - 1) {
          std::cout << ',';
        }
      }
      std::cout << '\n';
    }
  }

  return 0;
}
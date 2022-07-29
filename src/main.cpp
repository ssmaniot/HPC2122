#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "MappedData.hpp"
#include "MultivariateNormalDistribution.hpp"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << " !! Error, correct usage: ./main <file.bin>\n";
    return 1;
  }

  try {
    MappedData::MappedData<double> dm2{argv[1]};
    MappedData::MappedData<double> dm = std::move(dm2);
    dm2 = std::move(dm);
    std::cout << "for-each:\n";
    for (auto &p : dm) {
      std::cout << p << '\n';
    }
    dm = std::move(dm2);
    for (auto &p : dm) {
      std::cout << p << '\n';
    }
    std::cout << "for:\n";
    for (size_t i = 0; i < dm.length(); ++i) {
      std::cout << dm[i] << '\n';
    }
  } catch (const std::exception &e) {
    std::cout << "[EXCEPTION] Error: " << e.what() << '\n';
  }

  MappedData::MappedData<char> dm3{"hello.bin"};
  for (auto &c : dm3) std::cout << c;

  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> dist{5, 2};
  std::cout << dist.operator()(gen) << "\n\n";

  MappedData::Linalg::Matrix<double> m(2, 2, {{1.1, 2.2}, {3.3, 4.4}});
  std::cout << "made m\n";
  auto m2 = m;
  std::cout << "made m2\n";
  auto m3 = std::move(m);
  std::cout << "made m3 by moving m\n";
  m2 = std::move(m3);
  std::cout << "moved m3 into m2\n";
  for (size_t i = 0; i < m2.rows(); ++i) {
    for (size_t j = 0; j < m2.cols(); ++j) {
      std::cout << m2(i, j) << ' ';
    }
    std::cout << '\n';
  }

  std::cout << "------------------\n";

  // MappedData::Random::Matrix<double> A{2, 2, {{2, 1}, {1, 2}}};
  MappedData::Linalg::Matrix<double> A{2, 2, {{2, 1}, {1, 2}}};
  std::vector<double> v = {1, 2};
  MappedData::Random::MultivariateNormalDistribution<double> mnd{A, v};
  static constexpr size_t N = 1000000;

  // init chrono
  auto begin = std::chrono::high_resolution_clock::now();

  MappedData::Linalg::Matrix<double> r = mnd(N);

  // end timing
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

  // some nice output
  // std::cout << "-------------------------------\n";
  std::cout << "Multivariate Normal Distribution (n = " << r.rows() << ")\n";
  std::cout << "Time     : " << elapsed.count() << "ms.\n";

  std::vector<double> mean(A.rows(), 0);
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t n = 0; n < r.rows(); ++n) {
      mean[i] += r(n, i);
    }
    mean[i] /= r.rows();
  }

  MappedData::Linalg::Matrix<double> cov(2, 2, 0);
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = i; j < A.rows(); ++j) {
      for (size_t n = 0; n < r.rows(); ++n) {
        cov(i, j) += (r(n, i) - mean[i]) * (r(n, j) - mean[j]);
      }
      cov(i, j) /= r.rows() - 1;
      cov(j, i) = cov(i, j);
    }
  }

  std::cout << "Sample mean: (";
  for (size_t i = 0; i < mean.size(); ++i) {
    std::cout << mean[i];
    if (i != mean.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ")\n";

  std::cout << "Sample covariance matrix:\n";

  size_t rows = cov.rows(), columns = cov.cols();
  std::cout << std::fixed << std::setprecision(5);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      if (column > 0) std::cout << ' ';
      std::cout << std::setw(9) << cov(row, column);
    }
    std::cout << '\n';
  }

  return 0;
}

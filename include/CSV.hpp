#ifndef CSV_H
#define CSV_H

#include <exception>
#include <memory>

namespace CSV {

class CellProxy {
 public:
  CellProxy(const std::string& data, const std::string& dataType)
      : m_data{data}, m_dataType{dataType} {}

  double getNumeric() const;

  std::string getString() const { return m_data; }

 private:
  const std::string& m_data;
  const std::string& m_dataType;
};

class CSV {
 public:
  CSV();
  CSV(const std::string& fileName);
  CSV(const std::string& fileName, const std::string& delimiter);
  ~CSV();
  CSV(const CSV&) = delete;
  CSV(CSV&& other) noexcept;
  CSV& operator=(const CSV&) = delete;
  CSV& operator=(CSV&& other) noexcept;

  CellProxy operator()(size_t row, size_t column) const;
  size_t rows() const;
  size_t cols() const;

 private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl{nullptr};
};

}  // namespace CSV

#endif
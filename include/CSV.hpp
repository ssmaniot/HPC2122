#ifndef CSV_H
#define CSV_H

#include <memory>

namespace CSV {

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

 private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl{nullptr};
};

}  // namespace CSV

#endif
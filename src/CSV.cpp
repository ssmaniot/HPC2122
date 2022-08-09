#include "CSV.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace CSV {

class CSV::Impl {
 public:
  // Accessors

  CellProxy operator()(size_t row, size_t column) const {
    return CellProxy(m_data[row][column], m_dataTypes[column]);
  }

  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }

  const std::vector<std::string>& getHeader() const { return m_header; }
  const std::vector<std::string>& getDataTypes() const { return m_dataTypes; }

  // Constructor
  Impl(const std::string& fileName, const std::string& separator = ",") {
    m_fileName = fileName;
    m_separator = separator;

    std::ifstream str(m_fileName);
    std::string line;

    if (!std::getline(str, line)) {
      throw std::runtime_error("File \"" + fileName + "\" is empty.");
    }

    // Tokenize first line and determine if the file has header
    std::vector<std::string> tokens = tokenize(line);
    bool firstLineIsHeader = std::all_of(
        std::begin(tokens), std::end(tokens), [this](const std::string& token) {
          return computeDataType(token) == "String";
        });

    m_rows = 0;
    m_cols = tokens.size();

    // If first line is header, then parse another line
    if (firstLineIsHeader) {
      for (const std::string& token : tokens) {
        m_header.emplace_back(token);
      }
      if (std::getline(str, line)) {
        tokens = tokenize(line);
        if (tokens.size() != m_cols) {
          throw std::runtime_error(badLineFormatCSV(m_rows, tokens.size()));
        }
      }
    }

    // Determine value type for each column
    for (const auto& token : tokens) {
      m_dataTypes.push_back(computeDataType(token));
    }

    // Insert first row and continue parsing the rest of the file
    m_data.emplace_back(tokens);
    ++m_rows;

    while (std::getline(str, line)) {
#ifdef DEBUG
      std::cout << "Row " << m_rows << '\n';
#endif
      tokens = tokenize(line);
      if (tokens.size() != m_cols) {
        throw std::runtime_error(badLineFormatCSV(m_rows, tokens.size()));
      }
      for (size_t column = 0; column < tokens.size(); ++column) {
        if (computeDataType(tokens[column]) != m_dataTypes[column]) {
          throw std::runtime_error(
              badCellFormatCSV(m_rows, column, tokens[column]));
        }
      }
      m_data.emplace_back(tokens);
      ++m_rows;
    }
  }

 private:
  /**
   * Splits a line from the CSV file using the separator into tokens, and
   * returns an array containing the tokens.
   *
   * @param line String containing the line to be tokenized.
   * @return A vector of strings containing the tokens parsed from line.
   */
  std::vector<std::string> tokenize(const std::string& line) const {
    std::string s(line);
    std::vector<std::string> tokens;
    size_t pos;
    std::string token;
    size_t from = 0;
    size_t n = 0;
#ifdef DEBUG
    std::cout << "  Begin tokenizing:\n";
#endif
    while ((pos = s.substr(from, std::string::npos).find(m_separator)) !=
           std::string::npos) {
      token = s.substr(from, pos);
#ifdef DEBUG
      std::cout << "    Token " << ++n << ": \"" << token
                << "\", type: " << computeDataType(token) << '\n';
#endif
      tokens.emplace_back(token);
      from += pos + m_separator.length();
    }
#ifdef DEBUG
    std::cout << "  Done.\n";
#endif
    return tokens;
  }

  /**
   * Computes the type of a token and returns it.
   *
   * @param token Token to be typed.
   * @return A string representing the data type of the token.
   */
  std::string computeDataType(const std::string& token) const {
    std::istringstream iss(token);
    double d;
    iss >> std::noskipws >> d;
#ifdef DEBUG
    std::cout << "Token: \"" << token << "\", type: "
              << ((iss.eof() && !iss.fail()) ? "Numeric" : "String") << '\n';
#endif
    return (iss.eof() && !iss.fail()) ? "Numeric" : "String";
  }

  std::string badLineFormatCSV(size_t row, size_t numTokens) const {
    std::string msg = "Bad line format in file \"" + m_fileName + "\": line ";
    msg += std::to_string(row);
    msg += " has ";
    msg += std::to_string(numTokens);
    msg += " tokens, expected ";
    msg += std::to_string(m_cols);
    return msg;
  }

  std::string badCellFormatCSV(size_t row, size_t column,
                               std::string token) const {
    std::string msg = "Bad cell format in file \"" + m_fileName + "\": cell " +
                      std::to_string(column) + " on line " +
                      std::to_string(row) + " has value \"" + token +
                      "\" and type " + computeDataType(token) +
                      ", expected type is " + m_dataTypes[column];
    return msg;
  }

  std::string m_separator{","};
  size_t m_rows{0};
  size_t m_cols{0};
  std::string m_fileName{};
  std::vector<std::string> m_dataTypes{};
  std::vector<std::string> m_header{};
  std::vector<std::vector<std::string>> m_data{};
};

// CSV class interface implementation

// Constructors

CSV::CSV() = default;

CSV::CSV(const std::string& fileName)
    : m_pImpl{std::make_unique<Impl>(fileName)} {}

CSV::CSV(const std::string& fileName, const std::string& separator)
    : m_pImpl{std::make_unique<Impl>(fileName, separator)} {}

CSV::~CSV() = default;

CSV::CSV(CSV&& other) noexcept : m_pImpl{std::move(other.m_pImpl)} {}

CSV& CSV::operator=(CSV&& other) noexcept {
  this->m_pImpl = std::move(other.m_pImpl);
  return *this;
}

// Accessors

CellProxy CSV::operator()(size_t row, size_t column) const {
  return m_pImpl->operator()(row, column);
}

size_t CSV::rows() const { return m_pImpl->rows(); }
size_t CSV::cols() const { return m_pImpl->cols(); }

const std::vector<std::string>& CSV::getHeader() const {
  return m_pImpl->getHeader();
}

const std::vector<std::string>& CSV::getDataTypes() const {
  return m_pImpl->getDataTypes();
}

// CellProxy

double CellProxy::getNumeric() const {
  if (m_dataType != "Numeric") {
    throw std::runtime_error(
        "Attempting to cast data of type \"String\" into a \"Numeric\" "
        "value");
  }
  std::istringstream iss(m_data);
  double d;
  iss >> std::noskipws >> d;
  return d;
}

}  // namespace CSV
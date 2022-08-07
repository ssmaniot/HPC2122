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
  Impl(const std::string& fileName, const std::string& delimiter = ",") {
    m_fileName = fileName;
    m_delimiter = delimiter;

    std::ifstream str(m_fileName);
    std::string line;

    std::cout << "Impl(\"" << fileName << "\", \"" << delimiter << "\")\n";

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
      std::cout << "Row " << m_rows << '\n';
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
   * Splits a line from the CSV file using the delimiter into tokens, and
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
    std::cout << "  Begin tokenizing:\n";
    while ((pos = s.substr(from, std::string::npos).find(m_delimiter)) !=
           std::string::npos) {
      token = s.substr(from, pos);
      std::cout << "    Token " << ++n << ": \"" << token
                << "\", type: " << computeDataType(token) << '\n';
      tokens.emplace_back(token);
      from += pos + m_delimiter.length();
    }
    std::cout << "  Done.\n";
    return tokens;
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

  // Very basic check: if it contains a '.', check if it's floating point
  // number; otherwise, check if numeric.
  std::string computeDataType(const std::string& token) const {
    std::istringstream iss(token);
    double d;
    iss >> std::noskipws >> d;
    std::cout << "Token: \"" << token << "\", type: "
              << ((iss.eof() && !iss.fail()) ? "Numeric" : "String") << '\n';

    return (iss.eof() && !iss.fail()) ? "Numeric" : "String";
  }

  std::string m_delimiter{","};
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

CSV::CSV(const std::string& fileName, const std::string& delimiter)
    : m_pImpl{std::make_unique<Impl>(fileName, delimiter)} {}

CSV::~CSV() = default;

CSV::CSV(CSV&& other) noexcept : m_pImpl{std::move(other.m_pImpl)} {}

CSV& CSV::operator=(CSV&& other) noexcept {
  this->m_pImpl = std::move(other.m_pImpl);
  return *this;
}

}  // namespace CSV
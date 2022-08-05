#ifndef HPC_MAPPED_DATA_H
#define HPC_MAPPED_DATA_H

#include <cstring>
#include <system_error>

// For mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace MappedData {

template <class T>
class MappedData {
 public:
  MappedData() = default;
  MappedData(const char *fileName) {
    fname = fileName;
    fd = open(fileName, O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("open() failure");
    }

    // Obtain file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      throw std::runtime_error("fstat() failure");
    }

    fileSize = static_cast<size_t>(sb.st_size);
    size = fileSize / sizeof(T);

    dataMatrix =
        static_cast<T *>(mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0u));
    if (dataMatrix == MAP_FAILED) {
      throw std::runtime_error("mmap() failure");
    }
  }

  MappedData(const MappedData<T> &other) = delete;
  MappedData<T> &operator=(const MappedData<T> &other) = delete;

  MappedData(MappedData<T> &&other)
      : fname(other.fname),
        dataMatrix(other.dataMatrix),
        fileSize(other.fileSize),
        size(other.size),
        fd(other.fd) {
    other.fname = "";
    other.dataMatrix = nullptr;
    other.fileSize = 0;
    other.size = 0;
    other.fd = -1;
  }

  MappedData<T> &operator=(MappedData<T> &&other) {
    fname = other.fname;
    dataMatrix = other.dataMatrix;
    fileSize = other.fileSize;
    size = other.size;
    fd = other.fd;

    other.fname = "";
    other.dataMatrix = nullptr;
    other.fileSize = 0;
    other.size = 0;
    other.fd = -1;

    return *this;
  }

  ~MappedData() {
    if (dataMatrix) {
      munmap(dataMatrix, size);
      close(fd);
    }
  }

  const T *operator()() const noexcept { return dataMatrix; }
  const T &operator[](size_t i) {
    if (i >= size) {
      throw std::runtime_error("Out of bound");
    }
    return *(dataMatrix + i);
  }

  const T *begin() const noexcept { return dataMatrix; }
  const T *end() const noexcept { return dataMatrix + length(); }

  size_t length() const noexcept { return size; }

 private:
  std::string fname{};
  T *dataMatrix{nullptr};
  size_t fileSize{0};
  size_t size{0};
  int fd{0};
};

}  // namespace MappedData

#endif
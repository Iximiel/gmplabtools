#ifndef TRIANGULARMATRIX_HPP
#define TRIANGULARMATRIX_HPP
#include <vector>
namespace libpamm {
template <typename T> class triangularMatrix {
  // This is a lower triangular matrix, without the diagonal
public:
  inline size_t address(size_t i, size_t j) const {
    if (i < j)
      std::swap(i, j);
    return (i * (i - 1) / 2) + j;
  }
  size_t dim() const { return dim_; }
  triangularMatrix(size_t dim)
      : data_(new T[((dim - 1) * dim) / 2]), dim_(dim) {}

  triangularMatrix(const triangularMatrix &other)
      : data_(new T[((other.dim_ - 1) * other.dim_) / 2]), dim_(other.dim_) {
    const size_t len = ((dim_ - 1) * dim_) / 2;
    for (size_t i = 0; i < len; ++i) {
      this->data_[i] = other.data_[i];
    }
  }

  triangularMatrix(triangularMatrix &&other)
      : data_(other.data_), dim_(other.dim_) {
    other.data_ = nullptr;
  }

  triangularMatrix &operator=(triangularMatrix other) {
    if (&other != this) {
      swap(other);
    }
    return *this;
  }

  void swap(triangularMatrix &other) {
    if (dim_ != other.dim_) {
      throw "Error in triangularMatrix swap: different size";
    }
    std::swap(this->data_, other.data_);
  }

  ~triangularMatrix() { delete[] data_; }

  T &operator()(size_t i, size_t j) { return data_[address(i, j)]; }

  T operator()(size_t i, size_t j) const { return data_[address(i, j)]; }

private:
  T *data_{nullptr};
  const size_t dim_{0};
};
} // namespace libpamm
#endif // TRIANGULARMATRIX_HPP

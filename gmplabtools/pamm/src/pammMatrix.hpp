#ifndef TRIANGULARMATRIX_HPP
#define TRIANGULARMATRIX_HPP
//#include <iostream>
#include <vector>
namespace libpamm {
  template <typename T>
  class triangularMatrix {
    // This is a lower triangular matrix, without the diagonal
  public:
    inline size_t address (size_t i, size_t j) const {
      if (i < j)
        std::swap (i, j);
      return (i * (i - 1) / 2) + j;
    }
    size_t dim () const { return dim_; }
    triangularMatrix (size_t dim)
      : data_ (new T[((dim - 1) * dim) / 2]),
        dim_ (dim) {}

    triangularMatrix (const triangularMatrix &other)
      : data_ (new T[((other.dim_ - 1) * other.dim_) / 2]),
        dim_ (other.dim_) {
      const size_t len = ((dim_ - 1) * dim_) / 2;
      for (size_t i = 0; i < len; ++i) {
        this->data_[i] = other.data_[i];
      }
    }

    triangularMatrix (triangularMatrix &&other)
      : data_ (other.data_),
        dim_ (other.dim_) {
      other.data_ = nullptr;
    }

    triangularMatrix &operator= (triangularMatrix other) {
      if (&other != this) {
        swap (other);
      }
      return *this;
    }

    void swap (triangularMatrix &other) {
      if (dim_ != other.dim_) {
        throw "Error in triangularMatrix swap: different size";
      }
      std::swap (this->data_, other.data_);
    }

    ~triangularMatrix () { delete[] data_; }

    T &operator() (size_t i, size_t j) { return data_[address (i, j)]; }

    T operator() (size_t i, size_t j) const { return data_[address (i, j)]; }

  private:
    T *data_{nullptr};
    const size_t dim_{0};
  };
  template <typename T>
  class dynamicMatrix {
  public:
    // dynamicMatrix (size_t dim);
    dynamicMatrix (size_t rows, size_t columns)
      : ROWS_ (rows),
        COLUMNS_ (columns),
        addresses_ (new T *[rows]),
        data_ (new T[columns * rows]) {
      for (size_t I = 0; I < ROWS_; ++I) {
        addresses_[I] = data_ + I * COLUMNS_;
      }
    }
    dynamicMatrix (const dynamicMatrix &other)
      : ROWS_ (other.ROWS_),
        COLUMNS_ (other.COLUMNS_),
        addresses_ (new T *[other.ROWS_]),
        data_ (new T[other.COLUMNS_ * other.ROWS_]) {
      for (size_t I = 0; I < ROWS_; ++I) {
        addresses_[I] = data_ + I * COLUMNS_;
      }
      for (size_t I = 0; I < ROWS_ * COLUMNS_; ++I) {
        data_[I] = other.data_[I];
      }
    }
    dynamicMatrix (dynamicMatrix &&other)
      : ROWS_ (other.ROWS_),
        COLUMNS_ (other.COLUMNS_),
        addresses_ (other.addresses_),
        data_ (other.data_) {
      other.addresses_ = nullptr;
      other.data_ = nullptr;
    }
    virtual ~dynamicMatrix () {
      delete[] data_;
      delete[] addresses_;
    }
    dynamicMatrix &operator= (dynamicMatrix other) { this->swap (other); }
    void swap (dynamicMatrix &other) {
      std::swap (ROWS_, other.ROWS_);
      std::swap (COLUMNS_, other.COLUMNS_);
      std::swap (addresses_, other.addresses_);
      std::swap (data_, other.data_);
    }
    T *operator[] (size_t I) { return addresses_[I]; }
    T *operator[] (size_t I) const { return addresses_[I]; }
    T *data () { return data_; }
    T *data () const { return data_; }
    T **addresses () { return addresses_; }
    T **addresses () const { return addresses_; }
    size_t Rows () const { return ROWS_; }
    size_t Columns () const { return COLUMNS_; }

  private:
    size_t ROWS_{0};
    size_t COLUMNS_{0};
    T **addresses_{nullptr};
    T *data_{nullptr};
  };

  template <typename T>
  dynamicMatrix<T>
  matMul (const dynamicMatrix<T> &A, const dynamicMatrix<T> &B) {
    if (A.Columns () != B.Rows ()) {
      throw "matMul: cannot multiply: the matrices are not mXn nXp";
    }
    dynamicMatrix<T> toReturn (A.Rows (), B.Columns ());
    for (size_t I = 0; I < toReturn.Rows (); ++I) {
      for (size_t J = 0; J < toReturn.Columns (); ++J) {
        toReturn[I][J] = static_cast<T> (0);
        for (size_t K = 0; K < A.Columns (); ++K) {
          toReturn[I][J] += A[I][K] * B[K][J];
        }
        // std::cerr << toReturn[I][J] << ' ';
      }
      // std::cerr << '\n';
    }
    return toReturn;
  }
  template <typename T>
  dynamicMatrix<T> Transpose (const dynamicMatrix<T> &A) {
    dynamicMatrix<T> toReturn (A.Columns (), A.Rows ());
    for (size_t I = 0; I < toReturn.Rows (); ++I) {
      for (size_t J = 0; J < toReturn.Columns (); ++J) {
        toReturn[I][J] = A[J][I];
      }
    }
    return toReturn;
  }
} // namespace libpamm
#endif // TRIANGULARMATRIX_HPP

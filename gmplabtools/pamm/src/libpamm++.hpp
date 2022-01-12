/*
#ifdef __cplusplus
extern "C" {
#endif
double __libpamm_MOD_mahalanobis(int, double *period, double *x, double *y,
                                 double *Qinv);
#ifdef __cplusplus
}
#endif
*/
#include <vector>

namespace libpamm {
void clusteringMode();
double SOAPDistance(size_t dim, double *x, double *y);
double SOAPDistance(size_t dim, double *x, double *y, double xNorm,
                    double yNorm);

class distanceMatrix {
  // This is a upper triangular matrix, without the diagonal
public:
  distanceMatrix(size_t dim);
  ~distanceMatrix();
  double &operator()(size_t, size_t);
  double operator()(size_t, size_t) const;
  inline size_t address(size_t i, size_t j) const {
    if (i > j)
      std::swap(i, j);
    return (dim_ * (i + 1) - dim_ - i * (i + 1) / 2) + j - i - 1;
    // return dim_ * i * (i + 1) / 2 + j - (1 + i);
    // return i * (i + 1) / 2 + j;
  }

private:
  double *data_{nullptr};
  const size_t dim_{0};
};
class pammgrid {
public:
  pammgrid(size_t gridsize, size_t Dim);
  double &grid(size_t, size_t);
  double grid(size_t, size_t) const;
  size_t &samples(size_t);
  size_t samples(size_t) const;

private:
  std::vector<double> grid_{};
  std::vector<size_t> samples_{};
  size_t dim_ = 0;
};
} // namespace libpamm

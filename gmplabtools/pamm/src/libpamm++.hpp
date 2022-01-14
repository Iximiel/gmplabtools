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
double SOAPDistance(size_t dim, const double *x, const double *y);
double SOAPDistance(size_t dim, const double *x, const double *y,
                    double xyNormProduct);
double SOAPDistanceNormalized(size_t dim, const double *x, const double *y);

class distanceMatrix {
  // This is a upper triangular matrix, without the diagonal
public:
  distanceMatrix(size_t dim);
  distanceMatrix(const distanceMatrix &);
  distanceMatrix(distanceMatrix &&);
  distanceMatrix &operator=(distanceMatrix);
  void swap(distanceMatrix &);
  ~distanceMatrix();
  double &operator()(size_t, size_t);
  double operator()(size_t, size_t) const;
  inline size_t address(size_t i, size_t j) const {
    if (i < j)
      std::swap(i, j);
    return (i * (i - 1) / 2) + j;
  }
  size_t dim() const;

private:
  double *data_{nullptr};
  const size_t dim_{0};
};

class pammClustering {
public:
  pammClustering();
  pammClustering(const pammClustering &) = delete;
  pammClustering(pammClustering &&) = delete;
  virtual ~pammClustering();

  void work();
  void testLoadData();
  struct gridSimplified {
    /// Contains the information for the grid
    gridSimplified() = delete;
    gridSimplified(size_t);
    std::vector<size_t> grid{};
    std::vector<size_t> ni{};
    std::vector<double> wi{};
  };

private:
  size_t dim{0};
  size_t nsamples{0};
  size_t gridDim{0};
  std::vector<double> dataWeights{};
  double **data = nullptr; /// TODO: correct this
  gridSimplified createGrid(distanceMatrix distances, size_t firstPoint = 0);
};

} // namespace libpamm

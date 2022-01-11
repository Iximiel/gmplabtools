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

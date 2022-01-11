#include "libpamm++.hpp"
#include <cmath>
#include <limits>
#include <numeric>
namespace libpamm {
void clusteringMode() {}
double SOAPDistance(size_t dim, double *x, double *y, double xNorm,
                    double yNorm) {
  return std::sqrt(2.0 - 2.0 * std::inner_product(x, x + dim, y, 0.0) /
                             (xNorm * yNorm));
}

double SOAPDistance(size_t dim, double *x, double *y) {
  return SOAPDistance(dim, x, y, std::inner_product(x, x + dim, x, 0.0),
                      std::inner_product(y, y + dim, y, 0.0));
}

pammgrid::pammgrid(size_t gridsize, size_t Dim)
    : grid_(gridsize * Dim), samples_(gridsize), dim_(Dim) {}

double &pammgrid::grid(size_t i, size_t j) { return grid_[i * dim_ + j]; }
double pammgrid::grid(size_t i, size_t j) const { return grid_[i * dim_ + j]; }

size_t &pammgrid::samples(size_t i) { return samples_[i]; }
size_t pammgrid::samples(size_t i) const { return samples_[i]; }

pammgrid createGrid(size_t dim, size_t nsample, size_t gridDim, double **points,
                    size_t firstPoint = 0) {

  pammgrid grid(gridDim, dim);
  /*SUBROUTINE mkgrid(D,period,nsamples,ngrid,x,wj,y,ni,iminij, &
                      ineigh,wi,saveidx,idxgrid,ofile)
       ! Select ngrid grid points from nsamples using minmax and
       ! the voronoi polyhedra around them.
       !
       ! Args:
       !    nsamples: total points number
       !    ngrid: number of grid points
       !    x: array containing the data samples
       !    y: array that will contain the grid points
       !    ni: array cotaing the number of samples inside the Voronoj
     polyhedron of each grid point !    iminij: array containg the neighbor list
     for data samples

       INTEGER, INTENT(IN) :: D
       DOUBLE PRECISION, INTENT(IN) :: period(D)
       INTEGER, INTENT(IN) :: nsamples
       INTEGER, INTENT(IN) :: ngrid
       DOUBLE PRECISION, DIMENSION(D,nsamples), INTENT(IN) :: x
       DOUBLE PRECISION, DIMENSION(nsamples), INTENT(IN) :: wj

       DOUBLE PRECISION, DIMENSION(D,ngrid), INTENT(OUT) :: y
       INTEGER, DIMENSION(ngrid), INTENT(OUT) :: ni
       INTEGER, DIMENSION(ngrid), INTENT(OUT) :: ineigh
       INTEGER, DIMENSION(nsamples), INTENT(OUT) :: iminij
       DOUBLE PRECISION, DIMENSION(ngrid), INTENT(OUT) :: wi
       INTEGER, DIMENSION(ngrid), INTENT(OUT) :: idxgrid
       CHARACTER(LEN=1024), INTENT(IN) :: ofile
       LOGICAL, INTENT(IN) :: saveidx
*/
  for (auto k = 0U; k < dim; ++k) {
    grid.grid(0, k) = points[firstPoint][k];
  }
  grid.samples(0) = firstPoint;

  std::vector<double> Dmins(nsample, std::numeric_limits<double>::max());
  std::vector<size_t> Imins(nsample, 0);
  size_t jmax = 0;
  double dMax, dNeighMin;
  double dij;
  for (auto i = 0U; i < gridDim - 1; ++i) {
    dMax = 0.0;
    dNeighMin = std::numeric_limits<double>::max();
    for (auto j = 0U; j < nsample; ++j) {
      dij = SOAPDistance(dim, points[i], points[j]);
      if (dij < Dmins[j]) {
        Dmins[j] = dij;
        Imins[j] = i;
      }
      if (dMax < Dmins[j]) {
        dMax = Dmins[j];
        jmax = j;
      }
      if (dij < dNeighMin && (0.0 < dij)) {
        dNeighMin = dij;
        Imins[i] = j;
      }
    }
    for (auto k = 0U; k < dim; ++k) {
      grid.grid(i + 1, k) = points[jmax][k];
    }
    grid.samples(i + 1) = jmax;
  }
  return grid;
}
} // namespace libpamm
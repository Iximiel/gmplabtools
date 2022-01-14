#include "libpamm++.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
//#include <functional>
namespace libpamm {
void clusteringMode() {}
double SOAPDistance(size_t dim, const double *x, const double *y,
                    double xyNormProduct) {
  // using already initialized NormProduct
  xyNormProduct = std::sqrt(2.0 - 2.0 * std::inner_product(x, x + dim, y, 0.0) /
                                      xyNormProduct);
  return xyNormProduct;
}

double SOAPDistance(size_t dim, const double *x, const double *y) {
  return SOAPDistance(dim, x, y,
                      sqrt(std::inner_product(x, x + dim, x, 0.0) *
                           std::inner_product(y, y + dim, y, 0.0)));
}

double SOAPDistanceNormalized(size_t dim, const double *x, const double *y) {
  return std::sqrt(2.0 - 2.0 * std::inner_product(x, x + dim, y, 0.0));
}

distanceMatrix::distanceMatrix(size_t dim)
    : data_(new double[((dim - 1) * dim) / 2]), dim_(dim) {}

distanceMatrix::distanceMatrix(const distanceMatrix &other)
    : data_(new double[((other.dim_ - 1) * other.dim_) / 2]), dim_(other.dim_) {
  const size_t len = ((dim_ - 1) * dim_) / 2;
  for (size_t i = 0; i < len; ++i) {
    this->data_[i] = other.data_[i];
  }
}

distanceMatrix::distanceMatrix(distanceMatrix &&other)
    : data_(other.data_), dim_(other.dim_) {
  other.data_ = nullptr;
}

distanceMatrix &distanceMatrix::operator=(distanceMatrix other) {
  if (&other != this) {
    distanceMatrix::swap(other);
  }
  return *this;
}

void distanceMatrix::swap(distanceMatrix &other) {
  if (dim_ != other.dim_) {
    throw "Error in distanceMatrix swap: different size";
  }
  std::swap(this->data_, other.data_);
}

distanceMatrix::~distanceMatrix() { delete[] data_; }

double &distanceMatrix::operator()(size_t i, size_t j) {
  return data_[address(i, j)];
}

double distanceMatrix::operator()(size_t i, size_t j) const {
  return data_[address(i, j)];
}

distanceMatrix CalculateDistanceMatrixSOAP(double **data, size_t dataDim,
                                           size_t dim) {
  distanceMatrix distances(dataDim);
  std::vector<double> norms(dataDim);
  for (auto i = 0U; i < dataDim; ++i) {
    norms[i] = sqrt(std::inner_product(data[i], data[i] + dim, data[i], 0.0));
  }
  double dmax = 0.0;
  size_t ti, tj;
  for (auto i = 0U; i < dataDim; ++i) {
    for (auto j = 0; j < i; ++j) {
      distances(i, j) =
          SOAPDistance(dim, data[i], data[j], norms[i] * norms[j]);
      if (distances(i, j) > dmax) {
        dmax = distances(i, j);
        ti = i;
        tj = j;
      }
    }
  }
  return distances;
}

/**/
pammClustering::pammClustering() {}

pammClustering::~pammClustering() {
  if (nsamples > 0) {
    delete[] data[0];
  }
  delete[] data;
}
pammClustering::gridSimplified::gridSimplified(size_t gridDim)
    : grid(gridDim, 0), ni(gridDim, 0), wi(gridDim, 0.0) {}

pammClustering::gridSimplified
pammClustering::createGrid(distanceMatrix distances, size_t firstPoint) {
  gridSimplified grid(gridDim);

  std::vector<double> Dmins(nsamples, std::numeric_limits<double>::max());
  std::vector<size_t> closestGridIndex(nsamples, 0);
  std::vector<size_t> voronoiAssociationIndex(gridDim, 0);
  size_t jmax = 0;
  grid.grid[0] = firstPoint;
  Dmins[firstPoint] = 0.0;
  closestGridIndex[firstPoint] = 0;
  double dij;
  {
    double dMax, dNeighMin;
    for (auto i = 0U; i < gridDim - 1; ++i) {
      dMax = 0.0;
      dNeighMin = std::numeric_limits<double>::max();
      auto gridIndex = grid.grid[i];
      // find the farthest point from gridIndex
      for (auto j = 0U; j < nsamples; ++j) {
        if (gridIndex == j) {
          continue;
        }
        dij = distances(gridIndex, j);
        if (dij < Dmins[j]) {
          Dmins[j] = dij;
          // keep track of the Voronoi attribution
          closestGridIndex[j] = i;
        }
        if (dMax < Dmins[j]) {
          dMax = Dmins[j];
          jmax = j;
        }
        if (dij < dNeighMin && (0.0 < dij)) {
          dNeighMin = dij;
          // store index of closest sample neighbor to grid point
          voronoiAssociationIndex[i] = j;
        }
      }
      grid.grid[i + 1] = jmax;
      Dmins[jmax] = 0.0;
      closestGridIndex[jmax] = i + 1;
    }
  }
  // completes the voronoi attribuition for the last point in the grid
  {
    auto gridIndex = grid.grid[gridDim - 1];
    auto dNeighMin = std::numeric_limits<double>::max();
    for (auto j = 0U; j < nsamples; ++j) {
      if (gridIndex == j) {
        continue;
      }
      dij = distances(gridIndex, j);
      if (dij < Dmins[j]) {
        Dmins[j] = dij;
        closestGridIndex[j] = gridDim - 1;
      }
      if (dij < dNeighMin && (0.0 < dij)) {
        dNeighMin = dij;
        voronoiAssociationIndex[gridDim - 1] = j;
      }
    }
  }
  // Assign neighbor list pointer of voronois
  // Number of points in each voronoi polyhedra

  for (auto j = 0U; j < nsamples; ++j) {
    ++grid.ni[closestGridIndex[j]];
    grid.wi[closestGridIndex[j]] += dataWeights[j];
  }
  return grid;
}

void pammClustering::work() {

  auto distances = CalculateDistanceMatrixSOAP(data, nsamples, dim);
  size_t randomGeneratedFirstPoint = 1;
  // no need for precalculated distances
  auto grid = createGrid(distances, randomGeneratedFirstPoint);
  // TODO:voronoi if loading grid
  //~MAYBE: export voronoi

  // TODO: generate Neigh list between voronoi sets
  // TODO: generate distance matrix between grid points
  // TODO: Gabriel Graphs

  //~MAYBE: global covariance on grid
  // TODO: localization- Kernel Density Bandwidths + warning on grid dimension
  // // TODO: localization with fractionofpoint or fractionofspread
  // TODO: Bandwidths from localization
  // //~->covariance->oracle shrinkage->invert Covariance

  // TODO: Kernel Density Estimation
  // TODO: Kernel Density Estimation Statical error
  // //TODO: bootstrap
  // //TODO: binomial-distribution ansatz to estimate the error

  // TODO: Quick-Shift (also for bootstrap)
  // TODO: Determine cluster Centers, merging the outliers
  // completing the work

  // file to save: bs dim grid pamm

  std::ofstream f("test_grid.soap");
  std::ofstream g("test_grid.dat");
  for (auto i = 0; i < gridDim; ++i) {
    for (auto j = 0; j < dim; ++j) {
      f << ((j == 0) ? "" : " ") << data[grid.grid[i]][j];
    }
    f << '\n';
    g << grid.grid[i] << '\n';
  }
  f.close();
}

void pammClustering::testLoadData() {
  std::ifstream f("test.soap");
  dim = 324;
  nsamples = 30900;
  // nsamples = 30;
  data = new double *[nsamples];
  data[0] = new double[nsamples * dim];
  for (auto i = 0; i < nsamples; ++i) {
    if (i > 0) {
      data[i] = data[0] + i * dim;
    }
    for (auto j = 0; j < dim; ++j) {
      f >> data[i][j];
    }
  }
  dataWeights = std::vector<double>(nsamples, 1.0);
  gridDim = 1000;
}

} // namespace libpamm

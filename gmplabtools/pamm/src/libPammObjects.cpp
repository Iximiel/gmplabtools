#include "libpamm++.hpp"

namespace libpamm {

  gridInfo::gridInfo (size_t gridDim, size_t dataDim)
    : grid (gridDim, dataDim),
      // NofSamples (gridDim, 0),
      gridIndexes (gridDim, 0),
      VoronoiWeights (gridDim, 0.0),
      voronoiAssociationIndex (gridDim, 0),
      gridNearestNeighbours (gridDim, 0),
      samplesIndexes (gridDim, std::vector<size_t> (0)),
      gridDistancesSquared (gridDim) {}

  size_t gridInfo::size () const { return grid.rows (); }

  gridErrorProbabilities::gridErrorProbabilities (
    size_t gridDim)
    : absolute (gridDim, 0),
      relative (gridDim, 0) {}
  size_t gridErrorProbabilities::size () const {
    return absolute.size ();
  }

  double EuclideanDistance (size_t dim, const double *x, const double *y) {
    return std::sqrt (std::inner_product (
      x, x + dim, y, 0.0, std::plus<> (), [] (double x, double y) {
        x -= y;
        return x * x;
      }));
  }

  double
  EuclideanDistanceSquared (size_t dim, const double *x, const double *y) {
    return std::inner_product (
      x, x + dim, y, 0.0, std::plus<> (), [] (double x, double y) {
        x -= y;
        return x * x;
      });
  }

  double SOAPDistance (
    size_t dim, const double *x, const double *y, double xyNormProduct) {
    // using already initialized NormProduct
    xyNormProduct = std::sqrt (
      2.0 - 2.0 * std::inner_product (x, x + dim, y, 0.0) / xyNormProduct);
    return xyNormProduct;
  }

  double SOAPDistance (size_t dim, const double *x, const double *y) {
    return SOAPDistance (
      dim, x, y,
      sqrt (
        std::inner_product (x, x + dim, x, 0.0) *
        std::inner_product (y, y + dim, y, 0.0)));
  }

  double SOAPDistanceNormalized (size_t dim, const double *x, const double *y) {
    return std::sqrt (2.0 - 2.0 * std::inner_product (x, x + dim, y, 0.0));
  }

  double SOAPDistanceSquared (
    size_t dim, const double *x, const double *y, double xyNormProduct) {
    // using already initialized NormProduct
    xyNormProduct =
      2.0 - 2.0 * std::inner_product (x, x + dim, y, 0.0) / xyNormProduct;
    return xyNormProduct;
  }

  double SOAPDistanceSquared (size_t dim, const double *x, const double *y) {
    return SOAPDistanceSquared (
      dim, x, y,
      sqrt (
        std::inner_product (x, x + dim, x, 0.0) *
        std::inner_product (y, y + dim, y, 0.0)));
  }

  double
  SOAPDistanceSquaredNormalized (size_t dim, const double *x, const double *y) {
    return 2.0 - 2.0 * std::inner_product (x, x + dim, y, 0.0);
  }

/*
  double pammClustering::calculateMahalanobisDistanceSquared (
    const double *A, const double *B, const Matrix &invCov) const {
    auto vA = Eigen::Map<Eigen::Matrix<const double, -1, 1>> (
      A, invCov.rows ()); // (A,invCov.rows ());
    // Eigen::VectorXd vA (invCov.rows (), A);
    auto vB =
      Eigen::Map<Eigen::Matrix<const double, -1, 1>> (B, invCov.rows ());
    // this may be euclidean?
    return vA.transpose () * (invCov * vB);
  }*/

  double calculateMahalanobisDistanceSquared (
    const Eigen::VectorXd &A,
    const Eigen::VectorXd &B,
    const Matrix &invCov) {
    auto D = A - B;
    // this may be euclidean?
    return D.transpose () * (invCov * B);
  }

  distanceMatrix
  CalculateDistanceMatrixSOAP (double **data, size_t dataDim, size_t dim) {
    distanceMatrix distances (dataDim);
    std::vector<double> norms (dataDim);
    for (auto i = 0U; i < dataDim; ++i) {
      norms[i] =
        sqrt (std::inner_product (data[i], data[i] + dim, data[i], 0.0));
    }

    for (auto i = 0U; i < dataDim; ++i) {
      for (auto j = 0; j < i; ++j) {
        distances (i, j) =
          SOAPDistance (dim, data[i], data[j], norms[i] * norms[j]);
      }
    }
    return distances;
  }
} // namespace libpamm
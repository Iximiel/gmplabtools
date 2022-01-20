#include "libpamm++.hpp"
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>
namespace libpamm {
  const double log2PI = std::log (TWOPI);
  gridInfo::gridInfo (size_t gridDim, size_t dataDim)
    : grid (gridDim, dataDim),
      // NofSamples (gridDim, 0),
      gridIndexes (gridDim, 0),
      VoronoiWeights (gridDim, 0.0),
      voronoiAssociationIndex (gridDim, 0),
      gridNearestNeighbours (gridDim, 0),
      // localWeightSum (gridDim, 0), //will be initializated when needed
      // sigmaSQ (gridDim, 0),//will be initializated when needed
      // pointProbabilities (gridDim),//will be initializated when needed
      samplesIndexes (gridDim, std::vector<size_t> (0)),
      gridDistancesSquared (gridDim) {}

  size_t gridInfo::size () const { return grid.rows (); }
  size_t gridInfo::dimensionality () const { return grid.cols (); }

  gridErrorProbabilities::gridErrorProbabilities (size_t gridDim)
    : absolute (gridDim, 0),
      relative (gridDim, 0) {}
  size_t gridErrorProbabilities::size () const { return absolute.size (); }

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
    const Eigen::VectorXd &A, const Eigen::VectorXd &B, const Matrix &invCov) {
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
  gaussian::gaussian (size_t N) : D (N), center (N), cov (N, N), icov (N, N) {}
  gaussian::gaussian (
    const size_t N,
    const size_t idK /*gridID*/,
    const size_t nmsopt,
    const double normpks,
    const gridInfo &grid,
    const quickShiftOutput &clusterInfo,
    Matrix HiInverse,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob)
    : D (N),
      center (N),
      cov (N, N),
      icov (N, N) {
    // optional mean shift for estimation of clustermode
    for (size_t ms = 0; ms < nmsopt; ++ms) {
      Eigen::VectorXd msmu = Eigen::VectorXd::Zero (D);
      // variables to set GM covariances
      double tmppks = -std::numeric_limits<double>::max ();
      for (size_t GI = 0; GI < grid.size (); ++GI) {
        double mahalanobis = calculateMahalanobisDistanceSquared (
          grid.grid.row (GI), grid.grid.row (idK), HiInverse);
        double msw = -0.5 * (normkernel[idK] + mahalanobis) + prob[GI];
        auto delta = grid.grid.row (GI) - grid.grid.row (idK);

        msmu += exp (msw) * delta;

        // log-sum-exp
        if (msw < tmppks) {
          tmppks += log (1.0 + exp (msw - tmppks));
        } else {
          tmppks = msw + log (1.0 + exp (tmppks - msw));
        }
      } // GI

      // TODO::if(periodic){}
      { center += msmu / exp (tmppks); }
    } // mean shifts

    // compute the covariance
    // TODO::if(periodic){}
    {
      // If we have a cluster with one point we compute the weighted
      // covariance with the points in the Voronoi
      if (
        std::count (
          clusterInfo.gridToClusterIdx.begin (),
          clusterInfo.gridToClusterIdx.end (), idK) == 1) {
        // CALL
        // getcovcluster(D,period,nsamples,wj,x,iminij,clustercenters(k),clusters(k)%cov)
        std::cerr << " Warning: single point cluster!!! \n";
      }
      double accumulatedLogSum =
        accumulateLogsumexp_if (clusterInfo.gridToClusterIdx, prob, idK);
      cov = oracleShrinkage (
        CalculateLogCovarianceMatrix (idK, grid, clusterInfo, normkernel, prob),
        accumulatedLogSum);
      weight = exp (accumulatedLogSum - normpks);
    }
  }
  void gaussian::prepare () {
    det = cov.determinant ();
    icov = cov.inverse ();
    lnorm = log (1.0 / sqrt (pow (TWOPI, D) * det));
  }

  std::string myformat (double x) {
    char buf[22];
    // fortran original formats with  ES21.8E4, C cannot serve the 4 digit for
    // the exponent, or at least I don't know how and google do not help
    std::snprintf (buf, 22, "%21.8E", x);
    return {buf};
  }

  std::ostream &operator<< (std::ostream &stream, gaussian g) {
    stream << " " << myformat (g.weight);
    for (const auto m : g.center) {
      stream << " " << myformat (m);
    }
    for (size_t row = 0; row < g.D; ++row) {
      for (size_t col = 0; col < g.D; ++col) {
        stream << " " << myformat (g.cov (row, col));
      }
    }
    return stream;
  }
} // namespace libpamm
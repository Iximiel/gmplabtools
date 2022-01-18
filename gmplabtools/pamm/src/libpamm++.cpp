#include "libpamm++.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
//#include <functional>
namespace libpamm {

  Matrix oracleShrinkage (Matrix m, double dimParameter) {
    const size_t D = m.rows ();

    double trQ = m.trace ();
    const double tr2Q = trQ * trQ;
    const double trQ2 = [=] (double t) {
      for (size_t i = 0; i < D; ++i) {
        t += m (i, i) * m (i, i);
      }
      return t;
    }(0.0);

    // apply oracle approximating shrinkage algorithm on m
    const double phi =
      ((1.0 - 2.0 / static_cast<double> (D)) * trQ2 + tr2Q) /
      ((dimParameter + 1.0 - 2.0 / static_cast<double> (D)) * trQ2 -
       tr2Q / static_cast<double> (D));

    const double rho = std::min (1.0, phi);

    // regularized local covariance matrix for grid point
    m *= (1.0 - rho);
    trQ = trQ / static_cast<double> (D);
    for (size_t i = 0; i < D; ++i) {
      m (i, i) += rho * trQ;
    }
    return m;
  }

  double RoyVetterliDimensionality (const Matrix &square) {
    assert (square.rows () == square.cols ());
    size_t D = square.rows ();
    auto eigenvalues = [] (auto t) {
      std::vector<double> x (t.size ());
      for (size_t i = 0; i < t.size (); ++i) {
        x[i] = t[i].real ();
      }
      return x;
    }(square.eigenvalues ());
    double eigenAccumulation =
      std::accumulate (eigenvalues.begin (), eigenvalues.end (), 0.0);
    std::transform (
      eigenvalues.begin (), eigenvalues.end (), eigenvalues.begin (),
      [=] (double x) -> double {
        if (x <= 0) {
          x = 0;
        } else {
          x /= eigenAccumulation;
          x *= std::log (x);
          if (std::isnan (x)) {
            x = 0;
          }
        }

        return x;
      });
    eigenAccumulation =
      std::accumulate (eigenvalues.begin (), eigenvalues.end (), 0.0);
    eigenAccumulation = std::exp (-eigenAccumulation);
    return eigenAccumulation;
  }

  void clusteringMode () {}
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

  /**/
  pammClustering::pammClustering () {}

  pammClustering::~pammClustering () {
    /*
    if (nsamples > 0) {
      delete[] data[0];
    }
    delete[] data;
    */
  }

  pammClustering::gridInfo::gridInfo (size_t gridDim, size_t dataDim)
    : grid (gridDim, dataDim),
      // NofSamples (gridDim, 0),
      VoronoiWeights (gridDim, 0.0),
      voronoiAssociationIndex (gridDim, 0),
      gridNearestNeighbours (gridDim, 0),
      samplesIndexes (gridDim, std::vector<size_t> (0)),
      gridDistances (gridDim) {}

  size_t pammClustering::gridInfo::size () const { return grid.rows (); }
  pammClustering::gridInfo
  pammClustering::createGrid (size_t firstPoint) const {
    gridInfo grid (gridDim, dim);
    std::vector<double> Dmins (nsamples, std::numeric_limits<double>::max ());
    std::vector<size_t> closestGridIndex (nsamples, 0);
    std::vector<size_t> gridIndexes (gridDim, 0);
    /*
    auto copyPoint = [this] (auto &row, const double *dataPoint) {
      std::copy (dataPoint, dataPoint + dim, row.begin ());
    };*/
    size_t jmax = 0;
    gridIndexes[0] = firstPoint;
    /*
    std::copy (
      data[firstPoint], data[firstPoint] + dim, grid.grid.row (0).begin ());
*/
    grid.grid.row (0) = data.row (firstPoint);
    Dmins[firstPoint] = 0.0;
    closestGridIndex[firstPoint] = 0;
    double dij;
    {
      double dMax, dNeighMin;
      for (size_t i = 0; i < gridDim - 1; ++i) {
        grid.samplesIndexes[i].reserve (nsamples / gridDim);
        dMax = 0.0;
        dNeighMin = std::numeric_limits<double>::max ();
        size_t gridIndex = gridIndexes[i];
        // find the farthest point from gridIndex
        for (auto j = 0U; j < nsamples; ++j) {
          if (gridIndex == j) {
            continue;
          }
          dij = distanceCalculator (gridIndex, j);
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
            grid.voronoiAssociationIndex[i] = j;
          }
        }
        grid.grid.row (i + 1) = data.row (jmax);
        // copyPoint (grid.grid.row (i + 1), data[jmax]);
        gridIndexes[i + 1] = jmax;
        Dmins[jmax] = 0.0;
        closestGridIndex[jmax] = i + 1;
      }
    }
    // completes the voronoi attribuition for the last point in the grid
    {
      auto gridIndex = gridIndexes[gridDim - 1];
      grid.samplesIndexes[gridDim - 1].reserve (nsamples / gridDim);
      auto dNeighMin = std::numeric_limits<double>::max ();
      for (auto j = 0U; j < nsamples; ++j) {
        if (gridIndex == j) {
          continue;
        }
        dij = distanceCalculator (gridIndex, j);
        if (dij < Dmins[j]) {
          Dmins[j] = dij;
          closestGridIndex[j] = gridDim - 1;
        }
        if (dij < dNeighMin && (0.0 < dij)) {
          dNeighMin = dij;
          grid.voronoiAssociationIndex[gridDim - 1] = j;
        }
      }
    }
    // Assign neighbor list pointer of voronois
    // Number of points in each voronoi polyhedra

    for (auto j = 0U; j < nsamples; ++j) {
      grid.VoronoiWeights[closestGridIndex[j]] += dataWeights[j];
      // TODO: push_back may result in poor performances: need to improve
      grid.samplesIndexes[closestGridIndex[j]].push_back (j);
    }
    return grid;
  }

  void pammClustering::work () {
    if (initialized_) {
      bool normalizeDataset = true;
      if (normalizeDataset) {
        doNormalizeDataset ();
      }
      // auto distances = CalculateDistanceMatrixSOAP(data, nsamples, dim);
      size_t randomGeneratedFirstPoint = 1;
      double totalWeight =
        std::accumulate (dataWeights.begin (), dataWeights.end (), 0.0);
      // no need for precalculated distances
      auto grid = createGrid (randomGeneratedFirstPoint);
      // this normalizes the gridweights
      for (auto &weight : grid.VoronoiWeights) {
        weight /= totalWeight;
      }
      totalWeight = 1.0;
      // if(loadGrid){
      // auto grid = loadGrid();
      // TODO:voronoi if loading grid
      // }
      //~MAYBE: export voronoi

      // DONE: generate Neigh list between voronoi sets:moved increateGrid
      // DONE: generate distance matrix between grid points #445
      // TODO: can we do this while generating the grid?
      GenerateGridDistanceMatrix (grid);
      Matrix covariance =
        CalculateCovarianceMatrix (grid, grid.VoronoiWeights, totalWeight);

      double weightAccumulation = double (nsamples);
      //~MAYBE: Gabriel Graphs //gs is gabriel clusterign flag in pamm #473
      // Matrix covariance = CalculateCovarianceMatrix (grid, totalWeight);
      bandwidthEstimation (grid, covariance, weightAccumulation);
      // TODO: localization weights #527
      // CALL
      // localization(D,period,ngrid,sigma2(i),y,wi,y(:,i),wlocal,flocal(i))
      // loalization #1307
      // TODO: localization- Kernel Density Bandwidths + warning on grid
      // dimension
      // // TODO: localization with fractionofpoint or fractionofspread
      // TODO: Bandwidths from localization
      // //~->covariance->oracle shrinkage->invert Covariance

      // TODO: Kernel Density Estimation
      // TODO: Kernel Density Estimation Statical error
      // //TODO: bootstrap
      // //TODO: binomial-distribution ansatz to estimate the error

      // TODO: Quick-Shift (also for bootstrap)
      // TODO: Determine cluster Centers, merging the outliers
      // completing the work:
      // TODO: Gaussian for each cluster and covariance
      // Output?
      // file to save: bs dim grid pamm

      std::ofstream f ("test_grid.soap");
      std::ofstream g ("test_grid.dat");
      for (auto i = 0; i < gridDim; ++i) {
        for (auto j = 0; j < dim; ++j) {
          f << ((j == 0) ? "" : " ") << grid.grid (i, j);
        }
        f << '\n';
        g << grid.grid.row (i);
        for (const auto j : grid.samplesIndexes[i]) {
          g << ' ' << j;
        }
        g << '\n';
      }
      f.close ();
      g.close ();
    } else {
      std::cerr << "Not initalized" << std::endl;
    }
  }

  void pammClustering::testLoadData () {
    std::ifstream f ("test.soap");
    dim = 324;
    nsamples = 30900;
    // nsamples = 30;
    data = Matrix (nsamples, dim);

    for (auto i = 0; i < nsamples; ++i) {
      for (auto j = 0; j < dim; ++j) {
        f >> data (i, j);
      }
    }
    dataWeights = std::vector<double> (nsamples, 1.0);
    gridDim = 1000;
    this->initialized_ = true;
    this->dataSetNormalized_ = false;
  }

  void pammClustering::doNormalizeDataset () {
    if (!dataSetNormalized_) {
      dataSetNormalized_ = true;
      for (auto i = 0; i < nsamples; ++i) {
        auto row = data.row (i);
        double norm = sqrt (
          std::inner_product (row.begin (), row.end (), row.begin (), 0.0));
        std::transform (
          row.begin (), row.end (), row.begin (),
          [=] (double x) -> double { return x / norm; });
      }
    }
  }
  double
  pammClustering::distanceCalculator (const size_t i, const size_t j) const {
    return distanceCalculator (data.row (i).data (), data.row (j).data ());
  }

  double pammClustering::distanceCalculator (
    const double *pointI, const double *pointJ) const {
    // TODO: make the user able to choose the distance algorithm
    // here we assume that the data is normalized:
    return SOAPDistanceNormalized (dim, pointI, pointJ);
  }
  void pammClustering::GenerateGridDistanceMatrix (gridInfo &grid) const {
    double d;
    for (auto i = 0; i < grid.grid.rows (); ++i) {
      auto NNdist = std::numeric_limits<double>::max ();
      for (auto j = i + 1; j < grid.grid.rows (); ++j) {
        d = SOAPDistanceNormalized (
          dim, grid.grid.row (i).data (), grid.grid.row (j).data ());
        grid.gridDistances (i, j) = d;
        if (d < NNdist) {
          grid.gridNearestNeighbours[i] = j;
          NNdist = d;
        }
      }
    }
  }
  /*
    void pammClustering::GenerateGridNeighbourList (gridInfo &grid) const {
      double d;
      //#row 1960
      for (auto i = 0; i < grid.grid.rows (); ++i) {
        auto NNdist = std::numeric_limits<double>::max ();
        for (auto j = i + 1; j < grid.grid.rows (); ++j) {
          d = SOAPDistanceNormalized (dim, grid.grid[i], grid.grid[j]);
          grid.gridDistances (i, j) = d;
          if (d < NNdist) {
            grid.gridNearestNeighbours[i] = j;
            NNdist = d;
          }
        }
      }
    }
  */
  Matrix pammClustering::CalculateCovarianceMatrix (
    const gridInfo &grid,
    const std::vector<double> &weights,
    const double totalWeight) const {
    // CALL covariance(D,period,ngrid,normwj,wi,y,Q)
    //     covariance(D,period,N    ,wnorm,w,x,Q)
    std::vector<double> means (dim);
    Matrix deltafromMeans (grid.grid.rows (), dim);
    Matrix deltafromMeansWeighted (grid.grid.rows (), dim);
    for (auto D = 0; D < dim; ++D) {
      means[D] = 0.0;
      // weighted mean:
      for (auto gID = 0; gID < grid.grid.rows (); ++gID) {
        means[D] += grid.grid (gID, D) * weights[gID];
      }
      means[D] /= totalWeight;
      for (auto gID = 0; gID < grid.grid.rows (); ++gID) {
        deltafromMeans (gID, D) = grid.grid (gID, D) - means[D];
        deltafromMeansWeighted (gID, D) =
          deltafromMeans (gID, D) * weights[gID] / totalWeight;
      }
    }
    double wSumSquared = std::accumulate (
      weights.begin (), weights.end (), 0.0, [=] (double x, double y) {
        y /= totalWeight;
        return x + y * y;
      });

    // TODO: see if it the correct covariance!
    Matrix covariance = deltafromMeansWeighted.transpose () * deltafromMeans;

    std::cerr << covariance.rows () << " " << covariance.cols () << " "
              << wSumSquared << std::endl;

    covariance *= (1.0 - wSumSquared);
    return covariance;
  }

  void pammClustering::bandwidthEstimation (
    const gridInfo &grid, const Matrix &covariance, const double totalWeight) {
    std::vector<double> localWeights (grid.size ());
    std::vector<double> LocalDimensionality (grid.size ());

    double delta = totalWeight / static_cast<double> (nsamples);
    tune = 0.0;
    for (auto D = 0; D < dim; ++D) {
      tune += covariance (D, D);
    }
    std::vector<double> sigmaSQ (grid.size (), tune);
    bool useFractionOfPoints = true;
    for (auto GI = 0; GI < grid.size (); ++GI) {
      double localWeightSum = estimateGaussianLocalization (
        grid, GI, sigmaSQ[GI], localWeights.data ());
      // two strategies:
      if (useFractionOfPoints) {
        fractionOfPointsLocalization (
          grid, GI, delta, localWeightSum, sigmaSQ[GI], localWeights.data ());
      } else {
        fractionOfSpreadLocalization (
          grid, GI, delta, localWeightSum, sigmaSQ[GI], localWeights.data ());
      }
      //#619
      // local covariance based on the grid aproxximation:
      // CALL covariance(D,period,ngrid,flocal(i),wlocal,y,Qi)
      auto localCovariance =
        CalculateCovarianceMatrix (grid, localWeights, localWeightSum);
      // number of local points:
      double nlocal = localWeightSum * nsamples;
      // estimate local dimensionality
      LocalDimensionality[GI] = RoyVetterliDimensionality (localCovariance);
      // oracle shrinkage of covariance matrix
      localCovariance = oracleShrinkage (localCovariance, nlocal);
      // auto inverseLocalCovariance = localCovariance.inverse ();

      // inverse local covariance matrix and store it
      //  CALL invmatrix (D, Qi, Qiinv)
    }
  }
  /*
      Matrix matrixOfDistances (const Matrix &points, const double *point) {
        Matrix deltas (points.rows (), points.cols ());
        for (auto R = 0; R < points.rows (); ++R) {
          for (auto D = 0; D < points.cols (); ++D) {
            deltas (R, D) = points (R, D) - point[D];
          }
        }
        return deltas;
      }*/
  double pammClustering::estimateGaussianLocalization (
    const gridInfo &grid,
    const double *point,
    const double sigmaSQ,
    double *outweights) const {
    double dSum;
    double localWeightSum = 0.0;
    for (auto gI = 0; gI < grid.size (); ++gI) {
      dSum = distanceCalculator (point, grid.grid.row (gI).data ());
      dSum *= dSum;
      // estimate weights for localization as product from
      // spherical gaussian weights and weights in voronoi
      outweights[gI] =
        std::exp (-0.5 / sigmaSQ * dSum) * grid.VoronoiWeights[gI];
      localWeightSum += outweights[gI];
    }
    return localWeightSum;
  }

  double pammClustering::estimateGaussianLocalization (
    const gridInfo &grid,
    const size_t gridPoint,
    const double sigmaSQ,
    double *outweights) const {
    double dSum;
    double localWeightSum = 0.0;
    for (auto gI = 0; gI < grid.size (); ++gI) {
      dSum = grid.gridDistances (gridPoint, gI);
      dSum *= dSum;
      // estimate weights for localization as product from
      // spherical gaussian weights and weights in voronoi
      outweights[gI] =
        std::exp (-0.5 / sigmaSQ * dSum) * grid.VoronoiWeights[gI];
      localWeightSum += outweights[gI];
    }
    return localWeightSum;
  }

  void pammClustering::fractionOfPointsLocalization (
    const gridInfo &grid,
    const size_t gID,
    const double delta,
    double &weight,
    double &sigmaSQ,
    double *localWeights) {
    double lim = fractionOfPointsVal;
    if (fractionOfPointsVal < grid.VoronoiWeights[gID]) {
      lim = weight + delta;
      std::cerr << " Warning: localization smaller than voronoi, increase grid "
                   "size (meanwhile adjusted localization)!"
                << std::endl;
    }
    // quick approach by "tune" steps
    while (weight < lim) {
      sigmaSQ += tune;
      weight = estimateGaussianLocalization (grid, gID, sigmaSQ, localWeights);
    }
    // using bisection to fine tune the localizatiom
    double bisectionParameter = 2.0;
    while ((weight - lim < delta) && (weight - lim > -delta)) { //(true){}
      sigmaSQ += tune / (bisectionParameter) * (weight < lim) ? 1.0 : -1.0;
      weight = estimateGaussianLocalization (grid, gID, sigmaSQ, localWeights);
      /*
       if ((weight - lim < delta) && (weight - lim > -delta)) {
        //(abs(weight-lim) < delta))?
        break;
      }
      */
      bisectionParameter *= 2.0;
    }
  }
  void pammClustering::fractionOfSpreadLocalization (
    const gridInfo &grid,
    const size_t gID,
    const double delta,
    double &weight,
    double &sigmaSQ,
    double *localWeights) {
    double mindist = grid.gridDistances (gID, grid.gridNearestNeighbours[gID]);
    if (sigmaSQ < mindist) {
      sigmaSQ = mindist;
      std::cerr << " Warning: localization smaller than Voronoi diameter, "
                   "increase grid size (meanwhile adjusted localization)!"
                << std::endl;
      weight = estimateGaussianLocalization (grid, gID, sigmaSQ, localWeights);
    }
  }
} // namespace libpamm

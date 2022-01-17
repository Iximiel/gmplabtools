#include "libpamm++.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
//#include <functional>
namespace libpamm {
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
    if (nsamples > 0) {
      delete[] data[0];
    }
    delete[] data;
  }

  pammClustering::gridInfo::gridInfo (size_t gridDim)
    : grid (gridDim, 0),
      // NofSamples (gridDim, 0),
      // WeightOfSamples (gridDim, 0.0),
      voronoiAssociationIndex (gridDim, 0),
      gridNearestNeighbours (gridDim, 0),
      samplesIndexes (gridDim, std::vector<size_t> (0)),
      gridDistances (gridDim) {}

  pammClustering::gridInfo
  pammClustering::createGrid (size_t firstPoint) const {
    gridInfo grid (gridDim);

    std::vector<double> Dmins (nsamples, std::numeric_limits<double>::max ());
    std::vector<size_t> closestGridIndex (nsamples, 0);
    size_t jmax = 0;
    grid.grid[0] = firstPoint;
    Dmins[firstPoint] = 0.0;
    closestGridIndex[firstPoint] = 0;
    double dij;
    {
      double dMax, dNeighMin;
      for (auto i = 0U; i < gridDim - 1; ++i) {
        grid.samplesIndexes[i].reserve (nsamples / gridDim);
        dMax = 0.0;
        dNeighMin = std::numeric_limits<double>::max ();
        auto gridIndex = grid.grid[i];
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
        grid.grid[i + 1] = jmax;
        Dmins[jmax] = 0.0;
        closestGridIndex[jmax] = i + 1;
      }
    }
    // completes the voronoi attribuition for the last point in the grid
    {
      auto gridIndex = grid.grid[gridDim - 1];
      grid.samplesIndexes[gridIndex].reserve (nsamples / gridDim);
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
      // grid.WeightOfSamples[closestGridIndex[j]] += dataWeights[j];
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
      // no need for precalculated distances
      auto grid = createGrid (randomGeneratedFirstPoint);
      // auto grid = loadGrid();
      // TODO:voronoi if loading grid
      //~MAYBE: export voronoi

      // DONE: generate Neigh list between voronoi sets:moved increateGrid
      // DONE: generate distance matrix between grid points #445
      // TODO: can we do this while generating the grid?
      CalculateGridDistanceMatrix (grid);
      // not using weight, yet:
      double weightAccululation = double (nsamples);
      //~MAYBE: Gabriel Graphs //gs is gabriel clusterign flag in pamm #473

      //~MAYBE: global covariance on grid
      // normwj:accumulator for wj, wj is the weight of the samples
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
          f << ((j == 0) ? "" : " ") << data[grid.grid[i]][j];
        }
        f << '\n';
        g << grid.grid[i];
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
    // dataWeights = std::vector<double> (nsamples, 1.0);
    gridDim = 1000;
    this->initialized_ = true;
    this->dataSetNormalized_ = false;
  }

  void pammClustering::doNormalizeDataset () {
    if (!dataSetNormalized_) {
      dataSetNormalized_ = true;
      for (auto i = 0; i < nsamples; ++i) {
        double norm =
          sqrt (std::inner_product (data[i], data[i] + dim, data[i], 0.0));
        std::transform (
          data[i], data[i] + dim, data[i],
          [=] (double x) -> double { return x / norm; });
      }
    }
  }
  double pammClustering::distanceCalculator (size_t i, size_t j) const {
    // TODO: make the user able to choose the distance algorithm
    // here we assume that the data is normalized:
    return SOAPDistanceNormalized (dim, data[i], data[j]);
  }

  void pammClustering::CalculateGridDistanceMatrix (gridInfo &grid) const {
    double d;
    for (auto i = 0; i < grid.grid.size (); ++i) {
      auto idxI = grid.grid[i];
      auto NNdist = std::numeric_limits<double>::max ();
      for (auto j = i + 1; j < grid.grid.size (); ++j) {
        auto idxJ = grid.grid[j];
        d = distanceCalculator (idxI, idxJ);
        grid.gridDistances (i, j) = d;
        if (d < NNdist) {
          grid.gridNearestNeighbours[i] = j;
          NNdist = d;
        }
      }
    }
  }
  using dynamicMatrices::matMul;
  using dynamicMatrices::Transpose;
  void pammClustering::CalculateCovarianceMatrix (gridInfo &grid) const {
    using dynamicMatrices::matMul;
    using dynamicMatrices::Transpose;
    // constexpr double wnorm = 1.0;
    // assuming all the weight==1
    // CALL covariance(D,period,ngrid,normwj,wi,y,Q)
    //     covariance(D,period,N    ,wnorm,w,x,Q)
    std::vector<double> means (dim);
    Matrix deltafromMeans (grid.grid.size (), dim);
    // dynamicMatrix<double>deltafromMeansWeighted (grid.grid.size (),dim);
    for (auto D = 0; D < dim; ++D) {
      means[D] = 0.0;
      for (const auto gridIndex : grid.grid) {
        means[D] += data[gridIndex][D];
      }
      means[D] /= static_cast<double> (grid.grid.size ());
      for (const auto gridIndex : grid.grid) {
        deltafromMeans[gridIndex][D] = data[gridIndex][D] - means[D];
        // deltafromMeansWeighted[gridIndex][D]=deltafromMeans[gridIndex][D]*w[gridIndex];
      }
    }
    auto deltafromMeansT = Transpose (deltafromMeans /*Weighted*/);
    Matrix covariance = matMul (deltafromMeans, deltafromMeansT);
    // covariance /= (1.0 -) // Q = Q / (1.0d0-SUM((w/wnorm)**2.0d0))
  }

} // namespace libpamm

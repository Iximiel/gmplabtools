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

  size_t pammClustering::gridInfo::size () const { return grid.Rows (); }
  pammClustering::gridInfo
  pammClustering::createGrid (size_t firstPoint) const {
    gridInfo grid (gridDim, dim);
    std::vector<double> Dmins (nsamples, std::numeric_limits<double>::max ());
    std::vector<size_t> closestGridIndex (nsamples, 0);
    std::vector<size_t> gridIndexes (gridDim, 0);
    auto copyPoint = [this] (double *gridPoint, double *dataPoint) {
      std::copy (gridPoint, gridPoint + dim, dataPoint);
    };
    size_t jmax = 0;
    gridIndexes[0] = firstPoint;
    copyPoint (grid.grid[0], data[firstPoint]);
    Dmins[firstPoint] = 0.0;
    closestGridIndex[firstPoint] = 0;
    double dij;
    {
      double dMax, dNeighMin;
      for (auto i = 0U; i < gridDim - 1; ++i) {
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
        copyPoint (grid.grid[i + 1], data[jmax]);
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
      Matrix covariance = CalculateCovarianceMatrix (grid, totalWeight);

      double weightAccumulation = double (nsamples);
      //~MAYBE: Gabriel Graphs //gs is gabriel clusterign flag in pamm #473
      Matrix covariance = CalculateCovarianceMatrix (grid, totalWeight);
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
          f << ((j == 0) ? "" : " ") << grid.grid[i][j];
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
    data = Matrix (nsamples, dim);

    for (auto i = 0; i < nsamples; ++i) {
      for (auto j = 0; j < dim; ++j) {
        f >> data[i][j];
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

  void pammClustering::GenerateGridDistanceMatrix (gridInfo &grid) const {
    double d;
    for (auto i = 0; i < grid.grid.Rows (); ++i) {
      auto NNdist = std::numeric_limits<double>::max ();
      for (auto j = i + 1; j < grid.grid.Rows (); ++j) {
        d = SOAPDistanceNormalized (dim, grid.grid[i], grid.grid[j]);
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
      for (auto i = 0; i < grid.grid.Rows (); ++i) {
        auto NNdist = std::numeric_limits<double>::max ();
        for (auto j = i + 1; j < grid.grid.Rows (); ++j) {
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
    gridInfo &grid, const double totalWeight) const {
    using dynamicMatrices::matMul;
    using dynamicMatrices::Transpose;
    // CALL covariance(D,period,ngrid,normwj,wi,y,Q)
    //     covariance(D,period,N    ,wnorm,w,x,Q)
    std::vector<double> means (dim);
    Matrix deltafromMeans (grid.grid.Rows (), dim);
    Matrix deltafromMeansWeighted (grid.grid.Rows (), dim);
    for (auto D = 0; D < dim; ++D) {
      means[D] = 0.0;
      // weighted mean:
      for (auto gID = 0; gID < grid.grid.Rows (); ++gID) {
        means[D] += grid.grid[gID][D] * grid.VoronoiWeights[gID];
      }
      means[D] /= totalWeight;
      for (auto gID = 0; gID < grid.grid.Rows (); ++gID) {
        deltafromMeans[gID][D] = grid.grid[gID][D] - means[D];
        deltafromMeansWeighted[gID][D] =
          deltafromMeans[gID][D] * grid.VoronoiWeights[gID] / totalWeight;
      }
    }
    double wSumSquared = std::accumulate (
      grid.VoronoiWeights.begin (), grid.VoronoiWeights.end (), 0.0,
      [=] (double x, double y) {
        y /= totalWeight;
        return x + y * y;
      });

    auto deltafromMeansWT = Transpose (deltafromMeansWeighted);
    Matrix covariance = matMul (deltafromMeansWT, deltafromMeans);
    /*
    std::cerr << covariance.Rows () << " " << covariance.Columns () << " "
              << wSumSquared << std::endl;
    */
    covariance *= (1.0 - wSumSquared);
    return covariance;
  }

  void pammClustering::bandwidthEstimation (
    const gridInfo &grid, const Matrix &covariance, const double totalWeight) {
    std::vector<double> localWeight (grid.size ());

    double delta = totalWeight / static_cast<double> (nsamples);
    tune = 0.0;
    for (auto D = 0; D < dim; ++D) {
      tune += covariance[D][D];
    }
    std::vector<double> sigmaSQ (grid.size (), tune);
    bool useFractionOfPoints = true;
    for (auto GI = 0; GI < grid.size (); ++GI) {
      double localWeightSum = estimateGaussianLocalization (
        grid, grid.grid[GI], sigmaSQ[GI], localWeight.data ());
      // two strategies:
      if (useFractionOfPoints) {
        fractionOfPointsLocalization (
          grid, GI, delta, localWeightSum, sigmaSQ[GI], localWeight.data ());
      } else {
        fractionOfSpreadLocalization (grid);
      }
    }
  }
  /*
      Matrix matrixOfDistances (const Matrix &points, const double *point) {
        Matrix deltas (points.Rows (), points.Columns ());
        for (auto R = 0; R < points.Rows (); ++R) {
          for (auto D = 0; D < points.Columns (); ++D) {
            deltas (R, D) = points (R, D) - point[D];
          }
        }
        return deltas;
      }*/
  double pammClustering::estimateGaussianLocalization (
    const gridInfo &grid,
    const double *point,
    double sigmaSQ,
    double *outweights) const {
    // localization(D,period,ngrid,sigma2(i),y,wi,y(:,i),wlocal,flocal(i))
    // SUBROUTINE localization(D,period,N,s2,x,w,y,wl,num)
    double delta, dSum;
    for (auto gI = 0; gI < grid.size (); ++gI) {
      dSum = 0;
      for (auto D = 0; D < grid.grid.Columns (); ++D) {
        delta = grid.grid[gI][D] - point[D];
        dSum += delta * delta;
      }
      // estimate weights for localization as product from
      // spherical gaussian weights and weights in voronoi
      outweights[gI] = exp (-0.5 / sigmaSQ * dSum) * grid.VoronoiWeights[gI];
    }
    double localWeightSum =
      std::accumulate (outweights, outweights + grid.size (), 0.0);
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
    while (weight < lim) {
      sigmaSQ += tune;
      weight = estimateGaussianLocalization (
        grid, grid.grid[gID], sigmaSQ, localWeights);
    }
  }
  void pammClustering::fractionOfSpreadLocalization (const gridInfo &) {}
} // namespace libpamm

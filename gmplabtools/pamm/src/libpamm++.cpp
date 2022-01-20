#include "libpamm++.hpp"
//#include "gcem.hpp"
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
  constexpr double TWOPI = 2.0 * M_PI;
  // constexpr double log2PI = gcem::log (M_2_PI);
  const double log2PI = std::log (TWOPI);
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

  /************************************************************************/
  pammClustering::pammClustering () {}

  pammClustering::~pammClustering () {
    /*
    if (nsamples > 0) {
      delete[] data[0];
    }
    delete[] data;
    */
  }

  gridInfo
  pammClustering::createGrid (size_t firstPoint) const {
    gridInfo grid (gridDim, dim);
    std::vector<double> Dmins (nsamples, std::numeric_limits<double>::max ());
    std::vector<size_t> closestGridIndex (nsamples, 0);
    /*
    auto copyPoint = [this] (auto &row, const double *dataPoint) {
      std::copy (dataPoint, dataPoint + dim, row.begin ());
    };*/
    size_t jmax = 0;
    grid.gridIndexes[0] = firstPoint;
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
        size_t gridIndex = grid.gridIndexes[i];
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
        grid.gridIndexes[i + 1] = jmax;
        Dmins[jmax] = 0.0;
        closestGridIndex[jmax] = i + 1;
      }
    }
    // completes the voronoi attribuition for the last point in the grid
    {
      auto gridIndex = grid.gridIndexes[gridDim - 1];
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
      const double kdecut2 = 9.0 * [=] () {
        double t = sqrt (static_cast<double> (dim)) + 1.0;
        return t * t;
      }();
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

      // TODO: warning on grid dimension
      std::vector<double> normkernel;
      Eigen::VectorXd localDimensionality;
      std::tie (normkernel, localDimensionality) =
        bandwidthEstimation (grid, covariance, weightAccumulation);

      auto gridPointProbabilities =
        KernelDensityEstimation (grid, normkernel, totalWeight, kdecut2);
      auto errors = StatisticalErrorFromKDE (
        grid, normkernel, gridPointProbabilities, localDimensionality,
        totalWeight, kdecut2);
      //#871
      // determine the clusters with quickShift
      auto clusters = quickShift (grid, gridPointProbabilities);
      auto mergedClusters =
        clusterMerger (thrpcl, grid, clusters, errors, gridPointProbabilities);
      gridOutput (
        grid, mergedClusters, errors, gridPointProbabilities,
        localDimensionality);

      // completing the work:
      // TODO: Gaussian for each cluster and covariance
      classification (grid, mergedClusters);
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
        d = SOAPDistanceSquaredNormalized (
          dim, grid.grid.row (i).data (), grid.grid.row (j).data ());
        grid.gridDistancesSquared (i, j) = d;
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

  std::pair<std::vector<double>, Eigen::VectorXd>
  pammClustering::bandwidthEstimation (
    const gridInfo &grid, const Matrix &covariance, const double totalWeight) {
    std::vector<double> localWeights (grid.size ());
    std::vector<double> logDetHi (grid.size ());
    QuickShiftCutSQ.resize (grid.size ());
    std::vector<double> normkernel (grid.size ());
    Eigen::VectorXd LocalDimensionality (grid.size ());
    HiInvStore.resize (grid.size ());
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
      auto inverseLocalCovariance = localCovariance.inverse ();

      // stimate bandwidth from normal (Scott's) reference rule
      auto Hi = pow (
                  4.0 / (LocalDimensionality[GI] + 2.0),
                  2.0 / (LocalDimensionality[GI] + 4.0)) *
                pow (nlocal, -2.0 / (LocalDimensionality[GI] + 4.0)) *
                inverseLocalCovariance;
      HiInvStore[GI] = Hi.inverse ();
      // estimate logarithmic determinant of local BW H's
      logDetHi[GI] = std::log (Hi.determinant ());
      // estimate the logarithmic normalization constants
      normkernel[GI] = static_cast<double> (dim) * log2PI + logDetHi[GI];
      // adaptive QS cutoff from local covariance
      QuickShiftCutSQ[GI] = localCovariance.trace ();
      QuickShiftCutSQ[GI] *= QuickShiftCutSQ[GI];
    }
    return {normkernel, LocalDimensionality};
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
      dSum = grid.gridDistancesSquared (gridPoint, gI);
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
    double mindist =
      grid.gridDistancesSquared (gID, grid.gridNearestNeighbours[gID]);
    if (sigmaSQ < mindist) {
      sigmaSQ = mindist;
      std::cerr << " Warning: localization smaller than Voronoi diameter, "
                   "increase grid size (meanwhile adjusted localization)!"
                << std::endl;
      weight = estimateGaussianLocalization (grid, gID, sigmaSQ, localWeights);
    }
  }

  Eigen::VectorXd pammClustering::KernelDensityEstimation (
    const gridInfo &grid,
    const std::vector<double> &normkernel,
    const double weightNorm,
    const double kdecut2) {
    // probabilities at grid points
    Eigen::VectorXd prob = Eigen::VectorXd ::Constant (
      grid.size (), -std::numeric_limits<double>::max ());
    for (size_t GI = 0; GI < grid.size (); ++GI) {
      for (size_t GJ = 0; GJ < grid.size (); ++GJ) {
        // mahalanobis distance is more or less the number of standard
        // deviations from the center of a gaussian distribution, however in the
        // SOAP case we simply divide the distance by the standard deviation: as
        // now I do not know how to calculate the equivalent of the mahalanobis
        // for the SOAP distance
        double mahalanobisDistance = calculateMahalanobisDistanceSquared (
          grid.grid.row (GI), grid.grid.row (GJ), HiInvStore[GI]);
        if (mahalanobisDistance > kdecut2) {
          // assume distribution in far away grid point is narrow and store sum
          // of all contributions in grid point
          // exponent of the gaussian
          // natural logarithm of kernel
          double lnK = -0.5 * (normkernel[GJ] + mahalanobisDistance) +
                       log (grid.VoronoiWeights[GJ]);
          if (prob[GI] > lnK) {
            prob[GI] += log (1.0 + exp (lnK - prob[GI]));
          } else {
            prob[GI] = lnK + log (1.0 + exp (prob[GI] - lnK));
          }
        } else {
          // cycle just inside the polyhedra using the neighbour list
          for (const auto DK : grid.samplesIndexes[GJ]) {
            // this is the self correction
            if (DK == grid.gridIndexes[GI]) {
              continue;
            }
            // exponent of the gaussian
            mahalanobisDistance = calculateMahalanobisDistanceSquared (
              grid.grid.row (GI), grid.grid.row (GJ), HiInvStore[GJ]);

            // weighted natural logarithm of kernel
            double lnK = -0.5 * (normkernel[GJ] + mahalanobisDistance) +
                         log (dataWeights[DK]);
            if (prob[GI] > lnK) {
              prob[GI] += log (1.0 + exp (lnK - prob[GI]));
            } else {
              prob[GI] = lnK + log (1.0 + exp (prob[GI] - lnK));
            }
          }
        } // if/else(mahalanobisDistance > kdecut2)
      }   // GJ
      prob[GI] -= weightNorm;
    } // GI
    /*
        for (size_t GI = 0; GI < grid.size (); ++GI) {
          prob[GI] -= weightNorm;
        }
        */
    return prob;
  }

  gridErrorProbabilities
  pammClustering::StatisticalErrorFromKDE (
    const gridInfo &grid,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob,
    const Eigen::VectorXd &localDimensionality,
    const double weightNorm,
    const double kdecut2) {
    // errors
    gridErrorProbabilities errors (grid.size ());
    if (bootStraps > 0) {
      // probabilities for each bootstrap step
      Matrix probboot (bootStraps, grid.size ());
      // bootstrapping:
      for (size_t boot = 0; boot < bootStraps; ++boot) {
        // rather than selecting nsel random points, we select a random number
        // of points from each voronoi. this makes it possible to apply some
        // simplifications and avoid computing distances from far-away voronoi
        size_t nbstot = 0;
        for (size_t GJ = 0; GJ < grid.size (); ++GJ) {
          // here we select points and assign them to grid points (i.e. this is
          // an "inside out" version of the KDE code) we want to loop over grid
          // points and know how many points we should pick from a bootstrapping
          // sample. this is given by a binomial distribution -- the total
          // number of samples will not be *exactly* nsamples, but will be close
          // enough
          std::binomial_distribution<size_t> random_binomial (
            nsamples, static_cast<double> (grid.samplesIndexes[GJ].size ()) /
                        static_cast<double> (nsamples));
          size_t nbssample = random_binomial (randomEngine);
          if (nbssample == 0) {
            continue;
          }

          // calculate "scaled" weight for contribution from far away voronoi we
          // take into account the fact that we might have selected a number of
          // samples different from ni(j)
          double dummd2 = log (
                            static_cast<double> (nbssample) /
                            grid.samplesIndexes[GJ].size ()) *
                          log (grid.VoronoiWeights[GJ]);

          nbstot += nbssample;
          for (size_t GI = 0; GI < grid.size (); ++GI) {
            // this is the distance between the grid point from which we're
            // sampling (j) and the one on which we're accumulating the KDE (i)
            double dummd1 = calculateMahalanobisDistanceSquared (
              grid.grid.row (GI), grid.grid.row (GJ), HiInvStore[GJ]);
            if (dummd1 < kdecut2) {
              // if the two cells are far apart, we just compute an "average
              // contribution" from the far away Voronoi
              double lnK = -0.5 * (normkernel[GJ] + dummd1) + dummd2;
              if (probboot (boot, GI) > lnK) {
                probboot (boot, GI) +=
                  log (1.0 + exp (lnK - probboot (boot, GI)));
              } else {
                probboot (boot, GI) =
                  lnK + log (1.0 + exp (probboot (boot, GI) - lnK));
              }
            } else {
              // do the actual bootstrapping selection for this Voronoi
              std::uniform_int_distribution<size_t> randomIndex (
                0, grid.samplesIndexes[GJ].size () - 1);
              for (size_t k = 0; k < nbssample; ++k) {
                size_t rndidx =
                  grid.samplesIndexes[GJ][randomIndex (randomEngine)];

                if (rndidx == grid.gridIndexes[GI]) {
                  continue;
                }
                double dummd1 = calculateMahalanobisDistanceSquared (
                  grid.grid.row (GI), data.row (rndidx), HiInvStore[GJ]);
                double lnK =
                  -0.5 * (normkernel[GJ] + dummd1) + log (dataWeights[rndidx]);
                if (probboot (boot, GI) > lnK) {
                  probboot (boot, GI) =
                    probboot (boot, GI) +
                    log (1.0 + exp (lnK - probboot (boot, GI)));
                } else {
                  probboot (boot, GI) =
                    lnK + log (1.0 + exp (probboot (boot, GI) - lnK));
                }
              }
            }
          } // GI
        }   // GJ
        // normalizes the probability estimate, keeping into account that we
        // might have used a different number of sample points than nsamples
        {
          double subtract =
            log (weightNorm) +
            log (static_cast<double> (nbstot) / static_cast<double> (nsamples));
          std::transform (
            probboot.row (boot).begin (), probboot.row (boot).end (),
            probboot.row (boot).begin (),
            [subtract] (double x) { return x - subtract; });
        }
        //#774: quick-shift
        auto clusterIndexes = quickShift (grid, probboot.row (boot));
        //#810: output for bootstrap

      } // bootstrap cycle
      // computes the bootstrap error from the statistics of the nbootstrap KDE
      // runs
      //#828
      for (size_t GI = 0; GI < grid.size (); ++GI) {
        for (size_t boot = 0; boot < bootStraps; ++boot) {
          if (prob[GI] > probboot (boot, GI)) {
            errors.absolute[GI] += exp (
              2.0 * (probboot (boot, GI) +
                     log (1.0 - exp (prob[GI] - probboot (boot, GI)))));
          } else {
            errors.absolute[GI] += exp (
              2.0 *
              (prob[GI] + log (1.0 - exp (probboot (boot, GI) - prob[GI]))));
          }
        }
        errors.absolute[GI] =
          log (sqrt (errors.absolute[GI] / (bootStraps - 1.0)));
        errors.relative[GI] = errors.absolute[GI] - prob[GI];
      }
    } else {
      // use a binomial-distribution ansatz to estimate the error
      for (size_t GI = 0; GI < grid.size (); ++GI) {
        auto i = GI;
        double mindist =
          grid.gridDistancesSquared (GI, grid.gridNearestNeighbours[GI]);
        errors.relative[i] = log (sqrt (
          (((pow (mindist * TWOPI, -localDimensionality[i])) / exp (prob[i])) -
           1.0) /
          nsamples));

        errors.absolute[i] = errors.relative[i] + prob[i];
      }
    }
    return errors;
  }

  quickShiftOutput pammClustering::quickShift (
    const gridInfo &grid, const Eigen::VectorXd &probabilities) {
    // Vedaldi, A.; Soatto, S. In Computer Vision - ECCV 2008:10th European
    // Conference on Computer Vision, Marseille, France, October 12–18, 2008,
    // Proceedings, Part IV; Forsyth, D.; Torr, P.; Zisserman, A., Eds.;
    // Springer: Berlin, 2008; pp 705– 718.

    std::vector<size_t> roots (grid.size (), 0);

    using indexVect = Eigen::Matrix<size_t, 1, -1>;
    indexVect qspath (grid.size ());
    for (size_t GI = 0; GI < grid.size (); ++GI) {
      if (roots[GI] != 0) {
        continue;
      }
      /*IF(verbose .AND. (modulo(i,1000).EQ.0)) &
         WRITE(*,*) i,"/",ngrid
         */
      qspath = indexVect::Zero (grid.size ());
      qspath[0] = GI;
      size_t counter = 0;
      while (qspath[counter] != roots[qspath[counter]]) {
        // TODO: Gabriel graph computation
        /*
          if (doGabrielGraphs) {
             roots[qspath[counter]] =
          gs_next[ngrid,qspath[counter],probabilities,distmm,gabriel,gs)
          }else*/
        {
          roots[qspath[counter]] = QuickShift_nextPoint (
            grid.size (), qspath[counter],
            grid.gridNearestNeighbours[qspath[counter]],
            QuickShiftCutSQ[qspath[counter]], probabilities,
            grid.gridDistancesSquared);
        }
        if (roots[roots[qspath[counter]]] != 0) {
          break;
        }
        ++counter;
        qspath[counter] = roots[qspath[counter - 1]];
      }
      for (size_t i = 0; i < counter; ++i) {
        // we found a new root, and we now set this point as the root for all
        // the point that are in this qspath
        roots[qspath[i]] = roots[roots[qspath[counter]]];
      }
    }

    // get a set with the unique cluster centers
    return {std::set<size_t>{roots.begin (), roots.end ()}, roots};
  }
  size_t QuickShift_nextPoint (
    const size_t ngrid,
    const size_t idx,
    const size_t idxn,
    const double lambda,
    const Eigen::VectorXd &probnmm,
    const distanceMatrix &distmm) {
    // Args:
    //    ngrid: number of grid points
    //    idx: current point
    //    qscut: cut-off in the jump
    //    probnmm: density estimations
    //    distmm: distances matrix

    double dmin = std::numeric_limits<double>::max ();
    size_t qs_next = (probnmm (idxn) < probnmm (idx)) ? idx : idxn;
    for (size_t GJ = 0; GJ < ngrid; ++GJ) {
      if (probnmm (idx) < probnmm (GJ)) {
        if ((distmm (idx, GJ) < dmin) && (distmm (idx, GJ) < lambda)) {
          dmin = distmm (idx, GJ);
          qs_next = GJ;
        }
      }
    }
    return qs_next;
  }

  template <typename VecType>
  double accumulateLogsumexp (
    const std::vector<size_t> &indexes, const VecType &probabilities) {
    double sum = -std::numeric_limits<double>::max ();
    for (size_t i = 0; i < indexes.size (); ++i) {
      if (probabilities[i] < sum) {
        sum += log (1.0 + exp (probabilities[i] - sum));
      } else {
        sum = probabilities[i] + log (1.0 + exp (sum - probabilities[i]));
      }
    }
    return sum;
  }

  template <typename VecType>
  double accumulateLogsumexp_if (
    const std::vector<size_t> &indexes,
    const VecType &probabilities,
    size_t sum_if) {
    double sum = -std::numeric_limits<double>::max ();
    for (size_t i = 0; i < indexes.size (); ++i) {
      if (indexes[i] == sum_if) {
        if (probabilities[i] < sum) {
          sum += log (1.0 + exp (probabilities[i] - sum));
        } else {
          sum = probabilities[i] + log (1.0 + exp (sum - probabilities[i]));
        }
      }
    }
    return sum;
  }

  quickShiftOutput pammClustering::clusterMerger (
    const double mergeThreshold,
    const gridInfo &grid,
    const quickShiftOutput &qsOut,
    const gridErrorProbabilities &errors,
    const Eigen::VectorXd &prob) {
    //#907
    // get sum of the probs, the normalization factor
    double normpks = accumulateLogsumexp (qsOut.gridToClusterIdx, prob);
    std::vector<size_t> newClusterCenters (
      qsOut.clustersIndexes.begin (), qsOut.clustersIndexes.end ());
    std::vector<size_t> newgridToClusterIdx = qsOut.gridToClusterIdx;
    // check if there are outliers that should be merged to the others
    std::vector<bool> mergeornot (qsOut.clustersIndexes.size (), false);
    size_t k = 0;
    for (auto idK : qsOut.clustersIndexes) {
      // compute the relative weight of the cluster
      double mergeParameter = exp (
        accumulateLogsumexp_if (qsOut.gridToClusterIdx, prob, idK) - normpks);
      /*
      std::cerr << idK << " " << k << ": " << mergeParameter << " "
                << ((mergeParameter < mergeThreshold) ? "merge"
                                                      : "do not merge")
                << "\n";
      */
      mergeornot[k] = mergeParameter < mergeThreshold;
      ++k;
    }
    // merge the outliers
    k = 0;
    for (auto idK : qsOut.clustersIndexes) {
      if (mergeornot[k]) {
        double minDistSQ = std::numeric_limits<double>::max ();
        size_t j = 0;
        for (auto idJ : qsOut.clustersIndexes) {
          if (!mergeornot[j]) {
            double distSQ = grid.gridDistancesSquared (idK, idJ);
            if (distSQ < minDistSQ) {
              minDistSQ = distSQ;
              newClusterCenters[k] = newClusterCenters[j];
            }
          }
          ++j;
        }
        for (size_t i = 0; i < newgridToClusterIdx.size (); ++i) {
          if (newgridToClusterIdx[i] == idK) {
            newgridToClusterIdx[i] = newClusterCenters[k];
          }
        }
      }
      ++k;
    }
    std::set<size_t> clustercenters{
      newClusterCenters.begin (), newClusterCenters.end ()};
    if (std::any_of (
          mergeornot.begin (), mergeornot.end (), [] (bool x) { return x; })) {
      std::cout << qsOut.clustersIndexes.size () - clustercenters.size ()
                << " clusters where merged into other clusters\n";

      // get the real maxima in the cluster, considering the errorbar
      for (auto idK : clustercenters) {
        double maxP = 0.0;
        size_t newCentroidIndex = idK;
        // search for the index of the grid point(centroid) with
        // max prob within abs err
        for (size_t i = 0; i < newgridToClusterIdx.size (); ++i) {
          if (newgridToClusterIdx[i] == idK) {
            double tempP = exp (prob[i]) + exp (errors.absolute[i]);
            if (maxP < tempP) {
              maxP = tempP;
              newCentroidIndex = i;
            }
          }
        }
        // reassign the cluster to the new centroid
        if (idK != newCentroidIndex) {
          for (size_t i = 0; i < newgridToClusterIdx.size (); ++i) {
            if (newgridToClusterIdx[i] == idK) {
              newgridToClusterIdx[i] = newCentroidIndex;
            }
          }
        }
        clustercenters.erase (idK);
        clustercenters.insert (newCentroidIndex);
      }
    }

    return {clustercenters, newgridToClusterIdx};
  }
  void pammClustering::gridOutput (
    const gridInfo &grid,
    const quickShiftOutput &clusterInfo,
    const gridErrorProbabilities &errors,
    const Eigen::VectorXd &prob,
    const Eigen::VectorXd &localDimensionality) const {
    // TODO
    /*
    IF(verbose) write(*,*) "Writing out"
    OPEN(UNIT=11,FILE=trim(outputfile)//".grid",STATUS='REPLACE',ACTION='WRITE')
    OPEN(UNIT=13,FILE=trim(outputfile)//".dim",STATUS='REPLACE',ACTION='WRITE')
    DO i=1,ngrid
       WRITE(13,"((A1,ES15.4E4))") " ", Di(i)
       DO j=1,D
          WRITE(11,"((A1,ES15.4E4))",ADVANCE = "NO") " ", y(j,i)
       ENDDO

       CALL invmatrix(D,Hiinv(:,:,i),Hi)

       !print out grid file with additional information on probability, errors,
       localization, weights in voronoi, dim
       WRITE(11,"blabla") &
          " " , MINLOC(ABS(clustercenters-idxroot(i)),1) ,      &
          " " , prob(i) ,      &
          " " , pabserr(i),    &
          " " , prelerr(i),    &
          " " , sigma2(i),     &
          " " , flocal(i),     &
          " " , wi(i),         &
          " " , Di(i),         &
          " " , trmatrix(D,Hi)/DBLE(D)

    ENDDO

    CLOSE(UNIT=11)
    */
  }
  struct gaussian {
    /// dimensionality of the Gaussian
    size_t D;
    /// weight associated with the Gaussian cluster (not included in the
    /// normalization!)
    double weight{};
    /// logarithm of the normalization factor
    double lnorm{};
    /// determinant of the covariance matrix
    double det{};
    /// mean of the gaussian
    Eigen::VectorXd center;
    /// convariance matrix
    Matrix cov;
    /// inverse convariance matrix
    Matrix icov;
    gaussian (size_t N) : D (N), center (N), cov (N, N), icov (N, N) {}
    gaussian (const size_t N, const size_t idK/*gridID*/, const size_t nmsopt, const double normpks,const gridInfo &grid,const quickShiftOutput &clusterInfo,Matrix HiInverse,    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob)
    : D (N), center (N), cov (N, N), icov (N, N) {
  
      // optional mean shift for estimation of clustermode
      for (size_t ms = 0; ms < nmsopt; ++ms) {
        auto msmu = Eigen::VectorXd::Zero (D);
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
          CalculateLogCovarianceMatrix (
            idK, grid, clusterInfo, normkernel, prob),
          accumulatedLogSum);
        weight = exp (accumulatedLogSum - normpks);
      }
    }
    void prepare(){
      det=cov.determinant();
      icov=cov.inverse();
      lnorm = log(1.0/sqrt(pow(TWOPI,D)*det));

}
};
  void pammClustering::classification (
    const gridInfo &grid,
    const quickShiftOutput &clusterInfo,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob) const {
    //#977
    // vonMises distribution with type -> periodic
    // gaussians distribution with type -> non periodic
    /*
    ! Structure that contains the parameters needed to define and
    ! estimate a Von Mises distribution
    TYPE vm_type
       INTEGER D ! dimensionality of the Gaussian
       DOUBLE PRECISION weight ! weight associated with the Gaussian cluster
       (not included in the normalization!)
       DOUBLE PRECISION lnorm ! logarithm of the normalization factor
       DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: period
       DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: mean
       DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: cov
       ! convariance matrix
       DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: icov
       ! inverse convariance matrix
    END TYPE
    */
    const size_t nClusters = clusterInfo.clustersIndexes.size ();
    std::vector<gaussian> clusters;clusters.reserve (nClusters);
    
    double normpks = accumulateLogsumexp (qsOut.gridToClusterIdx, prob);
    for (const size_t idK : clusterInfo.clusterIndexes) {
      
      
      
clusters.emplace_back(gaussian(dim,  idK, nmsopt, normpks,grid,clusterInfo,HiInvStore[idK],normkernel,
    prob));

    }
    // output
    /*#1073
      ! write the Gaussians
      ! write a 2-lines header
      WRITE(comment,*) "# PAMMv2 clusters analysis. NSamples: ", nsamples, "
      NGrid: ",ngrid, " QSLambda: ", qs, ACHAR(10), "#
      Dimensionality/NClusters//Pk/Mean/Covariance"

      OPEN(UNIT=12,FILE=trim(outputfile)//".pamm",STATUS='REPLACE',ACTION='WRITE')

      CALL writeclusters(12, comment, nk, clusters)
      CLOSE(UNIT=12)
      ! maybe I should deallocate better..
      DEALLOCATE(clusters)
      */
  }

  Matrix CalculateLogCovarianceMatrix (
    const size_t clusterIndex,
    const gridInfo &grid,
    const quickShiftOutput &clusterInfo,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob) const {
    double norm =
      accumulateLogsumexp_if (clusterInfo.gridToClusterIdx, prob, clusterIndex);
    std::vector<double> weights (grid.size (), 0.0);
    for (size_t GI; GI < grid.size (); ++GI) {
      if (clusterIndex == clusterInfo.gridToClusterIdx[GI]) {
        weights[GI] = exp (prob[GI] - norm);
      }
    }

    return CalculateCovarianceMatrix (grid, weights, 1.0);
  }
} // namespace libpamm

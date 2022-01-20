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
  const double log2PI = std::log (TWOPI);
  /************************************************************************/
  pammClustering::pammClustering () {}

  pammClustering::~pammClustering () {
    /*
    if (nsamples_ > 0) {
      delete[] data[0];
    }
    delete[] data;
    */
  }

  gridInfo pammClustering::createGrid (size_t firstPoint) const {
    gridInfo grid (gridDim_, dim_);
    std::vector<double> Dmins (nsamples_, std::numeric_limits<double>::max ());
    std::vector<size_t> closestGridIndex (nsamples_, 0);
    /*
    auto copyPoint = [this] (auto &row, const double *dataPoint) {
      std::copy (dataPoint, dataPoint + dim_, row.begin ());
    };*/
    size_t jmax = 0;
    grid.gridIndexes[0] = firstPoint;
    /*
    std::copy (
      data[firstPoint], data[firstPoint] + dim_, grid.grid.row (0).begin ());
*/
    grid.grid.row (0) = data.row (firstPoint);
    Dmins[firstPoint] = 0.0;
    closestGridIndex[firstPoint] = 0;
    double dij;
    {
      double dMax, dNeighMin;
      for (size_t i = 0; i < gridDim_ - 1; ++i) {
        grid.samplesIndexes[i].reserve (nsamples_ / gridDim_);
        dMax = 0.0;
        dNeighMin = std::numeric_limits<double>::max ();
        size_t gridIndex = grid.gridIndexes[i];
        // find the farthest point from gridIndex
        for (auto j = 0U; j < nsamples_; ++j) {
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
      auto gridIndex = grid.gridIndexes[gridDim_ - 1];
      grid.samplesIndexes[gridDim_ - 1].reserve (nsamples_ / gridDim_);
      auto dNeighMin = std::numeric_limits<double>::max ();
      for (auto j = 0U; j < nsamples_; ++j) {
        if (gridIndex == j) {
          continue;
        }
        dij = distanceCalculator (gridIndex, j);
        if (dij < Dmins[j]) {
          Dmins[j] = dij;
          closestGridIndex[j] = gridDim_ - 1;
        }
        if (dij < dNeighMin && (0.0 < dij)) {
          dNeighMin = dij;
          grid.voronoiAssociationIndex[gridDim_ - 1] = j;
        }
      }
    }
    // Assign neighbor list pointer of voronois
    // Number of points in each voronoi polyhedra

    for (auto j = 0U; j < nsamples_; ++j) {
      grid.VoronoiWeights[closestGridIndex[j]] += dataWeights_[j];
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
        double t = sqrt (static_cast<double> (dim_)) + 1.0;
        return t * t;
      }();
      // auto distances = CalculateDistanceMatrixSOAP(data, nsamples_, dim_);
      size_t randomGeneratedFirstPoint = 1;
      double totalWeight =
        std::accumulate (dataWeights_.begin (), dataWeights_.end (), 0.0);
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

      double weightAccumulation = double (nsamples_);
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
        clusterMerger (thrpcl_, grid, clusters, errors, gridPointProbabilities);
      gridOutput (
        grid, mergedClusters, errors, gridPointProbabilities,
        localDimensionality);

      // completing the work:
      // TODO: Gaussian for each cluster and covariance
      auto gaussianClusters = classification (
        grid, mergedClusters, normkernel, gridPointProbabilities);

      printClusters (gaussianClusters);
      // Output?
      // file to save: bs dim grid pamm

      std::ofstream f ("test_grid.soap");
      std::ofstream g ("test_grid.dat");
      for (auto i = 0; i < gridDim_; ++i) {
        for (auto j = 0; j < dim_; ++j) {
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
    dim_ = 324;
    nsamples_ = 30900;
    // nsamples_ = 30;
    data = Matrix (nsamples_, dim_);

    for (auto i = 0; i < nsamples_; ++i) {
      for (auto j = 0; j < dim_; ++j) {
        f >> data (i, j);
      }
    }
    dataWeights_ = std::vector<double> (nsamples_, 1.0);
    gridDim_ = 1000;
    this->initialized_ = true;
    this->dataSetNormalized_ = false;
  }

  void pammClustering::doNormalizeDataset () {
    if (!dataSetNormalized_) {
      dataSetNormalized_ = true;
      for (auto i = 0; i < nsamples_; ++i) {
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
    return SOAPDistanceNormalized (dim_, pointI, pointJ);
  }
  void pammClustering::GenerateGridDistanceMatrix (gridInfo &grid) const {
    double d;
    for (auto i = 0; i < grid.size (); ++i) {
      auto NNdist = std::numeric_limits<double>::max ();
      for (auto j = i + 1; j < grid.size (); ++j) {
        d = SOAPDistanceSquaredNormalized (
          dim_, grid.grid.row (i).data (), grid.grid.row (j).data ());
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
      for (auto i = 0; i < grid.size(); ++i) {
        auto NNdist = std::numeric_limits<double>::max ();
        for (auto j = i + 1; j < grid.size(); ++j) {
          d = SOAPDistanceNormalized (dim_, grid.grid[i], grid.grid[j]);
          grid.gridDistances (i, j) = d;
          if (d < NNdist) {
            grid.gridNearestNeighbours[i] = j;
            NNdist = d;
          }
        }
      }
    }
  */
  std::pair<std::vector<double>, Eigen::VectorXd>
  pammClustering::bandwidthEstimation (
    const gridInfo &grid, const Matrix &covariance, const double totalWeight) {
    std::vector<double> localWeights (grid.size ());
    std::vector<double> logDetHi (grid.size ());
    QuickShiftCutSQ_.resize (grid.size ());
    std::vector<double> normkernel (grid.size ());
    Eigen::VectorXd LocalDimensionality (grid.size ());
    HiInvStore_.resize (grid.size ());
    double delta = totalWeight / static_cast<double> (nsamples_);
    tune_ = 0.0;
    for (auto D = 0; D < dim_; ++D) {
      tune_ += covariance (D, D);
    }
    std::vector<double> sigmaSQ (grid.size (), tune_);
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
      double nlocal = localWeightSum * nsamples_;
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
      HiInvStore_[GI] = Hi.inverse ();
      // estimate logarithmic determinant of local BW H's
      logDetHi[GI] = std::log (Hi.determinant ());
      // estimate the logarithmic normalization constants
      normkernel[GI] = static_cast<double> (dim_) * log2PI + logDetHi[GI];
      // adaptive QS cutoff from local covariance
      QuickShiftCutSQ_[GI] = localCovariance.trace ();
      QuickShiftCutSQ_[GI] *= QuickShiftCutSQ_[GI];
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
    double lim = fractionOfPointsVal_;
    if (fractionOfPointsVal_ < grid.VoronoiWeights[gID]) {
      lim = weight + delta;
      std::cerr << " Warning: localization smaller than voronoi, increase grid "
                   "size (meanwhile adjusted localization)!"
                << std::endl;
    }
    // quick approach by steps of  "tune_" dimension
    while (weight < lim) {
      sigmaSQ += tune_;
      weight = estimateGaussianLocalization (grid, gID, sigmaSQ, localWeights);
    }
    // using bisection to fine tune_ the localizatiom
    double bisectionParameter = 2.0;
    while ((weight - lim < delta) && (weight - lim > -delta)) { //(true){}
      sigmaSQ += tune_ / (bisectionParameter) * (weight < lim) ? 1.0 : -1.0;
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
          grid.grid.row (GI), grid.grid.row (GJ), HiInvStore_[GI]);
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
              grid.grid.row (GI), grid.grid.row (GJ), HiInvStore_[GJ]);

            // weighted natural logarithm of kernel
            double lnK = -0.5 * (normkernel[GJ] + mahalanobisDistance) +
                         log (dataWeights_[DK]);
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

  gridErrorProbabilities pammClustering::StatisticalErrorFromKDE (
    const gridInfo &grid,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob,
    const Eigen::VectorXd &localDimensionality,
    const double weightNorm,
    const double kdecut2) {
    // errors
    gridErrorProbabilities errors (grid.size ());
    if (bootStraps_ > 0) {
      // probabilities for each bootstrap step
      Matrix probboot (bootStraps_, grid.size ());
      // bootstrapping:
      for (size_t boot = 0; boot < bootStraps_; ++boot) {
        // rather than selecting nsel random points, we select a random number
        // of points from each voronoi. this makes it possible to apply some
        // simplifications and avoid computing distances from far-away voronoi
        size_t nbstot = 0;
        for (size_t GJ = 0; GJ < grid.size (); ++GJ) {
          // here we select points and assign them to grid points (i.e. this is
          // an "inside out" version of the KDE code) we want to loop over grid
          // points and know how many points we should pick from a bootstrapping
          // sample. this is given by a binomial distribution -- the total
          // number of samples will not be *exactly* nsamples_, but will be
          // close enough
          std::binomial_distribution<size_t> random_binomial (
            nsamples_, static_cast<double> (grid.samplesIndexes[GJ].size ()) /
                         static_cast<double> (nsamples_));
          size_t nbssample = random_binomial (randomEngine_);
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
              grid.grid.row (GI), grid.grid.row (GJ), HiInvStore_[GJ]);
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
                  grid.samplesIndexes[GJ][randomIndex (randomEngine_)];

                if (rndidx == grid.gridIndexes[GI]) {
                  continue;
                }
                double dummd1 = calculateMahalanobisDistanceSquared (
                  grid.grid.row (GI), data.row (rndidx), HiInvStore_[GJ]);
                double lnK =
                  -0.5 * (normkernel[GJ] + dummd1) + log (dataWeights_[rndidx]);
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
        // might have used a different number of sample points than nsamples_
        {
          double subtract =
            log (weightNorm) +
            log (
              static_cast<double> (nbstot) / static_cast<double> (nsamples_));
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
        for (size_t boot = 0; boot < bootStraps_; ++boot) {
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
          log (sqrt (errors.absolute[GI] / (bootStraps_ - 1.0)));
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
          nsamples_));

        errors.absolute[i] = errors.relative[i] + prob[i];
      }
    }
    return errors;
  }

  quickShiftOutput pammClustering::quickShift (
    const gridInfo &grid, const Eigen::VectorXd &probabilities) const {
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
            QuickShiftCutSQ_[qspath[counter]], probabilities,
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

  std::vector<gaussian> pammClustering::classification (
    const gridInfo &grid,
    const quickShiftOutput &clusterInfo,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob) const {
    //#977
    // vonMises distribution with type -> periodic
    // gaussians distribution with type -> non periodic
    const size_t nClusters = clusterInfo.clustersIndexes.size ();
    std::vector<gaussian> clusters;
    clusters.reserve (nClusters);

    double normpks = accumulateLogsumexp (clusterInfo.gridToClusterIdx, prob);
    for (const size_t idK : clusterInfo.clustersIndexes) {
      clusters.emplace_back (gaussian (
        dim_, idK, nmsopt_, normpks, grid, clusterInfo, HiInvStore_[idK],
        normkernel, prob));
    }
    return clusters;
  }

  void pammClustering::printClusters (std::vector<gaussian> clusters) const {
    std::ofstream fout (outputFilesNames_ + ".pamm");
    fout << "# PAMM++ clusters analysis. NSamples: " << nsamples_
         << " NGrid: " << gridDim_ << " QSLambda: " << quickShiftLambda_
         << '\n';
    fout << "# Dimensionality/NClusters//Pk/Mean/Covariance";
  }

} // namespace libpamm

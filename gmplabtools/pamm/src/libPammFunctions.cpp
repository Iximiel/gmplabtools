#include "libpamm++.hpp"
#include <Eigen/Eigenvalues>
#include <iostream>

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

  Matrix CalculateLogCovarianceMatrix (
    const size_t clusterIndex,
    const gridInfo &grid,
    const quickShiftOutput &clusterInfo,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob) {
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

  Matrix CalculateCovarianceMatrix (
    const gridInfo &grid,
    const std::vector<double> &weights,
    const double totalWeight) {
    const size_t dim = grid.dimensionality ();
    std::vector<double> means (dim);
    Matrix deltafromMeans (grid.size (), dim);
    Matrix deltafromMeansWeighted (grid.size (), dim);
    for (auto D = 0; D < dim; ++D) {
      means[D] = 0.0;
      // weighted mean:
      for (auto gID = 0; gID < grid.size (); ++gID) {
        means[D] += grid.grid (gID, D) * weights[gID];
      }
      means[D] /= totalWeight;
      for (auto gID = 0; gID < grid.size (); ++gID) {
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

  quickShiftOutput clusterMerger (
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
  template <typename VecType>
  inline double accumulateLogsumexpTEMPL (
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

  double accumulateLogsumexp (
    const std::vector<size_t> &indexes,
    const std::vector<size_t> &probabilities) {
    return accumulateLogsumexpTEMPL (indexes, probabilities);
  }

  double accumulateLogsumexp (
    const std::vector<size_t> &indexes, const Eigen::VectorXd &probabilities) {
    return accumulateLogsumexpTEMPL (indexes, probabilities);
  }

  template <typename VecType>
  inline double accumulateLogsumexp_ifTEMPL (
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

  double accumulateLogsumexp_if (
    const std::vector<size_t> &indexes,
    const std::vector<size_t> &probabilities,
    size_t sum_if) {
    return accumulateLogsumexp_ifTEMPL (indexes, probabilities, sum_if);
  }

  double accumulateLogsumexp_if (
    const std::vector<size_t> &indexes,
    const Eigen::VectorXd &probabilities,
    size_t sum_if) {
    return accumulateLogsumexp_ifTEMPL (indexes, probabilities, sum_if);
  }

} // namespace libpamm
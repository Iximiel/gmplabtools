#include "libpamm++.hpp"
#include <catch2/catch.hpp>
#include <iostream>

void print (
  const libpamm::pammClustering::quickShiftOutput &t, std::ostream &stream) {
  for (auto i : t.clustersIndexes) {
    stream << i << " ";
  }
  stream << "\n";
  for (auto i = 0; i < t.gridToClusterIdx.size (); ++i) {
    stream << "{" << i << ", " << t.gridToClusterIdx[i] << "} ";
  }
  stream << "\n";
}
TEST_CASE (
  "Testing the cluster collapse algorithm with fake data",
  "[ClusterCollapse]") {
  libpamm::pammClustering::quickShiftOutput t{
    {1, 3, 5, 7}, {1, 1, 5, 3, 1, 5, 7, 7}};
  // print (t, std::cerr);
  Eigen::VectorXd probs (8);
  probs << 0.1, .01, .05, .6, .1, .1, .2, .3;
  libpamm::pammClustering::gridErrorProbabilities errors (8);
  errors.absolute = {0.1, 0.3, 0.2, 0.01, 0.1, 0.5, 0.6, 0.1};
  errors.relative = {0.1, 0.3, 0.2, 0.01, 0.1, 0.5, 0.6, 0.1};
  libpamm::pammClustering::gridInfo grid (8, 2);
  grid.grid.row (0) << 0.1, 0.0;
  grid.grid.row (1) << 0.0, 0.0;
  grid.grid.row (2) << 0.0, 2.0;
  grid.grid.row (3) << 2.0, 1.0;
  grid.grid.row (4) << 0.0, 0.1;
  grid.grid.row (5) << 0.0, 2.0;
  grid.grid.row (6) << 2.0, 1.8;
  grid.grid.row (7) << 2.0, 2.0;
  for (size_t i = 0; i < grid.size (); ++i) {
    for (size_t j = i + 1; j < grid.size (); ++j) {
      auto distV = grid.grid.row (i) - grid.grid.row (j);
      grid.gridDistancesSquared (i, j) = distV.squaredNorm ();
      // std::cerr << i << ", " << j << " " << grid.gridDistancesSquared (i, j)
      // << '\n';
    }
  }
  /*should produce this:
    clustername mergeWeight
    1           0.329613
    3           0.186497
    5           0.220716
    7           0.263174
    // point 6 has a probaility higher than 7 (this should provoke some
    movements)
    // point 5 has a probaility higher than 1 (this should provoke some
    movements)
*/
  {
    auto out =
      libpamm::pammClustering::clusterMerger (0.21, grid, t, errors, probs);
    // print (out, std::cerr);
    CHECK (out.clustersIndexes.count (1) != 0);
    CHECK (out.clustersIndexes.count (3) == 0);
    CHECK (out.clustersIndexes.count (5) != 0);
    CHECK (out.clustersIndexes.count (6) != 0);
    CHECK (out.clustersIndexes.count (7) == 0); // 7->6
  }
  {
    auto out =
      libpamm::pammClustering::clusterMerger (0.23, grid, t, errors, probs);
    // print (out, std::cerr);
    CHECK (out.clustersIndexes.count (1) == 0); // 1->5
    CHECK (out.clustersIndexes.count (3) == 0);
    CHECK (out.clustersIndexes.count (5) != 0);
    CHECK (out.clustersIndexes.count (6) != 0);
    CHECK (out.clustersIndexes.count (7) == 0); // 7->6
  }

  {
    auto out =
      libpamm::pammClustering::clusterMerger (0.1, grid, t, errors, probs);
    // print (out, std::cerr);
    // no movements
    CHECK (out.clustersIndexes.count (1) != 0);
    CHECK (out.clustersIndexes.count (3) != 0);
    CHECK (out.clustersIndexes.count (5) != 0);
    CHECK (out.clustersIndexes.count (7) != 0);
  }
}
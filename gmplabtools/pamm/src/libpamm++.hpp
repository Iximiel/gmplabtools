/*
#ifdef __cplusplus
extern "C" {
#endif
double __libpamm_MOD_mahalanobis(int, double *period, double *x, double *y,
                                 double *Qinv);
#ifdef __cplusplus
}
#endif
*/
#include "dynamicMatrices/dynamicMatrices.hpp"
//#include "pammMatrix.hpp"
#include <vector>
namespace libpamm {
  void clusteringMode ();
  double SOAPDistance (size_t dim, const double *x, const double *y);
  double SOAPDistance (
    size_t dim, const double *x, const double *y, double xyNormProduct);
  double SOAPDistanceNormalized (size_t dim, const double *x, const double *y);
  using distanceMatrix = dynamicMatrices::triangularMatrix<double>;
  using Matrix = dynamicMatrices::dynamicMatrix<double>;
  // using NNMatrix = triangularMatrix<size_t>;

  class pammClustering final {
  public:
    pammClustering ();
    pammClustering (const pammClustering &) = delete;
    pammClustering (pammClustering &&) = delete;
    ~pammClustering ();

    struct gridInfo {
      /// Contains the information for the grid
      gridInfo () = delete;
      gridInfo (size_t, size_t);
      Matrix grid{0, 0};
      // std::vector<size_t> NofSamples{};// ni is .size of samplesIndexes
      std::vector<double> VoronoiWeights{};          // wi
      std::vector<size_t> voronoiAssociationIndex{}; // ineigh: closest sample
      std::vector<size_t> gridNearestNeighbours{};
      std::vector<std::vector<size_t>> samplesIndexes{};
      distanceMatrix gridDistances{0};
      size_t size () const;
    };

    void work ();
    void testLoadData ();
    void doNormalizeDataset ();
    double distanceCalculator (size_t, size_t) const;
    void GenerateGridDistanceMatrix (gridInfo &) const;
    double estimateGaussianLocalization (
      const gridInfo &,
      const double *point,
      double sigma2,
      double *outweights) const;
    Matrix
    CalculateCovarianceMatrix (gridInfo &, const double totalWeight) const;
    void bandwidthEstimation (const gridInfo &, const Matrix &, const double);
    void fractionOfPointsLocalization (
      const gridInfo &grid,
      const size_t gID,
      const double delta,
      double &weight,
      double &sigmaSQ,
      double *localWeights);
    void fractionOfSpreadLocalization (const gridInfo &);

  private:
    size_t dim{0};
    size_t nsamples{0};
    size_t gridDim{0};

    // TODO: setup
    double fractionOfPointsVal{0.1};
    double fractionOfSpreadVal{0.1};
    double tune{0.01};
    std::vector<double> dataWeights{};
    Matrix data{0, 0}; /// TODO: correct this
    gridInfo createGrid (size_t firstPoint = 0) const;
    bool initialized_{false};
    bool dataSetNormalized_{false};
  };

} // namespace libpamm

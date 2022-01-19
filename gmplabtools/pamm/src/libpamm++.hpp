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
#include <Eigen/Core>
#include <dynamicMatrices/dynamicMatrices.hpp>
#include <random>
#include <set>
#include <vector>
namespace libpamm {

  double SOAPDistance (size_t dim, const double *x, const double *y);

  double SOAPDistance (
    size_t dim, const double *x, const double *y, double xyNormProduct);
  double
  SOAPDistanceSquaredNormalized (size_t dim, const double *x, const double *y);
  double SOAPDistanceSquared (size_t dim, const double *x, const double *y);

  double SOAPDistanceSquared (
    size_t dim, const double *x, const double *y, double xyNormProduct);
  double SOAPDistanceNormalized (size_t dim, const double *x, const double *y);
  using distanceMatrix = dynamicMatrices::triangularMatrix<double>;
  using Matrix = Eigen::MatrixXd;

  // using NNMatrix = triangularMatrix<size_t>;
  size_t QuickShift_nextPoint (
    const size_t ngrid,
    const size_t idx,
    const size_t idxn,
    const double lambda,
    const Eigen::VectorXd &probnmm,
    const distanceMatrix &distmm);
  class pammClustering final {
  public:
    pammClustering ();
    pammClustering (const pammClustering &) = delete;
    pammClustering (pammClustering &&) = delete;
    ~pammClustering ();

    struct gridInfo {
      /// Contains the information for the grid
      gridInfo () = delete;
      gridInfo (size_t gridDim, size_t dataDim);
      Matrix grid{0, 0};
      std::vector<size_t> gridIndexes{}; // idxgrid
      // std::vector<size_t> NofSamples{};// ni is .size of samplesIndexes
      std::vector<double> VoronoiWeights{};          // wi
      std::vector<size_t> voronoiAssociationIndex{}; // ineigh: closest sample
      std::vector<size_t> gridNearestNeighbours{};
      std::vector<std::vector<size_t>> samplesIndexes{};
      distanceMatrix gridDistancesSquared{0};
      size_t size () const;
    };

    struct gridErrorProbabilities {
      gridErrorProbabilities () = delete;
      gridErrorProbabilities (size_t);
      std::vector<double> absolute{}; // pabserr
      std::vector<double> relative{}; // prelerr
      size_t size () const;
    };

    void work ();
    void testLoadData ();
    void doNormalizeDataset ();

    double distanceCalculator (const size_t, const size_t) const;

    double distanceCalculator (const double *, const double *) const;
    /*
    double calculateMahalanobisDistance (
      const double *, const double *, const Matrix &) const;
*/
    double calculateMahalanobisDistance (
      const Eigen::VectorXd &A,
      const Eigen::VectorXd &B,
      const Matrix &invCov) const;
    void GenerateGridDistanceMatrix (gridInfo &) const;
    double estimateGaussianLocalization (
      const gridInfo &,
      const double *point,
      const double sigma2,
      double *outweights) const;
    double estimateGaussianLocalization (
      const gridInfo &,
      const size_t gridPoint,
      const double sigma2,
      double *outweights) const;
    Matrix CalculateCovarianceMatrix (
      const gridInfo &,
      const std::vector<double> &weights,
      const double totalWeight) const;
    std::pair<std::vector<double>, Eigen::VectorXd>
    bandwidthEstimation (const gridInfo &, const Matrix &, const double);
    void fractionOfPointsLocalization (
      const gridInfo &grid,
      const size_t gID,
      const double delta,
      double &weight,
      double &sigmaSQ,
      double *localWeights);
    void fractionOfSpreadLocalization (
      const gridInfo &grid,
      const size_t gID,
      const double delta,
      double &weight,
      double &sigmaSQ,
      double *localWeights);
    Eigen::VectorXd KernelDensityEstimation (
      const gridInfo &,
      const std::vector<double> &,
      const double weightNorm,
      const double kdecut2);
    gridErrorProbabilities StatisticalErrorFromKDE (
      const gridInfo &grid,
      const std::vector<double> &normkernel,
      const Eigen::VectorXd &prob,
      const Eigen::VectorXd &localDimensionality,
      const double weightNorm,
      const double kdecut2);

    struct quickShiftOutput {
      const std::set<size_t> clustersIndexes;
      const std::vector<size_t> gridToClusterIdx;
    };

    quickShiftOutput
    quickShift (const gridInfo &grid, const Eigen::VectorXd &probabilities);
    static quickShiftOutput clusterMerger (
      const double mergeThreshold,
      const gridInfo &grid,
      const quickShiftOutput &qsOut,
      const gridErrorProbabilities &errors,
      const Eigen::VectorXd &prob);

    void gridOutput (
      const gridInfo &grid,
      const quickShiftOutput &clusterInfo,
      const gridErrorProbabilities &errors,
      const Eigen::VectorXd &prob,
      const Eigen::VectorXd &localDimensionality) const;
    void classification (
      const gridInfo &grid,
      const quickShiftOutput &clusterInfo,
      const std::vector<double> &normkernel,
      const Eigen::VectorXd &prob) const;
    Matrix CalculateLogCovarianceMatrix (
      const size_t clusterIndex,
      const gridInfo &grid,
      const quickShiftOutput &clusterInfo,
      const std::vector<double> &normkernel,
      const Eigen::VectorXd &prob) const;

  private:
    size_t dim{0};
    size_t nsamples{0};
    size_t gridDim{0};
    size_t bootStraps{73};
    size_t nmsopt{1};
    // TODO: setup
    double fractionOfPointsVal{0.1};
    double fractionOfSpreadVal{0.1};
    /// parmeter controlling the merging of the outlier clusters
    double thrpcl{0.15};
    double tune{0.01};
    std::vector<double> dataWeights{};
    std::vector<double> QuickShiftCutSQ{};
    Matrix data{0, 0}; /// TODO: correct this
    gridInfo createGrid (size_t firstPoint = 0) const;
    std::vector<Matrix> HiInvStore{};
    std::mt19937_64 randomEngine{1};
    bool initialized_{false};
    bool dataSetNormalized_{false};
  };

} // namespace libpamm

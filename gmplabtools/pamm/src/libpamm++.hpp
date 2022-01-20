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
  using distanceMatrix = dynamicMatrices::triangularMatrix<double>;
  using Matrix = Eigen::MatrixXd;
  constexpr double TWOPI = 2.0 * M_PI;
  // constexpr double log2PI = gcem::log (M_2_PI);

  size_t QuickShift_nextPoint (
    const size_t ngrid,
    const size_t idx,
    const size_t idxn,
    const double lambda,
    const Eigen::VectorXd &probnmm,
    const distanceMatrix &distmm);

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
    size_t dimensionality () const;
  };

  struct gridErrorProbabilities {
    gridErrorProbabilities () = delete;
    gridErrorProbabilities (size_t);
    std::vector<double> absolute{}; // pabserr
    std::vector<double> relative{}; // prelerr
    size_t size () const;
  };
  struct quickShiftOutput {
    const std::set<size_t> clustersIndexes;
    const std::vector<size_t> gridToClusterIdx;
  };

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
    gaussian (size_t N);
    gaussian (
      const size_t N,
      const size_t gridAddr,
      const size_t nmsopt,
      const double normpks,
      const gridInfo &grid,
      const quickShiftOutput &clusterInfo,
      Matrix HiInverse,
      const std::vector<double> &normkernel,
      const Eigen::VectorXd &prob);
    void prepare ();

    // friend std::ostream& operator << (std::ostream&,gaussian);
  };

  class pammClustering final {
  public:
    pammClustering ();
    pammClustering (const pammClustering &) = delete;
    pammClustering (pammClustering &&) = delete;
    ~pammClustering ();
    void work ();
    void testLoadData ();
    void doNormalizeDataset ();

    double distanceCalculator (const size_t, const size_t) const;

    double distanceCalculator (const double *, const double *) const;
    /*
    double calculateMahalanobisDistanceSquared (
      const double *, const double *, const Matrix &) const;
*/
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

    quickShiftOutput quickShift (
      const gridInfo &grid, const Eigen::VectorXd &probabilities) const;

    void gridOutput (
      const gridInfo &grid,
      const quickShiftOutput &clusterInfo,
      const gridErrorProbabilities &errors,
      const Eigen::VectorXd &prob,
      const Eigen::VectorXd &localDimensionality) const;
    std::vector<gaussian> classification (
      const gridInfo &grid,
      const quickShiftOutput &clusterInfo,
      const std::vector<double> &normkernel,
      const Eigen::VectorXd &prob) const;
    void printClusters (std::vector<gaussian> clusters) const;

  private:
    size_t dim_{0};
    size_t nsamples_{0};
    size_t gridDim_{0};
    size_t bootStraps_{0};
    size_t nmsopt_{0};
    // TODO: setup
    double fractionOfPointsVal_{0.15};
    double fractionOfSpreadVal_{-1.0};
    double quickShiftLambda_{1.0};
    /// parmeter controlling the merging of the outlier clusters
    double thrpcl_{0.0};
    /// tune is not a setting!
    double tune_{0.00};
    std::string outputFilesNames_{"default"};
    std::vector<double> dataWeights_{};
    std::vector<double> QuickShiftCutSQ_{};
    Matrix data{0, 0}; /// TODO: correct this
    gridInfo createGrid (size_t firstPoint = 0) const;
    std::vector<Matrix> HiInvStore_{};
    size_t rngSeed_{12345};
    std::mt19937_64 randomEngine_{rngSeed_};
    bool initialized_{false};
    bool dataSetNormalized_{false};
  };

  // Function definitions
  double SOAPDistance (size_t dim, const double *x, const double *y);

  double SOAPDistance (
    size_t dim, const double *x, const double *y, double xyNormProduct);
  double
  SOAPDistanceSquaredNormalized (size_t dim, const double *x, const double *y);
  double SOAPDistanceSquared (size_t dim, const double *x, const double *y);

  double SOAPDistanceSquared (
    size_t dim, const double *x, const double *y, double xyNormProduct);
  double SOAPDistanceNormalized (size_t dim, const double *x, const double *y);
  double calculateMahalanobisDistanceSquared (
    const Eigen::VectorXd &A, const Eigen::VectorXd &B, const Matrix &invCov);

  double accumulateLogsumexp_if (
    const std::vector<size_t> &indexes,
    const std::vector<size_t> &probabilities,
    size_t sum_if);

  double accumulateLogsumexp_if (
    const std::vector<size_t> &indexes,
    const Eigen::VectorXd &probabilities,
    size_t sum_if);
  double accumulateLogsumexp (
    const std::vector<size_t> &indexes,
    const std::vector<size_t> &probabilities);

  double accumulateLogsumexp (
    const std::vector<size_t> &indexes, const Eigen::VectorXd &probabilities);

  Matrix CalculateCovarianceMatrix (
    const gridInfo &grid,
    const std::vector<double> &weights,
    const double totalWeight);

  Matrix CalculateLogCovarianceMatrix (
    const size_t clusterIndex,
    const gridInfo &grid,
    const quickShiftOutput &clusterInfo,
    const std::vector<double> &normkernel,
    const Eigen::VectorXd &prob);

  Matrix oracleShrinkage (Matrix m, double dimParameter);

  quickShiftOutput clusterMerger (
    const double mergeThreshold,
    const gridInfo &grid,
    const quickShiftOutput &qsOut,
    const gridErrorProbabilities &errors,
    const Eigen::VectorXd &prob);

  double RoyVetterliDimensionality (const Matrix &square);

} // namespace libpamm

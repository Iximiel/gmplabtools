
#include "libpamm++.hpp"
#include "tests.hpp"
#include <algorithm>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE ("Testing SOAP distance for known cases", "[SOAP]") {
  double normX = sqrt (std::inner_product (
    pammTests::SOAPTestX.begin (), pammTests::SOAPTestX.end (),
    pammTests::SOAPTestX.begin (), 0.0));
  double normY = sqrt (std::inner_product (
    pammTests::SOAPTestY.begin (), pammTests::SOAPTestY.end (),
    pammTests::SOAPTestY.begin (), 0.0));
  double testVal = libpamm::SOAPDistance (
    pammTests::SOAPDim, pammTests::SOAPTestX.data (),
    pammTests::SOAPTestY.data ());
  // with external calculated norm
  REQUIRE (
    libpamm::SOAPDistance (
      pammTests::SOAPDim, pammTests::SOAPTestX.data (),
      pammTests::SOAPTestY.data (), normX * normY) == Approx (testVal));
  std::array<double, pammTests::SOAPDim> SOAPTestYNorm, SOAPTestXNorm;
  std::transform (
    pammTests::SOAPTestX.begin (), pammTests::SOAPTestX.end (),
    SOAPTestXNorm.begin (), [=] (double x) -> double { return x / normX; });
  std::transform (
    pammTests::SOAPTestY.begin (), pammTests::SOAPTestY.end (),
    SOAPTestYNorm.begin (), [=] (double y) -> double { return y / normY; });
  // on normalized vectors:
  REQUIRE (
    libpamm::SOAPDistance (
      pammTests::SOAPDim, SOAPTestXNorm.data (), SOAPTestYNorm.data ()) ==
    Approx (testVal));
  REQUIRE (
    libpamm::SOAPDistance (
      pammTests::SOAPDim, SOAPTestXNorm.data (), SOAPTestYNorm.data (), 1.0) ==
    Approx (testVal));
  REQUIRE (
    libpamm::SOAPDistanceNormalized (
      pammTests::SOAPDim, SOAPTestXNorm.data (), SOAPTestYNorm.data ()) ==
    Approx (testVal));
  // with static result
  REQUIRE (testVal == Approx (0.4744757897));
  // auto std::c
}

TEST_CASE ("Testing Dynamic Matrix", "[Matrix]") {
  libpamm::dynamicMatrix<int> t (5, 2);
  int n = 0;
  for (int i = 0; i < t.Rows (); ++i) {
    for (int j = 0; j < t.Columns (); ++j) {
      t[i][j] = n;
      ++n;
    }
  }
  libpamm::dynamicMatrix<int> t2 (t);
  for (int i = 0; i < t.Rows (); ++i) {
    for (int j = 0; j < t.Columns (); ++j) {
      REQUIRE (t[i][j] == i * t.Columns () + j);
      REQUIRE (t[i][j] == t2[i][j]);
    }
  }
}

TEST_CASE (
  "Testing Dynamic Matrix product",
  "[Matrix][Product]") { // matrix product of two vectors
  auto t = [] (
             const libpamm::dynamicMatrix<int> &A,
             const libpamm::dynamicMatrix<int> B) {
    auto res = libpamm::matMul (A, B);
    REQUIRE (res.Rows () == A.Rows ());
    REQUIRE (res.Columns () == B.Columns ());
    return res;
  };

  libpamm::dynamicMatrix<int> vA (5, 1), vB (1, 5);
  for (int i = 0; i < vA.Rows (); ++i) {
    vA[i][0] = 1;
    vB[0][i] = 1;
  }
  libpamm::dynamicMatrix<int> vC (4, 1), vD (1, 4);
  for (int i = 0; i < vC.Rows (); ++i) {
    vC[i][0] = 2;
    vD[0][i] = 2;
  }

  auto newmat = t (vA, vB);
  REQUIRE (newmat[0][0] == 1);
  auto newscal = t (vB, vA);
  REQUIRE (newscal[0][0] == 5);
  auto newmat2 = t (vA, vD);
  REQUIRE (newmat2[0][0] == 2);

  REQUIRE_THROWS (libpamm::matMul (vA, vC));

  libpamm::dynamicMatrix<int> ldA (2, 3), ldB (3, 3);
  // clang-format off
  ldA[0][0] = 0; ldA[0][1] = 1; ldA[0][2] = 2;
  ldA[1][0] = 2; ldA[1][1] = 1; ldA[1][2] = 0;
  
  ldB[0][0] = 1; ldB[0][1] = 4; ldB[0][2] = 7;
  ldB[1][0] = 2; ldB[1][1] = 5; ldB[1][2] = 8;
  ldB[2][0] = 3; ldB[2][1] = 6; ldB[2][2] = 9;
  // clang-format on
  auto ldR = t (ldA, ldB);
  REQUIRE (ldR[0][0] == (0 * 1 + 1 * 2 + 2 * 3));
  REQUIRE (ldR[0][1] == (0 * 4 + 1 * 5 + 2 * 6));
  REQUIRE (ldR[0][2] == (0 * 7 + 1 * 8 + 2 * 9));
  REQUIRE (ldR[1][0] == (2 * 1 + 1 * 2 + 0 * 3));
  REQUIRE (ldR[1][1] == (2 * 4 + 1 * 5 + 0 * 6));
  REQUIRE (ldR[1][2] == (2 * 7 + 1 * 8 + 0 * 9));
}

TEST_CASE (
  "Testing Dynamic Matrix Transpose", "[Matrix][Transpose]") { // transpose
  libpamm::dynamicMatrix<int> t (5, 2);
  int n = 0;
  for (int i = 0; i < t.Rows (); ++i) {
    for (int j = 0; j < t.Columns (); ++j) {
      t[i][j] = n;
      ++n;
    }
  }
  auto tt = libpamm::Transpose (t);

  REQUIRE (tt.Rows () == t.Columns ());
  REQUIRE (t.Rows () == tt.Columns ());
  for (int i = 0; i < t.Rows (); ++i) {
    for (int j = 0; j < t.Columns (); ++j) {
      REQUIRE (tt[j][i] == t[i][j]);
    }
  }
  libpamm::dynamicMatrix<int> vA (5, 1);
  for (int i = 0; i < vA.Rows (); ++i) {
    vA[i][0] = i;
  }
  auto tA = libpamm::Transpose (vA);
  REQUIRE (tA.Rows () == vA.Columns ());
  REQUIRE (vA.Rows () == tA.Columns ());
  REQUIRE_NOTHROW (libpamm::matMul (tA, vA));
}

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
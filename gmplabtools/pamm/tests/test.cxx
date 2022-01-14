
#include "libpamm++.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("Testing SOAP distance for known cases", "[SOAP]") {

  REQUIRE(== Approx(1147332688.4281545));
}
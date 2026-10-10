#define _MAIN
#include "meraxes.h"
#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

// Include the cooling source to access static functions
#include "core/cooling.c"

// Test fixture setup
void setup(void)
{
  int argc = 0;
  char** argv = NULL;

  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  run_globals.mpi_rank = 0;
  run_globals.mpi_size = 1;

  // Absolute source-tree path set by CMake, so the test works from ctest's build directory.
  snprintf(run_globals.params.CoolingFuncsDir, sizeof(run_globals.params.CoolingFuncsDir), "%s", COOLING_FUNCS_DIR);

  // Cosmology (Genesis simulations); Mcool_SV -> Tvir_to_Mvir needs E(z)
  run_globals.params.OmegaM = 0.3121;
  run_globals.params.OmegaK = 0.0;
  run_globals.params.OmegaLambda = 0.6879;

  // Read the cooling function tables
  read_cooling_functions();
}

void teardown(void)
{
  MPI_Finalize();
}

TestSuite(cooling, .init = setup, .fini = teardown);

// Test that cooling rate interpolation returns sensible values
Test(cooling, interpolate_cooling_rate_basic)
{
  // Test at solar metallicity (logZ ~ -1.7 for Z=0.02)
  double logZ_solar = log10(0.02);

  // At high temperature (logT = 7.0), cooling should be significant
  double rate_high_T = interpolate_cooling_rate(7.0, logZ_solar);
  cr_expect_gt(rate_high_T, 0.0, "Cooling rate should be positive at high temp");

  // At moderate temperature (logT = 5.5)
  double rate_mid_T = interpolate_cooling_rate(5.5, logZ_solar);
  cr_expect_gt(rate_mid_T, 0.0, "Cooling rate should be positive at mid temp");

  // At low temperature (logT = 4.5)
  double rate_low_T = interpolate_cooling_rate(4.5, logZ_solar);
  cr_expect_gt(rate_low_T, 0.0, "Cooling rate should be positive at low temp");
}

// Test boundary conditions for temperature
Test(cooling, interpolate_cooling_rate_temp_boundaries)
{
  double logZ_solar = log10(0.02);

  // Below minimum temperature (MIN_TEMP = 4.0) should return 10^(-27)
  double rate_below_min = interpolate_cooling_rate(3.0, logZ_solar);
  cr_expect_float_eq(rate_below_min, pow(10, -27.0), 1e-35, "Rate below MIN_TEMP should be 10^-27");

  // At minimum temperature
  double rate_at_min = interpolate_cooling_rate(MIN_TEMP, logZ_solar);
  cr_expect_gt(rate_at_min, 0.0, "Rate at MIN_TEMP should be positive");

  // At maximum temperature
  double rate_at_max = interpolate_cooling_rate(MAX_TEMP, logZ_solar);
  cr_expect_gt(rate_at_max, 0.0, "Rate at MAX_TEMP should be positive");

  // Above maximum temperature (should clamp)
  double rate_above_max = interpolate_cooling_rate(9.0, logZ_solar);
  cr_expect_gt(rate_above_max, 0.0, "Rate above MAX_TEMP should be positive");
}

// Test metallicity boundary conditions
Test(cooling, interpolate_cooling_rate_metallicity_boundaries)
{
  double logT = 6.0;

  // Very low metallicity (below table minimum)
  double rate_low_Z = interpolate_cooling_rate(logT, -10.0);
  cr_expect_gt(rate_low_Z, 0.0, "Rate at very low Z should be positive");

  // Very high metallicity (above table maximum)
  double rate_high_Z = interpolate_cooling_rate(logT, 1.0);
  cr_expect_gt(rate_high_Z, 0.0, "Rate at very high Z should be positive");
}

// Test that cooling rate increases with metallicity at fixed temperature
Test(cooling, cooling_rate_metallicity_trend)
{
  double logT = 5.5; // Temperature where metal cooling dominates

  double rate_low_Z = interpolate_cooling_rate(logT, -4.0);
  double rate_mid_Z = interpolate_cooling_rate(logT, -2.0);
  double rate_high_Z = interpolate_cooling_rate(logT, 0.0);

  // Generally, cooling increases with metallicity due to metal line cooling
  // This may not be strictly monotonic at all temperatures, but overall trend
  cr_expect_gt(rate_high_Z, rate_low_Z, "High metallicity should have higher cooling rate than low Z");
}

// Test the static interpolation function directly
Test(cooling, interpolate_temp_dependant_cooling_rate_direct)
{
  // Test interpolation at a few metallicity indices
  for (int i_m = 0; i_m < N_METALLICITIES; i_m++) {
    double rate = interpolate_temp_dependant_cooling_rate(i_m, 6.0);
    // Rate is in log space, so should typically be between -27 and -20
    cr_expect_lt(rate, 0.0, "Log cooling rate should be negative");
    cr_expect_gt(rate, -30.0, "Log cooling rate should be > -30");
  }
}

// Parameterized test for various temperature values
struct temp_param
{
  double logTemp;
  bool expect_positive;
};

ParameterizedTestParameters(cooling, temperature_sweep)
{
  static struct temp_param params[] = {
    { 3.5, true }, // Below MIN_TEMP
    { 4.0, true }, // At MIN_TEMP
    { 4.5, true }, { 5.0, true }, { 5.5, true }, { 6.0, true }, { 6.5, true },
    { 7.0, true }, { 7.5, true }, { 8.0, true }, { 8.5, true }, // At MAX_TEMP
    { 9.0, true },                                              // Above MAX_TEMP
  };
  return cr_make_param_array(struct temp_param, params, sizeof(params) / sizeof(params[0]));
}

ParameterizedTest(struct temp_param* param, cooling, temperature_sweep)
{
  double logZ_solar = log10(0.02);
  double rate = interpolate_cooling_rate(param->logTemp, logZ_solar);

  if (param->expect_positive) {
    cr_expect_gt(rate, 0.0, "Cooling rate should be positive at logT=%.1f", param->logTemp);
  }
  cr_expect(isfinite(rate), "Cooling rate should be finite at logT=%.1f", param->logTemp);
}

#if USE_MINI_HALOS
// Tests for mini-halo functions (only compiled if USE_MINI_HALOS is enabled)

Test(cooling, LTE_Mcool_basic)
{
  // Test LTE molecular cooling at typical conditions
  double Temp = 1000.0; // 1000 K
  double nH = 1.0;      // 1 cm^-3

  double LTE = LTE_Mcool(Temp, nH);
  cr_expect_gt(LTE, 0.0, "LTE cooling should be positive");
  cr_expect(isfinite(LTE), "LTE cooling should be finite");
}

Test(cooling, LTE_Mcool_temperature_dependence)
{
  double nH = 1.0;

  double LTE_500K = LTE_Mcool(500.0, nH);
  double LTE_1000K = LTE_Mcool(1000.0, nH);
  double LTE_2000K = LTE_Mcool(2000.0, nH);

  // LTE cooling should generally increase with temperature
  cr_expect_gt(LTE_1000K, LTE_500K, "LTE cooling should increase with T");
  cr_expect_gt(LTE_2000K, LTE_1000K, "LTE cooling should increase with T");
}

Test(cooling, LTE_Mcool_density_dependence)
{
  double Temp = 1000.0;

  double LTE_low_n = LTE_Mcool(Temp, 0.1);
  double LTE_high_n = LTE_Mcool(Temp, 10.0);

  // LTE cooling per particle scales inversely with density
  cr_expect_gt(LTE_low_n, LTE_high_n, "LTE cooling per particle should decrease with density");
}

Test(cooling, Mcool_SV_basic)
{
  // Test streaming velocity cooling mass at z=20
  double redshift = 20.0;
  int n_sigma = 0; // No streaming velocity

  double Mcool = Mcool_SV(redshift, n_sigma);
  cr_expect_gt(Mcool, 0.0, "Mcool_SV should be positive");
  cr_expect(isfinite(Mcool), "Mcool_SV should be finite");
}

Test(cooling, Mcool_SV_streaming_velocity_effect)
{
  double redshift = 20.0;

  double Mcool_0sigma = Mcool_SV(redshift, 0);
  double Mcool_1sigma = Mcool_SV(redshift, 1);
  double Mcool_2sigma = Mcool_SV(redshift, 2);

  // Higher streaming velocity should increase minimum cooling mass
  cr_expect_gt(Mcool_1sigma, Mcool_0sigma, "1-sigma SV should increase Mcool");
  cr_expect_gt(Mcool_2sigma, Mcool_1sigma, "2-sigma SV should further increase Mcool");
}
#endif

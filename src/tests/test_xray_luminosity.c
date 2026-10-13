#define _MAIN
#include <criterion/criterion.h>
#include <math.h>
#include <string.h>

#include "meraxes.h"

// Follow test_cooling.c: include the source to test its static functions.
#include "core/reionization.c"

static void setup(void)
{
  memset(&run_globals, 0, sizeof(run_globals));
  run_globals.params.Flag_InstantaneousSFR = 1;
  run_globals.units.UnitMass_in_g = SOLAR_MASS;
  run_globals.units.UnitTime_in_s = SEC_PER_YEAR;

  physics_params_t* params = &run_globals.params.physics;
  params->Flag_UseXrayMetallicity = 1;
  params->LXrayGal = 3.16e40;
  params->XrayLogLNorm = 40.68;
  params->XrayMetallicitySlope = -1.22;
  params->XrayMetallicitySolar = 0.0134;
  params->XrayOxygenAbundanceFactor = 1.0;
  params->NuXrayThreshold = 500.0;
  params->NuXraySoftCut = 2000.0;
  params->SpecIndexXrayGal = 1.0;

#if USE_STOCHASTICITY
  run_globals.random_generator = gsl_rng_alloc(gsl_rng_mt19937);
  cr_assert_not_null(run_globals.random_generator);
  gsl_rng_set(run_globals.random_generator, 1809);
#endif
}

static void teardown(void)
{
#if USE_STOCHASTICITY
  gsl_rng_free(run_globals.random_generator);
  run_globals.random_generator = NULL;
#endif
}

static galaxy_t make_galaxy(void)
{
  galaxy_t gal;
  memset(&gal, 0, sizeof(gal));
  gal.Sfr = 1.0;
  gal.GrossStellarMass = 8.0;
  gal.ColdGas = 2.0;
  gal.MetalsColdGas = 0.0268;
  return gal;
}

static void expect_luminosity(double luminosity, double expected)
{
  cr_assert(isfinite(luminosity));
  cr_assert_gt(expected, 0.0);
  cr_expect_float_eq(luminosity / expected, 1.0, 1e-11);
}

TestSuite(xray_luminosity, .init = setup, .fini = teardown);

Test(xray_luminosity, flag_off_uses_legacy_relation)
{
  galaxy_t gal = make_galaxy();
  physics_params_t* params = &run_globals.params.physics;
  params->Flag_UseXrayMetallicity = 0;

  // Disabled metallicity parameters must not affect the original relation.
  params->XrayLogLNorm = NAN;
  params->XrayMetallicitySlope = NAN;
  params->XrayMetallicitySolar = 0.0;
  gal.MetalsColdGas = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 3.16);
  gal.MetalsColdGas = 0.2;
  params->NuXrayThreshold = 250.0;
  params->NuXraySoftCut = 4000.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 3.16);
}

Test(xray_luminosity, metallicity_relation_uses_linear_relative_oxygen_abundance)
{
  galaxy_t gal = make_galaxy();
  const double relative_abundances[] = { 0.0, 0.1, 1.0, 2.0 };
  const double expected_log_lx_sfr[] = { 40.68, 40.558, 39.46, 38.24 };

  for (int i = 0; i < 4; i++) {
    gal.MetalsColdGas = gal.ColdGas * 0.0134 * relative_abundances[i];
    double luminosity = calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0);
    cr_assert_gt(luminosity, 0.0);
    cr_expect_float_eq(log10(luminosity * XRAY_LUMINOSITY_UNIT), expected_log_lx_sfr[i], 1e-11);
  }
}

Test(xray_luminosity, source_units_and_sfr_scaling)
{
  galaxy_t gal = make_galaxy();
  run_globals.units.UnitMass_in_g = 10.0 * SOLAR_MASS;
  run_globals.units.UnitTime_in_s = 2.0 * SEC_PER_YEAR;

  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    double lx_per_sfr = flag ? pow(10.0, 39.46) / XRAY_LUMINOSITY_UNIT : 3.16;
    gal.Sfr = 0.4; // 2 Msun/yr with the units above.
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 2.0 * lx_per_sfr);
    gal.Sfr = 1.2;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 6.0 * lx_per_sfr);
  }
}

Test(xray_luminosity, time_averaged_sfr_uses_gross_stellar_mass)
{
  galaxy_t gal = make_galaxy();
  gal.Sfr = 99.0;
  run_globals.params.Flag_InstantaneousSFR = 0;

  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    double lx_per_sfr = flag ? pow(10.0, 39.46) / XRAY_LUMINOSITY_UNIT : 3.16;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 4.0, 0.0), 2.0 * lx_per_sfr);
  }
}

Test(xray_luminosity, energy_band_conversion)
{
  galaxy_t gal = make_galaxy();
  physics_params_t* params = &run_globals.params.physics;
  double reference = calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0);
  params->NuXrayThreshold = 250.0;
  params->NuXraySoftCut = 4000.0;

  // Integral ratios relative to 500-2000 eV: 2 for E^-1 and 2.5 for E^-2.
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 2.0 * reference);
  params->SpecIndexXrayGal = 2.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 2.5 * reference);
}

Test(xray_luminosity, oxygen_conversion_and_solar_metallicity)
{
  galaxy_t gal = make_galaxy();
  physics_params_t* params = &run_globals.params.physics;
  params->XrayOxygenAbundanceFactor = 2.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), pow(10.0, 38.24) / XRAY_LUMINOSITY_UNIT);
  params->XrayOxygenAbundanceFactor = 1.0;
  params->XrayMetallicitySolar = 0.0268;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), pow(10.0, 40.07) / XRAY_LUMINOSITY_UNIT);
}

Test(xray_luminosity, cold_gas_abundance_takes_priority)
{
  galaxy_t gal = make_galaxy();
  gal.NewStars[0] = 1.0;
  gal.StellarMass = 1.0;
#if USE_MINI_HALOS
  gal.NewStars_II[0] = 1.0;
  gal.StellarMass_II = 1.0;
#endif
  gal.NewMetals[0] = 0.0;
  gal.MetalsStellarMass = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), pow(10.0, 39.46) / XRAY_LUMINOSITY_UNIT);
}

Test(xray_luminosity, exhausted_gas_uses_new_stars)
{
  galaxy_t gal = make_galaxy();
  gal.ColdGas = 0.0;
  gal.MetalsColdGas = 0.0;
  gal.NewStars[0] = 2.0;
  gal.NewMetals[0] = 0.0268;
  gal.StellarMass = 10.0;
  gal.MetalsStellarMass = 0.0;
#if USE_MINI_HALOS
  gal.NewStars_II[0] = 2.0;
  gal.NewStars[0] = 20.0; // The abundance estimate must use the Pop II component.
  gal.StellarMass_II = 10.0;
#endif
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), pow(10.0, 39.46) / XRAY_LUMINOSITY_UNIT);
}

Test(xray_luminosity, exhausted_gas_uses_stellar_mass_if_no_new_stars)
{
  galaxy_t gal = make_galaxy();
  gal.ColdGas = 0.0;
  gal.MetalsColdGas = 0.0;
  gal.StellarMass = 2.0;
  gal.MetalsStellarMass = 0.0268;
#if USE_MINI_HALOS
  gal.StellarMass_II = 2.0;
  gal.StellarMass = 20.0;
#endif
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), pow(10.0, 39.46) / XRAY_LUMINOSITY_UNIT);
}

Test(xray_luminosity, unavailable_abundance_uses_legacy_relation)
{
  galaxy_t gal = make_galaxy();
  gal.ColdGas = 0.0;
  gal.MetalsColdGas = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 3.16);
}

Test(xray_luminosity, zero_sfr_has_zero_luminosity)
{
  galaxy_t gal = make_galaxy();
  gal.Sfr = 0.0;
  gal.GrossStellarMass = 0.0;
  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    for (int instant = 0; instant <= 1; instant++) {
      run_globals.params.Flag_InstantaneousSFR = instant;
      cr_expect_float_eq(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.3), 0.0, 0.0);
    }
  }
}

#if USE_STOCHASTICITY
Test(xray_luminosity, flag_off_preserves_legacy_scatter_and_rng_sequence)
{
  galaxy_t gal = make_galaxy();
  run_globals.params.physics.Flag_UseXrayMetallicity = 0;

  double expected[16];
  for (int i = 0; i < 16; i++)
    expected[i] = apply_lognormal_scatter(3.16, 0.3);
  unsigned long next = gsl_rng_get(run_globals.random_generator);

  gsl_rng_set(run_globals.random_generator, 1809);
  for (int i = 0; i < 16; i++)
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.3), expected[i]);
  cr_expect_eq(gsl_rng_get(run_globals.random_generator), next);
}

Test(xray_luminosity, metallicity_scatter_is_a_gaussian_residual_in_dex)
{
  galaxy_t gal = make_galaxy();
  gsl_rng* reference_rng = gsl_rng_clone(run_globals.random_generator);
  cr_assert_not_null(reference_rng);

  for (int i = 0; i < 32; i++) {
    double residual = 0.3 * gsl_cdf_ugaussian_Pinv(gsl_rng_uniform(reference_rng));
    double luminosity = calculate_galaxy_xray_luminosity(&gal, 1.0, 0.3);
    cr_assert_gt(luminosity, 0.0);
    cr_expect_float_eq(log10(luminosity * XRAY_LUMINOSITY_UNIT) - 39.46, residual, 1e-11);
  }
  cr_expect_eq(gsl_rng_get(run_globals.random_generator), gsl_rng_get(reference_rng));
  gsl_rng_free(reference_rng);
}

Test(xray_luminosity, zero_scatter_and_zero_sfr_do_not_consume_rng)
{
  galaxy_t gal = make_galaxy();
  gsl_rng* reference_rng = gsl_rng_clone(run_globals.random_generator);
  cr_assert_not_null(reference_rng);

  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    gal.Sfr = 1.0;
    double expected = flag ? pow(10.0, 39.46) / XRAY_LUMINOSITY_UNIT : 3.16;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), expected);
    gal.Sfr = 0.0;
    cr_expect_float_eq(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.3), 0.0, 0.0);
  }
  cr_expect_eq(gsl_rng_get(run_globals.random_generator), gsl_rng_get(reference_rng));
  gsl_rng_free(reference_rng);
}
#else
Test(xray_luminosity, scatter_is_ignored_without_stochasticity)
{
  galaxy_t gal = make_galaxy();
  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    double expected = flag ? pow(10.0, 39.46) / XRAY_LUMINOSITY_UNIT : 3.16;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.3), expected);
  }
}
#endif

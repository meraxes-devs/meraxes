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
  params->XrayMetallicityExponent = 0.3;
  params->XrayMetallicityTurnover = 0.008;
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
  gal.GrossStellarMetals = 8.0 * 0.0134;
  gal.StarFormingGasMetallicity = 0.0134;
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
  params->XrayMetallicityExponent = NAN;
  params->XrayMetallicityTurnover = 0.0;
  gal.MetalsColdGas = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 3.16);
  gal.MetalsColdGas = 0.2;
  params->NuXrayThreshold = 250.0;
  params->NuXraySoftCut = 4000.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 3.16);
}

Test(xray_luminosity, fragos_kaur_shape_uses_absolute_metal_mass_fraction)
{
  galaxy_t gal = make_galaxy();
  const double metallicities[] = { 0.0, 0.0001, 0.001, 0.002, 0.008, 0.0134, 0.02, 0.04 };
  // Independently evaluated with 50-digit decimal arithmetic, normalized at the low-Z plateau.
  const double expected_log_lx_sfr[] = {
    40.499687082618404, 40.499687082618404, 40.499687082618404, 40.464885759984931,
    39.895503757383320, 39.287708200795179, 38.514885759984931, 36.105194758684125
  };
  double previous = INFINITY;

  for (int i = 0; i < 8; i++) {
    gal.StarFormingGasMetallicity = metallicities[i];
    double luminosity = calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0);
    cr_assert_gt(luminosity, 0.0);
    cr_expect_float_eq(log10(luminosity * XRAY_LUMINOSITY_UNIT), expected_log_lx_sfr[i], 1e-11);
    cr_expect_leq(luminosity, previous);
    previous = luminosity;
  }
}

Test(xray_luminosity, low_metallicity_plateau_is_continuous)
{
  galaxy_t gal = make_galaxy();
  const double plateau_edge = 0.0010423067565678044;
  const double offsets[] = { -1e-6, 0.0, 1e-6 };

  for (int i = 0; i < 3; i++) {
    gal.StarFormingGasMetallicity = plateau_edge * (1.0 + offsets[i]);
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 3.16);
  }
}

Test(xray_luminosity, source_units_and_sfr_scaling)
{
  galaxy_t gal = make_galaxy();
  run_globals.units.UnitMass_in_g = 10.0 * SOLAR_MASS;
  run_globals.units.UnitTime_in_s = 2.0 * SEC_PER_YEAR;

  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    double lx_per_sfr = flag ? pow(10.0, 39.287708200795179) / XRAY_LUMINOSITY_UNIT : 3.16;
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
    double lx_per_sfr = flag ? pow(10.0, 39.287708200795179) / XRAY_LUMINOSITY_UNIT : 3.16;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 4.0, 0.0), 2.0 * lx_per_sfr);
  }
}

Test(xray_luminosity, supplied_normalization_has_no_automatic_band_conversion)
{
  galaxy_t gal = make_galaxy();
  physics_params_t* params = &run_globals.params.physics;
  double reference = calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0);
  const double spectral_indices[] = { 0.5, 1.0, 1.5, 2.0 };
  params->NuXrayThreshold = 250.0;
  params->NuXraySoftCut = 4000.0;
  for (int i = 0; i < 4; i++) {
    params->SpecIndexXrayGal = spectral_indices[i];
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), reference);
  }
}

Test(xray_luminosity, shared_normalization_scales_both_models)
{
  galaxy_t gal = make_galaxy();
  physics_params_t* params = &run_globals.params.physics;
  for (int flag = 0; flag <= 1; flag++) {
    params->Flag_UseXrayMetallicity = flag;
    params->LXrayGal = 3.16e40;
    double reference = calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0);
    params->LXrayGal *= 2.0;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 2.0 * reference);
  }
  gal.StarFormingGasMetallicity = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 6.32);
}

Test(xray_luminosity, heating_conversion_preserves_source_normalization)
{
  const double luminosities[] = { 0.0, 1e-40, 1e-30, 1e-20 };
  const double normalizations[] = { 1e38, 3.16e40, 1e42 };
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      double source = xray_luminosity_to_sfr_density(luminosities[i], normalizations[j]);
      double reconstructed = source * normalizations[j] * SEC_PER_YEAR;
      cr_assert(isfinite(source));
      if (luminosities[i] > 0.0)
        cr_expect_float_eq(reconstructed / luminosities[i], 1.0, 1e-14);
      else
        cr_expect_eq(reconstructed, 0.0);
    }
  }
}

Test(xray_luminosity, zero_normalization_disables_xrays)
{
  galaxy_t gal = make_galaxy();
  run_globals.params.physics.LXrayGal = 0.0;
  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    cr_expect_eq(calculate_galaxy_xray_luminosity(&gal, 1.0, flag ? 0.0 : 0.3), 0.0);
  }
  cr_expect_eq(xray_luminosity_to_sfr_density(0.0, 0.0), 0.0);
}

Test(xray_luminosity, turnover_rescales_the_metallicity_axis)
{
  galaxy_t gal = make_galaxy();
  double reference = calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0);
  run_globals.params.physics.XrayMetallicityTurnover *= 2.0;
  gal.StarFormingGasMetallicity *= 2.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), reference);
}

Test(xray_luminosity, exponent_sets_the_plateau_edge)
{
  galaxy_t gal = make_galaxy();
  run_globals.params.physics.XrayMetallicityExponent = 0.6;
  // The plateau now extends to Z = 0.0020846135131356088.
  gal.StarFormingGasMetallicity = 0.002;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), 3.16);
  gal.StarFormingGasMetallicity = 0.008;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0),
                    pow(10.0, 40.110702434749847) / XRAY_LUMINOSITY_UNIT);
}

Test(xray_luminosity, birth_abundance_survives_feedback_and_gas_exhaustion)
{
  galaxy_t gal = make_galaxy();
  double reference = pow(10.0, 39.287708200795179) / XRAY_LUMINOSITY_UNIT;
  gal.MetalsColdGas = 0.2; // Feedback-enriched gas is not the birth abundance.
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), reference);
  gal.ColdGas = 0.0;
  gal.MetalsColdGas = 0.0;
  gal.NewStars[0] = gal.NewMetals[0] = 0.0;
  gal.StellarMass = gal.MetalsStellarMass = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), reference);
}

Test(xray_luminosity, averaged_source_uses_cumulative_birth_abundance)
{
  galaxy_t gal = make_galaxy();
  run_globals.params.Flag_InstantaneousSFR = 0;
  gal.Sfr = 0.0;
  gal.StarFormingGasMetallicity = 0.0;
  gal.ColdGas = gal.MetalsColdGas = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 8.0, 0.0),
                    pow(10.0, 39.287708200795179) / XRAY_LUMINOSITY_UNIT);
  gal.GrossStellarMetals = 0.0;
  expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 8.0, 0.0), 3.16);
}

#if USE_STOCHASTICITY
Test(xray_luminosity, no_sfr_rescaling_preserves_birth_abundance)
{
  galaxy_t gal = make_galaxy();
  gal.SfrNoScatter = 2.0 * gal.Sfr;
  gal.GrossStellarMassNoScatter = 3.0 * gal.GrossStellarMass;
  const double factors[] = { 3.0, 2.0 };
  for (int instant = 0; instant <= 1; instant++) {
    run_globals.params.Flag_InstantaneousSFR = instant;
    galaxy_t view = xray_no_sfr_source_view(&gal);
    double target = calculate_galaxy_xray_luminosity(&gal, 8.0, 0.0);
    expect_luminosity(calculate_galaxy_xray_luminosity(&view, 8.0, 0.0), factors[instant] * target);
  }
}

Test(xray_luminosity, imputed_sfr_uses_zero_instantaneous_birth_abundance)
{
  galaxy_t gal = make_galaxy();
  gal.Sfr = 0.0;
  gal.StarFormingGasMetallicity = 0.0;
  gal.SfrNoScatter = 1.0;
  gal.GrossStellarMassNoScatter = 1.0;
  galaxy_t view = xray_no_sfr_source_view(&gal);
  cr_expect_eq(galaxy_xray_metallicity(&gal), 0.0);
  expect_luminosity(calculate_galaxy_xray_luminosity(&view, 1.0, 0.0), 3.16);
  gal.GrossStellarMass = gal.GrossStellarMetals = 0.0;
  for (int instant = 0; instant <= 1; instant++) {
    run_globals.params.Flag_InstantaneousSFR = instant;
    view = xray_no_sfr_source_view(&gal);
    expect_luminosity(calculate_galaxy_xray_luminosity(&view, 1.0, 0.0), 3.16);
  }
}
#endif

Test(xray_luminosity, zero_sfr_has_zero_luminosity)
{
  galaxy_t gal = make_galaxy();
  gal.Sfr = 0.0;
  gal.GrossStellarMass = 0.0;
  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    for (int instant = 0; instant <= 1; instant++) {
      run_globals.params.Flag_InstantaneousSFR = instant;
      cr_expect_float_eq(calculate_galaxy_xray_luminosity(&gal, 1.0, flag ? 0.0 : 0.3), 0.0, 0.0);
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

Test(xray_luminosity, zero_scatter_and_zero_sfr_do_not_consume_rng)
{
  galaxy_t gal = make_galaxy();
  gsl_rng* reference_rng = gsl_rng_clone(run_globals.random_generator);
  cr_assert_not_null(reference_rng);

  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    gal.Sfr = 1.0;
    double expected = flag ? pow(10.0, 39.287708200795179) / XRAY_LUMINOSITY_UNIT : 3.16;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), expected);
    gal.Sfr = 0.0;
    cr_expect_float_eq(calculate_galaxy_xray_luminosity(&gal, 1.0, flag ? 0.0 : 0.3), 0.0, 0.0);
  }
  cr_expect_eq(gsl_rng_get(run_globals.random_generator), gsl_rng_get(reference_rng));
  gsl_rng_free(reference_rng);
}
#else
// Parameter validation rejects nonzero scatter before this helper is reached.
Test(xray_luminosity, non_stochastic_helper_is_deterministic)
{
  galaxy_t gal = make_galaxy();
  for (int flag = 0; flag <= 1; flag++) {
    run_globals.params.physics.Flag_UseXrayMetallicity = flag;
    double expected = flag ? pow(10.0, 39.287708200795179) / XRAY_LUMINOSITY_UNIT : 3.16;
    expect_luminosity(calculate_galaxy_xray_luminosity(&gal, 1.0, 0.0), expected);
  }
}
#endif

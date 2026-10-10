#define _MAIN
#include <criterion/criterion.h>
#include <math.h>
#include <string.h>

#include "core/init.h"
#include "meraxes.h"
#include "physics/emission_lines.h"

static void reset_run_globals(void)
{
  memset(&run_globals, 0, sizeof(run_globals));
}

static void configure_default_units_and_params(void)
{
  run_globals.params.Hubble_h = 0.6774;

  // Typical cosmological code units used by existing tests.
  run_globals.units.UnitLength_in_cm = 3.085678e24;   // 1 Mpc/h
  run_globals.units.UnitMass_in_g = 1.989e43;         // 1e10 Msun/h
  run_globals.units.UnitVelocity_in_cm_per_s = 1.0e5; // 1 km/s

  set_units();
}

static galaxy_t make_valid_test_galaxy(void)
{
  galaxy_t gal;
  memset(&gal, 0, sizeof(gal));

  gal.Mvir = 10.0;
  gal.Rvir = 0.08;
  gal.Vmax = 150.0;
  gal.Spin = 0.04;
  gal.dt = 0.01;

  gal.StellarMass = 0.5;
  gal.ColdGas = 0.7;
  gal.MetalsColdGas = 0.01;
  gal.Mcool = 0.2;
  gal.Sfr = 0.5;
  gal.DiskScaleLength = 0.003;

  return gal;
}

TestSuite(emission_lines);

Test(emission_lines, set_OIII_coeffs_generates_finite_coefficients)
{
  reset_run_globals();
  set_OIII_coeffs(1e4);

  cr_expect(isfinite(run_globals.loiii_params.oxygen_abundance_over_z_sun));
  cr_expect(isfinite(run_globals.loiii_params.excitation_rate));
  cr_expect(isfinite(run_globals.loiii_params.branching_ratio));

  cr_expect_gt(run_globals.loiii_params.oxygen_abundance_over_z_sun, 0.0);
  cr_expect_gt(run_globals.loiii_params.excitation_rate, 0.0);
  cr_expect_gt(run_globals.loiii_params.branching_ratio, 0.0);
}

Test(emission_lines, set_OIII_coeffs_non_positive_temperature_falls_back_to_1e4)
{
  reset_run_globals();

  set_OIII_coeffs(1e4);
  const loiii_params_t ref = run_globals.loiii_params;

  set_OIII_coeffs(0.0);

  cr_expect_float_eq(run_globals.loiii_params.oxygen_abundance_over_z_sun, ref.oxygen_abundance_over_z_sun, 1e-15);
  cr_expect_float_eq(run_globals.loiii_params.excitation_rate, ref.excitation_rate, 1e-15);
  cr_expect_float_eq(run_globals.loiii_params.branching_ratio, ref.branching_ratio, 1e-15);
}

Test(emission_lines, compute_LOIII_valid_inputs_give_finite_positive_outputs)
{
  reset_run_globals();
  configure_default_units_and_params();
  set_OIII_coeffs(1e4);

  double zz[] = { 7.0 };
  run_globals.ZZ = zz;

  galaxy_t gal = make_valid_test_galaxy();
  compute_LOIII(&gal, 0);

  cr_expect(isfinite(gal.LOIII));
  cr_expect_gt(gal.LOIII, 0.0);
  cr_expect(isfinite(gal.ionization_param));
  cr_expect_gt(gal.ionization_param, 0.0);
}

// compute_LOIII sets the snapshot value rather than accumulating, so a
// second call on the same galaxy must not change it.
Test(emission_lines, compute_LOIII_does_not_accumulate)
{
  reset_run_globals();
  configure_default_units_and_params();
  set_OIII_coeffs(1e4);

  double zz[] = { 7.0 };
  run_globals.ZZ = zz;

  galaxy_t gal = make_valid_test_galaxy();
  compute_LOIII(&gal, 0);
  const double loiii = gal.LOIII;
  const double ionization_param = gal.ionization_param;

  compute_LOIII(&gal, 0);
  cr_expect_float_eq(gal.LOIII, loiii, 1e-12 * loiii);
  cr_expect_float_eq(gal.ionization_param, ionization_param, 1e-12 * ionization_param);
}

Test(emission_lines, compute_LOIII_invalid_inputs_set_zero)
{
  reset_run_globals();
  configure_default_units_and_params();
  set_OIII_coeffs(1e4);

  double zz[] = { 7.0 };
  run_globals.ZZ = zz;

  galaxy_t gal = make_valid_test_galaxy();

  gal.LOIII = 123.0;

  gal.ColdGas = 0.0;
  compute_LOIII(&gal, 0);
  cr_expect_float_eq(gal.LOIII, 0.0, 0.0);

  gal = make_valid_test_galaxy();
  gal.LOIII = 456.0;
  gal.Mvir = 0.0;
  compute_LOIII(&gal, 0);
  cr_expect_float_eq(gal.LOIII, 0.0, 0.0);
}

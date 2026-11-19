#define _MAIN
#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/galaxies.h"
#include "core/reionization.h"
#include "meraxes.h"
#include "physics/emission_lines.h"
#include "physics/mergers.h"
#include "physics/star_formation.h"
#include "physics/supernova_feedback.h"
// Synthetic energy tables for the production merger/SN lifecycle test.
#include "core/stellar_feedback.c"

static void setup(void)
{
  static double redshifts[] = { 10.0, 9.0, 8.0 };
  static double lookback_times[] = { 3.0, 2.0, 1.0 };
  memset(&run_globals, 0, sizeof(run_globals));
  run_globals.ZZ = redshifts;
  run_globals.LTTime = lookback_times;
  run_globals.params.Hubble_h = 0.7;
  run_globals.params.physics.Flag_IRA = 1;
  run_globals.params.physics.MinMergerStellarMass = 1e10;
  // Nontrivial X-ray factors: every metallicity used below lies past the plateau.
  run_globals.params.physics.Flag_UseXrayMetallicity = 1;
  run_globals.params.physics.XrayMetallicityExponent = 0.3;
  run_globals.params.physics.XrayMetallicityTurnover = 0.008;
  // Units and coefficients for the per-event [O III] luminosity.
  run_globals.units.UnitMass_in_g = 1e10 * SOLAR_MASS;
  run_globals.units.UnitTime_in_s = 1e8 * SEC_PER_YEAR;
  set_OIII_coeffs();
  run_globals.params.physics.ReionNionPhotPerBary = 4000.0;
  run_globals.random_generator = gsl_rng_alloc(gsl_rng_mt19937);
  cr_assert_not_null(run_globals.random_generator);
#ifdef CALC_MAGS
  // All target snapshots precede the test bursts, so no spectral data are needed.
  static double unused_spectrum;
  run_globals.mag_params.working = &unused_spectrum;
  run_globals.mag_params.inBC = &unused_spectrum;
  run_globals.mag_params.outBC = &unused_spectrum;
#if USE_MINI_HALOS
  run_globals.mag_params.workingIII = &unused_spectrum;
#endif
#endif
}

static void teardown(void)
{
  gsl_rng_free(run_globals.random_generator);
}

static galaxy_t make_galaxy(double gas, double metallicity, double dt)
{
  galaxy_t gal = { 0 };
  gal.ColdGas = gas;
  gal.MetalsColdGas = gas * metallicity;
  gal.dt = dt;
  gal.ghost_flag = true;
#if USE_MINI_HALOS
  gal.Galaxy_Population = 2;
#endif
  return gal;
}

// X-ray factor of stars born at metallicity Z (its shape is tested in test_xray_luminosity).
static double g(double metallicity)
{
  return xray_metallicity_factor(metallicity);
}

// [O III] luminosity of one event with this SFR and gas metallicity (its form is
// tested in test_emission_lines).
static double loiii_of(double sfr, double metallicity)
{
  galaxy_t gal = { 0 };
  compute_LOIII(&gal, sfr, metallicity, 1);
  return gal.LOIII;
}

static void expect_loiii(double loiii, double expected)
{
  cr_assert_gt(expected, 0.0);
  cr_expect_float_eq(loiii, expected, 1e-12 * expected);
}

TestSuite(star_forming_metallicity, .init = setup, .fini = teardown);

Test(star_forming_metallicity, new_galaxy_initializes_birth_abundances)
{
  galaxy_t* gal = new_galaxy(0, 1);
  cr_assert_not_null(gal);
  cr_expect_eq(gal->LOIII, 0.0);
  cr_expect_eq(gal->XrayWeightedSfr, 0.0);
  cr_expect_eq(gal->XrayWeightedGSM, 0.0);
  free(gal);
}

Test(star_forming_metallicity, exhausted_gas_retains_birth_abundance_through_feedback)
{
  galaxy_t gal = make_galaxy(1.0, 0.01, 1.0);
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);
  cr_expect_eq(gal.ColdGas, 0.0);
  expect_loiii(gal.LOIII, loiii_of(1.0, 0.01));
  cr_expect_float_eq(gal.XrayWeightedSfr, g(0.01), 1e-14);
  cr_expect_float_eq(gal.XrayWeightedGSM, g(0.01), 1e-14);

  // Recycled gas and new metals change the gas abundance, not the recorded birth Z.
  update_reservoirs_from_sn_feedback(&gal, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.004);
  cr_expect_float_eq(gal.MetalsColdGas / gal.ColdGas, 0.02, 1e-14);
  expect_loiii(gal.LOIII, loiii_of(1.0, 0.01));
  cr_expect_float_eq(gal.XrayWeightedSfr, g(0.01), 1e-14);
  cr_expect_float_eq(gal.XrayWeightedGSM, g(0.01), 1e-14);
}

Test(star_forming_metallicity, multiple_episodes_follow_instantaneous_and_cumulative_source_weights)
{
  galaxy_t gal = make_galaxy(4.0, 0.01, 1.0);
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);
  gal.MetalsColdGas = gal.ColdGas * 0.03;
  gal.dt = 0.5;
  update_reservoirs_from_sf(&gal, 1.0, 1, MERGER);

  cr_expect_float_eq(gal.Sfr, 3.0, 1e-14);
  expect_loiii(gal.LOIII, loiii_of(1.0, 0.01) + loiii_of(2.0, 0.03));
  cr_expect_float_eq(gal.XrayWeightedSfr, g(0.01) + 2.0 * g(0.03), 1e-14);
  cr_expect_float_eq(gal.XrayWeightedGSM, g(0.01) + g(0.03), 1e-14);

  // L_X/SFR is nonlinear in Z: the factor of the SFR-weighted mean Z underestimates
  // the per-event sum by ~12x here.
  cr_expect_gt(gal.XrayWeightedSfr, 10.0 * gal.Sfr * g(0.07 / 3.0));
}

Test(star_forming_metallicity, snapshot_reset_preserves_cumulative_xray_weights)
{
  galaxy_t gal = make_galaxy(3.0, 0.01, 1.0);
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);
  reset_galaxy_properties(&gal, 2);
  cr_expect_eq(gal.Sfr, 0.0);
  cr_expect_eq(gal.LOIII, 0.0);
  cr_expect_eq(gal.XrayWeightedSfr, 0.0);
  cr_expect_float_eq(gal.XrayWeightedGSM, g(0.01), 1e-14);
  cr_expect_float_eq(gal.GrossStellarMass, 1.0, 1e-14);

  gal.MetalsColdGas = gal.ColdGas * 0.03;
  update_reservoirs_from_sf(&gal, 1.0, 2, INSITU);
  expect_loiii(gal.LOIII, loiii_of(1.0, 0.03));
  cr_expect_float_eq(gal.XrayWeightedSfr, g(0.03), 1e-14);
  cr_expect_float_eq(gal.XrayWeightedGSM, g(0.01) + g(0.03), 1e-14);
}

Test(star_forming_metallicity, mergers_combine_birth_abundances_with_source_weights)
{
  galaxy_t parent = make_galaxy(2.0, 0.01, 1.0);
  galaxy_t satellite = make_galaxy(2.0, 0.03, 0.5);
  update_reservoirs_from_sf(&parent, 1.0, 1, INSITU);
  update_reservoirs_from_sf(&satellite, 1.0, 1, INSITU);
  satellite.MergerTarget = &parent;
  int dead = 0;
  merge_with_target(&satellite, &dead, 1);

  cr_expect_eq(dead, 1);
  cr_expect_float_eq(parent.Sfr, 3.0, 1e-14);
  expect_loiii(parent.LOIII, loiii_of(1.0, 0.01) + loiii_of(2.0, 0.03));
  cr_expect_float_eq(parent.XrayWeightedSfr, g(0.01) + 2.0 * g(0.03), 1e-14);
  cr_expect_float_eq(parent.XrayWeightedGSM, g(0.01) + g(0.03), 1e-14);
}

// Merges a satellite carrying this snapshot's [O III] state into a parent and returns
// the parent. Neither galaxy forms stars in the merger, so only the inputs combine.
static galaxy_t merge_loiii(double parent_loiii, double parent_u, double satellite_loiii, double satellite_u)
{
  galaxy_t parent = make_galaxy(1.0, 0.01, 1.0);
  galaxy_t satellite = make_galaxy(1.0, 0.03, 1.0);
  parent.LOIII = parent_loiii;
  parent.ionization_param = parent_u;
  satellite.LOIII = satellite_loiii;
  satellite.ionization_param = satellite_u;
  satellite.MergerTarget = &parent;
  int dead = 0;
  merge_with_target(&satellite, &dead, 1);
  cr_assert_eq(dead, 1);
  cr_assert_eq(parent.Sfr, 0.0);
  return parent;
}

// Luminosities add and the ionization parameter is their LOIII-weighted mean; a
// galaxy without [O III] this snapshot leaves the other's parameter unchanged.
Test(star_forming_metallicity, mergers_weight_ionization_param_by_loiii)
{
  galaxy_t merged = merge_loiii(2.0, 3e9, 6.0, 1e9);
  cr_expect_float_eq(merged.LOIII, 8.0, 1e-14);
  cr_expect_float_eq(merged.ionization_param, 1.5e9, 1e-14 * 1.5e9);

  merged = merge_loiii(2.0, 3e9, 0.0, 0.0);
  cr_expect_eq(merged.LOIII, 2.0);
  cr_expect_eq(merged.ionization_param, 3e9);

  merged = merge_loiii(0.0, 0.0, 6.0, 1e9);
  cr_expect_eq(merged.LOIII, 6.0);
  cr_expect_eq(merged.ionization_param, 1e9);

  merged = merge_loiii(0.0, 0.0, 0.0, 0.0);
  cr_expect_eq(merged.LOIII, 0.0);
  cr_expect_eq(merged.ionization_param, 0.0);
}

// Gives the galaxy a disc state for which its events have a positive ionization
// parameter (the galaxies above leave it zero).
static void give_valid_disc(galaxy_t* gal, double disk_scale_length)
{
  run_globals.units.UnitLength_in_cm = 3.085678e24;
  run_globals.units.UnitVelocity_in_cm_per_s = 1e5;
  run_globals.params.physics.Y_He = 0.24;
  gal->Mvir = 10.0;
  gal->Rvir = 0.08;
  gal->Vmax = 150.0;
  gal->Spin = 0.04;
  gal->StellarMass = 0.5;
  gal->Mcool = 0.2;
  gal->DiskScaleLength = disk_scale_length;
}

// The ionization parameters that in-situ events give the parent and the satellite
// reach the merged galaxy weighted by their [O III] luminosities.
Test(star_forming_metallicity, mergers_combine_event_ionization_params)
{
  galaxy_t parent = make_galaxy(2.0, 0.01, 1.0);
  galaxy_t satellite = make_galaxy(2.0, 0.03, 0.5);
  give_valid_disc(&parent, 0.003);
  give_valid_disc(&satellite, 0.006);
  update_reservoirs_from_sf(&parent, 1.0, 1, INSITU);
  update_reservoirs_from_sf(&satellite, 1.0, 1, INSITU);

  const double parent_loiii = parent.LOIII, parent_u = parent.ionization_param;
  const double satellite_loiii = satellite.LOIII, satellite_u = satellite.ionization_param;
  cr_assert_gt(parent_u, 0.0);
  cr_assert_gt(satellite_u, 0.0);
  cr_assert_gt(fabs(satellite_u / parent_u - 1.0), 1e-3);
  cr_assert_gt(fabs(satellite_loiii / parent_loiii - 1.0), 1e-3);

  satellite.MergerTarget = &parent;
  int dead = 0;
  merge_with_target(&satellite, &dead, 1);

  const double expected_u =
    (parent_loiii * parent_u + satellite_loiii * satellite_u) / (parent_loiii + satellite_loiii);
  expect_loiii(parent.LOIII, parent_loiii + satellite_loiii);
  cr_expect_float_eq(parent.ionization_param, expected_u, 1e-12 * expected_u);
}

Test(star_forming_metallicity, reidentified_ghost_records_birth_metals_independently_of_history_bin)
{
  galaxy_t gal = make_galaxy(2.0, 0.01, 3.0);
  gal.LastIdentSnap = 0;
  run_globals.params.physics.Flag_IRA = 0;
  update_reservoirs_from_sf(&gal, 1.0, 2, INSITU);

  cr_expect_eq(gal.NewStars[0], 0.0);
  cr_expect_float_eq(gal.NewStars[1], 1.0, 1e-14);
  expect_loiii(gal.LOIII, loiii_of(1.0 / 3.0, 0.01));
  cr_expect_float_eq(gal.XrayWeightedSfr, g(0.01) / 3.0, 1e-14);
  cr_expect_float_eq(gal.XrayWeightedGSM, g(0.01), 1e-14);
}

Test(star_forming_metallicity, merger_without_current_star_formation_has_finite_zero_birth_abundance)
{
  galaxy_t parent = make_galaxy(1.0, 0.01, 1.0);
  galaxy_t satellite = make_galaxy(1.0, 0.03, 1.0);
  satellite.MergerTarget = &parent;
  int dead = 0;
  merge_with_target(&satellite, &dead, 1);
  cr_expect_eq(parent.LOIII, 0.0);
  cr_expect_eq(parent.XrayWeightedSfr, 0.0);
  cr_expect_eq(parent.XrayWeightedGSM, 0.0);
}

#if USE_MINI_HALOS
Test(star_forming_metallicity, popiii_episodes_do_not_change_popii_birth_abundances)
{
  galaxy_t gal = make_galaxy(3.0, 0.01, 1.0);
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);
  gal.Galaxy_Population = 3;
  gal.MetalsColdGas = 0.0;
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);

  cr_expect_float_eq(gal.Sfr, 1.0, 1e-14);
  cr_expect_float_eq(gal.SfrIII, 1.0, 1e-14);
  expect_loiii(gal.LOIII, loiii_of(1.0, 0.01));
  cr_expect_float_eq(gal.GrossStellarMass, 1.0, 1e-14);
  cr_expect_float_eq(gal.XrayWeightedSfr, g(0.01), 1e-14);
  cr_expect_float_eq(gal.XrayWeightedGSM, g(0.01), 1e-14);
}
#endif

Test(star_forming_metallicity, merger_starburst_records_birth_abundance_before_feedback)
{
  physics_params_t* params = &run_globals.params.physics;
  params->MinMergerStellarMass = 0.0;
  params->MinMergerRatioForBurst = 0.0;
  params->MergerBurstFactor = 0.5;
  params->MergerBurstScaling = 1.0;
  params->SfRecycleFraction = 0.2;
  params->Yield = 0.01;
  params->SnModel = 1;
  params->SnReheatNorm = params->SnEjectionNorm = 100.0;

  // Finite synthetic SN tables; energy-coupling efficiencies remain zero.
  energy_tables[NAGE - 1] = 1.0;
  for (int i = 0; i < NMETAL; i++)
    energy_tables_working[0][i] = 1.0;

  galaxy_t parent = make_galaxy(2.0, 0.01, 1.0);
  galaxy_t satellite = make_galaxy(2.0, 0.03, 0.5);
  fof_group_t fof = { 0 };
  halo_t halo = { 0 };

  fof.Vvir = parent.Vvir = parent.Vmax = 100.0;
  halo.FOFGroup = &fof;
  parent.Halo = &halo;

  update_reservoirs_from_sf(&parent, 1.0, 1, INSITU);
  update_reservoirs_from_sf(&satellite, 1.0, 1, INSITU);
  satellite.MergerTarget = &parent;
  int dead = 0;
  merge_with_target(&satellite, &dead, 1);

  cr_expect_eq(dead, 1);
  cr_expect_float_eq(parent.MergerBurstMass, 1.0, 1e-14);
  cr_expect_float_eq(parent.Sfr, 4.0, 1e-14);
  cr_expect_float_eq(parent.GrossStellarMass, 3.0, 1e-14);
  // The burst forms from the merged gas at Z = 0.02, before its feedback enriches it.
  expect_loiii(parent.LOIII, loiii_of(1.0, 0.01) + loiii_of(2.0, 0.03) + loiii_of(1.0, 0.02));
  cr_expect_float_eq(parent.XrayWeightedSfr, g(0.01) + 2.0 * g(0.03) + g(0.02), 1e-14);
  cr_expect_float_eq(parent.XrayWeightedGSM, g(0.01) + g(0.03) + g(0.02), 1e-14);
  cr_expect_float_eq(parent.MetalsColdGas / parent.ColdGas, 0.025, 1e-14);
}

// With the empirical relation the weighted sources must reproduce Sfr and
// GrossStellarMass bit for bit, so Flag_UseXrayMetallicity=0 runs are unchanged.
Test(star_forming_metallicity, disabled_metallicity_weights_equal_sources_exactly)
{
  run_globals.params.physics.Flag_UseXrayMetallicity = 0;
  galaxy_t parent = make_galaxy(4.0, 0.01, 0.7);
  galaxy_t satellite = make_galaxy(2.0, 0.03, 0.3);
  update_reservoirs_from_sf(&parent, 0.9, 1, INSITU);
  parent.MetalsColdGas = parent.ColdGas * 0.02;
  update_reservoirs_from_sf(&parent, 1.3, 1, MERGER);
  update_reservoirs_from_sf(&satellite, 0.7, 1, INSITU);
  satellite.MergerTarget = &parent;
  int dead = 0;
  merge_with_target(&satellite, &dead, 1);

  cr_expect_gt(parent.Sfr, 0.0);
  cr_expect_eq(parent.XrayWeightedSfr, parent.Sfr);
  cr_expect_eq(parent.XrayWeightedGSM, parent.GrossStellarMass);
}
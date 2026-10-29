#define _MAIN
#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>

#include "core/galaxies.h"
#include "meraxes.h"
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

TestSuite(star_forming_metallicity, .init = setup, .fini = teardown);

Test(star_forming_metallicity, new_galaxy_initializes_birth_abundances)
{
  galaxy_t* gal = new_galaxy(0, 1);
  cr_assert_not_null(gal);
  cr_expect_eq(gal->StarFormingGasMetallicity, 0.0);
  cr_expect_eq(gal->GrossStellarMetals, 0.0);
  free(gal);
}

Test(star_forming_metallicity, exhausted_gas_retains_birth_abundance_through_feedback)
{
  galaxy_t gal = make_galaxy(1.0, 0.01, 1.0);
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);
  cr_expect_eq(gal.ColdGas, 0.0);
  cr_expect_float_eq(gal.StarFormingGasMetallicity, 0.01, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMetals, 0.01, 1e-14);

  // Recycled gas and new metals change the gas abundance, not the recorded birth Z.
  update_reservoirs_from_sn_feedback(&gal, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.004);
  cr_expect_float_eq(gal.MetalsColdGas / gal.ColdGas, 0.02, 1e-14);
  cr_expect_float_eq(gal.StarFormingGasMetallicity, 0.01, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMetals / gal.GrossStellarMass, 0.01, 1e-14);
}

Test(star_forming_metallicity, multiple_episodes_follow_instantaneous_and_cumulative_source_weights)
{
  galaxy_t gal = make_galaxy(4.0, 0.01, 1.0);
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);
  gal.MetalsColdGas = gal.ColdGas * 0.03;
  gal.dt = 0.5;
  update_reservoirs_from_sf(&gal, 1.0, 1, MERGER);

  cr_expect_float_eq(gal.Sfr, 3.0, 1e-14);
  cr_expect_float_eq(gal.StarFormingGasMetallicity, 0.07 / 3.0, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMetals, 0.04, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMetals / gal.GrossStellarMass, 0.02, 1e-14);
}

Test(star_forming_metallicity, snapshot_reset_preserves_cumulative_birth_metals)
{
  galaxy_t gal = make_galaxy(3.0, 0.01, 1.0);
  update_reservoirs_from_sf(&gal, 1.0, 1, INSITU);
  reset_galaxy_properties(&gal, 2);
  cr_expect_eq(gal.Sfr, 0.0);
  cr_expect_eq(gal.StarFormingGasMetallicity, 0.0);
  cr_expect_float_eq(gal.GrossStellarMetals, 0.01, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMass, 1.0, 1e-14);

  gal.MetalsColdGas = gal.ColdGas * 0.03;
  update_reservoirs_from_sf(&gal, 1.0, 2, INSITU);
  cr_expect_float_eq(gal.StarFormingGasMetallicity, 0.03, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMetals / gal.GrossStellarMass, 0.02, 1e-14);
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
  cr_expect_float_eq(parent.StarFormingGasMetallicity, 0.07 / 3.0, 1e-14);
  cr_expect_float_eq(parent.GrossStellarMetals / parent.GrossStellarMass, 0.02, 1e-14);
}

Test(star_forming_metallicity, reidentified_ghost_records_birth_metals_independently_of_history_bin)
{
  galaxy_t gal = make_galaxy(2.0, 0.01, 3.0);
  gal.LastIdentSnap = 0;
  run_globals.params.physics.Flag_IRA = 0;
  update_reservoirs_from_sf(&gal, 1.0, 2, INSITU);

  cr_expect_eq(gal.NewStars[0], 0.0);
  cr_expect_float_eq(gal.NewStars[1], 1.0, 1e-14);
  cr_expect_float_eq(gal.StarFormingGasMetallicity, 0.01, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMetals, 0.01, 1e-14);
}

Test(star_forming_metallicity, merger_without_current_star_formation_has_finite_zero_birth_abundance)
{
  galaxy_t parent = make_galaxy(1.0, 0.01, 1.0);
  galaxy_t satellite = make_galaxy(1.0, 0.03, 1.0);
  satellite.MergerTarget = &parent;
  int dead = 0;
  merge_with_target(&satellite, &dead, 1);
  cr_expect_eq(parent.StarFormingGasMetallicity, 0.0);
  cr_expect_eq(parent.GrossStellarMetals, 0.0);
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
  cr_expect_float_eq(gal.StarFormingGasMetallicity, 0.01, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMass, 1.0, 1e-14);
  cr_expect_float_eq(gal.GrossStellarMetals, 0.01, 1e-14);
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
  cr_expect_float_eq(parent.StarFormingGasMetallicity, 0.0225, 1e-14);
  cr_expect_float_eq(parent.GrossStellarMass, 3.0, 1e-14);
  cr_expect_float_eq(parent.GrossStellarMetals, 0.06, 1e-14);
  cr_expect_float_eq(parent.MetalsColdGas / parent.ColdGas, 0.025, 1e-14);
}
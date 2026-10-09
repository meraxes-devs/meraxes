#define _MAIN
#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <math.h>
#include "meraxes.h"
#include <mpi.h>
#include <string.h>

// Include the blackhole_feedback source to access static functions
#include "physics/blackhole_feedback.c"

// Forward declaration
void set_units(void);

// Test fixture setup
void setup(void)
{
  int argc = 0;
  char** argv = NULL;

  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  run_globals.mpi_rank = 0;
  run_globals.mpi_size = 1;

  // Set up basic parameters needed for blackhole feedback calculations
  run_globals.params.Hubble_h = 0.6774;

  // Set up units (typical cosmological simulation units)
  run_globals.units.UnitLength_in_cm = 3.085678e24;     // 1 Mpc/h
  run_globals.units.UnitMass_in_g = 1.989e43;           // 1e10 Msun/h
  run_globals.units.UnitVelocity_in_cm_per_s = 1.0e5;   // 1 km/s

  // Initialize derived units
  set_units();

  // Set physics parameters for blackhole feedback
  run_globals.params.physics.EddingtonRatio = 1.0;
  run_globals.params.physics.RadioModeEff = 0.08;
  run_globals.params.physics.QuasarModeEff = 0.02;
  run_globals.params.physics.BlackHoleGrowthRate = 0.03;
  run_globals.params.physics.quasar_mode_scaling = 0.0;
  run_globals.params.physics.quasar_fobs = 1.0;
  run_globals.params.physics.Flag_BHObscuedIonization = 1;
  run_globals.params.physics.ReionNionPhotPerBary = 5000.0;
  run_globals.params.physics.Flag_BHARExponentialCut = 0;
}

void teardown(void)
{
  MPI_Finalize();
}

TestSuite(blackhole_feedback, .init = setup, .fini = teardown);

// Test calculate_BHemissivity function
Test(blackhole_feedback, calculate_BHemissivity_basic)
{
  double BlackHoleMass = 1e-3;  // 1e7 Msun in internal units (1e10 Msun/h)
  double accreted_mass = 1e-4; // 1e6 Msun accreted
  double emissivity, accretion_time, quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  calculate_BHemissivity(BlackHoleMass, accreted_mass, &emissivity, &accretion_time,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  cr_expect_gt(emissivity, 0.0, "Emissivity should be positive");
  cr_expect_gt(accretion_time, 0.0, "Accretion time should be positive");
  cr_expect(isfinite(emissivity), "Emissivity should be finite");
  cr_expect(isfinite(accretion_time), "Accretion time should be finite");
}

Test(blackhole_feedback, calculate_BHemissivity_zero_accretion)
{
  double BlackHoleMass = 1e-3;
  double accreted_mass = 0.0;
  double emissivity, accretion_time, quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  calculate_BHemissivity(BlackHoleMass, accreted_mass, &emissivity, &accretion_time,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  // With zero accretion, emissivity should be zero or very small
  cr_expect_float_eq(accretion_time, 0.0, 1e-15, "Accretion time should be zero for zero accretion");
}

Test(blackhole_feedback, calculate_BHemissivity_mass_scaling)
{
  double accreted_mass = 1e-4;
  double emissivity1, emissivity2, accretion_time1, accretion_time2;
  double quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  // Larger BH mass should have higher luminosity
  calculate_BHemissivity(1e-3, accreted_mass, &emissivity1, &accretion_time1,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);
  calculate_BHemissivity(1e-2, accreted_mass, &emissivity2, &accretion_time2,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  cr_expect_gt(emissivity2, emissivity1, "Larger BH should have higher emissivity");
}

Test(blackhole_feedback, calculate_BHemissivity_accretion_scaling)
{
  double BlackHoleMass = 1e-3;
  double emissivity1, emissivity2, accretion_time1, accretion_time2;
  double quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  // More accretion should lead to higher emissivity
  calculate_BHemissivity(BlackHoleMass, 1e-5, &emissivity1, &accretion_time1,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);
  calculate_BHemissivity(BlackHoleMass, 1e-4, &emissivity2, &accretion_time2,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  cr_expect_gt(emissivity2, emissivity1, "More accretion should increase emissivity");
  cr_expect_gt(accretion_time2, accretion_time1, "More accretion should increase accretion time");
}

// Test ETA constant is correctly defined
Test(blackhole_feedback, eta_constant)
{
  cr_expect_float_eq(ETA, 0.06, 1e-10, "ETA should be 0.06 (6%% efficiency)");
}

// Test velocity scale constant
Test(blackhole_feedback, velocity_scale_constant)
{
  cr_expect_float_eq(VELOCITY_SCALE, 280.0, 1e-10, "VELOCITY_SCALE should be 280 km/s");
}

// Parameterized test for various BH masses
struct bh_mass_param {
  double mass;  // in internal units (1e10 Msun/h)
};

ParameterizedTestParameters(blackhole_feedback, mass_sweep)
{
  static struct bh_mass_param params[] = {
    { 1e-6 },  // 1e4 Msun
    { 1e-5 },  // 1e5 Msun
    { 1e-4 },  // 1e6 Msun
    { 1e-3 },  // 1e7 Msun
    { 1e-2 },  // 1e8 Msun
    { 1e-1 },  // 1e9 Msun
    { 1.0 },   // 1e10 Msun
  };
  return cr_make_param_array(struct bh_mass_param, params, sizeof(params) / sizeof(params[0]));
}

ParameterizedTest(struct bh_mass_param* param, blackhole_feedback, mass_sweep)
{
  double accreted_mass = param->mass * 0.1;  // 10% of BH mass accreted
  double emissivity, accretion_time, quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  calculate_BHemissivity(param->mass, accreted_mass, &emissivity, &accretion_time,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  cr_expect_gt(emissivity, 0.0, "Emissivity should be positive for mass %.2e", param->mass);
  cr_expect_gt(accretion_time, 0.0, "Accretion time should be positive for mass %.2e", param->mass);
  cr_expect(isfinite(emissivity), "Emissivity should be finite for mass %.2e", param->mass);
  cr_expect(isfinite(accretion_time), "Accretion time should be finite for mass %.2e", param->mass);
}

// Test Eddington ratio effect
Test(blackhole_feedback, eddington_ratio_effect)
{
  double BlackHoleMass = 1e-3;
  double accreted_mass = 1e-4;
  double emissivity1, emissivity2, accretion_time1, accretion_time2;
  double quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  // Lower Eddington ratio
  run_globals.params.physics.EddingtonRatio = 0.1;
  calculate_BHemissivity(BlackHoleMass, accreted_mass, &emissivity1, &accretion_time1,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  // Higher Eddington ratio
  run_globals.params.physics.EddingtonRatio = 1.0;
  calculate_BHemissivity(BlackHoleMass, accreted_mass, &emissivity2, &accretion_time2,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  // Higher Eddington ratio should lead to shorter accretion time
  cr_expect_lt(accretion_time2, accretion_time1, "Higher Eddington ratio should reduce accretion time");
}

// Test quasar observation fraction effect
Test(blackhole_feedback, quasar_fobs_effect)
{
  double BlackHoleMass = 1e-3;
  double accreted_mass = 1e-4;
  double emissivity1, emissivity2, accretion_time1, accretion_time2;
  double quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  // Lower fobs
  run_globals.params.physics.quasar_fobs = 0.5;
  calculate_BHemissivity(BlackHoleMass, accreted_mass, &emissivity1, &accretion_time1,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  // Higher fobs
  run_globals.params.physics.quasar_fobs = 1.0;
  calculate_BHemissivity(BlackHoleMass, accreted_mass, &emissivity2, &accretion_time2,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);

  // Emissivity should scale with fobs
  cr_expect_gt(emissivity2, emissivity1, "Higher fobs should increase emissivity");
  cr_expect_float_eq(emissivity2 / emissivity1, 2.0, 0.01, "Emissivity should scale linearly with fobs");
}

// Test numerical stability for extreme values
Test(blackhole_feedback, extreme_mass_ratio)
{
  double emissivity, accretion_time, quasar_luv, quasar_lx, quasar_lx_soft, xray_emissivity;

  // Very small accretion relative to BH mass
  calculate_BHemissivity(1.0, 1e-10, &emissivity, &accretion_time,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);
  cr_expect(isfinite(emissivity), "Should handle small mass ratios");
  cr_expect(isfinite(accretion_time), "Should handle small mass ratios");

  // Large accretion equal to BH mass
  calculate_BHemissivity(1e-3, 1e-3, &emissivity, &accretion_time,
                         &quasar_luv, &quasar_lx, &quasar_lx_soft, &xray_emissivity);
  cr_expect(isfinite(emissivity), "Should handle equal masses");
  cr_expect(isfinite(accretion_time), "Should handle equal masses");
}

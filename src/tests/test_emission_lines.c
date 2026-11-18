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

static void setup_loiii(void)
{
  reset_run_globals();
  configure_default_units_and_params();
  set_OIII_coeffs(1e4);

  static double zz[] = { 7.0 };
  run_globals.ZZ = zz;
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

static const double event_sfr = 0.5;
static const double event_metallicity = 0.01 / 0.7;

// One event on a galaxy with no earlier [O III] this snapshot.
static galaxy_t single_event(double sfr, double metallicity)
{
  galaxy_t gal = make_valid_test_galaxy();
  compute_LOIII(&gal, sfr, metallicity, 0);
  return gal;
}

Test(emission_lines, compute_LOIII_valid_event_gives_finite_positive_outputs)
{
  setup_loiii();
  galaxy_t gal = single_event(event_sfr, event_metallicity);

  cr_expect(isfinite(gal.LOIII));
  cr_expect_gt(gal.LOIII, 0.0);
  cr_expect(isfinite(gal.ionization_param));
  cr_expect_gt(gal.ionization_param, 0.0);
}

// The normalization written out with textbook constants rather than the code's
// macros: 1 Msun/yr forming stars from solar-metallicity gas at T = 1e4 K gives
// L[O III] 5007 = 9.09e41 erg/s. The constants differ from the code's in the 4th
// digit, so this pins every factor but not their rounding.
Test(emission_lines, compute_LOIII_matches_independent_normalization)
{
  setup_loiii();

  // 1 Msun/yr in g/s, then in code units (1.989e43 g per (3.085678e24 cm / 1e5 cm/s)).
  const double sfr_gs = 1.98847e33 / 3.15576e7;
  const double sfr = sfr_gs / (1.989e43 / (3.085678e24 / 1.0e5));
  const double metallicity = 0.0134;

  galaxy_t gal = make_valid_test_galaxy();
  compute_LOIII(&gal, sfr, metallicity, 0);

  // 4000 ionizing photons per baryon, all absorbed in case-B HII regions:
  // recombinations balance ionizations, so n_e n_H V = Q / alpha_B.
  const double photon_rate = 4000.0 * sfr_gs / 1.67262e-24;
  const double emission_measure = photon_rate / 2.6e-13;
  // 12 + log(O/H) = 8.69 at Z_sun = 0.0134.
  const double oxygen_over_hydrogen = pow(10.0, 8.69 - 12.0);
  // Collisional excitation of 1D2 from the ground term, directly and via 1S0 -> 1D2:
  // q = 8.629e-6 / sqrt(T) * gamma * exp(-E / kT) [cm^3/s], with the code's
  // collision-strength fits at 1e4 K (gamma = 0.243 and 0.0321).
  const double q_1d2 = 8.629e-6 / 100.0 * 0.243 * exp(-29169.0 / 1e4);
  const double q_1s0 = 8.629e-6 / 100.0 * 0.0321 * exp(-61207.0 / 1e4) * 1.7 / (1.7 + 0.215);
  const double branching_5007 = 3.52e-5 / (3.52e-5 + 4.57e-6);
  const double photon_energy = 6.62607e-27 * 2.99792458e10 / 5008e-8; // [erg]
  const double expected_erg_s = oxygen_over_hydrogen * emission_measure * (q_1d2 + q_1s0) * branching_5007 *
                                photon_energy * 0.8; // [O III] fills 80% of the HII volume

  cr_expect_float_eq(gal.LOIII, expected_erg_s / 1e40, 1e-3 * expected_erg_s / 1e40);
  cr_expect_float_eq(gal.LOIII, 90.9, 0.1);
}

// Events add their luminosities; the ionization parameter is LOIII-weighted.
Test(emission_lines, compute_LOIII_adds_events_and_weights_ionization_param)
{
  setup_loiii();
  const galaxy_t first = single_event(event_sfr, event_metallicity);
  const galaxy_t second = single_event(3.0 * event_sfr, 0.5 * event_metallicity);

  galaxy_t gal = make_valid_test_galaxy();
  compute_LOIII(&gal, event_sfr, event_metallicity, 0);
  compute_LOIII(&gal, 3.0 * event_sfr, 0.5 * event_metallicity, 0);

  const double total = first.LOIII + second.LOIII;
  const double weighted_u = (first.LOIII * first.ionization_param + second.LOIII * second.ionization_param) / total;
  cr_expect_float_eq(gal.LOIII, total, 1e-12 * total);
  cr_expect_float_eq(gal.ionization_param, weighted_u, 1e-12 * weighted_u);
  cr_expect_gt(fabs(second.ionization_param / first.ionization_param - 1.0), 1e-3);
}

Test(emission_lines, compute_LOIII_is_linear_in_metallicity_and_sfr)
{
  setup_loiii();
  const double reference = single_event(event_sfr, event_metallicity).LOIII;

  const double doubled_z = single_event(event_sfr, 2.0 * event_metallicity).LOIII;
  const double tripled_sfr = single_event(3.0 * event_sfr, event_metallicity).LOIII;
  cr_expect_float_eq(doubled_z, 2.0 * reference, 1e-12 * reference);
  cr_expect_float_eq(tripled_sfr, 3.0 * reference, 1e-12 * reference);
}

// The bubble count cancels in LOIII, so the disc state (including the cold gas
// metals, which the event metallicity replaces) only sets the ionization parameter.
Test(emission_lines, compute_LOIII_luminosity_is_independent_of_disc_state)
{
  setup_loiii();
  const galaxy_t reference = single_event(event_sfr, event_metallicity);

  galaxy_t gal = make_valid_test_galaxy();
  gal.MetalsColdGas = 0.03;
  gal.DiskScaleLength *= 2.0;
  gal.StellarMass *= 3.0;
  gal.Mcool *= 0.5;
  gal.Spin *= 1.5;
  compute_LOIII(&gal, event_sfr, event_metallicity, 0);

  cr_expect_float_eq(gal.LOIII, reference.LOIII, 1e-12 * reference.LOIII);
  cr_expect_gt(fabs(gal.ionization_param / reference.ionization_param - 1.0), 1e-3);
}

// An invalid disc state gives the event no ionization parameter but keeps its luminosity.
Test(emission_lines, compute_LOIII_invalid_disc_state_zeroes_only_ionization_param)
{
  setup_loiii();
  const double reference = single_event(event_sfr, event_metallicity).LOIII;

  galaxy_t gal = make_valid_test_galaxy();
  gal.Mvir = 0.0;
  compute_LOIII(&gal, event_sfr, event_metallicity, 0);
  cr_expect_float_eq(gal.LOIII, reference, 1e-12 * reference);
  cr_expect_eq(gal.ionization_param, 0.0);
}

Test(emission_lines, compute_LOIII_without_sfr_or_metals_adds_nothing)
{
  setup_loiii();
  galaxy_t gal = make_valid_test_galaxy();
  gal.LOIII = 123.0;
  gal.ionization_param = 4.0;

  compute_LOIII(&gal, 0.0, event_metallicity, 0);
  compute_LOIII(&gal, event_sfr, 0.0, 0);
  cr_expect_eq(gal.LOIII, 123.0);
  cr_expect_eq(gal.ionization_param, 4.0);
}

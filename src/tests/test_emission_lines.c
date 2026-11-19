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
  set_OIII_coeffs();
  run_globals.params.physics.ReionNionPhotPerBary = 4000.0;

  static double zz[] = { 7.0 };
  run_globals.ZZ = zz;
}

TestSuite(emission_lines);

Test(emission_lines, set_OIII_coeffs_sets_abundance_branching_ratio_and_prefactor)
{
  reset_run_globals();
  set_OIII_coeffs();

  const double oxygen_abundance = pow(10.0, 8.69 - 12.0) / 0.0134;
  cr_expect_float_eq(run_globals.loiii_params.oxygen_abundance_over_z_sun, oxygen_abundance, 1e-12 * oxygen_abundance);
  // A(5007) / (A(5007) + A(4959)) = 0.745 (NIST)
  cr_expect_float_eq(run_globals.loiii_params.branching_ratio, 1.81e-2 / (1.81e-2 + 6.21e-3), 1e-14);
  // h^2 / (2 pi m_e)^(3/2) / sqrt(k) = 8.629e-6 cm^3 s^-1 K^1/2
  cr_expect_float_eq(run_globals.loiii_params.excitation_prefactor, 8.629e-6, 1e-3 * 8.629e-6);
}

// The Yang et al. (2023) fit, held at its values at 0.016 and 1.6 Z_sun outside them.
Test(emission_lines, electron_temperature_follows_metallicity_fit)
{
  reset_run_globals();
  cr_expect_float_eq(oiii_electron_temperature(Z_SUN), 1.08e4, 1e-8);
  cr_expect_float_eq(oiii_electron_temperature(0.1 * Z_SUN), 1.803e4, 1e-8);

  // Beyond the ends: T4 = 3.556 at 0.016 Z_sun and 1.135 at 1.6 Z_sun.
  const double low_z_end = oiii_electron_temperature(1e-3 * Z_SUN);
  const double high_z_end = oiii_electron_temperature(10.0 * Z_SUN);
  cr_expect_float_eq(low_z_end, 3.5562e4, 1.0);
  cr_expect_float_eq(high_z_end, 1.1349e4, 1.0);
  cr_expect_float_eq(oiii_electron_temperature(0.016 * Z_SUN), low_z_end, 1e-8);
  cr_expect_float_eq(oiii_electron_temperature(1.6 * Z_SUN), high_z_end, 1e-8);
  cr_expect_eq(oiii_electron_temperature(0.0), low_z_end);
  cr_expect_eq(oiii_electron_temperature(NAN), low_z_end);
  cr_expect_eq(oiii_electron_temperature(3.0 * Z_SUN), high_z_end);
}

Test(emission_lines, case_b_recombination_coefficient_scales_as_temperature_to_minus_0p7)
{
  cr_expect_float_eq(case_b_recombination_coefficient(1e4), 2.59e-13, 1e-27);
  cr_expect_float_eq(case_b_recombination_coefficient(3e4), 2.59e-13 * pow(3.0, -0.7), 1e-27);
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
// macros: 1 Msun/yr forming stars from solar-metallicity gas (T = 1.08e4 K) with
// 6000 ionizing photons per stellar baryon gives L[O III] 5007 = 1.466e42 erg/s.
// The constants differ from the code's in the 4th or 5th digit, so this pins every
// factor but not their rounding.
Test(emission_lines, compute_LOIII_matches_independent_normalization)
{
  setup_loiii();
  run_globals.params.physics.ReionNionPhotPerBary = 6000.0;

  // 1 Msun/yr in g/s, then in code units (1.989e43 g per (3.085678e24 cm / 1e5 cm/s)).
  const double sfr_gs = 1.98847e33 / 3.15576e7;
  const double sfr = sfr_gs / (1.989e43 / (3.085678e24 / 1.0e5));
  const double metallicity = 0.0134;

  galaxy_t gal = make_valid_test_galaxy();
  compute_LOIII(&gal, sfr, metallicity, 0);

  // At Z_sun the temperature fit gives T4 = 1.08.
  const double t4 = 1.08;
  const double temperature = 1.08e4;
  // All ionizing photons are absorbed in case-B HII regions: recombinations balance
  // ionizations, so n_e n_H V = Q / alpha_B.
  const double photon_rate = 6000.0 * sfr_gs / 1.67262e-24;
  const double emission_measure = photon_rate / (2.59e-13 * pow(t4, -0.7));
  // 12 + log(O/H) = 8.69 at Z_sun = 0.0134.
  const double oxygen_over_hydrogen = pow(10.0, 8.69 - 12.0);
  // Collisional excitation of 1D2 from the ground term, directly and via 1S0 -> 1D2:
  // q = 8.629e-6 / sqrt(T) * Omega(T) * exp(-E / kT) [cm^3/s], with the Draine (2011)
  // collision-strength fits and 1D2 and 1S0 at 29169 K and 62134 K.
  const double q_1d2 =
    8.629e-6 / sqrt(temperature) * 0.243 * pow(t4, 0.120 + 0.031 * log(t4)) * exp(-29169.0 / temperature);
  const double q_1s0 = 8.629e-6 / sqrt(temperature) * 0.0321 * pow(t4, 0.118 + 0.057 * log(t4)) *
                       exp(-62134.0 / temperature) * 1.71 / (1.71 + 0.215);
  const double branching_5007 = 1.81e-2 / (1.81e-2 + 6.21e-3);
  const double photon_energy = 6.62607e-27 * 2.99792458e10 / 5008e-8; // [erg]
  const double expected_erg_s = oxygen_over_hydrogen * emission_measure * (q_1d2 + q_1s0) * branching_5007 *
                                photon_energy * 0.8; // [O III] fills 80% of the HII volume

  cr_expect_float_eq(gal.LOIII, expected_erg_s / 1e40, 1e-3 * expected_erg_s / 1e40);
  cr_expect_float_eq(gal.LOIII, 146.6, 0.1);
}

// The ionizing photon rate comes from the reionization parameter.
Test(emission_lines, compute_LOIII_scales_with_photons_per_baryon)
{
  setup_loiii();
  const galaxy_t reference = single_event(event_sfr, event_metallicity);

  run_globals.params.physics.ReionNionPhotPerBary = 6000.0;
  const galaxy_t more_photons = single_event(event_sfr, event_metallicity);
  cr_expect_float_eq(more_photons.LOIII, 1.5 * reference.LOIII, 1e-12 * reference.LOIII);
  // q grows as the cube root of the photon rate per bubble.
  cr_expect_float_eq(
    more_photons.ionization_param, cbrt(1.5) * reference.ionization_param, 1e-12 * reference.ionization_param);
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

// Linear in the SFR. In metallicity only where the temperature fit is held fixed
// (Z >= 1.6 Z_sun); below that the gas is hotter, so L/Z rises as Z falls.
Test(emission_lines, compute_LOIII_is_linear_in_sfr_and_follows_temperature_in_metallicity)
{
  setup_loiii();
  const double reference = single_event(event_sfr, event_metallicity).LOIII;
  const double tripled_sfr = single_event(3.0 * event_sfr, event_metallicity).LOIII;
  cr_expect_float_eq(tripled_sfr, 3.0 * reference, 1e-12 * reference);

  const double high_z = 2.0 * Z_SUN;
  const double at_high_z = single_event(event_sfr, high_z).LOIII;
  const double at_double_high_z = single_event(event_sfr, 2.0 * high_z).LOIII;
  cr_expect_float_eq(at_double_high_z, 2.0 * at_high_z, 1e-12 * at_high_z);

  const double low_z = 0.1 * Z_SUN;
  cr_expect_gt(single_event(event_sfr, low_z).LOIII / low_z, 2.0 * at_high_z / high_z);
}

// Through the Stromgren radius, q is proportional to alpha_B^(2/3) at fixed disc
// state and SFR, so it follows the temperature of the gas the stars form from.
Test(emission_lines, compute_LOIII_ionization_param_follows_recombination_coefficient)
{
  setup_loiii();
  const double hot_z = 0.1 * Z_SUN;
  const double cool_z = 2.0 * Z_SUN;
  const double hot_u = single_event(event_sfr, hot_z).ionization_param;
  const double cool_u = single_event(event_sfr, cool_z).ionization_param;

  const double alpha_ratio = case_b_recombination_coefficient(oiii_electron_temperature(hot_z)) /
                             case_b_recombination_coefficient(oiii_electron_temperature(cool_z));
  cr_expect_float_eq(hot_u / cool_u, pow(alpha_ratio, 2.0 / 3.0), 1e-12);
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

/**
 * @file main_blackhole_feedback.c
 * @brief Standalone driver for testing black hole feedback functions
 *
 * This program allows independent testing and exploration of the black hole
 * feedback module without running the full Meraxes simulation.
 *
 * Usage:
 *   ./main_blackhole_feedback [options]
 *
 * Options:
 *   -M, --bh-mass MASS      Black hole mass in 1e10 Msun/h (default: 1e-3 = 1e7 Msun)
 *   -a, --accreted MASS     Accreted mass in 1e10 Msun/h (default: 1e-4 = 1e6 Msun)
 *   -e, --eddington RATIO   Eddington ratio (default: 1.0)
 *   -s, --sweep             Sweep through BH mass range
 *   -h, --help              Show this help message
 *
 * Example:
 *   ./main_blackhole_feedback -M 1e-3 -a 1e-4
 *   ./main_blackhole_feedback --sweep
 */

#define _MAIN
#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "meraxes.h"
#include "physics/blackhole_feedback.h"

void set_units(void);

static void print_usage(const char* progname)
{
  printf("Usage: %s [options]\n", progname);
  printf("\nStandalone driver for testing Meraxes black hole feedback functions.\n");
  printf("\nOptions:\n");
  printf("  -M, --bh-mass MASS      Black hole mass in 1e10 Msun/h (default: 1e-3)\n");
  printf("  -a, --accreted MASS     Accreted mass in 1e10 Msun/h (default: 1e-4)\n");
  printf("  -e, --eddington RATIO   Eddington ratio (default: 1.0)\n");
  printf("  -f, --fobs FRAC         Quasar observation fraction (default: 1.0)\n");
  printf("  -s, --sweep             Sweep through BH mass range\n");
  printf("  -h, --help              Show this help message\n");
  printf("\nExamples:\n");
  printf("  %s -M 1e-3 -a 1e-4\n", progname);
  printf("  %s --sweep\n", progname);
  printf("\nNote: Internal mass units are 1e10 Msun/h\n");
  printf("      e.g., -M 1e-3 corresponds to 1e7 Msun BH\n");
}

static void init_globals(void)
{
  // Set up basic parameters
  run_globals.params.Hubble_h = 0.6774;

  // Set up units (typical cosmological simulation units)
  run_globals.units.UnitLength_in_cm = 3.085678e24;   // 1 Mpc/h
  run_globals.units.UnitMass_in_g = 1.989e43;         // 1e10 Msun/h
  run_globals.units.UnitVelocity_in_cm_per_s = 1.0e5; // 1 km/s

  // Initialize derived units
  set_units();

  // Set physics parameters
  run_globals.params.physics.EddingtonRatio = 1.0;
  run_globals.params.physics.RadioModeEff = 0.08;
  run_globals.params.physics.QuasarModeEff = 0.02;
  run_globals.params.physics.BlackHoleGrowthRate = 0.03;
  run_globals.params.physics.quasar_mode_scaling = 0.0;
  run_globals.params.physics.quasar_fobs = 1.0;
  run_globals.params.physics.SpecIndexUVAGNSoft = 0.61; // input/params/defaults.par values
  run_globals.params.physics.SpecIndexUVAGNHard = 1.70;
  run_globals.params.physics.ReionNionPhotPerBary = 5000.0;
}

static void sweep_bh_emissivity(void)
{
  printf("\n=== Black Hole Emissivity Sweep ===\n\n");

  // BH mass range: 1e4 to 1e10 Msun
  double bh_masses[] = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
  const char* mass_labels[] = { "1e4 Msun", "1e5 Msun", "1e6 Msun", "1e7 Msun", "1e8 Msun", "1e9 Msun", "1e10 Msun" };
  int n_masses = sizeof(bh_masses) / sizeof(bh_masses[0]);

  // Accretion fractions
  double acc_fractions[] = { 0.001, 0.01, 0.1, 1.0 };
  const char* frac_labels[] = { "0.1%", "1%", "10%", "100%" };
  int n_fracs = sizeof(acc_fractions) / sizeof(acc_fractions[0]);

  // Print header
  printf("%-12s", "BH Mass");
  for (int j = 0; j < n_fracs; j++) {
    printf("%18s", frac_labels[j]);
  }
  printf("\n");

  printf("%-12s", "");
  for (int j = 0; j < n_fracs; j++) {
    printf("  %8s %7s", "Emiss", "t_acc");
  }
  printf("\n");

  printf("%-12s", "------------");
  for (int j = 0; j < n_fracs; j++) {
    printf("%18s", "------------------");
  }
  printf("\n");

  // Print emissivity values
  for (int i = 0; i < n_masses; i++) {
    printf("%-12s", mass_labels[i]);
    for (int j = 0; j < n_fracs; j++) {
      double accreted_mass = bh_masses[i] * acc_fractions[j];
      double emissivity, accretion_time, quasar_luv, quasar_lx, quasar_lx_soft;
      calculate_BHemissivity(
        bh_masses[i], accreted_mass, &emissivity, &accretion_time, &quasar_luv, &quasar_lx, &quasar_lx_soft);

      // Convert accretion time to Myr
      double t_acc_myr = accretion_time * run_globals.units.UnitTime_in_Megayears / run_globals.params.Hubble_h;
      printf("  %8.2e %7.1f", emissivity, t_acc_myr);
    }
    printf("\n");
  }

  printf("\n[Note: Emissivity in 1e60 photons, accretion time in Myr]\n");
  printf("[Physics: ETA=%.2f, Eddington ratio=%.1f, fobs=%.1f]\n",
         ETA,
         run_globals.params.physics.EddingtonRatio,
         run_globals.params.physics.quasar_fobs);
}

static void print_constants(void)
{
  printf("\n=== Black Hole Feedback Constants ===\n");
  printf("  ETA (radiative efficiency)     = %.4f (%.1f%%)\n", ETA, ETA * 100);
  printf("  BONDI_HOYLE_COEFFICIENT        = %.4f\n", BONDI_HOYLE_COEFFICIENT);
  printf("  VELOCITY_SCALE                 = %.1f km/s\n", VELOCITY_SCALE);
  printf("  EMISSIVITY_CONVERTOR           = %.5e\n", EMISSIVITY_CONVERTOR);
  printf("  LUMINOSITY_CONVERTOR           = %.4f\n", LUMINOSITY_CONVERTOR);
  printf("  LUV2EMISSIVITY                 = %.6e\n", LUV2EMISSIVITY);
  printf("\n=== Derived Quantities ===\n");
  printf("  G (internal units)             = %.5e\n", run_globals.G);
  printf("  c^2 (internal units)           = %.5e\n", run_globals.Csquare);
  printf("  Eddington timescale            = %.2f (internal units)\n", run_globals.EddingtonTimescale);
  printf("  Eddington timescale            = %.2f Myr\n",
         run_globals.EddingtonTimescale * run_globals.units.UnitTime_in_Megayears / run_globals.params.Hubble_h);
}

int main(int argc, char* argv[])
{
  // Default values
  double bh_mass = 1e-3;       // 1e7 Msun in internal units
  double accreted_mass = 1e-4; // 1e6 Msun
  double eddington_ratio = 1.0;
  double fobs = 1.0;
  int do_sweep = 0;

  // Parse command line options
  static struct option long_options[] = { { "bh-mass", required_argument, 0, 'M' },
                                          { "accreted", required_argument, 0, 'a' },
                                          { "eddington", required_argument, 0, 'e' },
                                          { "fobs", required_argument, 0, 'f' },
                                          { "sweep", no_argument, 0, 's' },
                                          { "help", no_argument, 0, 'h' },
                                          { 0, 0, 0, 0 } };

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "M:a:e:f:sh", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'M':
        bh_mass = atof(optarg);
        break;
      case 'a':
        accreted_mass = atof(optarg);
        break;
      case 'e':
        eddington_ratio = atof(optarg);
        break;
      case 'f':
        fobs = atof(optarg);
        break;
      case 's':
        do_sweep = 1;
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  // Initialize MPI
  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(run_globals.mpi_comm, &run_globals.mpi_rank);
  MPI_Comm_size(run_globals.mpi_comm, &run_globals.mpi_size);

  // Initialize globals
  init_globals();

  // Apply command line physics parameters
  run_globals.params.physics.EddingtonRatio = eddington_ratio;
  run_globals.params.physics.quasar_fobs = fobs;

  printf("=== Meraxes Black Hole Feedback Test Driver ===\n");

  print_constants();

  if (do_sweep) {
    sweep_bh_emissivity();
  } else {
    // Single point evaluation
    double emissivity, accretion_time, quasar_luv, quasar_lx, quasar_lx_soft;
    calculate_BHemissivity(
      bh_mass, accreted_mass, &emissivity, &accretion_time, &quasar_luv, &quasar_lx, &quasar_lx_soft);

    double t_acc_myr = accretion_time * run_globals.units.UnitTime_in_Megayears / run_globals.params.Hubble_h;
    double bh_mass_msun = bh_mass * 1e10 / run_globals.params.Hubble_h;
    double acc_mass_msun = accreted_mass * 1e10 / run_globals.params.Hubble_h;

    printf("\n=== Single Point Evaluation ===\n");
    printf("  Input:\n");
    printf("    BH mass (internal)     = %.3e (1e10 Msun/h)\n", bh_mass);
    printf("    BH mass                = %.3e Msun\n", bh_mass_msun);
    printf("    Accreted mass          = %.3e (1e10 Msun/h)\n", accreted_mass);
    printf("    Accreted mass          = %.3e Msun\n", acc_mass_msun);
    printf("    Mass ratio             = %.2f%%\n", accreted_mass / bh_mass * 100);
    printf("    Eddington ratio        = %.2f\n", eddington_ratio);
    printf("    fobs                   = %.2f\n", fobs);
    printf("\n  Output:\n");
    printf("    Emissivity             = %.6e (1e60 photons)\n", emissivity);
    printf("    Accretion time         = %.3e (internal units)\n", accretion_time);
    printf("    Accretion time         = %.2f Myr\n", t_acc_myr);

    // Compute approximate bolometric luminosity
    double Lbol = sqrt(1. + accreted_mass / bh_mass) * eddington_ratio * bh_mass / run_globals.params.Hubble_h *
                  LUMINOSITY_CONVERTOR;
    printf("    Bolometric luminosity  = %.3e (1e10 Lsun)\n", Lbol);
    printf("    Bolometric luminosity  = %.3e Lsun\n", Lbol * 1e10);
  }

  // Cleanup
  MPI_Finalize();

  return 0;
}

/**
 * @file main_cooling.c
 * @brief Standalone driver for testing cooling functions
 *
 * This program allows independent testing and exploration of the cooling
 * module without running the full Meraxes simulation.
 *
 * Usage:
 *   ./main_cooling [options]
 *
 * Options:
 *   -c, --cooling-dir DIR   Path to cooling functions directory (default: <source>/input/cooling_functions)
 *   -T, --temperature LOGT  Log10 temperature in Kelvin (default: 6.0)
 *   -Z, --metallicity LOGZ  Log10 metallicity (default: -1.7, ~solar)
 *   -s, --sweep             Sweep through temperature and metallicity range
 *   -h, --help              Show this help message
 *
 * Example:
 *   ./main_cooling -T 5.5 -Z -2.0
 *   ./main_cooling --sweep
 */

#define _MAIN
#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "meraxes.h"
#include "physics/cooling.h"

// Forward declarations from cooling.c
extern void read_cooling_functions(void);
extern double interpolate_cooling_rate(double logTemp, double logZ);

#if USE_MINI_HALOS
extern double LTE_Mcool(double Temp, double nH);
extern double Mcool_SV(double redshift, int n);
#endif

static void print_usage(const char* progname)
{
  printf("Usage: %s [options]\n", progname);
  printf("\nStandalone driver for testing Meraxes cooling functions.\n");
  printf("\nOptions:\n");
  printf("  -c, --cooling-dir DIR   Path to cooling functions directory\n");
  printf("                          (default: %s)\n", COOLING_FUNCS_DIR);
  printf("  -T, --temperature LOGT  Log10 temperature in Kelvin (default: 6.0)\n");
  printf("  -Z, --metallicity LOGZ  Log10 metallicity (default: -1.7, ~solar)\n");
  printf("  -s, --sweep             Sweep through temperature and metallicity range\n");
#if USE_MINI_HALOS
  printf("  -m, --mini-halos        Test mini-halo cooling functions\n");
#endif
  printf("  -h, --help              Show this help message\n");
  printf("\nExamples:\n");
  printf("  %s -T 5.5 -Z -2.0\n", progname);
  printf("  %s --sweep\n", progname);
  printf("  %s -c /path/to/cooling_functions -T 7.0\n", progname);
}

static void sweep_cooling_rates(void)
{
  printf("\n=== Cooling Rate Sweep ===\n\n");

  // Temperature range: 10^4 to 10^8.5 K
  double logT_values[] = { 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5 };
  int n_temps = sizeof(logT_values) / sizeof(logT_values[0]);

  // Metallicity range (relative to solar Z=0.02, so logZ_solar = -1.7)
  double logZ_values[] = { -6.7, -4.7, -3.7, -3.2, -2.7, -2.2, -1.7, -1.2 };
  const char* Z_labels[] = { "1e-5 Zsun", "1e-3 Zsun", "1e-2 Zsun", "0.03 Zsun",
                             "0.1 Zsun",  "0.3 Zsun",  "1.0 Zsun",  "3.0 Zsun" };
  int n_metals = sizeof(logZ_values) / sizeof(logZ_values[0]);

  // Print header
  printf("%-8s", "logT");
  for (int j = 0; j < n_metals; j++) {
    printf("%12s", Z_labels[j]);
  }
  printf("\n");

  // Print separator
  printf("%-8s", "--------");
  for (int j = 0; j < n_metals; j++) {
    printf("%12s", "------------");
  }
  printf("\n");

  // Print cooling rates
  for (int i = 0; i < n_temps; i++) {
    printf("%-8.1f", logT_values[i]);
    for (int j = 0; j < n_metals; j++) {
      double rate = interpolate_cooling_rate(logT_values[i], logZ_values[j]);
      printf("%12.3e", rate);
    }
    printf("\n");
  }

  printf("\n[Note: Cooling rates in erg cm^3 s^-1 (normalized lambda)]\n");
}

#if USE_MINI_HALOS
static void test_mini_halo_cooling(void)
{
  printf("\n=== Mini-Halo Cooling Functions ===\n\n");

  // Test LTE molecular cooling
  printf("LTE Molecular Cooling (H2):\n");
  printf("%-12s %-12s %-15s\n", "T [K]", "nH [cm^-3]", "Lambda_LTE");
  printf("%-12s %-12s %-15s\n", "------------", "------------", "---------------");

  double temps[] = { 300, 500, 1000, 2000, 5000, 8000 };
  double densities[] = { 0.1, 1.0, 10.0 };

  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 3; j++) {
      double LTE = LTE_Mcool(temps[i], densities[j]);
      printf("%-12.0f %-12.1f %-15.3e\n", temps[i], densities[j], LTE);
    }
  }

  // Test streaming velocity cooling mass
  printf("\nStreaming Velocity Cooling Mass:\n");
  printf("%-12s %-12s %-15s\n", "Redshift", "n*sigma", "Mcool [Msun]");
  printf("%-12s %-12s %-15s\n", "------------", "------------", "---------------");

  double redshifts[] = { 10, 15, 20, 25, 30 };
  int n_sigmas[] = { 0, 1, 2, 3 };

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      double Mcool = Mcool_SV(redshifts[i], n_sigmas[j]);
      printf("%-12.0f %-12d %-15.3e\n", redshifts[i], n_sigmas[j], Mcool);
    }
  }
}
#endif

int main(int argc, char* argv[])
{
  // Default values
  char cooling_dir[STRLEN] = COOLING_FUNCS_DIR;
  double logT = 6.0;
  double logZ = log10(0.02); // Solar metallicity
  int do_sweep = 0;
#if USE_MINI_HALOS
  int do_mini_halos = 0;
#endif

  // Parse command line options
  static struct option long_options[] = {
    { "cooling-dir", required_argument, 0, 'c' },
    { "temperature", required_argument, 0, 'T' },
    { "metallicity", required_argument, 0, 'Z' },
    { "sweep", no_argument, 0, 's' },
#if USE_MINI_HALOS
    { "mini-halos", no_argument, 0, 'm' },
#endif
    { "help", no_argument, 0, 'h' },
    { 0, 0, 0, 0 }
  };

#if USE_MINI_HALOS
  const char* optstring = "c:T:Z:smh";
#else
  const char* optstring = "c:T:Z:sh";
#endif

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
    switch (opt) {
      case 'c':
        strncpy(cooling_dir, optarg, STRLEN - 1);
        cooling_dir[STRLEN - 1] = '\0';
        break;
      case 'T':
        logT = atof(optarg);
        break;
      case 'Z':
        logZ = atof(optarg);
        break;
      case 's':
        do_sweep = 1;
        break;
#if USE_MINI_HALOS
      case 'm':
        do_mini_halos = 1;
        break;
#endif
      case 'h':
        print_usage(argv[0]);
        return 0;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  // Initialize MPI (required for reading cooling tables)
  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(run_globals.mpi_comm, &run_globals.mpi_rank);
  MPI_Comm_size(run_globals.mpi_comm, &run_globals.mpi_size);

  // Set up cooling functions directory
  strncpy(run_globals.params.CoolingFuncsDir, cooling_dir, STRLEN - 1);
  run_globals.params.CoolingFuncsDir[STRLEN - 1] = '\0';

  printf("=== Meraxes Cooling Function Test Driver ===\n");
  printf("Cooling functions directory: %s\n", run_globals.params.CoolingFuncsDir);

  // Read cooling function tables
  printf("Reading cooling function tables...\n");
  read_cooling_functions();
  printf("Done.\n");

  if (do_sweep) {
    sweep_cooling_rates();
  }
#if USE_MINI_HALOS
  else if (do_mini_halos) {
    test_mini_halo_cooling();
  }
#endif
  else {
    // Single point evaluation
    double rate = interpolate_cooling_rate(logT, logZ);

    printf("\n=== Single Point Evaluation ===\n");
    printf("  log10(T/K)    = %.3f\n", logT);
    printf("  T             = %.3e K\n", pow(10, logT));
    printf("  log10(Z)      = %.3f\n", logZ);
    printf("  Z/Zsun        = %.3e\n", pow(10, logZ) / 0.02);
    printf("  Cooling rate  = %.6e erg cm^3 s^-1\n", rate);
    printf("  log10(Lambda) = %.3f\n", log10(rate));
  }

  // Cleanup
  MPI_Finalize();

  return 0;
}

#include <assert.h>
#include <math.h>
#include <string.h>

#if USE_STOCHASTICITY
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_rng.h>
#endif

#include "debug.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "reionization.h"

void myexit(int signum)
{
  fprintf(stderr, "\n");
  fprintf(stderr, "================================================================================\n");
  fprintf(stderr, "PROGRAM TERMINATION REPORT\n");
  fprintf(stderr, "================================================================================\n");
  fprintf(stderr, "MPI Task:           %d / %d\n", run_globals.mpi_rank, run_globals.mpi_size);
  fprintf(stderr, "\n");
  fprintf(stderr, "For debugging information, check:\n");
  fprintf(stderr, "  - Compilation flags: Run 'cmake --version' and check CMakeCache.txt\n");
  fprintf(stderr, "  - Runtime parameters: Check input/params/*.par\n");
  fprintf(stderr, "  - Output logs: Check output/ directory\n");
  fprintf(stderr, "  - Detailed error: Enable DEBUG flag in CMakeLists.txt\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "For support, provide:\n");
  fprintf(stderr, "  - Full error message and backtrace\n");
  fprintf(stderr, "  - Your parameter file (input/params/*.par)\n");
  fprintf(stderr, "  - System configuration (OS, compiler version)\n");
  fprintf(stderr, "  - Complete stdout/stderr logs\n");
  fprintf(stderr, "================================================================================\n");
  fprintf(stderr, "\n");
  mpi_debug_here();
  cleanup();
  MPI_Finalize();
  exit(signum);
}

double calc_metallicity(double total_gas, double metals)
{
  double Z;

  if ((total_gas > 0) && (metals > 0))
    Z = metals / total_gas;
  else
    Z = 0.0;

  CLAMP_0_1(Z);

  return Z;
}

#if USE_STOCHASTICITY
double apply_lognormal_scatter(double mean_esc, double scatter_dex)
{
  double sigma_ln;
  double zeta;
  double u;
  double g;
  double scattered_fesc;

  if (scatter_dex <= 0.0 || mean_esc <= 0.0)
    return mean_esc;

  sigma_ln = log(10.0) * scatter_dex;
  zeta = log(mean_esc);
  u = gsl_rng_uniform(run_globals.random_generator);
  g = gsl_cdf_ugaussian_Pinv(u);

  scattered_fesc = exp(zeta + sigma_ln * g);
  return scattered_fesc;
}
#endif

int compare_ints(const void* a, const void* b)
{
  return *((int*)a) - *((int*)b);
}

int compare_longs(const void* a, const void* b)
{
  long value = (*((long*)a) - *((long*)b));

  if (value > 0)
    return 1;
  else if (value < 0)
    return -1;
  else
    return 0;
}

int compare_floats(const void* a, const void* b)
{
  float value = *(float*)a - *(float*)b;

  if (value > 0)
    return 1;
  else if (value < 0)
    return -1;
  else
    return 0;
}

int compare_doubles(const void* a, const void* b)
{
  double value = *(double*)a - *(double*)b;

  if (value > 0)
    return 1;
  else if (value < 0)
    return -1;
  else
    return 0;
}

int compare_ptrdiff(const void* a, const void* b)
{
  ptrdiff_t result = *(ptrdiff_t*)a - *(ptrdiff_t*)b;

  return (int)result;
}

int compare_int_long(const void* a, const void* b)
{
  long value = (*((int*)a) - *((long*)b));

  if (value > 0)
    return 1;
  else if (value < 0)
    return -1;
  else
    return 0;
}

int compare_slab_assign(const void* a, const void* b)
{
  int value = ((gal_to_slab_t*)a)->slab_ind - ((gal_to_slab_t*)b)->slab_ind;

  return value != 0 ? value : ((gal_to_slab_t*)a)->index - ((gal_to_slab_t*)b)->index;
}

static inline float apply_pbc_disp(float delta)
{
  float box_size = (float)(run_globals.params.BoxSize);

  if (fabs(delta - box_size) < fabs(delta))
    delta -= box_size;
  if (fabs(delta + box_size) < fabs(delta))
    delta += box_size;

  return delta;
}

float apply_pbc_pos(float x)
{
  float box_size = (float)(run_globals.params.BoxSize);

  if (x >= box_size)
    x -= box_size;
  else if (x < 0.0)
    x += box_size;

  return x;
}

int searchsorted(void* val,
                 void* arr,
                 int count,
                 size_t size,
                 int (*compare)(const void*, const void*),
                 int imin,
                 int imax)
{
  // check if we need to init imin and imax
  if ((imax < 0) && (imin < 0)) {
    imin = 0;
    imax = count - 1;
  }

  // test if we have found the result
  if ((imax - imin) < 0)
    return imax;
  else {
    // calculate midpoint to cut set in half
    int imid = imin + ((imax - imin) / 2);
    void* arr_val = (void*)(((char*)arr + imid * size));

    // three-way comparison
    if (compare(arr_val, val) > 0)
      // key is in lower subset
      return searchsorted(val, arr, count, size, compare, imin, imid - 1);
    else if (compare(arr_val, val) < 0)
      // key is in upper subset
      return searchsorted(val, arr, count, size, compare, imid + 1, imax);
    else
      // key has been found
      return imid;
  }
}

int pos_to_ngp(double x, double side, int nx)
{
  int ind = (int)nearbyint(x / side * (double)nx);

  if (ind > nx - 1)
    ind = 0;

  assert(ind > -1);

  return ind;
}

float comoving_distance(float a[3], float b[3])
{
  float dx = apply_pbc_disp(a[0] - b[0]);
  float dy = apply_pbc_disp(a[1] - b[1]);
  float dz = apply_pbc_disp(a[2] - b[2]);

  float dist = sqrtf(dx * dx + dy * dy + dz * dz);

  assert(dist <= (sqrtf(3.0) / 2.0 * run_globals.params.BoxSize));

  return dist;
}

double accurate_sumf(float* arr, int n)
{
  // inplace reorder and sum
  qsort(arr, (size_t)n, sizeof(float), compare_floats);

  double total = 0;
  for (int ii = 0; ii < n; ii++)
    total += (double)(arr[ii]);

  return total;
}

int grid_index(int i, int j, int k, int dim, index_type type)
{
  int ind = -1;

  switch (type) {
    case INDEX_PADDED:
      ind = k + (2 * (dim / 2 + 1)) * (j + dim * i);
      break;
    case INDEX_REAL:
      ind = k + dim * (j + dim * i);
      break;
    case INDEX_COMPLEX_HERM:
      ind = k + (dim / 2 + 1) * (j + dim * i);
      break;
    default:
      mlog_error("Unknown indexing type.");
      break;
  }

  return ind;
}

int grid_index_LC(int i, int j, int k, int dim, int dim_LC)
{
  int ind = -1;

  ind = k + dim_LC * (j + dim * i);

  return ind;
}

int grid_index_smoothedSFR(int radii, int i, int j, int k, int filter_steps, int dim)
{
  int ind = -1;

  ind = radii + filter_steps * (k + dim * (j + dim * i));

  return ind;
}

/// Numpy style isclose()
int isclosef(float a,
             float b,
             float rel_tol, ///< [in] = -1 for Numpy default
             float abs_tol) ///< [in] = -1 for Numpy default
{
  if (abs_tol < 0)
    abs_tol = 1e-8; ///< Numpy default
  if (rel_tol < 0)
    rel_tol = 1e-5; ///< Numpy default
  return fabs(a - b) <= (abs_tol + rel_tol * fabs(b));
}

int find_original_index(int index, int* lookup, int n_mappings)
{
  int new_index = -1;
  int* pointer = bsearch(&index, lookup, (size_t)n_mappings, sizeof(int), compare_ints);

  if (pointer)
    new_index = (int)(pointer - lookup);

  return new_index;
}

double interp(double xp, double* x, double* y, int nPts)
{
  /* Interpolate a given points */
  int idx0, idx1;
  if ((xp < x[0]) || (xp > x[nPts - 1])) {
    mlog_error("Beyond the interpolation region!");
    ABORT(EXIT_FAILURE);
  }
  if (xp == x[nPts - 1])
    return y[nPts - 1];
  else {
    idx0 = searchsorted(&xp, x, nPts, sizeof(double), compare_doubles, -1, -1);
    if (x[idx0] == xp)
      return y[idx0];
    idx1 = idx0 + 1;
    return y[idx0] + (y[idx1] - y[idx0]) * (xp - x[idx0]) / (x[idx1] - x[idx0]);
  }
}

double trapz_table(double* y, double* x, int nPts, double a, double b)
{
  /* Integrate tabular data from a to b */
  int i;
  int idx0, idx1;
  double ya, yb;
  double sum;
  if (x[0] > a) {
    mlog_error("Integration range is beyond the tabular data!");
    ABORT(EXIT_FAILURE);
  }
  if (x[nPts - 1] < b) {
    mlog_error("Integration range is beyond the tabular data!");
    ABORT(EXIT_FAILURE);
  }
  if (a > b) {
    mlog_error("Integration range is wrong!");
    ABORT(EXIT_FAILURE);
  }
  idx0 = searchsorted(&a, x, nPts, sizeof(double), compare_doubles, -1, -1);
  idx1 = idx0 + 1;

  ya = y[idx0] + (y[idx1] - y[idx0]) * (a - x[idx0]) / (x[idx1] - x[idx0]);
  if (b <= x[idx1]) {
    yb = y[idx0] + (y[idx1] - y[idx0]) * (b - x[idx0]) / (x[idx1] - x[idx0]);
    return (b - a) * (yb + ya) / 2.;
  } else
    sum = (x[idx1] - a) * (y[idx1] + ya) / 2.;

  for (i = idx1; i < nPts - 1; ++i) {
    if (x[i + 1] < b)
      sum += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.;
    else if (x[i] < b) {
      yb = y[i] + (y[i + 1] - y[i]) * (b - x[i]) / (x[i + 1] - x[i]);
      sum += (b - x[i]) * (yb + y[i]) / 2.;
    } else
      break;
  }
  return sum;
}

bool check_for_flag(int flag, int tree_flags)
{
  if ((tree_flags & flag) == flag)
    return true;
  else
    return false;
}

// Reads this rank's current resident set size from /proc/self/status (Linux-only)
// and MPI_Reduces it to rank 0, so a single per-checkpoint log line reports both
// the worst-offending rank and the total RAM footprint across the whole job.
// `ngal` is this rank's current live galaxy_t count (pass 0 if not tracked at the
// call site); it is summed across ranks and multiplied by sizeof(galaxy_t) to give
// an independent estimate of how much of the reported RSS is the galaxy array
// itself, versus halo storage / everything else.
void log_memory_usage(const char* label, int snapshot, int ngal)
{
  static bool printed_struct_sizes = false;
  if (!printed_struct_sizes) {
    mlog("MEMORY :: sizeof(halo_t) = %zu bytes, sizeof(fof_group_t) = %zu bytes, sizeof(galaxy_t) = %zu bytes",
         MLOG_MESG,
         sizeof(halo_t),
         sizeof(fof_group_t),
         sizeof(galaxy_t));
    printed_struct_sizes = true;
  }

  double vmrss_kb = 0.0;
  FILE* status_file = fopen("/proc/self/status", "r");
  if (status_file != NULL) {
    char line[256];
    while (fgets(line, sizeof(line), status_file) != NULL) {
      if (strncmp(line, "VmRSS:", 6) == 0) {
        sscanf(line + 6, "%lf", &vmrss_kb);
        break;
      }
    }
    fclose(status_file);
  }

  double vmrss_gb = vmrss_kb / (1024.0 * 1024.0);
  double max_rss_gb = 0.0;
  double sum_rss_gb = 0.0;
  MPI_Reduce(&vmrss_gb, &max_rss_gb, 1, MPI_DOUBLE, MPI_MAX, 0, run_globals.mpi_comm);
  MPI_Reduce(&vmrss_gb, &sum_rss_gb, 1, MPI_DOUBLE, MPI_SUM, 0, run_globals.mpi_comm);

  long ngal_local = (long)ngal;
  long ngal_total = 0;
  MPI_Reduce(&ngal_local, &ngal_total, 1, MPI_LONG, MPI_SUM, 0, run_globals.mpi_comm);
  double galaxy_gb = ((double)ngal_total * (double)sizeof(galaxy_t)) / (1024.0 * 1024.0 * 1024.0);

  // Sum this rank's halo/FOF-group counts over every snapshot slot loaded so
  // far (SnapshotTreesInfo[0..snapshot], clamped to what's actually been
  // allocated). Under FlagInteractive/FlagMCMC every snapshot's halos stay
  // resident for the whole run (needed for descendant lookups), so this is
  // the running total that actually drives RSS -- not just this snapshot's.
  long nhalo_local = 0;
  long nfof_local = 0;
  if (run_globals.SnapshotTreesInfo != NULL) {
    int max_ii = snapshot;
    if (max_ii > run_globals.NStoreSnapshots - 1)
      max_ii = run_globals.NStoreSnapshots - 1;
    for (int ii = 0; ii <= max_ii; ii++) {
      nhalo_local += run_globals.SnapshotTreesInfo[ii].n_halos;
      nfof_local += run_globals.SnapshotTreesInfo[ii].n_fof_groups;
    }
  }
  long nhalo_total = 0;
  long nfof_total = 0;
  MPI_Reduce(&nhalo_local, &nhalo_total, 1, MPI_LONG, MPI_SUM, 0, run_globals.mpi_comm);
  MPI_Reduce(&nfof_local, &nfof_total, 1, MPI_LONG, MPI_SUM, 0, run_globals.mpi_comm);
  double halo_gb = ((double)nhalo_total * (double)sizeof(halo_t)) / (1024.0 * 1024.0 * 1024.0);
  double fof_gb = ((double)nfof_total * (double)sizeof(fof_group_t)) / (1024.0 * 1024.0 * 1024.0);

  mlog("MEMORY [%s] snapshot %d :: max rank RSS = %.2f GB, total RSS (sum over ranks) = %.2f GB, "
       "live galaxies = %ld (est. galaxy_t footprint = %.2f GB), "
       "halos loaded so far = %ld (est. halo_t footprint = %.2f GB), "
       "FOF groups loaded so far = %ld (est. fof_group_t footprint = %.2f GB)",
       MLOG_MESG,
       label,
       snapshot,
       max_rss_gb,
       sum_rss_gb,
       ngal_total,
       galaxy_gb,
       nhalo_total,
       halo_gb,
       nfof_total,
       fof_gb);
}

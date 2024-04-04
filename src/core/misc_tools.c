#include <hdf5.h>
#include <hdf5_hl.h>

#include <assert.h>
#include <math.h>

#include "cn_exceptions.h"
#include "debug.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "reionization.h"

void myexit(int signum)
{
  fprintf(stderr, "Task: %d\tis exiting.\n\n\n", run_globals.mpi_rank);
  cn_quote();
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

  if (Z < 0)
    Z = 0.0;
  if (Z > 1)
    Z = 1.0;

  return Z;
}

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

#if USE_SCALING_REL
static double Delta[NDelta];
static double PopIIIz0[NDelta];
static double PopIIz0[NDelta];
static double PopIIIParams[NDelta * NPopIIIPars];
static double PopIIParams[NDelta * NPopIIPars];

void read_scaling_rel_tables(void)
{
  if (run_globals.mpi_rank == 0) {
    hid_t fdd;
    char fname[STRLEN];
    
    sprintf(fname, "%s/ScalingParameter_%d.hdf5", run_globals.params.ScalingRelDir,run_globals.params.ScalingRelModel);
    fdd = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    // Read Delta [Overdensity]
    H5LTread_dataset_double(fdd, "Delta", Delta);
    // Polinomial parameters for Pop. III
    H5LTread_dataset_double(fdd, "PopIIIPar", PopIIIParams);
    // z0 for Pop. III
    H5LTread_dataset_double(fdd, "PopIIIz0", PopIIIz0);
    // Polinomial parameters for Pop. II
    H5LTread_dataset_double(fdd, "PopIIPar", PopIIParams);
    // z0 for Pop. II
    H5LTread_dataset_double(fdd, "PopIIz0", PopIIz0);
    H5Fclose(fdd);
  }

  // Broadcast the values to all cores
  MPI_Bcast(Delta, sizeof(Delta), MPI_BYTE, 0, run_globals.mpi_comm);
  MPI_Bcast(PopIIIParams, sizeof(PopIIIParams), MPI_BYTE, 0, run_globals.mpi_comm);
  MPI_Bcast(PopIIParams, sizeof(PopIIParams), MPI_BYTE, 0, run_globals.mpi_comm);
  MPI_Bcast(PopIIIz0, sizeof(PopIIIz0), MPI_BYTE, 0, run_globals.mpi_comm);
  MPI_Bcast(PopIIz0, sizeof(PopIIz0), MPI_BYTE, 0, run_globals.mpi_comm);
}

void initialize_ScalingRel()
// A bit hardcoding now, it might be better to upload a table and do
// things in a nicer way.
{
  int Scaling_Model = run_globals.params.ScalingRelModel;
  int n_snaps = run_globals.params.SnaplistLength;
  
  double MuIII;
  double SigmaIII;
  double MuII;
  double SigmaII;
  
  run_globals.NormIII = malloc(sizeof(float) * (NDelta * n_snaps));
  run_globals.NormII = malloc(sizeof(float) * (NDelta * n_snaps));

  if (run_globals.mpi_rank == 0) {
    for (int i_delta = 0; i_delta < NDelta; ++i_delta) {
      double z0_II = PopIIz0[i_delta];
      double a0_II = PopIIParams[i_delta * 6 + 5];
      double a1_II = PopIIParams[i_delta * 6 + 4];
      double a2_II = PopIIParams[i_delta * 6 + 3];
      double a3_II = PopIIParams[i_delta * 6 + 2];
      double a4_II = PopIIParams[i_delta * 6 + 1];
      double a5_II = PopIIParams[i_delta * 6 + 0];
  
      double z0_III = PopIIIz0[i_delta];
      double a0_III = PopIIIParams[i_delta * 6 + 5];
      double a1_III = PopIIIParams[i_delta * 6 + 4];
      double a2_III = PopIIIParams[i_delta * 6 + 3];
      double a3_III = PopIIIParams[i_delta * 6 + 2];
      double a4_III = PopIIIParams[i_delta * 6 + 1];
      double a5_III = PopIIIParams[i_delta * 6 + 0];
      for (int snap = 0; snap < n_snaps; snap++) {
        run_globals.NormIII[i_delta, snap] = NormFitting_Function(snap, a0_III, a1_III, a2_III, a3_III, a4_III, a5_III, z0_III);
        run_globals.NormII[i_delta, snap] = NormFitting_Function(snap, a0_II, a1_II, a2_II, a3_II, a4_II, a5_II, z0_II);
      }
    }
  }
  
  MPI_Bcast(run_globals.NormIII, NDelta * n_snaps, MPI_FLOAT, 0, run_globals.mpi_comm); // MAYBE NDelta * SnaplistLength???
  MPI_Bcast(run_globals.NormII, NDelta * n_snaps, MPI_FLOAT, 0, run_globals.mpi_comm);
  
  switch (Scaling_Model) { 
    case 1:
      MuIII = -5.0;
      SigmaIII = 0.6;
      MuII = -3.5;
      SigmaII = 0.6;
      break;
    default:
      mlog_error("Unrecognised value for ScalingModel! Defaulting to 1.");
      MuIII = -5.0;
      SigmaIII = 0.6;
      MuII = -3.5;
      SigmaII = 0.6;
      break;
  }
  
  run_globals.mu_MCIII = MuIII;
  run_globals.mu_MCII = MuII;
  run_globals.sigma_MCIII = SigmaIII;
  run_globals.sigma_MCII = SigmaII;
  mlog("Init Scaling Rel quantities: muIII = %f, muII = %f, SigmaIII = %f, SigmaII = %f",
       MLOG_MESG,
       run_globals.mu_MCIII,
       run_globals.mu_MCII,
       run_globals.sigma_MCIII,
       run_globals.sigma_MCII);
}

double NormFitting_Function(double x, double a0, double a1, double a2, double a3, double a4, double a5, double x0)
{
  if (x > x0)
    return 0.0;
  else
    return a5 * pow(x,5) + a4 * pow(x,4) + a3 * pow(x,3) + a2 * pow(x,2) + a1 * x + a0;
}

double NormalRandNum(double ave, double std) // Generate normal random number
{

  double U1 = gsl_rng_uniform(run_globals.random_generator);
  double U2 = gsl_rng_uniform(run_globals.random_generator);

  double RanNorm = sqrt(-2*log(U1)) * cos(2*M_PI*U2);
  
  return ave + std * RanNorm;
}

int Find_DeltaIndex(double DeltaVal)
{
  int DeltaIndex;
  double DDelta = Delta[1] - Delta[0];
  for (int i = 0; i < NDelta; i++) {
    if (i == 0) {
      if (DeltaVal < Delta[i] + DDelta / 2.0); {
        DeltaIndex = i;
        break;
      }
    }
    else if (i == NDelta - 1) {
      if (DeltaVal >= Delta[i] - DDelta / 2.0) {
        DeltaIndex = i;
        break;
      }
      else
        mlog_error("Haven't found the Delta index");
    }
    else {
      if (fabs(DeltaVal - Delta[i]) < DDelta / 2.0) {
        DeltaIndex = i;
        break;
      }
    }
  }
  return DeltaIndex;  
}
#endif

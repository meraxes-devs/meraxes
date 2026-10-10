#include <assert.h>
#include <hdf5_hl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "dist_func.h"
#include "meraxes.h"

void df_init(distribution_function_t* df, double x_min, double x_max, int bins_per_dex, const char* description)
{
  assert(df != NULL);
  assert(x_max > x_min);
  assert(bins_per_dex > 0);

  df->x_min = x_min;
  df->x_max = x_max;

  // Logarithmic binning: bins per dex (or per magnitude for UVLF/DustyLF)
  df->n_bins = df_n_bins(x_min, x_max, bins_per_dex);
  df->bin_width = (x_max - x_min) / df->n_bins;

  // Allocate memory for bins and counts
  df->bins = (distribution_bin_t*)malloc(df->n_bins * sizeof(distribution_bin_t));
  df->bin_counts = (double*)calloc(df->n_bins, sizeof(double));
  df->bin_variance = (double*)calloc(df->n_bins, sizeof(double));
  assert(df->bins != NULL);
  assert(df->bin_counts != NULL);
  assert(df->bin_variance != NULL);

  // Store description
  if (description != NULL) {
    strncpy(df->description, description, 255);
    df->description[255] = '\0';
  } else {
    strcpy(df->description, "Distribution Function");
  }

  // Initialize bin centers
  for (int i = 0; i < df->n_bins; i++) {
    df->bins[i].center = x_min + (i + 0.5) * df->bin_width;
    df->bins[i].number_density = 0.0;
    df->bins[i].uncertainty = 0.0;
  }
}

int df_n_bins(double x_min, double x_max, int bins_per_dex)
{
  // Round rather than truncate: a range times bins_per_dex that should be an
  // integer can evaluate to just below it in floating point. A range narrower
  // than one bin still gets one, so bin_width stays finite.
  int n_bins = (int)lround((x_max - x_min) * bins_per_dex);
  return (n_bins > 0) ? n_bins : 1;
}

int df_bin_index(const distribution_function_t* df, double val)
{
  if (!(val >= df->x_min && val <= df->x_max))
    return -1;

  // (val - x_min) / bin_width reaches n_bins at val == x_max, and can round
  // up to it just below; both belong in the last bin.
  int idx = (int)((val - df->x_min) / df->bin_width);
  return (idx < df->n_bins) ? idx : df->n_bins - 1;
}

void df_free(distribution_function_t* df)
{
  assert(df != NULL);
  if (df->bins != NULL) {
    free(df->bins);
    df->bins = NULL;
  }
  if (df->bin_counts != NULL) {
    free(df->bin_counts);
    df->bin_counts = NULL;
  }
  if (df->bin_variance != NULL) {
    free(df->bin_variance);
    df->bin_variance = NULL;
  }
}

void df_mpi_reduce(distribution_function_t* df, int mpi_rank, int mpi_size)
{
  assert(df != NULL);

  if (mpi_size > 1) {
    // Sum counts across all processes to rank 0 only
    if (mpi_rank == 0) {
      // Rank 0 uses MPI_IN_PLACE to reduce in-place into df->bin_counts
      MPI_Reduce(MPI_IN_PLACE, df->bin_counts, df->n_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, df->bin_variance, df->n_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
      // Other ranks send their data (recvbuf is ignored for non-root ranks)
      MPI_Reduce(df->bin_counts, NULL, df->n_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(df->bin_variance, NULL, df->n_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
  }
}

void df_write_hdf5(hid_t file_id,
                   const char* group_name,
                   const distribution_function_t* df,
                   const char* dataset_prefix,
                   const char* units)
{
  assert(df != NULL);
  assert(df->bins != NULL);

  // Open group
  hid_t group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);
  if (group_id < 0) {
    mlog_error("Failed to open group %s for writing distribution function", group_name);
    return;
  }

  // Create 2D array: (n_bins, 3) for [center, density, uncertainty]
  hsize_t dims[2] = { (hsize_t)df->n_bins, 3 };
  double* data = (double*)malloc(df->n_bins * 3 * sizeof(double));
  assert(data != NULL);

  // Check if Bernoulli variance has been accumulated (sum > 0)
  double total_variance = 0.0;
  for (int i = 0; i < df->n_bins; i++) {
    total_variance += df->bin_variance[i];
  }
  int use_bernoulli = (total_variance > 0.0);

  for (int i = 0; i < df->n_bins; i++) {
    // Compute number density and uncertainty from bin counts
    double count = df->bin_counts[i];
    double bin_width = df->bin_width;

    // Column 0: bin center
    data[i * 3 + 0] = df->bins[i].center;

    // Column 1: number density N / (volume * bin_width)
    data[i * 3 + 1] = count / (df->volume * bin_width);

    // Column 2: uncertainty
    if (use_bernoulli) {
      // Bernoulli uncertainty: sqrt(sum(p*(1-p))) / (volume * bin_width)
      double var = df->bin_variance[i];
      data[i * 3 + 2] = sqrt(var) / (df->volume * bin_width);
    } else {
      // Poisson uncertainty: sqrt(N) / (volume * bin_width)
      if (count > 0) {
        data[i * 3 + 2] = sqrt(count) / (df->volume * bin_width);
      } else {
        data[i * 3 + 2] = 0.0;
      }
    }
  }

  // Write single 2D dataset
  H5LTmake_dataset_double(group_id, dataset_prefix, 2, dims, data);

  // Write metadata as attributes to the dataset
  H5LTset_attribute_int(group_id, dataset_prefix, "n_bins", (int*)&df->n_bins, 1);
  H5LTset_attribute_double(group_id, dataset_prefix, "x_min", (double*)&df->x_min, 1);
  H5LTset_attribute_double(group_id, dataset_prefix, "x_max", (double*)&df->x_max, 1);
  H5LTset_attribute_double(group_id, dataset_prefix, "bin_width", (double*)&df->bin_width, 1);
  H5LTset_attribute_double(group_id, dataset_prefix, "volume", (double*)&df->volume, 1);
  H5LTset_attribute_string(group_id, dataset_prefix, "description", df->description);
  H5LTset_attribute_string(group_id, dataset_prefix, "units", (char*)units);

  // Add column labels as attribute
  const char* column_names = "center,density,uncertainty";
  H5LTset_attribute_string(group_id, dataset_prefix, "columns", (char*)column_names);

  H5Gclose(group_id);
  free(data);
}

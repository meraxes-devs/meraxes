#ifndef DIST_FUNC_H
#define DIST_FUNC_H

#include "meraxes.h"

//! Generic bin structure for distribution functions (HMF, SMF, LF, etc.)
typedef struct distribution_bin_t
{
  double center;      //!< Bin center value (e.g., Log10(M) for HMF)
  double number_density;  //!< Number density [h^3 Mpc^-3]
  double uncertainty;     //!< Poisson uncertainty on number density
} distribution_bin_t;

//! Generic distribution function structure
typedef struct distribution_function_t
{
  int n_bins;         //!< Number of bins
  double x_min;       //!< Minimum value (e.g., Log10(M) minimum)
  double x_max;       //!< Maximum value
  double bin_width;   //!< Bin width in dex or linear units
  distribution_bin_t* bins;    //!< Array of bin data
  double volume;      //!< Comoving volume in (Mpc/h)^3
  double* bin_counts;    //!< Count array for use in MPI reductions (supports weighted counts)
  double* bin_variance;  //!< Variance accumulator for Bernoulli uncertainty: sum(p*(1-p))
  char description[256];  //!< Description of what this function represents
} distribution_function_t;

// Backward compatibility typedef
typedef distribution_function_t hmf_t;

#ifdef __cplusplus
extern "C"
{
#endif

  //! Initialize a generic distribution function with logarithmic binning
  //! \param[in,out] df Pointer to distribution function structure
  //! \param[in] x_min Lower bin edge (e.g., log10(M_min) for HMF)
  //! \param[in] x_max Upper bin edge (e.g., log10(M_max) for HMF)
  //! \param[in] bins_per_dex Number of bins per dex (or per magnitude for UVLF/DustyLF)
  //! \param[in] description Human-readable description of function (e.g., "Halo Mass Function")
  void df_init(distribution_function_t* df, double x_min, double x_max, int bins_per_dex, const char* description);

  //! Index of the bin containing val, or -1 if val is outside [x_min, x_max] (or NaN).
  //! The last bin is closed, so val == x_max falls in bin n_bins - 1.
  int df_bin_index(const distribution_function_t* df, double val);

  //! Accumulate counts from local process to global distribution (for MPI)
  //! Must be called on all processes before finalization
  //! \param[in,out] df Pointer to distribution function
  //! \param[in] mpi_rank Current MPI rank
  //! \param[in] mpi_size Total number of MPI processes
  void df_mpi_reduce(distribution_function_t* df, int mpi_rank, int mpi_size);

  //! Free distribution function structure
  //! \param[in,out] df Pointer to distribution function to be freed
  void df_free(distribution_function_t* df);

  //! Write distribution function to HDF5 file as a 2D array
  //! \param[in] file_id HDF5 file identifier
  //! \param[in] group_name Name of group to write to (e.g., "SnapXXX")
  //! \param[in] df Pointer to distribution function structure
  //! \param[in] dataset_prefix Dataset name (e.g., "HMF")
  //! \param[in] units Unit string for the data (e.g., "per Mpc^3 per dex" for HMF/SMF)
  //! 
  //! Creates a 2D dataset with shape (n_bins, 3) where columns are:
  //!   [0] = bin centers, [1] = number density, [2] = Poisson uncertainty
  void df_write_hdf5(hid_t file_id, const char* group_name, const distribution_function_t* df, 
                     const char* dataset_prefix, const char* units);

#ifdef __cplusplus
}
#endif

#endif  // DIST_FUNC_H

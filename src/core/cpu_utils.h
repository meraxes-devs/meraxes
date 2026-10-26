#ifndef MERAXES_CPU_UTILS_H
#define MERAXES_CPU_UTILS_H

#include <complex.h>
#include <fftw3.h>
#include <mpi.h>
#include <stddef.h>

/* Inputs and outputs have the usual FFTW half-complex slab layout. */
void filter_fields(const fftwf_complex* const* input,
                   fftwf_complex* const* output,
                   int n_fields,
                   int local_ix_start,
                   int slab_nx,
                   int dim,
                   float radius,
                   int filter_type,
                   int apply_filter);
void free_filter_window_cache(void);

/* Inspect physical cells, excluding FFTW row padding. All ranks participate.
 * Returns 1 for globally zero fields, 0 for a nonzero field, -1 for NaN/Inf. */
int cpu_sources_are_zero(const float* const* fields, int n_fields, int slab_nx, int dim, MPI_Comm comm);

/* Counts/offsets are in records, not bytes. All ranks participate, including empty ones.
 * The return buffer is grouped by sender, preserving each sender's order. */
void* exchange_records(const void* send,
                       const size_t* send_count,
                       size_t record_size,
                       size_t** recv_count,
                       size_t** recv_offset,
                       MPI_Comm comm);
void* cpu_checked_alloc(size_t count, size_t size);

#endif

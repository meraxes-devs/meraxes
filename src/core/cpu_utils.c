#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cpu_utils.h"
#include "meraxes.h"

void* cpu_checked_alloc(size_t count, size_t size)
{
  if (size && count > SIZE_MAX / size) {
    mlog_error("CPU workspace allocation overflows size_t.");
    ABORT(EXIT_FAILURE);
  }
  void* p = calloc(count ? count : 1, size ? size : 1);
  if (!p) {
    mlog_error("Cannot allocate CPU workspace (%zu records of %zu bytes).", count, size);
    ABORT(EXIT_FAILURE);
  }
  return p;
}

int cpu_sources_are_zero(const float* const* fields, int n_fields, int slab_nx, int dim, MPI_Comm comm)
{
  /* Bit classification also works in Release builds with -ffast-math. */
  _Static_assert(sizeof(float) == sizeof(uint32_t) && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
                 "Source classification requires IEEE binary32 floats");
  int status = 0;
  const size_t rows = (size_t)slab_nx * dim;
  const size_t stride = 2 * (dim / 2 + 1);
  for (int f = 0; f < n_fields; ++f)
    for (size_t row = 0; row < rows; ++row)
      for (int col = 0; col < dim; ++col) {
        uint32_t bits;
        memcpy(&bits, fields[f] + row * stride + col, sizeof(bits));
        bits &= UINT32_C(0x7fffffff); // both signs of zero are zero
        if (bits != 0)
          status |= 1;
        if (bits >= UINT32_C(0x7f800000))
          status |= 2;
      }
  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_BOR, comm);
  return status & 2 ? -1 : !(status & 1);
}

/* Windows recur across snapshots. Retain their exact float values and actions,
 * not an interpolated radial table. The sharp-k window is cheap and is left
 * uncached so heating radii do not evict the recurring bubble windows. */
#define FILTER_CACHE_ENTRIES 64
#define FILTER_CACHE_BYTES ((size_t)512 * 1024 * 1024)

typedef struct
{
  float* weight;
  unsigned char* action;
  uint32_t radius_bits;
  int type;
  int ready;
  size_t bytes;
  uint64_t last_use;
} filter_window_t;

static filter_window_t filter_cache[FILTER_CACHE_ENTRIES];
static size_t filter_cache_bytes;
static uint64_t filter_cache_clock;
static int filter_cache_dim, filter_cache_nx, filter_cache_start;
static uint32_t filter_cache_box_bits;

void free_filter_window_cache(void)
{
  for (int i = 0; i < FILTER_CACHE_ENTRIES; ++i)
    free(filter_cache[i].weight);
  memset(filter_cache, 0, sizeof(filter_cache));
  filter_cache_bytes = 0;
  filter_cache_clock = 0;
  filter_cache_dim = 0;
}

static filter_window_t* cached_filter_window(int start, int nx, int dim, float box, float radius, int type)
{
  uint32_t box_bits, radius_bits;
  memcpy(&box_bits, &box, sizeof(box_bits));
  memcpy(&radius_bits, &radius, sizeof(radius_bits));
  if (filter_cache_dim != dim || filter_cache_nx != nx || filter_cache_start != start ||
      filter_cache_box_bits != box_bits) {
    free_filter_window_cache();
    filter_cache_dim = dim;
    filter_cache_nx = nx;
    filter_cache_start = start;
    filter_cache_box_bits = box_bits;
  }
  const size_t count = (size_t)nx * dim * (dim / 2 + 1);
  if (type == 1 || count == 0 || count > FILTER_CACHE_BYTES / (sizeof(float) + sizeof(unsigned char)))
    return NULL;
  const size_t bytes = count * (sizeof(float) + sizeof(unsigned char));
  for (int i = 0; i < FILTER_CACHE_ENTRIES; ++i)
    if (filter_cache[i].weight && filter_cache[i].radius_bits == radius_bits && filter_cache[i].type == type) {
      filter_cache[i].last_use = ++filter_cache_clock;
      return filter_cache + i;
    }

  int slot;
  for (;;) {
    slot = -1;
    int oldest = -1;
    for (int i = 0; i < FILTER_CACHE_ENTRIES; ++i) {
      if (!filter_cache[i].weight)
        slot = i;
      else if (oldest < 0 || filter_cache[i].last_use < filter_cache[oldest].last_use)
        oldest = i;
    }
    if (slot >= 0 && filter_cache_bytes + bytes <= FILTER_CACHE_BYTES)
      break;
    filter_cache_bytes -= filter_cache[oldest].bytes;
    free(filter_cache[oldest].weight);
    memset(filter_cache + oldest, 0, sizeof(filter_cache[oldest]));
  }
  float* weights = malloc(bytes);
  if (!weights)
    return NULL; // caching is optional; calculate the same window directly
  filter_cache[slot] = (filter_window_t){ .weight = weights,
                                          .action = (unsigned char*)(weights + count),
                                          .radius_bits = radius_bits,
                                          .type = type,
                                          .bytes = bytes,
                                          .last_use = ++filter_cache_clock };
  filter_cache_bytes += bytes;
  return filter_cache + slot;
}

void filter_fields(const fftwf_complex* const* input,
                   fftwf_complex* const* output,
                   int n_fields,
                   int local_ix_start,
                   int slab_nx,
                   int dim,
                   float radius,
                   int filter_type,
                   int apply_filter)
{
  const int middle = dim / 2;
  const float box_size = (float)run_globals.params.BoxSize;
  const float delta_k = (float)(2.0 * M_PI / box_size);
  if (!apply_filter) {
    size_t bytes = (size_t)slab_nx * dim * (middle + 1) * sizeof(fftwf_complex);
    for (int f = 0; f < n_fields; ++f)
      if (input[f] != output[f])
        memcpy(output[f], input[f], bytes);
    return;
  }
  if (filter_type < 0 || filter_type > 2) {
    mlog_error("Unknown filter type %d.", filter_type);
    ABORT(EXIT_FAILURE);
  }
  filter_window_t* window = cached_filter_window(local_ix_start, slab_nx, dim, box_size, radius, filter_type);
  if (window && window->ready) {
    size_t count = (size_t)slab_nx * dim * (middle + 1);
    for (size_t i = 0; i < count; ++i) {
      fftwf_complex w = (fftwf_complex)window->weight[i];
      for (int f = 0; f < n_fields; ++f) {
        fftwf_complex v = input[f][i];
        if (window->action[i] == 2)
          v = 0.0f;
        else if (window->action[i] == 1)
          v *= w;
        output[f][i] = v;
      }
    }
    return;
  }
  for (int x = 0; x < slab_nx; ++x) {
    int gx = x + local_ix_start;
    float kx = (gx > middle ? gx - dim : gx) * delta_k;
    for (int y = 0; y < dim; ++y) {
      float ky = (y > middle ? y - dim : y) * delta_k;
      for (int z = 0; z <= middle; ++z) {
        float kz = z * delta_k;
        float kR = sqrtf(kx * kx + ky * ky + kz * kz) * radius;
        size_t i = z + (size_t)(middle + 1) * (y + (size_t)dim * x);
        fftwf_complex w = 1.0f;
        int multiply = 1, zero = 0;
        switch (filter_type) {
          case 0: // Real-space top-hat
            if (kR > 1e-4)
              w = (fftwf_complex)(3.0 * (sinf(kR) / powf(kR, 3) - cosf(kR) / powf(kR, 2)));
            else
              multiply = 0;
            break;
          case 1:              // k-space top-hat
            kR *= 0.413566994; // Match the real-space top-hat volume: (9pi/2)^(-1/3).
            zero = kR > 1;
            multiply = 0;
            break;
          case 2:        // Gaussian
            kR *= 0.643; // Match the real-space top-hat volume.
            w = (fftwf_complex)powf((float)M_E, (float)(-kR * kR / 2.0));
            break;
        }
        if (window) {
          window->weight[i] = crealf(w);
          window->action[i] = zero ? 2 : multiply ? 1 : 0;
        }
        for (int f = 0; f < n_fields; ++f) {
          fftwf_complex v = input[f][i];
          if (zero)
            v = 0.0f;
          else if (multiply)
            v *= w;
          output[f][i] = v;
        }
      }
    }
  }
  if (window)
    window->ready = 1;
}

void* exchange_records(const void* send,
                       const size_t* send_count,
                       size_t record_size,
                       size_t** recv_count,
                       size_t** recv_offset,
                       MPI_Comm comm)
{
  int ranks, rank;
  MPI_Comm_size(comm, &ranks);
  MPI_Comm_rank(comm, &rank);
  unsigned long long* counts = cpu_checked_alloc((size_t)ranks * 2, sizeof(*counts));
  for (int p = 0; p < ranks; ++p)
    counts[p] = send_count[p];
  MPI_Alltoall(counts, 1, MPI_UNSIGNED_LONG_LONG, counts + ranks, 1, MPI_UNSIGNED_LONG_LONG, comm);
  *recv_count = cpu_checked_alloc(ranks, sizeof(size_t));
  *recv_offset = cpu_checked_alloc((size_t)ranks + 1, sizeof(size_t));
  size_t* send_offset = cpu_checked_alloc((size_t)ranks + 1, sizeof(size_t));
  int small = 1;
  for (int p = 0; p < ranks; ++p) {
    (*recv_count)[p] = (size_t)counts[ranks + p];
    if ((*recv_count)[p] > SIZE_MAX - (*recv_offset)[p] || send_count[p] > SIZE_MAX - send_offset[p])
      ABORT(EXIT_FAILURE);
    (*recv_offset)[p + 1] = (*recv_offset)[p] + (*recv_count)[p];
    send_offset[p + 1] = send_offset[p] + send_count[p];
  }
  free(counts);
  if (!record_size || record_size > INT_MAX)
    ABORT(EXIT_FAILURE);
  if (send_offset[ranks] > SIZE_MAX / record_size)
    ABORT(EXIT_FAILURE);
  size_t limit = (size_t)INT_MAX / record_size;
  if (send_offset[ranks] > limit || (*recv_offset)[ranks] > limit)
    small = 0;
  MPI_Allreduce(MPI_IN_PLACE, &small, 1, MPI_INT, MPI_LAND, comm);
  char* recv = cpu_checked_alloc((*recv_offset)[ranks], record_size);
  char dummy = 0;
  if (!send)
    send = &dummy;
  if (small) {
    int* v = cpu_checked_alloc((size_t)ranks * 4, sizeof(int));
    for (int p = 0; p < ranks; ++p) {
      v[p] = (int)(send_count[p] * record_size);
      v[ranks + p] = (int)(send_offset[p] * record_size);
      v[2 * ranks + p] = (int)((*recv_count)[p] * record_size);
      v[3 * ranks + p] = (int)((*recv_offset)[p] * record_size);
    }
    MPI_Alltoallv(send, v, v + ranks, MPI_BYTE, recv, v + 2 * ranks, v + 3 * ranks, MPI_BYTE, comm);
    free(v);
  } else {
    /* MPI-3 uses int byte counts. Large jobs use bounded rounds; every rank
     * takes the same number of rounds so an empty peer cannot deadlock one. */
    for (int step = 0; step < ranks; ++step) {
      int to = (rank + step) % ranks;
      int from = (rank - step + ranks) % ranks;
      size_t ns = send_count[to], nr = (*recv_count)[from];
      unsigned long long rounds = (ns > nr ? ns : nr) / limit + 1;
      MPI_Allreduce(MPI_IN_PLACE, &rounds, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
      for (unsigned long long r = 0; r < rounds; ++r) {
        size_t off = (size_t)r * limit;
        size_t sc = off < ns ? (ns - off < limit ? ns - off : limit) : 0;
        size_t rc = off < nr ? (nr - off < limit ? nr - off : limit) : 0;
        const char* sp = sc ? (const char*)send + (send_offset[to] + off) * record_size : &dummy;
        char* rp = rc ? recv + ((*recv_offset)[from] + off) * record_size : &dummy;
        MPI_Sendrecv(sp,
                     (int)(sc * record_size),
                     MPI_BYTE,
                     to,
                     29117,
                     rp,
                     (int)(rc * record_size),
                     MPI_BYTE,
                     from,
                     29117,
                     comm,
                     MPI_STATUS_IGNORE);
      }
    }
  }
  free(send_offset);
  return recv;
}

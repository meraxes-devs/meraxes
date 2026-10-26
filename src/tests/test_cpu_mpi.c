/* Small, deterministic MPI checks; no simulation data or Criterion required. */
#define _MAIN
#include "core/cpu_grids.h"
#include "core/cpu_utils.h"
#include "core/misc_tools.h"
#include "meraxes.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      fprintf(stderr, "rank %d: %s:%d: %s\n", run_globals.mpi_rank, __FILE__, __LINE__, #condition);                   \
      MPI_Abort(run_globals.mpi_comm, 1);                                                                              \
    }                                                                                                                  \
  } while (0)

static size_t nc;
static ptrdiff_t nx, x_start;
static const int dim = 4;

void reference_filter(fftwf_complex* box, int local_ix_start, int slab_nx, int grid_dim, float R, int filter_type);

static void test_filters(void)
{
  fftwf_complex *src[3], *dst[3], *reference, *single;
  reference = fftwf_alloc_complex(nc ? nc : 1);
  single = fftwf_alloc_complex(nc ? nc : 1);
  for (int f = 0; f < 3; ++f) {
    src[f] = fftwf_alloc_complex(nc ? nc : 1);
    dst[f] = fftwf_alloc_complex(nc ? nc : 1);
    for (size_t i = 0; i < nc; ++i)
      src[f][i] = (float)(f + 1 + i * 0.03125) + I * (float)(0.5 + i * 0.0625);
  }
  size_t logical = (size_t)nx * dim * (dim / 2 + 1);
  for (int type = 0; type < 3; ++type)
    for (int r = 0; r < 3; ++r) {
      float radius = r == 0 ? 0.0f : r == 1 ? 0.31f : 8.0f;
      for (int repeat = 0; repeat < 2; ++repeat) {
        filter_fields((const fftwf_complex* const*)src, dst, 3, x_start, nx, dim, radius, type, 1);
        for (int f = 0; f < 3; ++f) {
          memcpy(reference, src[f], nc * sizeof(*reference));
          reference_filter(reference, x_start, nx, dim, radius, type);
          CHECK(memcmp(reference, dst[f], logical * sizeof(*reference)) == 0);
          memcpy(single, src[f], nc * sizeof(*single));
          filter(single, x_start, nx, dim, radius, type);
          CHECK(memcmp(reference, single, logical * sizeof(*reference)) == 0);
          for (size_t i = 0; i < logical; ++i)
            src[f][i] += 0.125f; // cache windows, never filtered source values
        }
      }
    }
  filter_fields((const fftwf_complex* const*)src, dst, 3, x_start, nx, dim, 1.0f, 0, 0);
  for (int f = 0; f < 3; ++f) {
    CHECK(memcmp(src[f], dst[f], logical * sizeof(*reference)) == 0);
    fftwf_free(src[f]);
    fftwf_free(dst[f]);
  }
  fftwf_free(reference);
  fftwf_free(single);
}

static void test_filter_cache_keys(void)
{
  const int sizes[] = { 4, 4, 4, 5, 5, 4 };
  const int starts[] = { 0, 1, 1, 1, 0, 0 };
  const int widths[] = { 2, 2, 1, 1, 2, 2 };
  const double boxes[] = { 4, 4, 4, 4, 7, 4 };
  fftwf_complex input[60], output[60], reference[60];
  const fftwf_complex* inputs[] = { input };
  fftwf_complex* outputs[] = { output };
  for (int layout = 0; layout < 6; ++layout) {
    int size = sizes[layout], start = starts[layout], width = widths[layout];
    size_t count = (size_t)width * size * (size / 2 + 1);
    CHECK(count <= 60);
    run_globals.params.BoxSize = boxes[layout];
    for (int type = 0; type < 3; ++type)
      // More windows than cache entries exercises bounded eviction and revisits.
      for (int r = 0; r < 71; ++r) {
        float radius = r == 70 ? 0.0f : r * 0.071f;
        for (int repeat = 0; repeat < 2; ++repeat) {
          for (size_t i = 0; i < count; ++i)
            input[i] = (float)(i + repeat) * 0.125f + I * (float)(r - (int)i) * 0.0625f;
          memcpy(reference, input, count * sizeof(*reference));
          reference_filter(reference, start, width, size, radius, type);
          filter_fields(inputs, outputs, 1, start, width, size, radius, type, 1);
          CHECK(memcmp(reference, output, count * sizeof(*reference)) == 0);
          filter(input, start, width, size, radius, type);
          CHECK(memcmp(reference, input, count * sizeof(*reference)) == 0);
        }
      }
  }
  free_filter_window_cache();
  run_globals.params.BoxSize = dim;
}

static void test_zero_sources(void)
{
  const size_t allocation = 2 * (nc ? nc : 1);
  float* fields[2] = { cpu_checked_alloc(allocation, sizeof(float)), cpu_checked_alloc(allocation, sizeof(float)) };
  const size_t stride = 2 * (dim / 2 + 1);
  uint32_t invalid = UINT32_C(0x7fc00000);
  for (int f = 0; f < 2; ++f)
    for (size_t row = 0; row < (size_t)nx * dim; ++row)
      for (int col = dim; col < (int)stride; ++col)
        memcpy(fields[f] + row * stride + col, &invalid, sizeof(invalid)); // ignore FFT padding
  CHECK(cpu_sources_are_zero((const float* const*)fields, 2, nx, dim, MPI_COMM_WORLD) == 1);
  int source_rank = nx > 0 ? run_globals.mpi_rank : -1;
  MPI_Allreduce(MPI_IN_PLACE, &source_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  CHECK(source_rank >= 0);
  uint32_t cases[] = { UINT32_C(0x80000000), UINT32_C(0x3f800000), UINT32_C(0x00000001), UINT32_C(0x7f800000),
                       UINT32_C(0xff800000), UINT32_C(0x7fc00000), UINT32_C(0) };
  const int expected[] = { 1, 0, 0, -1, -1, -1, 1 };
  for (int f = 0; f < 2; ++f)
    for (int c = 0; c < 7; ++c) {
      if (run_globals.mpi_rank == source_rank)
        memcpy(fields[f], cases + c, sizeof(cases[c]));
      CHECK(cpu_sources_are_zero((const float* const*)fields, 2, nx, dim, MPI_COMM_WORLD) == expected[c]);
    }
  free(fields[0]);
  free(fields[1]);
}

static void test_exchange(void)
{
  int rank = run_globals.mpi_rank, ranks = run_globals.mpi_size;
  size_t* counts = cpu_checked_alloc(ranks, sizeof(size_t));
  size_t total = 0;
  for (int p = 0; p < ranks; ++p) {
    counts[p] = ranks > 1 && rank == ranks - 1 ? 0 : (rank + p) % 3;
    total += counts[p];
  }
  uint64_t* send = cpu_checked_alloc(total, sizeof(uint64_t));
  size_t k = 0;
  for (int p = 0; p < ranks; ++p)
    for (size_t j = 0; j < counts[p]; ++j)
      send[k++] = (uint64_t)rank * 10000 + p * 100 + j;
  size_t *rc, *ro;
  uint64_t* recv = exchange_records(send, counts, sizeof(uint64_t), &rc, &ro, run_globals.mpi_comm);
  for (int p = 0; p < ranks; ++p) {
    size_t expected = ranks > 1 && p == ranks - 1 ? 0 : (p + rank) % 3;
    CHECK(rc[p] == expected);
    for (size_t j = 0; j < expected; ++j)
      CHECK(recv[ro[p] + j] == (uint64_t)p * 10000 + rank * 100 + j);
  }
  free(send);
  free(recv);
  free(counts);
  free(rc);
  free(ro);
}

static void test_sources_and_history(void)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  int rank = run_globals.mpi_rank, ranks = run_globals.mpi_size;
  run_globals.params.Flag_IncludeSpinTemp = 1;
  run_globals.params.Flag_InstantaneousSFR = 1;
  run_globals.params.ReionSfrTimescale = 2;
  run_globals.params.OmegaM = 1;
  run_globals.Hubble = 1;
  double redshift[] = { 0 };
  run_globals.ZZ = redshift;
  run_globals.NstoreSnapshots_Heating = 2;
  run_globals.params.physics.Flag_BHFeedback = 1;
  run_globals.params.physics.BlackHoleMassLimitReion = 0;
  run_globals.params.physics.Flag_IncludeAGNXray = 1;
  run_globals.params.physics.LXrayGal = 1e40;
  run_globals.units.UnitMass_in_g = SOLAR_MASS;
  run_globals.units.UnitTime_in_s = SEC_PER_YEAR;
#if USE_MINI_HALOS
  run_globals.params.Flag_IncludeLymanWerner = 1;
#endif
#define ALLOC_REAL(name) grids->name = fftwf_alloc_real(2 * (nc ? nc : 1))
  ALLOC_REAL(stars);
  ALLOC_REAL(weighted_sfr);
  ALLOC_REAL(effective_bhm);
  ALLOC_REAL(effective_bhar);
#if USE_MINI_HALOS
  ALLOC_REAL(starsIII);
  ALLOC_REAL(weighted_sfrIII);
#endif
#define ALLOC_HISTORY(name, history)                                                                                   \
  do {                                                                                                                 \
    ALLOC_REAL(name);                                                                                                  \
    grids->history = cpu_checked_alloc(4 * nc, sizeof(float));                                                         \
    grids->name##_unfiltered = fftwf_alloc_complex(nc ? nc : 1);                                                       \
    grids->name##_forward_plan = fftwf_mpi_plan_dft_r2c_3d(                                                            \
      dim, dim, dim, grids->name, grids->name##_unfiltered, run_globals.mpi_comm, FFTW_ESTIMATE);                      \
    CHECK(grids->name##_forward_plan != NULL);                                                                         \
  } while (0)
  ALLOC_HISTORY(sfr, sfr_histories);
  ALLOC_HISTORY(BHXrayEmissivity_hard, bh_xray_histories_hard);
  ALLOC_HISTORY(BHXrayEmissivity_soft, bh_xray_histories_soft);
#if USE_STOCHASTICITY
  ALLOC_HISTORY(xray_luminosity, xray_luminosity_histories);
#endif
#if USE_MINI_HALOS
  ALLOC_HISTORY(sfrIII, sfrIII_histories);
  ALLOC_HISTORY(BHUVEmissivity, bh_uv_histories);
#endif
#undef ALLOC_HISTORY
#undef ALLOC_REAL

  galaxy_t galaxies[7] = { 0 };
  gal_to_slab_t map[7];
  const int position[] = { 0, 0, 1, 1, 2, 2, 4 }; // duplicate cells and periodic boundary
  int n_live = ranks > 1 && rank == ranks - 1 ? 0 : 6;
  for (int i = 0; i < 7; ++i) {
    galaxies[i].Type = i == 2 ? 3 : 0; // a galaxy that died after the map was made
    galaxies[i].Pos[0] = position[i];
    int x = position[i] % dim, owner = -1;
    for (int p = 0; p < ranks; ++p)
      if (x >= grids->slab_ix_start[p] && x < grids->slab_ix_start[p] + grids->slab_nix[p])
        owner = p;
    CHECK(owner >= 0);
    map[i] = (gal_to_slab_t){ i, &galaxies[i], owner };
  }
  grids->galaxy_to_slab_map = n_live ? map : NULL;
  float previous[64] = { 0 };
  float* expected_real = fftwf_alloc_real(2 * (nc ? nc : 1));
  fftwf_complex* expected_fourier = fftwf_alloc_complex(nc ? nc : 1);
  fftwf_plan expected_plan =
    fftwf_mpi_plan_dft_r2c_3d(dim, dim, dim, expected_real, expected_fourier, run_globals.mpi_comm, FFTW_ESTIMATE);
  CHECK(expected_plan != NULL);
  float* feedback = cpu_checked_alloc((size_t)nx * dim * dim, sizeof(float));
  for (ptrdiff_t x = 0; x < nx; ++x)
    for (int y = 0; y < dim; ++y)
      for (int z = 0; z < dim; ++z)
        feedback[z + dim * (y + dim * x)] = (float)(z + dim * (y + dim * (x + x_start)));
  float* sampled = sample_grid_for_galaxies(feedback, n_live ? map : NULL, n_live ? 7 : 0);
  for (int i = 0; i < (n_live ? 7 : 0); ++i)
    CHECK(sampled[i] == (position[i] % dim) * dim * dim);
  free(sampled);
  free(feedback);

  // More snapshots than history slots exercises wraparound and overwrite.
  for (int step = 0; step < 5; ++step) {
    float expected[64] = { 0 };
    for (int i = 0; i < 7; ++i) {
      double value = (rank + 1) * (i + 1) * (step + 1);
      galaxy_t* g = &galaxies[i];
      g->FescWeightedGSM = value;
      g->FescWeightedSfr = 2 * value;
      g->EffectiveBHM = 3 * value;
      g->EffectiveBHAR = 4 * value;
      g->Sfr = 5 * value;
      g->GrossStellarMass = 10 * value;
      g->BHXrayEmissivity_hard = 6 * value;
      g->BHXrayEmissivity_soft = 7 * value;
#if USE_MINI_HALOS
      g->FescIIIWeightedGSM = 8 * value;
      g->FescIIIWeightedSfr = 9 * value;
      g->SfrIII = 10 * value;
      g->GrossStellarMassIII = 20 * value;
      g->QuasarLuv = 11 * value;
#endif
      if (n_live && g->Type <= 2)
        expected[(position[i] % dim) * dim * dim] += (float)value;
    }
    MPI_Allreduce(MPI_IN_PLACE, expected, 64, MPI_FLOAT, MPI_SUM, run_globals.mpi_comm);
    run_globals.params.Flag_InstantaneousSFR = step % 2;
    construct_baryon_grids(0, n_live);
    for (ptrdiff_t x = 0; x < nx; ++x)
      for (int y = 0; y < dim; ++y)
        for (int z = 0; z < dim; ++z) {
          size_t global = z + dim * (y + dim * (x + x_start));
          size_t padded = z + 2 * (dim / 2 + 1) * (y + dim * x);
          float value = expected[global];
          CHECK(grids->stars[padded] == value);
          CHECK(grids->weighted_sfr[padded] == 2 * value);
          CHECK(grids->effective_bhm[padded] == 3 * value);
          CHECK(grids->effective_bhar[padded] == 4 * value);
          CHECK(grids->sfr[padded] == 5 * value);
          CHECK(grids->BHXrayEmissivity_hard[padded] == 6 * value);
          CHECK(grids->BHXrayEmissivity_soft[padded] == 7 * value);
#if USE_STOCHASTICITY
          CHECK(fabsf(grids->xray_luminosity[padded] - 5 * value) <= 1e-5f * fmaxf(1, value));
#endif
#if USE_MINI_HALOS
          CHECK(grids->starsIII[padded] == 8 * value);
          CHECK(grids->weighted_sfrIII[padded] == 9 * value);
          CHECK(grids->sfrIII[padded] == 10 * value);
          CHECK(grids->BHUVEmissivity[padded] == 11 * value);
#endif
          expected_real[padded] = 0.25f * expected[global] + 0.75f * previous[global];
        }
    // Compare FFT of a real-space history sum to the cached spectral sum.
    fftwf_execute(expected_plan);
    load_reion_sfr_grids(0, 0.25f, 1);
    load_reion_sfr_grids(1, 0.75f, 0);
    load_reion_bh_grids(0, 0.25f, 1);
    load_reion_bh_grids(1, 0.75f, 0);
    size_t logical = (size_t)nx * dim * (dim / 2 + 1);
    for (size_t i = 0; i < logical; ++i) {
      fftwf_complex expected_value = expected_fourier[i] / 64.0f;
      float tolerance = 2e-5f * fmaxf(1, cabsf(expected_value));
      CHECK(cabsf(grids->sfr_unfiltered[i] - 5 * expected_value) < tolerance);
      CHECK(cabsf(grids->BHXrayEmissivity_hard_unfiltered[i] - 6 * expected_value) < tolerance);
      CHECK(cabsf(grids->BHXrayEmissivity_soft_unfiltered[i] - 7 * expected_value) < tolerance);
#if USE_STOCHASTICITY
      CHECK(cabsf(grids->xray_luminosity_unfiltered[i] - 5 * expected_value) < tolerance);
#endif
#if USE_MINI_HALOS
      CHECK(cabsf(grids->sfrIII_unfiltered[i] - 10 * expected_value) < tolerance);
      CHECK(cabsf(grids->BHUVEmissivity_unfiltered[i] - 11 * expected_value) < tolerance);
#endif
    }
    memcpy(previous, expected, sizeof(previous));
  }
  fftwf_destroy_plan(expected_plan);
  fftwf_free(expected_real);
  fftwf_free(expected_fourier);
#define FREE_HISTORY(name, history)                                                                                    \
  do {                                                                                                                 \
    fftwf_destroy_plan(grids->name##_forward_plan);                                                                    \
    fftwf_free(grids->name);                                                                                           \
    fftwf_free(grids->name##_unfiltered);                                                                              \
    free(grids->history);                                                                                              \
  } while (0)
  FREE_HISTORY(sfr, sfr_histories);
  FREE_HISTORY(BHXrayEmissivity_hard, bh_xray_histories_hard);
  FREE_HISTORY(BHXrayEmissivity_soft, bh_xray_histories_soft);
#if USE_STOCHASTICITY
  FREE_HISTORY(xray_luminosity, xray_luminosity_histories);
#endif
#if USE_MINI_HALOS
  FREE_HISTORY(sfrIII, sfrIII_histories);
  FREE_HISTORY(BHUVEmissivity, bh_uv_histories);
  fftwf_free(grids->starsIII);
  fftwf_free(grids->weighted_sfrIII);
#endif
#undef FREE_HISTORY
  fftwf_free(grids->stars);
  fftwf_free(grids->weighted_sfr);
  fftwf_free(grids->effective_bhm);
  fftwf_free(grids->effective_bhar);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);
  init_mlog(MPI_COMM_WORLD, stdout, stdout, stderr);
  fftwf_mpi_init();
  run_globals.params.ReionGridDim = dim;
  run_globals.params.BoxSize = dim;
  ptrdiff_t count = fftwf_mpi_local_size_3d(dim, dim, dim / 2 + 1, MPI_COMM_WORLD, &nx, &x_start);
  nc = (size_t)count;
  reion_grids_t* grids = &run_globals.reion_grids;
  grids->slab_nix = cpu_checked_alloc(run_globals.mpi_size, sizeof(ptrdiff_t));
  grids->slab_ix_start = cpu_checked_alloc(run_globals.mpi_size, sizeof(ptrdiff_t));
  grids->slab_n_complex = cpu_checked_alloc(run_globals.mpi_size, sizeof(ptrdiff_t));
  MPI_Allgather(&nx, sizeof(ptrdiff_t), MPI_BYTE, grids->slab_nix, sizeof(ptrdiff_t), MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgather(
    &x_start, sizeof(ptrdiff_t), MPI_BYTE, grids->slab_ix_start, sizeof(ptrdiff_t), MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgather(
    &count, sizeof(ptrdiff_t), MPI_BYTE, grids->slab_n_complex, sizeof(ptrdiff_t), MPI_BYTE, MPI_COMM_WORLD);
  test_filters();
  test_filter_cache_keys();
  test_zero_sources();
  test_exchange();
  test_sources_and_history();
  free(grids->slab_nix);
  free(grids->slab_ix_start);
  free(grids->slab_n_complex);
  free_filter_window_cache();
  fftwf_mpi_cleanup();
  if (run_globals.mpi_rank == 0)
    puts("\nCPU MPI tests passed (filters, sparse exchange, sources, feedback, circular FFT history).");
  MPI_Finalize();
  return 0;
}

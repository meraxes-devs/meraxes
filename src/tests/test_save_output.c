/* Exercise the actual output writer across buffer boundaries and empty ranks. */
#define _MAIN
#include "core/save.h"
#include "meraxes.h"
#include <hdf5_hl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      fprintf(stderr, "rank %d: %s:%d: %s\n", run_globals.mpi_rank, __FILE__, __LINE__, #condition);                   \
      MPI_Abort(run_globals.mpi_comm, 1);                                                                              \
    }                                                                                                                  \
  } while (0)

#ifdef CALC_MAGS
static void check_fluxes(const galaxy_t* gal)
{
  for (int i = 0; i < MAGS_N; ++i) {
    CHECK(gal->inBCFlux[i] == 1.0 + 0.25 * i);
    CHECK(gal->outBCFlux[i] == 2.0 + 0.5 * i);
#if USE_MINI_HALOS
    CHECK(gal->inBCFluxIII[i] == 3.0 + 0.25 * i);
    CHECK(gal->outBCFluxIII[i] == 4.0 + 0.5 * i);
#endif
  }
}
#endif

static void check_table(int snapshot, int count, galaxy_t* galaxies)
{
  char path[64], value[256];
  snprintf(path, sizeof(path), "Snap%03d/Galaxies", snapshot);
  hdf5_output_t* props = &run_globals.hdf5props;
  hsize_t fields = 0, rows = 0;
  CHECK(H5TBget_table_info(run_globals.output_file_id, path, &fields, &rows) >= 0);
  CHECK(fields == (hsize_t)props->n_props && rows == (hsize_t)count);
  CHECK(H5LTget_attribute_string(run_globals.output_file_id, path, "CLASS", value) >= 0);
  CHECK(strcmp(value, "TABLE") == 0);
  CHECK(H5LTget_attribute_string(run_globals.output_file_id, path, "TITLE", value) >= 0);
  CHECK(strcmp(value, "Galaxies") == 0);
  CHECK(H5LTget_attribute_string(run_globals.output_file_id, path, "VERSION", value) >= 0);
  for (int i = 0; i < props->n_props; ++i) {
    char attribute[64];
    snprintf(attribute, sizeof(attribute), "FIELD_%d_NAME", i);
    CHECK(H5LTget_attribute_string(run_globals.output_file_id, path, attribute, value) >= 0);
    CHECK(strcmp(value, props->field_names[i]) == 0);
  }

  hid_t dataset = H5Dopen2(run_globals.output_file_id, path, H5P_DEFAULT);
  CHECK(dataset >= 0);
  hid_t datatype = H5Dget_type(dataset);
  CHECK(H5Tget_size(datatype) == props->dst_size);
  for (int i = 0; i < props->n_props; ++i) {
    CHECK(H5Tget_member_offset(datatype, (unsigned)i) == props->dst_offsets[i]);
    hid_t field_type = H5Tget_member_type(datatype, (unsigned)i);
    CHECK(H5Tequal(field_type, props->field_types[i]) > 0);
    H5Tclose(field_type);
  }
  H5Tclose(datatype);
  hid_t space = H5Dget_space(dataset);
  hsize_t extent, maximum;
  CHECK(H5Sget_simple_extent_dims(space, &extent, &maximum) == 1);
  CHECK(extent == (hsize_t)count && maximum == H5S_UNLIMITED);
  H5Sclose(space);
  H5Dclose(dataset);

  if (count == 0)
    return;
  galaxy_output_t* actual = calloc((size_t)count, sizeof(*actual));
  CHECK(actual != NULL);
  CHECK(H5TBread_table(
          run_globals.output_file_id, path, props->dst_size, props->dst_offsets, props->dst_field_sizes, actual) >= 0);
  for (int i = 0; i < count; ++i) {
    galaxy_output_t expected = { 0 };
#ifdef CALC_MAGS
    check_fluxes(&galaxies[i]);
#endif
    prepare_galaxy_for_output(&galaxies[i], &expected, snapshot);
#ifdef CALC_MAGS
    check_fluxes(&galaxies[i]);
    if (snapshot == 1) {
      for (int band = 0; band < MAGS_N_BANDS; ++band) {
        CHECK(isfinite(actual[i].Mags[band]) && actual[i].Mags[band] < 900.0f);
        CHECK(isfinite(actual[i].DustyMags[band]) && actual[i].DustyMags[band] < 900.0f);
        CHECK(actual[i].DustyMags[band] > actual[i].Mags[band]);
#if USE_MINI_HALOS
        CHECK(isfinite(actual[i].MagsIII[band]) && actual[i].MagsIII[band] < 900.0f);
#endif
      }
    }
#endif
    CHECK(actual[i].ID == (unsigned long)(1000000 + i));
    for (int field = 0; field < props->n_props; ++field) {
      size_t offset = props->dst_offsets[field];
      CHECK(memcmp((const char*)&actual[i] + offset, (const char*)&expected + offset, props->dst_field_sizes[field]) ==
            0);
    }
  }
  free(actual);
}

static void check_links(int count)
{
  if (count == 0)
    return;
  int* indices = malloc((size_t)count * sizeof(*indices));
  CHECK(indices != NULL);
  CHECK(H5LTread_dataset_int(run_globals.output_file_id, "Snap001/DescendantIndices", indices) >= 0);
  for (int i = 0; i < count; ++i)
    CHECK(indices[i] == (i == count - 1 ? 0 : i));
  CHECK(H5LTread_dataset_int(run_globals.output_file_id, "Snap001/NextProgenitorIndices", indices) >= 0);
  for (int i = 0; i < count; ++i)
    CHECK(indices[i] == (i == 0 ? count - 1 : -1));
  CHECK(H5LTread_dataset_int(run_globals.output_file_id, "Snap002/FirstProgenitorIndices", indices) >= 0);
  for (int i = 0; i < count - 1; ++i)
    CHECK(indices[i] == i);
  free(indices);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);
  init_mlog(MPI_COMM_WORLD, stdout, stdout, stderr);

  char root[STRLEN] = "./meraxes-output-test-XXXXXX";
  if (run_globals.mpi_rank == 0)
    CHECK(mkdtemp(root) != NULL);
  MPI_Bcast(root, sizeof(root), MPI_CHAR, 0, MPI_COMM_WORLD);
  snprintf(run_globals.FNameOut, sizeof(run_globals.FNameOut), "%s/rank%d.h5", root, run_globals.mpi_rank);

  int snapshots[] = { 0, 1, 2, 3 };
  double times[] = { 4, 3, 2, 1 };
  run_globals.ListOutputSnaps = snapshots;
  run_globals.NOutputSnaps = 4;
  run_globals.LTTime = times;
  run_globals.params.BoxSize = 10;
  run_globals.params.Hubble_h = 0.7;
  run_globals.units.UnitMass_in_g = SOLAR_MASS;
  run_globals.units.UnitTime_in_s = SEC_PER_YEAR;
  run_globals.units.UnitTime_in_Megayears = 1;
#ifdef CALC_MAGS
  double redshifts[] = { 9, 8, 7, 6 };
  run_globals.ZZ = redshifts;
  run_globals.loiii_rest_band_mag_index = -1;
  for (int i = 0; i < MAGS_N_SNAPS; ++i)
    run_globals.mag_params.targetSnap[i] = 99 + i;
  run_globals.mag_params.targetSnap[0] = 1;
  for (int band = 0; band < MAGS_N_BANDS; ++band)
    run_globals.mag_params.allcentreWaves[0][band] = 1550.0 + 50.0 * band;
  run_globals.mag_params.tBC = 1e7;
  run_globals.params.DustMetallicityScale = 1.2;
  run_globals.params.DustTauUVISM = 1.0;
  run_globals.params.DustTauUVBC = 2.0;
  run_globals.params.DustNISM = -1.6;
  run_globals.params.DustNBC = -1.6;
  run_globals.params.DustAZ = -0.35;
#endif
  calc_hdf5_props();
  prep_hdf5_file();

  int last_count = 0;
  write_snapshot(0, 0, &last_count);
  check_table(0, 0, NULL);

  /* More than two 4 MiB blocks even in the smallest output configuration. */
  const int count = run_globals.mpi_rank == 0 ? 32771 : 0;
  galaxy_t* galaxies = count ? calloc((size_t)count, sizeof(*galaxies)) : NULL;
  CHECK(count == 0 || galaxies != NULL);
  for (int i = 0; i < count; ++i) {
    galaxies[i].ID = (unsigned long)(1000000 + i);
    galaxies[i].Type = 1;
    galaxies[i].ghost_flag = true;
    galaxies[i].output_index = -1;
    galaxies[i].StellarMass = 1 + i * 0.125;
    galaxies[i].ColdGas = 2 + i * 0.25;
#ifdef CALC_MAGS
    galaxies[i].MetalsColdGas = 0.02 * galaxies[i].ColdGas;
    galaxies[i].DiskScaleLength = 0.01;
    for (int flux = 0; flux < MAGS_N; ++flux) {
      galaxies[i].inBCFlux[flux] = 1.0 + 0.25 * flux;
      galaxies[i].outBCFlux[flux] = 2.0 + 0.5 * flux;
#if USE_MINI_HALOS
      galaxies[i].inBCFluxIII[flux] = 3.0 + 0.25 * flux;
      galaxies[i].outBCFluxIII[flux] = 4.0 + 0.5 * flux;
#endif
    }
#endif
    galaxies[i].Sfr = i * 0.5;
    galaxies[i].NewStars[0] = i * 0.125;
    galaxies[i].Pos[0] = i * 0.125f;
    galaxies[i].Next = i + 1 < count ? &galaxies[i + 1] : NULL;
  }
  run_globals.FirstGal = galaxies;
  write_snapshot(count, 1, &last_count);
  CHECK(last_count == count);
  check_table(1, count, galaxies);

  if (count) {
    galaxies[count - 1].Type = 3;
    galaxies[count - 1].MergerTarget = &galaxies[0];
  }
  const int remaining = count ? count - 1 : 0;
  write_snapshot(remaining, 2, &last_count);
  CHECK(last_count == remaining);
  check_table(2, remaining, galaxies);
  check_links(count);
  run_globals.params.FlagInteractive = 2;
  write_snapshot(remaining, 3, &last_count);
  CHECK(H5Lexists(run_globals.output_file_id, "Snap003", H5P_DEFAULT) > 0);
  CHECK(H5Lexists(run_globals.output_file_id, "Snap003/Galaxies", H5P_DEFAULT) == 0);
  close_hdf5_file();
  CHECK(unlink(run_globals.FNameOut) == 0);
  free(galaxies);
  MPI_Barrier(MPI_COMM_WORLD);
  if (run_globals.mpi_rank == 0) {
    CHECK(rmdir(root) == 0);
    puts("Output checks passed: empty ranks, multiple chunks, fields, table metadata and merger links.");
  }
  MPI_Finalize();
  return 0;
}

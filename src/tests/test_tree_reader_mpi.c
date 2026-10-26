/* Synthetic I/O integration test: no simulation data or Criterion required. */
#define _MAIN
#include "core/read_halos.h"
#include "meraxes.h"
#include "tree_flags.h"
#include <hdf5_hl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      fprintf(stderr, "rank %d: %s:%d: %s\n", run_globals.mpi_rank, __FILE__, __LINE__, #condition);                   \
      MPI_Abort(run_globals.mpi_comm, 1);                                                                              \
    }                                                                                                                  \
  } while (0)

/* Host 9999 and satellites 10000/10002 cross the 10000-row read boundary. */
enum
{
  NROWS = 20005,
  HOST = 9999
};
static const long id_base = 1000000000000L;

static int satellite(int row)
{
  return row == 10000 || row == 10002;
}

static long forest(int row)
{
  return satellite(row) ? 40 : 10 * (1 + row % 4);
}

static float property(int column, int row)
{
  switch (column) {
    case 0:
      return 100 + row % 17; /* Mass_200crit */
    case 1:
      return 200 + row % 23; /* Mass_tot */
    case 2:
      return 2 + 0.25f * (row % 5); /* R_200crit */
    case 3:
      return 4 + row % 7; /* Vmax */
    case 4:
      return -5 + row % 120; /* Xc */
    case 5:
      return 5 + row % 80; /* Yc */
    case 6:
      return 95 + row % 10; /* Zc */
    case 7:
      return -3 + row % 9; /* VXc */
    case 8:
      return 4 + row % 11; /* VYc */
    case 9:
      return 8 + row % 13; /* VZc */
    default:
      return 1 + 0.5f * (row % 19); /* AngMom */
  }
}

static void dataset(hid_t group, const char* name, hid_t type, const void* values, int count, int chunked)
{
  hsize_t length = count, chunk = 257;
  hid_t space = H5Screate_simple(1, &length, NULL);
  hid_t props = H5Pcreate(H5P_DATASET_CREATE);
  if (chunked && count)
    CHECK(H5Pset_chunk(props, 1, &chunk) >= 0);
  hid_t data = H5Dcreate(group, name, type, space, H5P_DEFAULT, props, H5P_DEFAULT);
  CHECK(data >= 0);
  if (count)
    CHECK(H5Dwrite(data, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) >= 0);
  CHECK(H5Dclose(data) >= 0);
  H5Pclose(props);
  H5Sclose(space);
}

static void fixture(const char* root, const char* subdir, int chunked)
{
  char dirname[STRLEN + 32], filename[STRLEN + 64];
  snprintf(dirname, sizeof(dirname), "%s/%s", root, subdir);
  CHECK(mkdir(dirname, 0700) == 0);
  snprintf(filename, sizeof(filename), "%s/fixture.h5", dirname);
  hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  CHECK(file >= 0);
  hid_t header = H5Gcreate(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t units = H5Gcreate(header, "Units", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  double massunit = 1e10, scale = 0.5;
  CHECK(H5LTset_attribute_double(file, "Header/Units", "Mass_unit_to_solarmass", &massunit, 1) >= 0);
  H5Gclose(units);
  H5Gclose(header);
  long* integers = calloc(NROWS, sizeof(long));
  unsigned long* unsigneds = calloc(NROWS, sizeof(unsigned long));
  float* floats = calloc(NROWS, sizeof(float));
  CHECK(integers && unsigneds && floats);
  const char* names[] = { "Mass_200crit", "Mass_tot", "R_200crit", "Vmax", "Xc",    "Yc",
                          "Zc",           "VXc",      "VYc",       "VZc",  "AngMom" };
  for (int snapshot = 1; snapshot <= 2; ++snapshot) {
    int count = snapshot == 1 ? NROWS : 0;
    const char* groupname = snapshot == 1 ? "Snap_001" : "Snap_002";
    hid_t group = H5Gcreate(file, groupname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(H5LTset_attribute_int(file, groupname, "NHalos", &count, 1) >= 0);
    CHECK(H5LTset_attribute_double(file, groupname, "scalefactor", &scale, 1) >= 0);
    for (int i = 0; i < count; ++i)
      integers[i] = forest(i);
    dataset(group, "ForestID", H5T_NATIVE_LONG, integers, count, chunked);
    for (int i = 0; i < count; ++i)
      integers[i] = i == 17 ? id_base + i + 1 : 2 * id_base + i + 2;
    dataset(group, "Head", H5T_NATIVE_LONG, integers, count, chunked);
    for (int i = 0; i < count; ++i)
      integers[i] = satellite(i) ? id_base + HOST + 1 : -1;
    dataset(group, "hostHaloID", H5T_NATIVE_LONG, integers, count, chunked);
    for (int c = 0; c < 11; ++c) {
      for (int i = 0; i < count; ++i)
        floats[i] = property(c, i);
      dataset(group, names[c], H5T_NATIVE_FLOAT, floats, count, chunked);
    }
    for (int i = 0; i < count; ++i)
      unsigneds[i] = id_base + i + 1;
    dataset(group, "ID", H5T_NATIVE_ULONG, unsigneds, count, chunked);
    for (int i = 0; i < count; ++i)
      unsigneds[i] = 30 + i % 101;
    dataset(group, "npart", H5T_NATIVE_ULONG, unsigneds, count, chunked);
    H5Gclose(group);
  }
  free(integers);
  free(unsigneds);
  free(floats);
  H5Fclose(file);
}

static void check_case(int mode, int snapshot)
{
  long subset[] = { 10, 40 }, absent = 999;
  int empty_rank = run_globals.mpi_size > 1 && run_globals.mpi_rank == run_globals.mpi_size - 1;
  run_globals.RequestedForestId = mode == 1 ? subset : mode == 2 || mode == 4 ? &absent : NULL;
  run_globals.NRequestedForests = mode == 0 ? -1 : mode == 1 ? 2 : mode == 4 ? 1 : 0;
  if (mode == 1 && empty_rank) {
    run_globals.RequestedForestId = NULL;
    run_globals.NRequestedForests = 0;
  }
  halo_t* halos = calloc(NROWS, sizeof(*halos));
  fof_group_t* groups = calloc(NROWS, sizeof(*groups));
  int* lookup = calloc(NROWS, sizeof(*lookup));
  CHECK(halos && groups && lookup);
  int count = -1, group_count = -1;
  read_trees__velociraptor(snapshot, halos, &count, groups, &group_count, mode == 0 ? NULL : lookup);
  int expected_count = 0, expected_groups = 0, host_index = -1;
  for (int row = 0; snapshot == 1 && row < NROWS; ++row) {
    int keep = mode == 0 || (mode == 1 && !empty_rank && (forest(row) == 10 || forest(row) == 40));
    if (!keep)
      continue;
    halo_t* halo = &halos[expected_count];
    if (mode != 0)
      CHECK(lookup[expected_count] == row);
    CHECK(halo->ID == (unsigned long)(id_base + row + 1));
    CHECK(halo->DescIndex == (row == 17 ? -1 : row + 1));
    CHECK(halo->SnapOffset == (row == 17 ? 0 : 1));
    CHECK(halo->ProgIndex == -1 && halo->TreeFlags == TREE_CASE_NO_PROGENITORS);
    CHECK(halo->Type == satellite(row));
    CHECK(halo->Len == 30 + row % 101 && halo->Galaxy == NULL);
    CHECK(halo->Mvir == property(1, row) * 0.5 && halo->Vmax == property(3, row));
    CHECK(halo->AngMom == property(10, row) * 0.5);
    CHECK(isfinite(halo->Rvir) && halo->Rvir > 0 && isfinite(halo->Vvir) && halo->Vvir > 0);
    for (int k = 0; k < 3; ++k) {
      float position = property(k + 4, row);
      if (position < 0)
        position += 100;
      if (position >= 100)
        position -= 100;
      CHECK(halo->Pos[k] == position && halo->Vel[k] == property(k + 7, row) * 2);
    }
    if (row == HOST)
      host_index = expected_count;
    if (satellite(row)) {
      CHECK(host_index >= 0 && halo->FOFGroup == halos[host_index].FOFGroup);
      if (row == 10000)
        CHECK(halos[host_index].NextHaloInFOFGroup == halo);
      if (row == 10002)
        CHECK(halos[host_index].NextHaloInFOFGroup->NextHaloInFOFGroup == halo);
    } else {
      CHECK(halo->FOFGroup == &groups[expected_groups]);
      CHECK(halo->FOFGroup->FirstHalo == halo);
      CHECK(halo->FOFGroup->Mvir == property(0, row) * 0.5);
      CHECK(halo->FOFGroup->Rvir == property(2, row) * 0.5);
      CHECK(halo->FOFGroup->FOFMvirModifier == 1);
      if (row != HOST)
        CHECK(halo->NextHaloInFOFGroup == NULL);
      ++expected_groups;
    }
    ++expected_count;
  }
  CHECK(count == expected_count && group_count == expected_groups);
  free(halos);
  free(groups);
  free(lookup);
  run_globals.RequestedForestId = NULL;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);
  init_mlog(MPI_COMM_WORLD, stdout, stdout, stderr);
  char root[STRLEN] = "./meraxes-tree-test-XXXXXX";
  if (run_globals.mpi_rank == 0) {
    CHECK(mkdtemp(root) != NULL);
    fixture(root, "trees", 0);
    fixture(root, "augmented_trees", 1);
  }
  MPI_Bcast(root, sizeof(root), MPI_CHAR, 0, MPI_COMM_WORLD);
  strcpy(run_globals.params.SimulationDir, root);
  strcpy(run_globals.params.CatalogFilePrefix, "fixture.h5");
  run_globals.params.Hubble_h = 0.5;
  run_globals.params.BoxSize = 100;
  run_globals.params.OmegaM = 1;
  run_globals.params.FlagIgnoreProgIndex = 1;
  run_globals.G = 1;
  double redshifts[] = { 0, 1, 2 }, critical[] = { 1, 1, 1 };
  run_globals.ZZ = redshifts;
  run_globals.rhocrit = critical;
  for (int layout = 0; layout < 2; ++layout) {
    run_globals.params.TreesID = layout == 0 ? VELOCIRAPTOR_TREES : VELOCIRAPTOR_TREES_AUG;
    for (int mode = 0; mode < 5; ++mode)
      for (int snapshot = 1; snapshot <= 2; ++snapshot)
        check_case(mode, snapshot);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (run_globals.mpi_rank == 0) {
    char path[STRLEN + 64];
    const char* dirs[] = { "trees", "augmented_trees" };
    for (int i = 0; i < 2; ++i) {
      snprintf(path, sizeof(path), "%s/%s/fixture.h5", root, dirs[i]);
      CHECK(unlink(path) == 0);
      snprintf(path, sizeof(path), "%s/%s", root, dirs[i]);
      CHECK(rmdir(path) == 0);
    }
    CHECK(rmdir(root) == 0);
    puts("Tree reader MPI checks passed: layouts, selection, row order, FOF links, properties, empty snapshots.");
  }
  MPI_Finalize();
  return 0;
}

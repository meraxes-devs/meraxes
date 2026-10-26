#include <assert.h>
#include <hdf5_hl.h>
#include <limits.h>
#include <math.h>

#include "cpu_utils.h"
#include "debug.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "modifiers.h"
#include "read_halos.h"
#include "tree_flags.h"
#include "virial_properties.h"

trees_info_t read_trees_info__velociraptor(const int snapshot)
{
  trees_info_t trees_info;

  if (run_globals.mpi_rank == 0) {
    // TODO: This could maybe only ever be done once and stored in run_globals.
    char fname[STRLEN + 34];
    switch (run_globals.params.TreesID) {
      case VELOCIRAPTOR_TREES:
        sprintf(fname, "%s/trees/meraxes_augmented_stats.h5", run_globals.params.SimulationDir);
        break;
      case VELOCIRAPTOR_TREES_AUG:
        sprintf(fname, "%s/augmented_trees/meraxes_augmented_stats.h5", run_globals.params.SimulationDir);
        break;
      default:
        mlog_error("Unrecognised input trees identifier (TreesID).");
        break;
    }

    hid_t fd = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fd < 0) {
      mlog("Failed to open file %s", MLOG_MESG, fname);
      ABORT(EXIT_FAILURE);
    }

    int n_snaps = 0;
    H5LTget_attribute_int(fd, "/", "n_snaps", &n_snaps);
    H5LTget_attribute_int(fd, "/", "n_halos_max", &(trees_info.n_halos_max));
    H5LTget_attribute_int(fd, "/", "n_fof_groups_max", &(trees_info.n_fof_groups_max));

    int* buffer = malloc(sizeof(int) * n_snaps);
    H5LTread_dataset_int(fd, "n_halos", buffer);
    trees_info.n_halos = buffer[snapshot];
    H5LTread_dataset_int(fd, "n_fof_groups", buffer);
    trees_info.n_fof_groups = buffer[snapshot];
    free(buffer);

    H5Fclose(fd);
  }

  // broadcast the snapshot info
  MPI_Bcast(&trees_info, sizeof(trees_info_t), MPI_BYTE, 0, run_globals.mpi_comm);

  return trees_info;
}

static int id_to_ind(long id)
{
  return (int)(((uint64_t)id % (uint64_t)1e12) - 1);
}

static int id_to_snap(long id)
{
  return (int)(id / 1e12l);
}

inline static void convert_input_virial_props(double* Mvir,
                                              double* Rvir,
                                              double* Vvir,
                                              double* FOFMvirModifier,
                                              const int len,
                                              const int snapshot,
                                              const bool fof_flag)
{
  // Update the virial properties for subhalos
  if (*Mvir == -1) {
    assert(len > 0);
    *Mvir = calculate_Mvir(*Mvir, len);
  } else {
    if (fof_flag && (run_globals.RequestedMassRatioModifier == 1)) {
      // Modifier the FoF mass and update the virial radius
      assert(FOFMvirModifier != NULL);
      *FOFMvirModifier =
        interpolate_modifier(run_globals.mass_ratio_modifier, log10(*Mvir / run_globals.params.Hubble_h) + 10.0);
      *Mvir *= *FOFMvirModifier;
    }
  }

  if (*Rvir == -1)
    *Rvir = calculate_Rvir(*Mvir, snapshot);

  if (*Vvir == -1)
    *Vvir = calculate_Vvir(*Mvir, *Rvir);
}

void read_trees__velociraptor(int snapshot,
                              halo_t* halos,
                              int* n_halos,
                              fof_group_t* fof_groups,
                              int* n_fof_groups,
                              int* index_lookup)
{
  switch (run_globals.params.TreesID) {
    case VELOCIRAPTOR_TREES:
      mlog("Reading velociraptor trees for snapshot %d...", MLOG_OPEN, snapshot);
      break;
    case VELOCIRAPTOR_TREES_AUG:
      mlog("Reading velociraptor augmented trees for snapshot %d...", MLOG_OPEN, snapshot);
      break;
    default:
      mlog_error("Unrecognised input trees identifier (TreesID).");
      break;
  }

  int n_tree_entries = 0;
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, run_globals.mpi_comm, MPI_INFO_NULL);

  double mass_unit_to_internal = 1.0;
  double scale_factor = -999.;

  *n_halos = 0;
  *n_fof_groups = 0;

  char fname[STRLEN * 2 + 8];
  switch (run_globals.params.TreesID) {
    case VELOCIRAPTOR_TREES:
      sprintf(fname, "%s/trees/%s", run_globals.params.SimulationDir, run_globals.params.CatalogFilePrefix);
      break;
    case VELOCIRAPTOR_TREES_AUG:
      sprintf(fname, "%s/augmented_trees/%s", run_globals.params.SimulationDir, run_globals.params.CatalogFilePrefix);
      break;
    default:
      mlog_error("Unrecognised input trees identifier (TreesID).");
      break;
  }

  hid_t fd = H5Fopen(fname, H5F_ACC_RDONLY, plist_id);
  if (fd < 0) {
    mlog("Failed to open file %s", MLOG_MESG, fname);
    ABORT(EXIT_FAILURE);
  }
  H5Pclose(plist_id);

  char snap_group_name[9];
  sprintf(snap_group_name, "Snap_%03d", snapshot);
  hid_t snap_group = H5Gopen(fd, snap_group_name, H5P_DEFAULT);

  H5LTget_attribute_int(fd, snap_group_name, "NHalos", &n_tree_entries);

  // check the units
  H5LTget_attribute_double(fd, "Header/Units", "Mass_unit_to_solarmass", &mass_unit_to_internal);
  mass_unit_to_internal /= 1.0e10;
  H5LTget_attribute_double(fd, snap_group_name, "scalefactor", &scale_factor);

  // Keep disk access contiguous. A directory of every forest and scattered
  // HDF5 point selections were slower than streaming on the production trees.
  // Bound the per-rank column workspace, including at late snapshots.
  size_t buffer_size = n_tree_entries > 100000 ? (size_t)n_tree_entries / 10 : 10000;
  if (buffer_size > 1048576)
    buffer_size = 1048576;
  if (buffer_size > (size_t)n_tree_entries)
    buffer_size = (size_t)n_tree_entries;
  long* ForestID = cpu_checked_alloc(buffer_size, sizeof(long));
  long* Head = cpu_checked_alloc(buffer_size, sizeof(long));
  long* hostHaloID = cpu_checked_alloc(buffer_size, sizeof(long));
  float* Mass_200crit = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Mass_tot = cpu_checked_alloc(buffer_size, sizeof(float));
  float* R_200crit = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Vmax = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Xc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Yc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Zc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* VXc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* VYc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* VZc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* AngMom = cpu_checked_alloc(buffer_size, sizeof(float));
  unsigned long* ID = cpu_checked_alloc(buffer_size, sizeof(unsigned long));
  unsigned long* npart = cpu_checked_alloc(buffer_size, sizeof(unsigned long));

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT); // or H5FD_MPIO_COLLECTIVE?
  hid_t fspace_id = H5Screate_simple(1, (hsize_t[1]){ n_tree_entries }, NULL);

  double hubble_h = run_globals.params.Hubble_h;

  struct
  {
    const char* name;
    hid_t type, dataset;
    void* data;
  } columns[] = {
    { "ForestID", H5T_NATIVE_LONG, -1, ForestID },
    { "Head", H5T_NATIVE_LONG, -1, Head },
    { "hostHaloID", H5T_NATIVE_LONG, -1, hostHaloID },
    { "Mass_200crit", H5T_NATIVE_FLOAT, -1, Mass_200crit },
    { "Mass_tot", H5T_NATIVE_FLOAT, -1, Mass_tot },
    { "R_200crit", H5T_NATIVE_FLOAT, -1, R_200crit },
    { "Vmax", H5T_NATIVE_FLOAT, -1, Vmax },
    { "Xc", H5T_NATIVE_FLOAT, -1, Xc },
    { "Yc", H5T_NATIVE_FLOAT, -1, Yc },
    { "Zc", H5T_NATIVE_FLOAT, -1, Zc },
    { "VXc", H5T_NATIVE_FLOAT, -1, VXc },
    { "VYc", H5T_NATIVE_FLOAT, -1, VYc },
    { "VZc", H5T_NATIVE_FLOAT, -1, VZc },
    { "AngMom", H5T_NATIVE_FLOAT, -1, AngMom },
    { "ID", H5T_NATIVE_ULONG, -1, ID },
    { "npart", H5T_NATIVE_ULONG, -1, npart },
  };
  const size_t n_columns = sizeof(columns) / sizeof(columns[0]);
  // All ranks open/close datasets in the same order, including ranks with no
  // selected rows. Only the independent H5Dread calls vary between ranks.
  for (size_t p = 0; p < n_columns; ++p) {
    columns[p].dataset = H5Dopen(snap_group, columns[p].name, H5P_DEFAULT);
    if (columns[p].dataset < 0)
      ABORT(EXIT_FAILURE);
  }
  int* selected = cpu_checked_alloc(buffer_size, sizeof(int));
  size_t n_read = 0;
  while (n_read < (size_t)n_tree_entries) {
    int n_to_read =
      (int)(((size_t)n_tree_entries - n_read < buffer_size) ? (size_t)n_tree_entries - n_read : buffer_size);
    if (H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, (hsize_t[1]){ n_read }, NULL, (hsize_t[1]){ n_to_read }, NULL) <
        0)
      ABORT(EXIT_FAILURE);
    hid_t memspace_id = H5Screate_simple(1, (hsize_t[1]){ n_to_read }, NULL);

    // Different ranks read different columns concurrently. Each column is read
    // once from disk, then broadcast, as in the original streaming reader.
    for (size_t p = 0; p < n_columns; ++p) {
      int reader = (int)((p + 1) % run_globals.mpi_size);
      if (run_globals.mpi_rank == reader &&
          H5Dread(columns[p].dataset, columns[p].type, memspace_id, fspace_id, plist_id, columns[p].data) < 0)
        ABORT(EXIT_FAILURE);
    }
    for (size_t p = 0; p < n_columns; ++p) {
      size_t bytes = H5Tget_size(columns[p].type) * (size_t)n_to_read;
      if (bytes > INT_MAX)
        ABORT(EXIT_FAILURE);
      MPI_Bcast(columns[p].data, (int)bytes, MPI_BYTE, (int)((p + 1) % run_globals.mpi_size), run_globals.mpi_comm);
    }
    H5Sclose(memspace_id);

    int n_selected = 0;
    for (int ii = 0; ii < n_to_read; ++ii) {
      bool keep = run_globals.RequestedForestId == NULL && run_globals.NRequestedForests < 0;
      if (run_globals.RequestedForestId != NULL && run_globals.NRequestedForests > 0)
        keep = bsearch(&ForestID[ii],
                       run_globals.RequestedForestId,
                       (size_t)run_globals.NRequestedForests,
                       sizeof(long),
                       compare_longs) != NULL;
      if (keep)
        selected[n_selected++] = ii;
    }

    for (int selected_index = 0; selected_index < n_selected; ++selected_index) {
      int ii = selected[selected_index];
      halo_t* halo = &(halos[*n_halos]);

      halo->ID = ID[ii];
      halo->DescIndex = id_to_ind(Head[ii]);

      if (run_globals.params.FlagIgnoreProgIndex)
        halo->ProgIndex = -1;
      // else
      //   halo->ProgIndex = id_to_ind(Tail[ii]);

      halo->NextHaloInFOFGroup = NULL;
      halo->Type = hostHaloID[ii] == -1 ? 0 : 1;
      halo->SnapOffset = id_to_snap(Head[ii]) - snapshot;

      // Any other tree flags need to be set using both the current and
      // progenitor halo information (stored in the galaxy), therefore we
      // need to leave setting those until later...
      if (run_globals.params.FlagIgnoreProgIndex)
        halo->TreeFlags = TREE_CASE_NO_PROGENITORS;
      // else
      //   halo->TreeFlags = (unsigned long)Tail[ii] != ID[ii] ? 0 : TREE_CASE_NO_PROGENITORS;

      // Here we have a cyclic pointer, indicating that this halo's life ends here
      if ((unsigned long)Head[ii] == ID[ii])
        halo->DescIndex = -1;

      if (index_lookup)
        index_lookup[*n_halos] = (int)n_read + ii;

      // TODO: What masses and radii should I use for centrals (inclusive vs. exclusive etc.)?
      if (halo->Type == 0) {
        fof_group_t* fof_group = &fof_groups[*n_fof_groups];

        // This check is to ensure sensible values of mass_200crit and avoid having
        // very weird halos. Put this check back if you feel that the N-body is weird.

        /*if ((Mass_200crit[ii] < 5 * Mass_tot[ii])) {
            fof_group->Mvir = Mass_200crit[ii] * hubble_h * mass_unit_to_internal;;
            fof_group->Rvir = R_200crit[ii] * hubble_h;
        }*/
        // else {
        //  BELOW_VIRIAL_THRESHOLD merger halo swammping
        if (Mass_200crit[ii] <= 0) {
          halo->TreeFlags |= TREE_CASE_BELOW_VIRIAL_THRESHOLD;
          fof_group->Mvir = Mass_tot[ii] * hubble_h * mass_unit_to_internal;
          fof_group->Rvir = -1;
        }

        else {
          fof_group->Mvir = (double)Mass_200crit[ii] * hubble_h * mass_unit_to_internal;
          fof_group->Rvir = (double)R_200crit[ii] * hubble_h;
        }

        fof_group->Vvir = -1;
        fof_group->FOFMvirModifier = 1.0;

        convert_input_virial_props(
          &fof_group->Mvir, &fof_group->Rvir, &fof_group->Vvir, &fof_group->FOFMvirModifier, -1, snapshot, true);

        halo->FOFGroup = &(fof_groups[*n_fof_groups]);
        fof_groups[(*n_fof_groups)++].FirstHalo = halo;
      } else {
        // We can take advantage of the fact that host halos always
        // seem to appear before their subhalos (checked below) in the
        // trees to immediately connect FOF group members.
        int host_index = id_to_ind(hostHaloID[ii]);

        if (index_lookup)
          host_index = find_original_index(host_index, index_lookup, *n_halos);

        assert(host_index > -1);
        assert(host_index < *n_halos);

        halo_t* prev_halo = &halos[host_index];
        halo->FOFGroup = prev_halo->FOFGroup;

        while (prev_halo->NextHaloInFOFGroup != NULL)
          prev_halo = prev_halo->NextHaloInFOFGroup;

        prev_halo->NextHaloInFOFGroup = halo;
      }

      halo->Len = (int)npart[ii];
      halo->Pos[0] = apply_pbc_pos(Xc[ii] * hubble_h / scale_factor);
      halo->Pos[1] = apply_pbc_pos(Yc[ii] * hubble_h / scale_factor);
      halo->Pos[2] = apply_pbc_pos(Zc[ii] * hubble_h / scale_factor);
      halo->Vel[0] = VXc[ii] / scale_factor;
      halo->Vel[1] = VYc[ii] / scale_factor;
      halo->Vel[2] = VZc[ii] / scale_factor;
      halo->Vmax = Vmax[ii];

      // TODO: What masses and radii should I use for satellites (inclusive vs. exclusive etc.)?
      halo->Mvir = (double)Mass_tot[ii] * hubble_h * mass_unit_to_internal;
      halo->Rvir = -1;
      halo->Vvir = -1;
      convert_input_virial_props(&halo->Mvir, &halo->Rvir, &halo->Vvir, NULL, -1, snapshot, false);

      halo->AngMom = AngMom[ii] * hubble_h;
      halo->Galaxy = NULL;

      (*n_halos)++;
    }

    n_read += n_to_read;
  }

  for (size_t p = 0; p < n_columns; ++p)
    H5Dclose(columns[p].dataset);
  free(selected);
  free(ForestID);
  free(Head);
  free(hostHaloID);
  free(Mass_200crit);
  free(Mass_tot);
  free(R_200crit);
  free(Vmax);
  free(Xc);
  free(Yc);
  free(Zc);
  free(VXc);
  free(VYc);
  free(VZc);
  free(AngMom);
  free(ID);
  free(npart);
  H5Pclose(plist_id);
  H5Sclose(fspace_id);
  H5Gclose(snap_group);
  H5Fclose(fd);

  mlog("...done", MLOG_CLOSE);
}

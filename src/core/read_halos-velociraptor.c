#include <assert.h>
#include <hdf5_hl.h>
#include <math.h>

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

  int mpi_size = run_globals.mpi_size;
  int mpi_rank = run_globals.mpi_rank;

  char snap_group_name[9];
  sprintf(snap_group_name, "Snap_%03d", snapshot);
  hid_t snap_group = H5Gopen(fd, snap_group_name, H5P_DEFAULT);

  H5LTget_attribute_int(fd, snap_group_name, "NHalos", &n_tree_entries);

  // check the units
  H5LTget_attribute_double(fd, "Header/Units", "Mass_unit_to_solarmass", &mass_unit_to_internal);
  mass_unit_to_internal /= 1.0e10;
  H5LTget_attribute_double(fd, snap_group_name, "scalefactor", &scale_factor);

  // Currently the chunk size is 10000, improving ~10% w.r.t. no chunk or 1k chunk
  int buffer_size = (n_tree_entries > 100000) ? n_tree_entries / 10 : 10000;
  buffer_size = buffer_size > n_tree_entries ? n_tree_entries : buffer_size;

  long* ForestID = malloc(sizeof(long) * buffer_size);
  long* Head = malloc(sizeof(long) * buffer_size);
  long* hostHaloID = malloc(sizeof(long) * buffer_size);
  float* Mass_200crit = malloc(sizeof(float) * buffer_size);
  float* Mass_tot = malloc(sizeof(float) * buffer_size);
  float* R_200crit = malloc(sizeof(float) * buffer_size);
  float* Vmax = malloc(sizeof(float) * buffer_size);
  float* Xc = malloc(sizeof(float) * buffer_size);
  float* Yc = malloc(sizeof(float) * buffer_size);
  float* Zc = malloc(sizeof(float) * buffer_size);
  float* VXc = malloc(sizeof(float) * buffer_size);
  float* VYc = malloc(sizeof(float) * buffer_size);
  float* VZc = malloc(sizeof(float) * buffer_size);
#if USE_ANG_MOM
  float* Lx = malloc(sizeof(float) * buffer_size);
  float* Ly = malloc(sizeof(float) * buffer_size);
  float* Lz = malloc(sizeof(float) * buffer_size);
#else
  float* AngMom = malloc(sizeof(float) * buffer_size);
#endif
  unsigned long* ID = malloc(sizeof(unsigned long) * buffer_size); 
  unsigned long* npart = malloc(sizeof(unsigned long) * buffer_size);

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT); // or H5FD_MPIO_COLLECTIVE?
  hid_t fspace_id = H5Screate_simple(1, (hsize_t[1]){ n_tree_entries }, NULL);

  double hubble_h = run_globals.params.Hubble_h;
  double box_size = run_globals.params.BoxSize;

  int n_read = 0;
  int n_to_read = buffer_size;
  while (n_read < n_tree_entries) {
    int n_remaining = n_tree_entries - n_read;
    if (n_remaining < n_to_read)
      n_to_read = n_remaining;


    // select a hyperslab in the filespace
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, (hsize_t[1]){ n_read }, NULL, (hsize_t[1]){ n_to_read }, NULL);
    hid_t memspace_id = H5Screate_simple(1, (hsize_t[1]){ n_to_read }, NULL);

#define READ_TREE_ENTRY_PROP(name, type, h5type, target_rank)                                                          \
{                                                                                                                      \
  if (mpi_rank == target_rank % mpi_size){                                                                             \
    hid_t dset_id = H5Dopen(snap_group, #name, H5P_DEFAULT);                                                           \
    herr_t status = H5Dread(dset_id, h5type, memspace_id, fspace_id, plist_id, name);                                  \
    assert(status >= 0);                                                                                               \
    H5Dclose(dset_id);                                                                                                 \
  }                                                                                                                    \
  MPI_Bcast(name, sizeof(type) * n_to_read, MPI_BYTE, target_rank % mpi_size, run_globals.mpi_comm);                   \
}                                                                                                                      \

	READ_TREE_ENTRY_PROP(ForestID,     long, H5T_NATIVE_LONG, 1);
	READ_TREE_ENTRY_PROP(Head,         long, H5T_NATIVE_LONG, 2);
	READ_TREE_ENTRY_PROP(hostHaloID,   long, H5T_NATIVE_LONG, 3);
	READ_TREE_ENTRY_PROP(Mass_200crit, float, H5T_NATIVE_FLOAT, 4);
	READ_TREE_ENTRY_PROP(Mass_tot,     float, H5T_NATIVE_FLOAT, 5);
	READ_TREE_ENTRY_PROP(R_200crit,    float, H5T_NATIVE_FLOAT, 6);
	READ_TREE_ENTRY_PROP(Vmax,         float, H5T_NATIVE_FLOAT, 7);
	READ_TREE_ENTRY_PROP(Xc,           float, H5T_NATIVE_FLOAT, 8);
	READ_TREE_ENTRY_PROP(Yc,           float, H5T_NATIVE_FLOAT, 9);
	READ_TREE_ENTRY_PROP(Zc,           float, H5T_NATIVE_FLOAT, 10);                                                                  
	READ_TREE_ENTRY_PROP(VXc,          float, H5T_NATIVE_FLOAT, 11);                                                                  
	READ_TREE_ENTRY_PROP(VYc,          float, H5T_NATIVE_FLOAT, 12);                                                                  
	READ_TREE_ENTRY_PROP(VZc,          float, H5T_NATIVE_FLOAT, 13);
#if USE_ANG_MOM
        READ_TREE_ENTRY_PROP(Lx,           float, H5T_NATIVE_FLOAT, 14);
        READ_TREE_ENTRY_PROP(Ly,           float, H5T_NATIVE_FLOAT, 15);
        READ_TREE_ENTRY_PROP(Lz,           float, H5T_NATIVE_FLOAT, 16);                                                                  
        READ_TREE_ENTRY_PROP(ID,           unsigned long, H5T_NATIVE_ULONG, 17);                                                          
        READ_TREE_ENTRY_PROP(npart,        unsigned long, H5T_NATIVE_ULONG, 18);
#else                                                                  
	READ_TREE_ENTRY_PROP(AngMom,       float, H5T_NATIVE_FLOAT, 14);                                                                  
	READ_TREE_ENTRY_PROP(ID,           unsigned long, H5T_NATIVE_ULONG, 15);                                                          
	READ_TREE_ENTRY_PROP(npart,        unsigned long, H5T_NATIVE_ULONG, 16); 
#endif                                                         

    H5Sclose(memspace_id);

    for (int ii = 0; ii < n_to_read; ++ii) {
      bool keep_this_halo = true;

      if ((run_globals.RequestedForestId != NULL) && (bsearch(&(ForestID[ii]),
                                                              run_globals.RequestedForestId,
                                                              (size_t)run_globals.NRequestedForests,
                                                              sizeof(long),
                                                              compare_longs)) == NULL)
        keep_this_halo = false;

      if (keep_this_halo) {
        halo_t* halo = &(halos[*n_halos]);

        halo->ID = ID[ii];
        halo->DescIndex = id_to_ind(Head[ii]);

        if (run_globals.params.FlagIgnoreProgIndex)
          halo->ProgIndex = -1;
        //else
        //  halo->ProgIndex = id_to_ind(Tail[ii]);

        halo->NextHaloInFOFGroup = NULL;
        halo->Type = hostHaloID[ii] == -1 ? 0 : 1;
        halo->SnapOffset = id_to_snap(Head[ii]) - snapshot;

        // Any other tree flags need to be set using both the current and
        // progenitor halo information (stored in the galaxy), therefore we
        // need to leave setting those until later...
        if (run_globals.params.FlagIgnoreProgIndex)
          halo->TreeFlags = TREE_CASE_NO_PROGENITORS;
        //else
        //  halo->TreeFlags = (unsigned long)Tail[ii] != ID[ii] ? 0 : TREE_CASE_NO_PROGENITORS;

        // Here we have a cyclic pointer, indicating that this halo's life ends here
        if ((unsigned long)Head[ii] == ID[ii])
          halo->DescIndex = -1;

        if (index_lookup)
          index_lookup[*n_halos] = ii + n_read;

        // TODO: What masses and radii should I use for centrals (inclusive vs. exclusive etc.)?
        if (halo->Type == 0) {
          fof_group_t* fof_group = &fof_groups[*n_fof_groups];
          
          // This check is to ensure sensible values of mass_200crit and avoid having 
          // very weird halos. Put this check back if you feel that the N-body is weird.
          
          /*if ((Mass_200crit[ii] < 5 * Mass_tot[ii])) {
              fof_group->Mvir = Mass_200crit[ii] * hubble_h * mass_unit_to_internal;;
              fof_group->Rvir = R_200crit[ii] * hubble_h;
          }*/
          //else {
            // BELOW_VIRIAL_THRESHOLD merger halo swammping
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
        halo->Pos[0] = fmax(0.0, fmin(Xc[ii] * hubble_h / scale_factor, box_size));
        halo->Pos[1] = fmax(0.0, fmin(Yc[ii] * hubble_h / scale_factor, box_size));
        halo->Pos[2] = fmax(0.0, fmin(Zc[ii] * hubble_h / scale_factor, box_size));
        halo->Vel[0] = VXc[ii] / scale_factor;
        halo->Vel[1] = VYc[ii] / scale_factor;
        halo->Vel[2] = VZc[ii] / scale_factor;
        halo->Vmax = Vmax[ii]; 

        // TODO: What masses and radii should I use for satellites (inclusive vs. exclusive etc.)?
        halo->Mvir = (double)Mass_tot[ii] * hubble_h * mass_unit_to_internal;
        halo->Rvir = -1;
        halo->Vvir = -1;
        convert_input_virial_props(&halo->Mvir, &halo->Rvir, &halo->Vvir, NULL, -1, snapshot, false);

#if USE_ANG_MOM
        halo->AngMom[0] = Lx[ii] / Mass_tot[ii] * hubble_h;
        halo->AngMom[1] = Ly[ii] / Mass_tot[ii] * hubble_h;
        halo->AngMom[2] = Lz[ii] / Mass_tot[ii] * hubble_h;
#else
        halo->AngMom = AngMom[ii] * hubble_h;
#endif
        halo->Galaxy = NULL;

        (*n_halos)++;
      }
    }

    n_read += n_to_read;
  }

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
#if USE_ANG_MOM
  free(Lx);
  free(Ly);
  free(Lz);
#else
  free(AngMom);
#endif
  free(ID);
  free(npart);
  H5Pclose(plist_id);
  H5Sclose(fspace_id);
  H5Gclose(snap_group);
  H5Fclose(fd);

  mlog("...done", MLOG_CLOSE);
}

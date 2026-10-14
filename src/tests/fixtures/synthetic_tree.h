#ifndef SYNTHETIC_TREE_H
#define SYNTHETIC_TREE_H

#include "meraxes.h"

#define SYNTHETIC_TREE_NSNAPS 4
#define SYNTHETIC_TREE_NGALAXIES 2

typedef struct
{
  int n_halos;
  int n_fof_groups;
  halo_t halos[2];
  fof_group_t fof_groups[2];
} synthetic_snapshot_t;

typedef struct
{
  synthetic_snapshot_t snapshots[SYNTHETIC_TREE_NSNAPS];
  double scale_factors[SYNTHETIC_TREE_NSNAPS];
  double redshifts[SYNTHETIC_TREE_NSNAPS];
  double lookback_times[SYNTHETIC_TREE_NSNAPS];
  galaxy_t* galaxies[SYNTHETIC_TREE_NGALAXIES];
  int snapshot;
  int n_galaxies;
  int n_created;
  int n_halo_mergers;
} synthetic_tree_t;

// Owns run_globals for an isolated test process; do not use in a running simulation.
// The fixture must stay at the same address until synthetic_tree_free().
int synthetic_tree_init(synthetic_tree_t* tree);
// Attach the existing galaxies to the next snapshot; physics is driven by the test.
int synthetic_tree_advance(synthetic_tree_t* tree);
void synthetic_tree_finish_snapshot(synthetic_tree_t* tree);
void synthetic_tree_free(synthetic_tree_t* tree);

#endif

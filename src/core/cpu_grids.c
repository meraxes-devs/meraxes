#include <stdlib.h>

#include "cpu_grids.h"
#include "cpu_utils.h"
#include "misc_tools.h"

typedef struct
{
  uint64_t cell;
  int rank, galaxy;
} cell_key_t;

static int compare_cell_key(const void* a, const void* b)
{
  const cell_key_t *x = a, *y = b;
  if (x->rank != y->rank)
    return (x->rank > y->rank) - (x->rank < y->rank);
  if (x->cell != y->cell)
    return (x->cell > y->cell) - (x->cell < y->cell);
  return (x->galaxy > y->galaxy) - (x->galaxy < y->galaxy);
}

cell_layout_t cells_for_galaxies(const gal_to_slab_t* map, int n)
{
  const int dim = run_globals.params.ReionGridDim;
  const double box = run_globals.params.BoxSize;
  const int ranks = run_globals.mpi_size;
  cell_layout_t layout = { 0 };
  layout.cell = cpu_checked_alloc(n, sizeof(*layout.cell));
  layout.slot = cpu_checked_alloc(n, sizeof(*layout.slot));
  layout.send_count = cpu_checked_alloc(ranks, sizeof(*layout.send_count));
  cell_key_t* keys = cpu_checked_alloc(n, sizeof(*keys));
  for (int i = 0; i < n; ++i) {
    const galaxy_t* gal = map[i].galaxy;
    int p = map[i].slab_ind;
    if (p < 0 || p >= ranks)
      ABORT(EXIT_FAILURE);
    ptrdiff_t x = pos_to_ngp(gal->Pos[0], box, dim) - run_globals.reion_grids.slab_ix_start[p];
    int y = pos_to_ngp(gal->Pos[1], box, dim);
    int z = pos_to_ngp(gal->Pos[2], box, dim);
    if (x < 0 || x >= run_globals.reion_grids.slab_nix[p])
      ABORT(EXIT_FAILURE);
    keys[i] = (cell_key_t){ z + (uint64_t)dim * (y + (uint64_t)dim * x), p, i };
  }
  qsort(keys, n, sizeof(*keys), compare_cell_key);
  for (int i = 0; i < n; ++i) {
    if (!i || keys[i].rank != keys[i - 1].rank || keys[i].cell != keys[i - 1].cell) {
      layout.cell[layout.count++] = keys[i].cell;
      ++layout.send_count[keys[i].rank];
    }
    layout.slot[keys[i].galaxy] = layout.count - 1;
  }
  free(keys);
  return layout;
}

void free_cell_layout(cell_layout_t* layout)
{
  free(layout->cell);
  free(layout->slot);
  free(layout->send_count);
  *layout = (cell_layout_t){ 0 };
}

float* sample_grid_for_galaxies(const float* grid, const gal_to_slab_t* map, int n)
{
  cell_layout_t layout = cells_for_galaxies(map, n);
  size_t *counts, *offsets, *back_counts, *back_offsets;
  uint64_t* requests =
    exchange_records(layout.cell, layout.send_count, sizeof(uint64_t), &counts, &offsets, run_globals.mpi_comm);
  const int dim = run_globals.params.ReionGridDim;
  uint64_t local_cells = (uint64_t)run_globals.reion_grids.slab_nix[run_globals.mpi_rank] * dim * dim;
  size_t received = offsets[run_globals.mpi_size];
  float* replies = cpu_checked_alloc(received, sizeof(float));
  for (size_t j = 0; j < received; ++j) {
    if (requests[j] >= local_cells || !grid)
      ABORT(EXIT_FAILURE);
    replies[j] = grid[requests[j]]; // Feedback grids have the unpadded real layout.
  }
  float* values = exchange_records(replies, counts, sizeof(float), &back_counts, &back_offsets, run_globals.mpi_comm);
  if (back_offsets[run_globals.mpi_size] != layout.count)
    ABORT(EXIT_FAILURE);
  float* samples = cpu_checked_alloc(n, sizeof(float));
  for (int i = 0; i < n; ++i)
    samples[i] = values[layout.slot[i]];
  free(requests);
  free(replies);
  free(values);
  free(counts);
  free(offsets);
  free(back_counts);
  free(back_offsets);
  free_cell_layout(&layout);
  return samples;
}

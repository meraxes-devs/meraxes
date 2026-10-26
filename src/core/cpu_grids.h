#ifndef MERAXES_CPU_GRIDS_H
#define MERAXES_CPU_GRIDS_H

#include "meraxes.h"
#include "reionization.h"
#include <stdint.h>

typedef struct
{
  size_t count;
  uint64_t* cell;
  size_t* slot;
  size_t* send_count;
} cell_layout_t;

cell_layout_t cells_for_galaxies(const gal_to_slab_t* map, int n);
void free_cell_layout(cell_layout_t* layout);
float* sample_grid_for_galaxies(const float* grid, const gal_to_slab_t* map, int n);

#endif

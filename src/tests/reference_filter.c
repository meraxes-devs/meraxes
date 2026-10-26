#include <math.h>

#include "core/misc_tools.h"
#include "meraxes.h"

// Independent pre-optimization implementation: keep this oracle separate from the production kernel.
void reference_filter(fftwf_complex* box, int local_ix_start, int slab_nx, int grid_dim, float R, int filter_type)
{
  int middle = grid_dim / 2;
  float box_size = (float)run_globals.params.BoxSize;
  float delta_k = (float)(2.0 * M_PI / box_size);

  // Loop through k-box
  for (int n_x = 0; n_x < slab_nx; n_x++) {
    float k_x;
    int n_x_global = n_x + local_ix_start;

    if (n_x_global > middle)
      k_x = (n_x_global - grid_dim) * delta_k;
    else
      k_x = n_x_global * delta_k;

    for (int n_y = 0; n_y < grid_dim; n_y++) {
      float k_y;

      if (n_y > middle)
        k_y = (n_y - grid_dim) * delta_k;
      else
        k_y = n_y * delta_k;

      for (int n_z = 0; n_z <= middle; n_z++) {
        float k_z = n_z * delta_k;

        float k_mag = sqrtf(k_x * k_x + k_y * k_y + k_z * k_z);

        float kR = k_mag * R; // Real space top-hat

        switch (filter_type) {
          case 0: // Real space top-hat
            if (kR > 1e-4)
              box[grid_index(n_x, n_y, n_z, grid_dim, INDEX_COMPLEX_HERM)] *=
                (fftwf_complex)(3.0 * (sinf(kR) / powf(kR, 3) - cosf(kR) / powf(kR, 2)));
            break;

          case 1:              // k-space top hat
            kR *= 0.413566994; // Equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
            if (kR > 1)
              box[grid_index(n_x, n_y, n_z, grid_dim, INDEX_COMPLEX_HERM)] = (fftwf_complex)0.0;
            break;

          case 2:        // Gaussian
            kR *= 0.643; // Equates integrated volume to the real space top-hat
            box[grid_index(n_x, n_y, n_z, grid_dim, INDEX_COMPLEX_HERM)] *=
              (fftwf_complex)(powf((float)M_E, (float)(-kR * kR / 2.0)));
            break;

          default:
            if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {
              mlog_error("ReionFilterType.c: Warning, ReionFilterType type %d is undefined!", filter_type);
              ABORT(EXIT_FAILURE);
            }
            break;
        }
      }
    }
  } // End looping through k box
}

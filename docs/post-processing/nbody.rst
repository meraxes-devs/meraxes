.. _dragons-nbody:

nbody
=====

nbody.io
--------

.. py:module:: dragons.nbody.io

Routines for reading nbody (gbpHalos, gbpTrees etc.) output files.

.. py:function:: read_density_grid(fname)

   .. deprecated:: 0.2.1
      The read_density_grid function is deprecated and may be removed in a future version. Use :py:func:`dragons.nbody.io.read_grid` instead.

   Read in a density grid produced by gbpCode.

   *Args*:
       fname : str
           Full path to input grid file.

   *Returns*:
       grid : array
           The density grid.

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/nbody/io.py#L66-L79>`__

.. py:function:: read_grid(fname, grid_name)

   Read in a real space grid produced by gbpCode.

   *Args*:
       fname : str
           Full path to input grid file.

       grid_name : str
           The name of the grid. Must be either `density`, `vx`, `vy`, or
           `vz`.

   *Returns*:
       grid : ndarray
           The requested grid.

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/nbody/io.py#L82-L132>`__

.. py:function:: read_halo_catalog(catalog_loc)

   Read in a halo catalog produced by gbpCode.

   *Args*:
       catalog_loc : str
           Full path to input catalog file or directory

   *Returns*:
       halo : array
           The catalog of halos

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/nbody/io.py#L135-L172>`__


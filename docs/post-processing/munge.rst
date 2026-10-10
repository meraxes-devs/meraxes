.. _dragons-munge:

munge
=====

.. py:module:: dragons.munge.munge

A collection of functions for doing common processing tasks.

.. py:function:: describe(arr, **kwargs)

   Run scipy.stats.describe and produce legible output.

   :param arr: Numpy ndarray
   :type arr: ndarray
   :param \*\*kwargs: passed to scipy.stats.describe

   :rtype: output of scipy.stats.describe

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/munge.py#L174-L201>`__

.. py:function:: edges_to_centers(edges, width=False)

   Convert **evenly spaced** bin edges to centers.

   :param edges: bin edges
   :type edges: ndarray
   :param width: also return the bin width
   :type width: bool

   :returns: * **centers** (*ndarray*) -- bin centers (size = edges.size-1)
             * **bin_width** (*float*) -- only returned if width = True

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/munge.py#L143-L171>`__

.. py:function:: mass_function(mass, volume, bins, range=None, poisson_uncert=False, return_edges=False, **kwargs)

   Generate a mass function.

   :param mass: an array of 'masses'
   :type mass: ndarray
   :param volume: volume of simulation cube/subset
   :type volume: float
   :param bins: passed to numpy.histogram
   :type bins: int or list or str
   :param range: range of data to be used for mass function
   :type range: len=2 list or array
   :param poisson_uncert: return poisson uncertainties in output array (default: False)
   :type poisson_uncert: bool
   :param return_edges: return the bin_edges (default: False)
   :type return_edges: bool
   :param \*\*kwargs: passed to numpy.histogram

   :returns: [bin centers, mass function vals]
             If poisson_uncert=True then array has 3rd column with uncertainties.
             If return_edges=True then the bin edges are also returned.
   :rtype: array

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/munge.py#L81-L140>`__

.. py:function:: ndarray_to_dataframe(arr, drop_vectors=False)

   Convert numpy ndarray to a pandas DataFrame, dealing with N(>1)
   dimensional datatypes.

   :param arr: Numpy ndarray
   :type arr: ndarray
   :param drop_vectors: only include single value datatypes in output DataFrame
   :type drop_vectors: bool

   :returns: **df** -- Pandas DataFrame
   :rtype: DataFrame

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/munge.py#L42-L78>`__

.. py:function:: power_spectrum(grid, side_length, n_bins, dimensional=False)

   Calculate the dimensionless and dimensional power spectra of a grid (G):

   .. math::

      \Delta^2 (k) = \frac{k^3 V}{2\pi^2} <|\hat G|^2>_k

      P(k) = <|\hat G|^2> V

   :param grid: The grid from which to construct the power spectrum
   :type grid: ndarray
   :param side_length: The side length of the grid (assumes all side lengths are equal)
   :type side_length: float
   :param n_bins: The number of k bins to use
   :type n_bins: int
   :param dimensional: Switch for calculating dimensional power spectrum
                       Default is False
   :type dimensional: Boolean (optional)

   :returns: * **kmean** (*ndarray*) -- The mean wavenumber of each bin
             * **power** (*ndarray*) -- The dimensionless power (:math:`\Delta^2 (k)`)
             * **uncert** (*ndarray*) -- The uncertainty of the dimensionless power within each k bin
             * **power_dim** (*ndarray*) -- The dimensional power (returned iff dimensional = True) (:math:`P(k)`)
             * **uncert_dim** (*ndarray*) -- The uncertainty of the dimensional power within each k bin (returned iff dimensional = True)

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/munge.py#L263-L347>`__

.. py:function:: pretty_print_dict(d, fmtlen=30)

   Pretty print a dictionary, dealing with recursion.

   :param d: the dictionary to print
   :type d: dict
   :param fmtlen: maximum length of dictionary key for print formatting
   :type fmtlen: int

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/munge.py#L18-L39>`__

.. py:function:: smooth_grid(grid, side_length, radius, filt='tophat')

   Smooth a grid by convolution with a filter.

   :param grid: The grid to be smoothed
   :type grid: ndarray
   :param side_length: The side length of the grid (assumes all side lengths are equal)
   :type side_length: float
   :param radius: The radius of the smoothing filter
   :type radius: float
   :param filt: The name of the filter.  Currently only "tophat" (real space) is
                implemented.  More filters will be added over time.
   :type filt: string, optional

   :returns: **smoothed_grid** -- The smoothed grid.
   :rtype: ndarray

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/munge.py#L204-L260>`__

.. py:module:: dragons.munge.regrid

.. py:function:: regrid(old_grid, n_cell)

   Downgrade the resolution of a 3 dimensional grid.

   :param old_grid: Grid to be resampled
   :type old_grid: np.ndarray[float32, ndim=3]
   :param n_cell: n cells per dimension of new grid
   :type n_cell: int

   :rtype: New, degraded resolution grid.

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/munge/regrid.pyx#L17-L51>`__


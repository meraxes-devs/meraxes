.. _dragons-meraxes:

meraxes
=======

meraxes.galaxy_history
----------------------

.. py:module:: dragons.meraxes.galaxy_history

Generate the full (first progenitor line) history of a galaxy.

.. py:function:: galaxy_history(fname, gal_id, snapshot, future_snapshot=-1, pandas=False, props=None)

   Read in the full first progenitor history of a galaxy at a given final
   snapshot.

   :param fname: Full path to input hdf5 master file.
   :type fname: str
   :param gal_id: Unique ID of the target galaxy.
   :type gal_id: int
   :param snapshot: Snapshot at which the history is to be traced from.
   :type snapshot: int
   :param props: A list of galaxy properties requested.  (default: All properties)
   :type props: list
   :param future_snapshot: Also read in the future of the galaxy object up to this snapshot.
                           (default: -1 [don't read in future])
   :type future_snapshot: int
   :param pandas: Return panads dataframe.  (default = False)
   :type pandas: bool

   :returns: * **history** (*ndarray or DataFrame*) -- The requested first progenitor history.  If future=True then the
               ndarray includes the future of this object.
             * **merged_snapshot** (*int*) -- If `future_snapshot != -1` then the snapshot at which the galaxy
               merged into another is also returned.  If `merged_snapshot = -1`
               then the galaxy remains until `future_snapshot.`

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/galaxy_history.py#L13-L94>`__

meraxes.io
----------

.. py:module:: dragons.meraxes.io

Routines for reading Meraxes output files.

.. py:function:: check_for_global_xH(fname, xH, tol=0.1)

   Check a Meraxes output file for the presence of a particular
   global neutral fraction.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param xH: Neutral fraction value
   :type xH: float
   :param tol: +- tolerance on neutral fraction value present.  An error will be
               thrown of no redshift within this tollerance is found.
   :type tol: float

   :returns: * **snapshot** (*int*) -- Closest snapshot
             * **redshift** (*float*) -- Closest corresponding redshift
             * **xH** (*float*) -- Closest corresponding redshift

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L520-L558>`__

.. py:function:: check_for_redshift(fname, redshift, tol=0.1)

   Check a Meraxes output file for the presence of a particular
   redshift.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param redshift: Redshift value
   :type redshift: float
   :param tol: +- tolerance on redshift value present.  An error will be thrown of
               no redshift within this tollerance is found.
   :type tol: float

   :returns: * **snapshot** (*int*) -- Closest snapshot
             * **redshift** (*float*) -- Closest corresponding redshift

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L484-L517>`__

.. py:function:: grab_redshift(fname, snapshot)

   Quickly grab the redshift value of a single snapshot from a Meraxes
   HDF5 file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot for which the redshift is to grabbed
   :type snapshot: int

   :returns: **redshift** -- Corresponding redshift value
   :rtype: float

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L561-L588>`__

.. py:function:: grab_unsampled_snapshot(fname, snapshot)

   Quickly grab the unsampled snapshot value of a single snapshot from a
   Meraxes HDF5 file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot for which the unsampled value is to be grabbed
   :type snapshot: int

   :returns: **redshift** -- Corresponding unsampled snapshot value
   :rtype: float

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L591-L613>`__

.. py:function:: list_grids(spec, fname, snapshot)

   List the available grids from a Meraxes HDF5 output file.

   :param spec: Specify the grid you want to read.
                0 -> Reionization grid
                1 -> Metal grid
   :type spec: int
   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot for which the grids are to be listed.
   :type snapshot: int

   :returns: **grids** -- A list of the available grids
   :rtype: list

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L894-L929>`__

.. py:function:: read_descendant_indices(fname, snapshot, pandas=False)

   Read the Descendant indices from the Meraxes HDF5 file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot from which the descendant dataset is to be read from.
   :type snapshot: int
   :param pandas: Return a pandas series instead of a numpy array.  (default = False)
   :type pandas: bool

   :returns: **desc_ind** -- NextProgenitor indices
   :rtype: array

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L744-L810>`__

.. py:function:: read_firstprogenitor_indices(fname, snapshot, pandas=False)

   Read the FirstProgenitor indices from the Meraxes HDF5 file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot from which the progenitors dataset is to be read from.
   :type snapshot: int
   :param pandas: Return a pandas series instead of a numpy array.  (default = False)
   :type pandas: bool

   :returns: **fp_ind** -- FirstProgenitor indices
   :rtype: array or series

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L616-L683>`__

.. py:function:: read_gals(fname, snapshot=None, props=None, sim_props=False, pandas=False, table=False, h=None, indices=None)

   Read in a Meraxes hdf5 output file.

   Reads in the default type of HDF5 file generated by the code.

   :param fname: Full path to input hdf5 master file.
   :type fname: str
   :param snapshot: The snapshot to read in.  (default: last present snapshot - usually
                    z=0)
   :type snapshot: int
   :param props: A list of galaxy properties requested.  (default: All properties)
   :type props: list
   :param sim_props: Output some simulation properties as well.  (default = False)
   :type sim_props: bool
   :param pandas: Ouput a pandas DataFrame instead of an astropy table.  (default =
                  False)
   :type pandas: bool
   :param table: Output an astropy Table instead of a numpy ndarray.  (default =
                 False)
   :type table: bool
   :param h: Hubble constant (/100) to scale the galaxy properties to.  If
             `None` then no scaling is made unless `set_little_h` was previously
             called.  (default = None)
   :type h: float
   :param indices: Indices of galaxies to be read.  If `None` then read all galaxies.
                   (default = None)
   :type indices: list or array

   :returns: * *An ndarray with the requested galaxies and properties.*
             * *If sim_props==True then output is a tuple of form (galaxies, sim_props)*

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L62-L270>`__

.. py:function:: read_git_info(fname)

   Read the git diff and ref saved in the master file.

   :param fname: Full path to input hdf5 master file.
   :type fname: str

   :returns: * **ref** (*str*) -- git ref of the model
             * **diff** (*str*) -- git diff of the model

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L410-L431>`__

.. py:function:: read_global_J_21(fname, snapshot)

   Read the volume weighted global J_21 from the Meraxes output.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot(s) from which the global J_21 is to be read
                    from.
   :type snapshot: int or list

   :returns: **global_J_21** -- Global J_21 value(s)
   :rtype: float or ndarray

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L1026-L1097>`__

.. py:function:: read_global_xH(fname, snapshot, weight='volume')

   Read global xH from the Meraxes HDF5 file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot(s) from which the global xH is to be read
                    from.
   :type snapshot: int or list
   :param weight: 'volume' -> volume weighted
                  'mass' -> mass weighted
   :type weight: str

   :returns: **global_xH** -- Global xH value(s)
   :rtype: float or ndarray

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L964-L1023>`__

.. py:function:: read_grid(spec, fname, snapshot, name, h=None, h_scaling={})

   Read a grid from the Meraxes HDF5 file.

   :param spec: Specify the grid you want to read.
                0 -> Reionization grid
                1 -> Metal grid
   :type spec: int
   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot from which the grid is to be read from.
   :type snapshot: int
   :param name: Name of the requested grid
   :type name: str
   :param h: Hubble constant (/100) to scale the galaxy properties to.  If
             `None` then no scaling is made unless `set_little_h` was previously
             called.  (default = None)
   :type h: float
   :param h_scaling: Dictionary of grid names (keys) and associated Hubble
                     constant scalings (values) as lambda functions. e.g.
                     | h_scaling = {"MassLikeGrid" : lambda x, h: x/h,}
   :type h_scaling: dict

   :returns: The requested grid
   :rtype: ndarray

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L813-L891>`__

.. py:function:: read_input_params(fname, h=None, raw=False)

   Read in the input parameters from a Meraxes hdf5 output file.

   :param fname: Full path to input hdf5 master file.
   :type fname: str
   :param h: Hubble constant (/100) to scale the galaxy properties to.  If
             `None` then no scaling is made unless `set_little_h` was previously
             called.  (default = None)
   :type h: float
   :param raw: Don't augment with extra useful quantities. (default = False)
   :type raw: bool

   :returns: All run properties.
   :rtype: dict

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L273-L339>`__

.. py:function:: read_nextprogenitor_indices(fname, snapshot, pandas=False)

   Read the NextProgenitor indices from the Meraxes HDF5 file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot from which the progenitors dataset is to be read from.
   :type snapshot: int
   :param pandas: Return a pandas series instead of a numpy array.  (default = False)
   :type pandas: bool

   :returns: **np_ind** -- NextProgenitor indices
   :rtype: array

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L686-L741>`__

.. py:function:: read_ps(fname, snapshot)

   Read 21cm power spectrum from the Meraxes HDF5 file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param snapshot: Snapshot from which the power spectrum is to be read from.
   :type snapshot: int

   :returns: * **kval** (*array*) -- k value (Mpc^-1)
             * **ps** (*array*) -- power value (should be dimensionless but actually might be power
               density i.e. with units [Mpc^-3])
             * **pserr** (*array*) -- error

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L932-L962>`__

.. py:function:: read_snaplist(fname, h=None)

   Read in the list of available snapshots from the Meraxes hdf5 file.

   :param fname: Full path to input hdf5 master file.
   :type fname: str
   :param h: Hubble constant (/100) to scale the galaxy properties to.  If
             `None` then no scaling is made unless `set_little_h` was previously
             called.  (default = None)
   :type h: float

   :returns: * **snaps** (*array*) -- snapshots
             * **redshifts** (*array*) -- redshifts
             * **lt_times** (*array*) -- light travel times (Myr)

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L434-L481>`__

.. py:function:: read_units(fname)

   Read in the units and hubble conversion information from a Meraxes hdf5
   output file.

   :param fname: Full path to input hdf5 master file.
   :type fname: str

   :returns: **units** -- A dict containing all units (Hubble conversions are stored with key
             `HubbleConversions`).
   :rtype: dict

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L342-L407>`__

.. py:function:: set_little_h(h=None)

   Set the value of little h to be used by all future meraxes.io calls
   where applicable.

   :param h: Little h value.  If a filename is passed as a string, then little h
             will be set to the simulation value read from that file.
             (default: None)
   :type h: float or str

   :returns: **h** -- Little h value.
   :rtype: float

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/io.py#L29-L59>`__

meraxes.reion
-------------

.. py:module:: dragons.meraxes.reion

Routines for reionisation related calculations.

.. py:function:: electron_optical_depth(fname, volume_weighted=False)

   Calculate the electron Thomson scattering optical depth from a Meraxes +
   21cmFAST run.  Note that this implementation assumes that the simulation
   volume is fully ionised before the final snapshot stored in the input file.

   :param fname: Full path to input hdf5 master file
   :type fname: str
   :param volume_weighted: This option is just for testing purposes as it can take a long time
                           to mass weight the neutral fraction depending on the grid size.
                           The optical depth obtained is often very similar to the correctly
                           mass weighted value, however, this should not be used to final
                           results.
   :type volume_weighted: bool

   :returns: * **z_list** (*ndarray*) -- Redshifts of each snapshot read in the input simulation.
             * **scattering_depth** (*ndarray*) -- Thomson scattering depth integrated between z=0 and each snapshot
               of the input simulation.

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/reion.py#L15-L104>`__

meraxes.plots
-------------

.. py:module:: dragons.meraxes.plots

`Module source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/meraxes/plots.py>`__


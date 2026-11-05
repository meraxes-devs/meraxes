# Build and run

The flags and outputs described in this guide are those of the
[`master` branch](https://github.com/meraxes-devs/meraxes/tree/master).

## Dependencies

Meraxes requires a C99 compiler and CMake **3.18 or newer**. The table lists
example Gadi module versions and their locations.

| Dependency | Version (Gadi example) | Gadi path | Use |
|---|---|---|---|
| GCC | 12.2.0 | `/apps/gcc/12.2.0/wrappers/gcc` | Compile C99 source. |
| CMake | 3.24.2 | `/apps/cmake/3.24.2` | Configure the build. |
| Open MPI | 4.1.4 | `/apps/openmpi/4.1.4` | Parallel execution. |
| Parallel HDF5 | 1.12.2p | `/apps/hdf5/1.12.2p` | Read and write data; requires the C and high-level libraries. |
| GSL and CBLAS | 2.7.1 | `$GSL_ROOT`, set by `module load gsl/2.7.1` | Integration, interpolation and random draws; includes `gslcblas`. |
| FFTW | 3.3.10 | `/apps/fftw3/3.3.10-nci1` | Distributed grids; requires `fftw3f` and `fftw3f_mpi`. |

Load compatible compiler, MPI, HDF5 and FFTW modules before configuring.
Use `module show <name>/<version>` to inspect include and library paths,
and `echo "$GSL_ROOT"` to print the GSL installation directory.
See the [Gadi software catalogue](https://opus.nci.org.au/spaces/Help/pages/248840422/Supported+Applications)
for available modules.

## Configure and compile

From the Meraxes source directory:

```sh
mkdir build
cd build
cmake ..
make
```

This creates `bin/meraxes` and `input.par` in the build directory.
Enable additional physics by adding the relevant option to `cmake ..`, then
run `make` again:

| Option | Capability |
|---|---|
| `-DUSE_STOCHASTICITY=ON` | Escape-fraction and X-ray scatter, no-SFR and source recalibration. |
| `-DUSE_MINI_HALOS=ON` | Molecular cooling, Pop. III stars and associated feedback. |
| `-DCALC_MAGS=ON -DSECTOR_ROOT=/path/to/sector/clib` | Stellar photometry. |
| `-DUSE_JWST=ON` / `-DUSE_HST=ON` | Observed filters for photometry builds. |
| `-DN_HISTORY_SNAPS=17` | Number of snapshots retained for delayed stellar feedback. |
| `-DMAGS_N_SNAPS=12 -DMAGS_N_BANDS=11` | Photometry array dimensions; match the selected snapshots and bands. |

The additional physics options are off by default. Stellar-feedback history
must cover the ages required by the feedback tables.

## Configure a run

Edit the generated `input.par`:

| Set | Purpose |
|---|---|
| `SimParamsFile` and `SimulationDir` | Select the simulation parameters and data. |
| Table directories | Locate cooling, stellar-feedback and enabled radiation/photometry tables. |
| `OutputDir`, `FileNameGalaxies`, `OutputSnapshots` | Choose where and when to save results. |
| Physics parameters and flags | Select the model and requested products. |

The [input reference](inputs.md) lists filenames, defaults and parameter meanings.

## Run

```sh
/path/to/meraxes /path/to/input.par
```

For cluster runs, see the [submission-file example](#submission-file).

### Submission file

On Gadi, save this minimal submission file as `submit.pbs` in the build directory:

```bash
#!/bin/bash
#PBS -P PROJECT
#PBS -l ncpus=32,mem=256GB,walltime=06:00:00
#PBS -l storage=gdata/PROJECT+scratch/PROJECT
#PBS -l wd
#PBS -V

mpirun -np "$PBS_NCPUS" ./bin/meraxes input.par
```

Replace `PROJECT`, adjust resources and include the storage projects used by
the run. Submit with the build modules loaded; `-V` inherits that environment.

```sh
qsub submit.pbs
```

See NCI's [PBS guide](https://opus.nci.org.au/spaces/Help/pages/90308829/PBS+Directives+Explained)
for scheduler options and [Post-processing tools](post-processing.md) for analysing results.

## Example: a small test volume

The quickest way to try Meraxes is the
[L10_N512 test set](https://doi.org/10.5281/zenodo.23281409): a 10 $h^{-1}$ Mpc box with
$512^3$ particles from the Genesis simulations, with merger trees and density grids down
to $z \approx 10$, 47 MB in all. The regression tests use the same data, and this example
runs the same model as their `default` setup.

Download and unpack it:

```sh
curl -L -o L10_N512.zip "https://zenodo.org/records/23281409/files/L10_N512.zip?download=1"
md5sum L10_N512.zip  # afe3719ce9144869c9f0b8629c62e9d5
unzip L10_N512.zip   # creates L10_N512/ with a_list.txt, grids/ and trees/
```

In the build directory, edit the generated `input.par`. Change the file name in
`SimParamsFile` and the value of `OutputSnapshots`, and add `SimulationDir`:

```text
SimParamsFile   : /path/to/meraxes/input/params/simulations/Genesis_L10_N512.par
SimulationDir   : /path/to/L10_N512
OutputSnapshots : 48, 63, 75  # z = 15, 12, 10
```

Then run it on two MPI ranks, which takes a minute or two:

```sh
mpirun -np 2 ./bin/meraxes input.par
```

The results are in `output/`: the master file `meraxes.hdf5`, one galaxy file per rank
and one grid file per saved snapshot (see [Outputs](outputs.md)). Read them with `h5py`:

```python
import h5py

with h5py.File("output/meraxes.hdf5", "r") as f:
    snap = f["Snap075"]
    print("z =", snap.attrs["Redshift"][0], "with", snap.attrs["NGalaxies"][0], "galaxies")
    print("neutral fraction:", snap["Grids/xH"].attrs["volume_weighted_global_xH"][0])
    smf = snap["SMF"][:]  # columns: log10(M*/Msun), number density (cMpc^-3 dex^-1), uncertainty
```

or with [DRAGONS](post-processing.md):

```python
from dragons import meraxes

gals = meraxes.read_gals("output/meraxes.hdf5", snapshot=75, props=["StellarMass", "Sfr"])
```

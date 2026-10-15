# Units and numerical definitions

This reference collects the unit conversions, discretization and statistical normalization used by the model. $h$ denotes the dimensionless Hubble parameter; $a$ and $z$ are scale factor and redshift.

<details>
<summary>On this page</summary>

```{contents}
:local:
:depth: 1
:backlinks: none
```

</details>

## Units

For cgs mass, length and velocity scales $U_M$, $U_L$ and $U_V$, the derived scales are

```{math}
:label: num-derived-units
\begin{aligned}
U_T&=\frac{U_L}{U_V},&
U_\rho&=\frac{U_M}{U_L^3},&
U_P&=\frac{U_M}{U_LU_T^2},\\
U_{\dot u}&=\frac{U_P}{U_T},&
U_E&=U_MU_V^2,&
G_{\rm int}&=G_{\rm cgs}\frac{U_MU_T^2}{U_L^3}.
\end{aligned}
```

*References:* [Meraxes unit definitions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/init.c).

The conventional internal mass and length units are $10^{10}h^{-1}M_\odot$ and $h^{-1}\mathrm{Mpc}$; velocity is measured in $\mathrm{km\,s^{-1}}$.

| Internal quantity | Physical cgs value |
|---|---|
| Mass $m$ | $mU_M/h$ |
| Proper radius $r$ | $rU_L/h$ |
| Comoving position $x$ | Proper position $axU_L/h$ |
| Velocity $v$ | $vU_V$ |
| Time $t$ | $tU_T/h$ |
| Mass rate $\dot m$ | $\dot mU_M/U_T$ |
| Proper density $\rho$ | $\rho U_\rho h^2$ |
| Comoving density $\rho_c$ | Proper density $\rho_cU_\rho h^2a^{-3}$ |
| Energy $e$ | $eU_E/h$ |
| Pressure $p$ | $pU_Ph^2$ |

A mass rate converted to solar masses per year is

```{math}
:label: num-rate-output
\frac{\dot M}{M_\odot\,\mathrm{yr}^{-1}}
=\dot m\frac{U_M}{U_T}\frac{1\,\mathrm{yr}}{M_\odot}.
```

*References:* [Meraxes output conversions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/save.c).

The cgs duration of one year and cgs solar mass are used on the right. The mass and time factors of $h$ cancel. Internal and physical Hubble rates satisfy

```{math}
:label: num-hubble-unit
H_{\rm phys}(z)=\frac{h}{U_T}H_{\rm int}(z).
```

*References:* [Meraxes unit definitions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/init.c).

## Expansion and time

A snapshot scale factor defines

```{math}
:label: run-redshift
z_i=a_i^{-1}-1.
```

*References:* {ref}`Hogg (1999) <ref-hogg1999>`.

The galaxy expansion factor $E_{\rm gal}$ follows Equation {eq}`gal-expansion`, with matter, curvature and a cosmological constant. Its internal unit conversion is

```{math}
:label: run-galaxy-expansion
\begin{aligned}
H_{\rm gal,int}(z)&=H_{100}U_TE_{\rm gal}(z),\\
\rho_{\rm crit,int}(z)&=\frac{3H_{\rm gal,int}^2(z)}{8\pi G_{\rm int}},
\end{aligned}
```

*References:* {ref}`Hogg (1999) <ref-hogg1999>`; [Meraxes expansion rate](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/virial_properties.c); [Meraxes unit definitions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/init.c).

where $H_{100}=100\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$. Internal lookback time is

```{math}
:label: num-lookback-time
L(z)=\frac{1}{H_{100}U_T}\int_{(1+z)^{-1}}^1
\frac{\mathrm da}{\sqrt{\Omega_m/a+\Omega_k+\Omega_\Lambda a^2}}.
```

*References:* {ref}`Hogg (1999) <ref-hogg1999>`; [Meraxes lookback-time calculation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/init.c).

For a galaxy last identified at snapshot $j$ and evolved at snapshot $i$,

```{math}
:label: run-timestep
\Delta t_{\rm gal}=\frac{L(z_j)-L(z_i)}{N_{\rm steps}},\qquad N_{\rm steps}=1.
```

*References:* [Meraxes galaxy time intervals](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/galaxies.c); [Meraxes time-step constraint](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_params.c).

New galaxies use the preceding snapshot, except at the first snapshot where their interval is zero. Physical time intervals multiply the internal result by $U_T/h$.

The IGM expansion rate includes radiation and omits curvature, as defined in Equation {eq}`igm-expansion-helpers`.

Its present-day density normalizations are

```{math}
:label: run-igm-density
\begin{aligned}
\rho_{\rm crit,0}&=\frac{3(hH_{100})^2}{8\pi G},&
\Omega_b&=f_b\Omega_m.
\end{aligned}
```

*References:* {ref}`Hogg (1999) <ref-hogg1999>`; [Meraxes IGM density definitions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.h).

$f_b$ is the cosmic baryon fraction. Hydrogen and helium number densities follow Equation {eq}`igm-number-densities`. Radiation-shell and lightcone interpolation use the matter–cosmological-constant time relation

```{math}
:label: num-cosmic-time-helper
t(z)=\frac{2\sqrt{1+\Omega_m/\Omega_\Lambda}}{3H_{100}h}
\operatorname{asinh}\!\left[\sqrt{\frac{\Omega_\Lambda}{\Omega_m}}(1+z)^{-3/2}\right].
```

*References:* [Meraxes cosmic-time relation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

This expression assumes flat matter–cosmological-constant cosmology and omits radiation. Lightcone interpolation is linear in this time coordinate.

## Mesh and source assignment

For a periodic box of comoving side $L_h$ in $h^{-1}\mathrm{Mpc}$ and $N$ cells per side,

```{math}
:label: num-cell-size
L=\frac{L_h}{h},\qquad
\Delta x_h=\frac{L_h}{N},\qquad
\Delta x=\frac{L_h}{hN},\qquad
\Delta x_{\rm proper}=\frac{L_h}{hN(1+z)}.
```

*References:* {ref}`Hogg (1999) <ref-hogg1999>`; [Meraxes mesh geometry](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ConstructLightcone.c).

$L$ and $\Delta x$ are in comoving Mpc. Nearest-grid-point assignment maps each coordinate to

```{math}
:label: num-ngp-index
i(x)=\operatorname{round}_{\rm nearest}\!\left(\frac{Nx}{L_h}\right),\qquad
 i=N\ \longrightarrow\ i=0.
```

*References:* [Meraxes mesh assignment](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

The source field sums galaxies within each cell, following Equation {eq}`igm-ngp`.

A galaxy contributes to one cell. Source masses and rates are cell sums; conversion to densities requires division by the cell volume. Dark-matter density is supplied independently.

For an input mesh of side $N_{\rm in}$, permitted downsampling satisfies

```{math}
:label: num-grid-resample
f_{\rm resample}=\frac{N}{N_{\rm in}}\leq1,\qquad
n_{\rm every}=\frac{N_{\rm in}}{N}\in\mathbb N.
```

*References:* [Meraxes grid resampling](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_grids.c).

The input field is smoothed with a real-space top-hat window of radius $L_h/(2N)$, then sampled every $n_{\rm every}$ cells. Its conversion to overdensity is

```{math}
:label: num-density-normalisation
C_\rho=\frac{L_{\rm file}^3h}{N_pm_{\rm part}},\qquad
\delta=\max\!\left(C_\rho\rho_{\rm file}-1,\delta_{\rm floor}\right).
```

*References:* [Meraxes VELOCIraptor density conversion](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_grids-velociraptor.c); [Meraxes gbpTrees density conversion](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_grids-gbptrees.c).

$L_{\rm file}$ is the input box side, $N_p$ the simulation particle count and $m_{\rm part}$ the particle mass in internal units. $\rho_{\rm file}$ follows the input density convention; its normalization includes the factor $h$. The numerical floor is either $-1$ or $-1+10^{-5}$, depending on the input format.

## Fourier transforms and filters

Forward and inverse unnormalized discrete transforms satisfy

```{math}
:label: num-fft-normalisation
\mathcal F^{-1}_{u}[\mathcal F(q)]=N^3q,\qquad
q_R=\mathcal F^{-1}_{u}\!\left[W(kR)\frac{\mathcal F(q)}{N^3}\right].
```

*References:* {ref}`FFTW documentation <ref-fftw>`; [Meraxes Fourier normalization](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_grids.c).

The signed wavevector mapping in the first two coordinates is

```{math}
:label: num-fft-wavevectors
\Delta k_h=\frac{2\pi}{L_h},\qquad
k_x=\begin{cases}
n_x\Delta k_h,&n_x\leq\lfloor N/2\rfloor,\\
(n_x-N)\Delta k_h,&n_x>\lfloor N/2\rfloor.
\end{cases}
```

*References:* {ref}`FFTW documentation <ref-fftw>`; [Meraxes Fourier modes](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

The same mapping applies in $y$. Hermitian storage retains only non-negative $k_z=n_z\Delta k_h$, up to $n_z=\lfloor N/2\rfloor$. Physical comoving wavenumbers are $k=hk_h$.

The available smoothing windows are

```{math}
:label: num-filter-windows
\begin{aligned}
W_{\rm TH}(u)&=3\left(\frac{\sin u}{u^3}-\frac{\cos u}{u^2}\right),\\
W_{\rm sharp}(u)&=\begin{cases}1,&0.413566994u\leq1,\\0,&0.413566994u>1,\end{cases}\\
W_{\rm G}(u)&=\exp\!\left[-\frac{(0.643u)^2}{2}\right],\qquad u=k_hR_h.
\end{aligned}
```

*References:* [Meraxes smoothing windows](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

The top-hat window is set to one for $u\leq10^{-4}$. The mesh's fundamental and axial Nyquist wavenumbers are

```{math}
:label: num-resolution-scales
k_{\rm fund}=\frac{2\pi}{L},\qquad
k_{\rm Nyquist}=\frac{\pi N}{L}.
```

*References:* {ref}`FFTW documentation <ref-fftw>`; [Meraxes Fourier mesh](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputePowerSpectrum.c).

## Array layout and decomposition

For local indices $i,j,k$, the final coordinate is contiguous. Linear offsets are

```{math}
:label: num-grid-layout
\begin{aligned}
I_{\rm real}(i,j,k)&=k+N(j+Ni),\\
I_{\rm padded}(i,j,k)&=k+2\bigl(\lfloor N/2\rfloor+1\bigr)(j+Ni),\\
I_{\rm complex}(i,j,k)&=k+\bigl(\lfloor N/2\rfloor+1\bigr)(j+Ni).
\end{aligned}
```

*References:* {ref}`FFTW documentation <ref-fftw>`; [Meraxes array indexing](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

Padded rows contain $N+2$ real values for even $N$ and $N+1$ for odd $N$; only $N$ are physical cells. With lightcone length $L_{\rm LC}$ and $F$ heating radii,

```{math}
:label: num-special-layouts
I_{\rm LC}(i,j,k)=k+L_{\rm LC}(j+Ni),\qquad
I_{\rm heat}(r,i,j,k)=r+F[k+N(j+Ni)].
```

*References:* [Meraxes array indexing](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

Merger forests are assigned to the least-loaded process as they are encountered in snapshot order. The load is measured by halo count at the last requested output:

```{math}
:label: num-forest-load
r_f=\operatorname*{arg\,min}_r C_r,\qquad
C_{r_f}\leftarrow C_{r_f}+N_{\rm halo,last}(f).
```

*References:* [Meraxes forest allocation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_halos.c).

Storage capacity sums the individual forest maxima:

```{math}
:label: num-forest-capacity
N_{\rm halo,max,r}=\sum_{f\in r}\max_sN_{\rm halo}(f,s),\qquad
N_{\rm FOF,max,r}=\sum_{f\in r}\max_sN_{\rm FOF}(f,s).
```

*References:* [Meraxes forest allocation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_halos.c).

The spatial mesh uses separate contiguous slabs. A process with $n_{x,r}$ local planes owns

```{math}
:label: num-slab-ownership
S_r=\sum_{p<r}n_{x,p},\qquad
S_r\leq i_{\rm global}<S_r+n_{x,r},\qquad
i_{\rm local}=i_{\rm global}-S_r.
```

*References:* [Meraxes slab decomposition](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

## Integration and tolerance

Thermal evolution takes one explicit step over each snapshot interval, following Equation {eq}`igm-euler-update`.

The generic tabulated interpolation and integration rules are linear and trapezoidal:

```{math}
:label: num-table-integration
\begin{aligned}
y(x)&=y_i+\frac{y_{i+1}-y_i}{x_{i+1}-x_i}(x-x_i),\\
\int_a^b y(x)\,\mathrm dx&\simeq\sum_j\frac{y(x_{j+1})+y(x_j)}{2}(x_{j+1}-x_j).
\end{aligned}
```

*References:* [Meraxes numerical helpers](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

The integration includes interpolated interval endpoints. Approximate floating-point equality uses

```{math}
:label: num-close-tolerance
|a-b|\leq\epsilon_{\rm abs}+\epsilon_{\rm rel}|b|,\qquad
\epsilon_{\rm abs}=10^{-8},\quad\epsilon_{\rm rel}=10^{-5}.
```

*References:* [Meraxes numerical helpers](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

## Distribution functions

For an output interval and nominal bin density $b$, the actual bins are

```{math}
:label: out-df-bins
N_{\rm bin}=\left\lfloor(x_{\max}-x_{\min})b\right\rfloor,\qquad
\Delta x=\frac{x_{\max}-x_{\min}}{N_{\rm bin}},\qquad
x_i=x_{\min}+\left(i+\frac12\right)\Delta x.
```

*References:* [Meraxes distribution functions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/dist_func.c); [Meraxes catalogue statistics](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/save.c).

The count contribution $w_g$ is either one or an activity/visibility weight. Summing over the galaxies $\mathcal G_{r,i}$ in bin $i$ on process $r$ gives

```{math}
:label: out-df-density
C_i=\sum_r\sum_{g\in\mathcal G_{r,i}}w_g,\qquad
V=\left(\frac{L_h}{h}\right)^3,\qquad
\phi_i=\frac{C_i}{V\Delta x}.
```

*References:* [Meraxes distribution functions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/dist_func.c); [Meraxes catalogue statistics](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/save.c).

$V$ is in comoving $\mathrm{Mpc}^3$; $\Delta x$ determines whether the density is per dex or per magnitude. For selection probability $p_g$, the reported count uncertainty is

```{math}
:label: out-df-uncertainty
\begin{aligned}
B_i&=\sum_r\sum_{g\in\mathcal G_{r,i}}p_g(1-p_g),\qquad B_{\rm tot}=\sum_iB_i,\\
\sigma_i&=\begin{cases}
\sqrt{B_i}/(V\Delta x),&B_{\rm tot}>0,\\
\sqrt{C_i}/(V\Delta x),&B_{\rm tot}\leq0,\ C_i>0,\\
0,&B_{\rm tot}\leq0,\ C_i\leq0.
\end{cases}
\end{aligned}
```

*References:* [Meraxes distribution functions](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/dist_func.c); [Meraxes catalogue statistics](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/save.c).

The Bernoulli or Poisson branch is selected for the whole distribution. These uncertainties describe counts; they do not include cosmic variance.

## Memory scaling

Let $n_{R,r}=n_{x,r}N^2$ be the local physical-cell count and $n_{C,r}$ the allocated complex-cell count. Array sizes in bytes are

```{math}
:label: num-grid-memory
\begin{aligned}
B_{\rm real,r}&=4n_{R,r},& B_{\rm padded,r}&=8n_{C,r},&B_{\rm complex,r}&=8n_{C,r},\\
B_{\rm history,r}&=8n_{C,r}H,&B_{\rm smooth,r}&=8n_{R,r}F,&B_{\rm LC,r}&=4n_{x,r}NL_{\rm LC}.
\end{aligned}
```

*References:* [Meraxes grid allocation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

$H$ is the number of retained heating snapshots. Real, padded and complex fields use single precision; heating-radius arrays use double precision. One unpadded cube requires $4N^3$ bytes, and a three-dimensional FFT scales approximately as $N^3\log N$.

For $N_{{\rm gal},r}$ galaxies and $b_{\rm gal}$ bytes per galaxy record,

```{math}
:label: num-galaxy-memory
B_{{\rm galaxies},r}\simeq N_{{\rm gal},r}b_{\rm gal},\qquad
B_{{\rm recent\ histories},r}=16H_{\rm SN}N_{{\rm gal},r}.
```

*References:* [Meraxes galaxy records](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/meraxes.h); [Meraxes memory accounting](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

The second term describes two double-precision stellar-history arrays of length $H_{\rm SN}$ within those records. Total job memory additionally includes halo storage, enabled source channels, transform workspace and I/O buffers.

# Galaxies, black holes and stellar populations

Gas supply, star formation and feedback follow {ref}`Mutch et al. (2016) <ref-mutch2016>`, with developments described by {ref}`Qin et al. (2018) <ref-qin2018xiv>` and {ref}`Qin et al. (2019) <ref-qin2019xv>`. Subscripts cold, hot, ej, star and BH identify the mass reservoirs; $Z=M_Z/M$ denotes metal mass fraction. $[x]_+=\max(x,0)$.

<details>
<summary>On this page</summary>

```{contents}
:local:
:depth: 1
:backlinks: none
```

</details>

## Halo dynamics and gas supply

### Units, expansion and virial quantities

$U_M,U_L,U_V$ are mass, length and velocity units. Numerical masses use $10^{10}h^{-1}M_\odot$, physical halo radii use $h^{-1}\mathrm{Mpc}$, and velocities use $\mathrm{km\,s^{-1}}$.

Derived unit scales are given in Equation {eq}`num-derived-units`.

```{math}
:label: gal-expansion
\begin{gathered}
E(z)=\left[\Omega_{m,0}(1+z)^3+\Omega_{k,0}(1+z)^2+\Omega_{\Lambda,0}\right]^{1/2},
\\ H(z)=H_0E(z),\\
\Omega_m(z)=\frac{\Omega_{m,0}(1+z)^3}{E^2(z)}.
\end{gathered}
```

*References:* {ref}`Barkana & Loeb (2001), §2.1 <ref-barkana2001>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/virial_properties.c).

```{math}
:label: gal-overdensity
x=\Omega_m(z)-1,\qquad
\Delta_{\rm vir}(z)=\frac{18\pi^2+82x-39x^2}{\Omega_m(z)}.
```

*References:* [Meraxes virial convention](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/virial_properties.c).

```{math}
:label: gal-virial-radius
\begin{gathered}
\rho_{\rm crit}(z)=\frac{3H^2(z)}{8\pi G},\\
R_{\rm vir}=\left[\frac{3M_{\rm vir}}{4\pi\Delta_{\rm vir}\rho_{\rm crit}}\right]^{1/3},
\\ V_{\rm vir}=\left(\frac{GM_{\rm vir}}{R_{\rm vir}}\right)^{1/2}.
\end{gathered}
```

*References:* [Meraxes virial convention](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/virial_properties.c).

The overdensity convention above is paired with critical density. A tabulated factor $f_M$ modifies the group mass; $j_{\rm halo}$ is specific angular momentum.

```{math}
:label: gal-halo-mass-modifier
\ell_M=\log_{10}\left(\frac{M_{\rm vir,FOF}^{\rm unmodified}}{h}\right)+10,\qquad
M_{\rm vir,FOF}=f_M(\ell_M)M_{\rm vir,FOF}^{\rm unmodified}.
```

*References:* {ref}`Qin et al. (2018), §3 <ref-qin2018xiv>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/read_halos-velociraptor.c).

```{math}
:label: gal-disk-radius
\lambda=\frac{j_{\rm halo}}{\sqrt{2}\,V_{\rm vir}R_{\rm vir}},\qquad
R_d=\frac{\lambda R_{\rm vir}}{\sqrt{2}},\qquad
R_{\rm SF}=3R_d.
```

*References:* {ref}`Mo, Mao & White (1998), eq. 12 <ref-mo1998>`; {ref}`Mutch et al. (2016), §2.4 <ref-mutch2016>`.

```{math}
:label: gal-virial-temperature
T_{\rm vir,AC}=35.9\left(\frac{V_{\rm vir}}{\mathrm{km\,s^{-1}}}\right)^2\mathrm{K},\qquad
T_{\rm vir,MC}=\min\left[73.8\left(\frac{V_{\rm vir}}{\mathrm{km\,s^{-1}}}\right)^2,10^4\right]\mathrm{K}.
```

*References:* {ref}`Barkana & Loeb (2001), eq. 26 <ref-barkana2001>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/virial_properties.c).

```{math}
:label: gal-temperature-mass
\frac{M_{\rm vir}(T,z)}{10^{10}h^{-1}M_\odot}
=0.01\left(\frac{\mu}{0.6}\right)^{-3/2}
\left[\frac{\Omega_{m,0}}{\Omega_m(z)}\frac{\Delta_{\rm vir}(z)}{18\pi^2}\right]^{-1/2}
\left(\frac{T}{1.98\times10^4\mathrm{K}}\right)^{3/2}
\left(\frac{1+z}{10}\right)^{-3/2},
```

*References:* {ref}`Barkana & Loeb (2001), virial scaling <ref-barkana2001>`; [Meraxes temperature–mass conversion](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/virial_properties.c).

The mean molecular weight is $\mu=1.22$ below approximately $10^4$ K and $0.59$ above. An evolution interval spans the galaxy's last identified halo snapshot to the current one.

The evolution interval is given in Equation {eq}`run-timestep`.

### Infall and baryon corrections

$f_b$ is the cosmic baryon fraction; $f_{\rm UVB}$ and $f_{b,\rm tab}$ describe photoheating and tabulated baryon suppression. Group sums include stellar remnants when present.

```{math}
:label: gal-baryon-budget
M_{b,\rm FOF}=\sum_g\left(M_{\star,g}+M_{\rm cold,g}+M_{\rm hot,g}
+M_{\rm ej,g}+M_{\rm BH,g}+M_{\rm BH,queued,g}\right).
```

*References:* {ref}`Mutch et al. (2016), eq. 1 <ref-mutch2016>`; {ref}`Qin et al. (2018), §3 <ref-qin2018xiv>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/infall.c).

```{math}
:label: gal-infall
f_{b,\rm eff}=f_{\rm UVB}f_{b,\rm tab},\qquad
\Delta M_{\rm infall}=f_{b,\rm eff}f_bM_{\rm vir,FOF}-M_{b,\rm FOF}.
```

*References:* {ref}`Mutch et al. (2016), eq. 1 <ref-mutch2016>`; {ref}`Qin et al. (2018), §3 <ref-qin2018xiv>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/infall.c).

```{math}
:label: gal-baryon-modifier-coordinate
\ell=\log_{10}\left(\frac{M_{\rm vir,FOF}}{f_Mh}\right)+10,
```

*References:* [Meraxes modifier-table coordinate](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/infall.c).

Modifier tables use the linear interpolation in Equation {eq}`num-table-integration`.

Modifier coordinates use numerical masses; interpolation holds the endpoint values outside the table. Positive infall adds hot gas; negative infall removes ejected gas, then hot gas.

## Cooling and reincorporation

### Atomic cooling

$\Lambda(T,Z)$ is the cooling coefficient. An isothermal hot atmosphere supplies cold gas over its halo dynamical time; $f_{\rm cool}$ normalises the supply rate.

```{math}
:label: gal-hot-profile
\rho_{\rm hot}(r)=\frac{M_{\rm hot}}{4\pi R_{\rm vir}r^2},\qquad
t_{\rm dyn,h}=\frac{R_{\rm vir}}{V_{\rm vir}}.
```

*References:* {ref}`Mutch et al. (2016), §2.3 <ref-mutch2016>`; {ref}`Sutherland & Dopita (1993) <ref-sutherland1993>`.

```{math}
:label: gal-cooling-radius
\begin{gathered}
\rho(r_{\rm cool})=\frac{3\mu m_pk_BT_{\rm vir}}{2\Lambda(T_{\rm vir},Z_{\rm hot})t_{\rm dyn,h}},\\
r_{\rm cool}=\left[\frac{M_{\rm hot}}{4\pi R_{\rm vir}\rho(r_{\rm cool})}\right]^{1/2}.
\end{gathered}
```

*References:* {ref}`Mutch et al. (2016), §2.3 <ref-mutch2016>`; {ref}`Sutherland & Dopita (1993) <ref-sutherland1993>`.

```{math}
:label: gal-cooling-mass
\Delta M_{\rm cool,0}=\min\left[
M_{\rm hot},\;
f_{\rm cool}\frac{M_{\rm hot}\delta t}{t_{\rm dyn,h}}
\min\left(1,\frac{r_{\rm cool}}{R_{\rm vir}}\right)
\right].
```

*References:* {ref}`Mutch et al. (2016), eq. 4 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/cooling.c).

```{math}
:label: gal-cooling-transfer
\Delta M_{\rm cool}=\min\left[M_{\rm hot},\max(0,\Delta M_{\rm cool,0}-\Delta M_{\rm heat,BH})\right],
\qquad \Delta M_{Z,\rm cool}=Z_{\rm hot}\Delta M_{\rm cool}.
```

*References:* {ref}`Qin et al. (2017b), §2.2 <ref-qin2017x>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/cooling.c).

### Molecular cooling

For molecular cooling, $\mu=1.22$, $T_3=T/(10^3\,\mathrm{K})$, and the interpolation adopts $n_H=100\,\mathrm{cm^{-3}}$. Temperatures are in kelvin and cooling coefficients in cgs units.

```{math}
:label: gal-molecular-lte
\begin{aligned}
\Lambda_{r,\rm LTE}&=\frac{1}{n_H}\left[
\frac{9.5\times10^{-22}T_3^{3.76}}{1+0.12T_3^{2.1}}
\exp\!\left(-\left[\frac{0.13}{T_3}\right]^3\right)
+3\times10^{-24}\exp\!\left(-\frac{0.51}{T_3}\right)\right],\\
\Lambda_{v,\rm LTE}&=\frac{1}{n_H}\left[
6.7\times10^{-19}\exp\!\left(-\frac{5.86}{T_3}\right)
+1.6\times10^{-18}\exp\!\left(-\frac{11.7}{T_3}\right)\right],\\
\Lambda_{\rm LTE}&=\Lambda_{r,\rm LTE}+\Lambda_{v,\rm LTE}.
\end{aligned}
```

*References:* {ref}`Hollenbach & McKee (1979), eqs. 6.37–6.38 <ref-hollenbach1979>`.

```{math}
:label: gal-molecular-cooling
\begin{aligned}
\log_{10}\Lambda_{\rm low}&=-103+97.59\log_{10}T
-48.05(\log_{10}T)^2+10.8(\log_{10}T)^3-0.9032(\log_{10}T)^4,\\
\Lambda_{\rm MC}&=\frac{\Lambda_{\rm LTE}}{1+\Lambda_{\rm LTE}/\Lambda_{\rm low}}.
\end{aligned}
```

*References:* {ref}`Galli & Palla (1998), eqs. A5–A7 <ref-galli1998>`.

### Return of ejected gas

The dynamical-time law uses dimensionless efficiency $\epsilon_{\rm reinc}$. The mass-dependent law uses timescale normalisation $\gamma_{\rm reinc}$ in Myr. Both return gas and its metals to the hot reservoir.

```{math}
:label: gal-reincorporation-one
\text{Model 1:}\qquad
\Delta M_{\rm reinc}=\min\left[M_{\rm ej},\;
\epsilon_{\rm reinc}M_{\rm ej}\frac{\delta t}{t_{\rm dyn,h}}\right].
```

*References:* {ref}`Croton et al. (2006) <ref-croton2006>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/reincorporation.c).

```{math}
:label: gal-reincorporation-two
\text{Model 2:}\qquad
t_{\rm reinc}=\max\left[t_{\rm dyn,h},\;
\gamma_{\rm reinc}\frac{10^{10}M_\odot}{M_{\rm vir,FOF}^{\rm physical}}\right],\qquad
\Delta M_{\rm reinc}=\min\left[M_{\rm ej},\;M_{\rm ej}\frac{\delta t}{t_{\rm reinc}}\right].
```

*References:* {ref}`Henriques et al. (2013) <ref-henriques2013>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/reincorporation.c).

## Star formation

### Critical-density law

$V_d$ is either maximum circular or virial velocity. $\alpha_{\rm SF,0}$ and $\beta_{\rm SF}$ set efficiency and redshift evolution; $C_{\rm SF}$ includes the unit conversion for the critical surface density.

```{math}
:label: gal-sf-efficiency
t_{\rm dyn,d}=\frac{R_{\rm SF}}{V_d},\qquad
\alpha_{\rm SF}(z)=\alpha_{\rm SF,0}\,(1+z)^{\beta_{\rm SF}}.
```

*References:* {ref}`Qin et al. (2018), §4.1 <ref-qin2018xiv>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/star_formation.c).

```{math}
:label: gal-critical-sf
M_{\rm crit}=C_{\rm SF}\,V_dR_{\rm SF},\qquad
\Delta M_{\star,0}=\alpha_{\rm SF}(z)
\frac{[M_{\rm cold}-M_{\rm crit}]_+}{t_{\rm dyn,d}}\delta t,
```

*References:* {ref}`Mutch et al. (2016), eqs. 5–7 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/star_formation.c).

### Molecular-pressure law

Gas and stars occupy exponential disks. The pressure relation uses SI units and integrates molecular gas to five scale radii.

```{math}
:label: gal-pressure-surfaces
\begin{gathered}
\Sigma_{g,0}=\frac{0.76M_{\rm cold}}{2\pi R_d^2},\\
\Sigma_{\star,0}=\frac{M_\star}{2\pi R_d^2},\\
\Sigma_g(r)=\Sigma_{g,0}e^{-r/R_d},\\
\Sigma_\star(r)=\Sigma_{\star,0}e^{-r/R_d}.
\end{gathered}
```

*References:* {ref}`Lagos et al. (2011), §2.3 <ref-lagos2011>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/star_formation.c).

```{math}
:label: gal-pressure-molecular-fraction
\begin{aligned}
C_\sigma&=\frac{10^4\,\mathrm{m\,s^{-1}}}{\sqrt{\pi G(0.14R_d)}},\\
P_{\rm ext}(r)&=\frac{\pi G}{2}\Sigma_g(r)
\left[\Sigma_g(r)+C_\sigma\sqrt{\Sigma_\star(r)}\right],\\
f_{\rm mol}(r)&=\left[1+\left(\frac{P_{\rm ext}(r)}{4.79\times10^{-13}\,\mathrm{Pa}}\right)^{-0.92}\right]^{-1}.
\end{aligned}
```

*References:* {ref}`Blitz & Rosolowsky (2006) <ref-blitz2006>`; {ref}`Lagos et al. (2011), Appendix B2 <ref-lagos2011>`.

```{math}
:label: gal-pressure-sf-integral
M_{\rm H_2,int}=2\pi\int_0^{5R_d}r\,f_{\rm mol}(r)\Sigma_g(r)\,dr,\qquad
\dot M_{\star,0}=\frac{\alpha_{\rm SF}(z)}{3\times10^8\,\mathrm{yr}}M_{\rm H_2,int}.
```

*References:* {ref}`Duffy et al. (2017), molecular star-formation model <ref-duffy2017>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/star_formation.c).

```{math}
:label: gal-hydrogen-masses
M_{\rm H_2}=\min\left[M_{\rm H_2,int},(1-Y_{\rm He})M_{\rm cold}\right],\qquad
M_{\rm HI}=(1-Y_{\rm He})M_{\rm cold}-M_{\rm H_2}.
```

*References:* [Meraxes molecular and atomic gas bookkeeping](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/star_formation.c).

$Y_{\rm He}$ is the helium mass fraction. The star-formation rate uses the uncapped molecular integral; the central molecular fraction differs from the integrated mass fraction.

### Dynamical-timescale law and mass conservation

```{math}
:label: gal-galform-sf
\tau_\star=\frac{t_{\rm dyn,d}}{0.029}
\left(\frac{200\,\mathrm{km\,s^{-1}}}{V_d}\right)^{1.5},\qquad
\Delta M_{\star,0}=\frac{M_{\rm cold}}{\tau_\star}\delta t.
```

*References:* {ref}`Cole et al. (2000), eq. 4.14 <ref-cole2000>`; [Meraxes GALFORM prescription](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/star_formation.c).

Accepted star formation is limited by available cold gas and simultaneous reheating. The following updates precede stellar recycling; gross stellar mass retains all mass ever formed.

```{math}
:label: gal-sf-reservoir-update
\begin{aligned}
M_{\rm cold}&\leftarrow M_{\rm cold}-\Delta M_\star,&
M_{Z,\rm cold}&\leftarrow M_{Z,\rm cold}-Z_{\rm cold}\Delta M_\star,\\
M_\star&\leftarrow M_\star+\Delta M_\star,&
M_{Z,\star}&\leftarrow M_{Z,\star}+Z_{\rm cold}\Delta M_\star,\\
M_{\star,\rm gross}&\leftarrow M_{\star,\rm gross}+\Delta M_\star,&
\dot M_\star&\leftarrow\dot M_\star+\frac{\Delta M_\star}{\delta t}.
\end{aligned}
```

*References:* [Meraxes reservoir updates](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/star_formation.c).

## Stellar recycling and supernova feedback

### Delayed and instantaneous returns

For formed mass $m_i$, initial metallicity $Z_i$ and age interval $[a_i,b_i]$, $\dot R$, $\dot Y_Z$ and $\mathcal E$ are tabulated mass-return, metal-return and cumulative-energy kernels per formed mass. $L_j$ is snapshot lookback time in Myr.

```{math}
:label: gal-metallicity
Z=\begin{cases}
\min[1,\max(0,M_Z/M)] & M>0\ \mathrm{and}\ M_Z>0,\\
0 & \mathrm{otherwise}.
\end{cases}
```

*References:* [Meraxes metallicity definition](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

```{math}
:label: gal-feedback-kernels
\begin{gathered}
R_i(Z)=\int_{a_i}^{b_i}\dot R(a,Z)\,da,\\
Y_i(Z)=\int_{a_i}^{b_i}\dot Y_Z(a,Z)\,da,\\
e_i(Z)=\mathcal E(b_i,Z)-\mathcal E(a_i,Z).
\end{gathered}
```

*References:* {ref}`Qin et al. (2019), stellar evolution <ref-qin2019xv>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/stellar_feedback.c); [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-delayed-feedback
\begin{gathered}
\Delta M_{\rm rec}=\sum_i m_iR_i(Z_i),\\
\Delta M_{Z,\rm ret}=\sum_i m_iY_i(Z_i),\\
E_{\rm SN,raw}=\sum_i m_i e_i(Z_i).
\end{gathered}
```

*References:* {ref}`Qin et al. (2019), stellar evolution <ref-qin2019xv>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/stellar_feedback.c); [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-feedback-ages
a_i=\frac{L_{s-i-1}+L_{s-i}}{2}-L_{s-1},\qquad
b_i=\frac{L_{s-i-1}+L_{s-i}}{2}-L_s,
```

*References:* {ref}`Mutch et al. (2016), §2.5.1 <ref-mutch2016>`; [Meraxes stellar-age bins](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/stellar_feedback.c).

Instantaneous recycling replaces the return kernels by fixed fractions $R$ and $y_Z$; current-burst supernova energy remains age dependent.

```{math}
:label: gal-ira
\Delta M_{\rm rec}=R\,\Delta M_\star,\qquad
\Delta M_{Z,\rm ret}=y_Z\,\Delta M_\star.
```

*References:* {ref}`Mutch et al. (2016), §§2.6–2.7 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

### Reheating and ejection

The two efficiency families control reheating ($q=\mathrm{rh}$) and energy coupling ($q=E$). Each has amplitude $\epsilon_{q,0}$, velocity scale $V_q$ and redshift exponent $\beta_q$, with either a smooth or broken velocity law.

```{math}
:label: gal-sn-guo
\epsilon_q(V_{\max},z)=\epsilon_{q,0}
\left(\frac{1+z}{4}\right)^{\beta_q}
\left[0.5+\left(\frac{V_{\max}}{V_q}\right)^{-\alpha_q}\right],
\qquad q\in\{\mathrm{rh},E\}.
```

*References:* {ref}`Guo et al. (2011) <ref-guo2011>`; [Meraxes redshift-dependent extension](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-sn-muratov
\epsilon_q(V_{\max},z)=\epsilon_{q,0}
\left(\frac{1+z}{4}\right)^{\beta_q}
\left(\frac{V_{\max}}{V_q}\right)^{-\alpha_q(V_{\max})},\qquad
\alpha_q(V)=\begin{cases}\alpha_{q,\rm low},&V<V_q,\\\alpha_{q,\rm high},&V\geq V_q.\end{cases}
```

*References:* {ref}`Muratov et al. (2015) <ref-muratov2015>`; [Meraxes feedback prescription](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-sn-caps
\epsilon_{\rm rh}\leftarrow\min(\epsilon_{\rm rh},\epsilon_{\rm rh,max}),\qquad
\epsilon_E\leftarrow\min(\epsilon_E,1).
```

*References:* {ref}`Mutch et al. (2016), eqs. 13–14 <ref-mutch2016>`; [Meraxes feedback limits](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-sn-reheating
\Delta M_{\rm rh,0}=\epsilon_{\rm rh}\frac{E_{\rm SN,raw}}{e_{\rm SN,tot}},\qquad
E_{\rm SN}=\epsilon_E E_{\rm SN,raw}.
```

*References:* {ref}`Mutch et al. (2016), §§2.5–2.5.1 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

$e_{\rm SN,tot}$ is lifetime energy per formed stellar mass; $V_h$ describes the galaxy or group potential. Insufficient energy reduces reheating to $2E_{\rm SN}/V_h^2$; ejection cannot exceed available hot gas.

```{math}
:label: gal-sn-energy-host
E_{\rm rh,h}=\frac12\Delta M_{\rm rh}V_h^2,\qquad
\Delta M_{\rm ej,h}=\frac{E_{\rm SN}-E_{\rm rh,h}}{V_h^2/2}.
```

*References:* {ref}`Mutch et al. (2016), eqs. 19–21 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-sn-energy-ejection
\Delta M_{\rm ej}=\left[\frac{2E_{\rm SN}}{V_{\rm vir,FOF}^2}-\Delta M_{\rm rh}\right]_+.
```

*References:* {ref}`Mutch et al. (2016), eqs. 19–21 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

When stars and reheating exceed cold gas, $f_{\rm avail}$ rescales stellar, reheated, recycled and remnant masses; precomputed energy and metal returns retain their values. $f_{Z,\rm ret}$ is the fraction of reheated metals retained in cold gas.

```{math}
:label: gal-sn-availability
f_{\rm avail}=\frac{M_{\rm cold}}{\Delta M_{\rm rh}+\Delta M_\star}<1,
```

*References:* [Meraxes mass and metal transfers](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-sn-metal-transfer
\Delta M_{Z,\rm rh}=(1-f_{Z,\rm ret})Z_{\rm cold}\Delta M_{\rm rh},\qquad
\Delta M_{Z,\rm ej}=Z_{\rm hot,central}\Delta M_{\rm ej}.
```

*References:* [Meraxes mass and metal transfers](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

## Galaxy mergers

The friction clock uses physical separation, halo particle counts $N_p$, satellite halo mass and timescale factor $f_{\rm merge}$. Burst efficiency follows the baryonic merger ratio, with normalisation $\alpha_{\rm burst}$ and exponent $\beta_{\rm burst}$. Remnants contribute to baryonic masses when present.

```{math}
:label: gal-merger-separation
r_{\rm phys}=\frac{r_{\rm com}}{1+z_{s-1}},\qquad
r=\min(r_{\rm phys},R_{\rm vir,mother}),\qquad
\ln\Lambda=\ln\left(1+\frac{N_{p,\rm mother}}{N_{p,\rm sat}}\right).
```

*References:* [Meraxes merger-clock inputs](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/mergers.c).

```{math}
:label: gal-merger-time
t_{\rm merge}=f_{\rm merge}\,
\frac{1.17r^2V_{\rm vir,mother}}{\ln\Lambda\,GM_{\rm vir,sat}}.
```

*References:* {ref}`Mutch et al. (2016), eq. 25 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/mergers.c).

```{math}
:label: gal-merger-ratio
B_g=M_{\star,g}+M_{\rm cold,g},\qquad
\mu_{\rm merge}=\frac{\min(B_1,B_2)}{\max(B_1,B_2)}.
```

*References:* {ref}`Mutch et al. (2016), eq. 26 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/mergers.c).

```{math}
:label: gal-merger-burst
\Delta M_{\star,\rm burst,0}=\min\left[
M_{\rm cold,combined},\;
\alpha_{\rm burst}\,\mu_{\rm merge}^{\beta_{\rm burst}}
M_{\rm cold,combined}\right].
```

*References:* {ref}`Mutch et al. (2016), eq. 26 <ref-mutch2016>`; [Meraxes implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/mergers.c).

## Black holes and AGN

### Accretion and mechanical feedback

{ref}`Qin et al. (2017) <ref-qin2017x>` describe black-hole growth and feedback. Fixed seeds grow by hot accretion and merger-fed cold disks. $\kappa_R$, $f_{\rm BH}$ and $\epsilon_Q$ set hot accretion, disk provisioning and mechanical coupling; $\lambda_{\rm Edd}$ is the Eddington ratio.

```{math}
:label: bh-seed
M_{\rm BH}=M_{\rm seed},\qquad
M_{\rm seed}=\mathrm{constant}.
```

*References:* {ref}`Qin et al. (2017b), §2.2 <ref-qin2017x>`.

```{math}
:label: bh-hot-accretion
\Delta M_{\rm hot}^{\rm trial}
=\kappa_{\rm R}\,G\,C_{\rm B}\,X\,M_{\rm BH}\,\Delta t,
\qquad C_{\rm B}=3.4754,
\qquad X=\frac{m_p k_B T_{\rm vir}}{\Lambda(T_{\rm vir},Z)}.
```

*References:* {ref}`Qin et al. (2017b), §2.2 <ref-qin2017x>`.

```{math}
:label: bh-hot-cap
\Delta M_{\rm hot}
=\min\!\left[
\Delta M_{\rm hot}^{\rm trial},\,
M_{\rm BH}\exp\!\left(\frac{\lambda_{\rm Edd}\Delta t}{\eta t_E}\right),\,
M_{\rm hot}\right],
\qquad \eta=0.06,\quad t_E=450.514890\ {\rm Myr}.
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c#L272-L297).

```{math}
:label: bh-radio-heating
\Delta M_{\rm heat,R}
=\frac{2\eta c^2}{V_{\rm vir}^2}\Delta M_{\rm hot}.
```

*References:* {ref}`Qin et al. (2017b), Eq. (18) <ref-qin2017x>`; {ref}`Croton et al. (2006) <ref-croton2006>`.

Radio heating and hot accretion are reduced together if heating exceeds cooling. The hot cap is an exponential mass, whereas cold accretion uses its increment. For cold episodes, $u$ is uniform on $[0,1)$ and $L_i$ is lookback time.

```{math}
:label: bh-merger-reservoir
\Delta M_{\rm disk}
=\min\!\left[
M_{\rm cold},\,
\frac{f_{\rm BH}\,\mu\,(1+z)^{p_{\rm Q}}}
{1+(280\ {\rm km\ s^{-1}}/V_{\rm vir})^2}
M_{\rm cold}\right].
```

*References:* {ref}`Qin et al. (2017b), §2.2.2 <ref-qin2017x>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c#L324-L351).

```{math}
:label: bh-cold-interval
\Delta t_{\rm snap}=L_{i-1}-L_i,
\qquad
\Delta t_{\rm eff}=
\begin{cases}
(1-u)\Delta t_{\rm snap},&\text{random start within the first interval},\\
\Delta t_{\rm snap},&\text{otherwise}.
\end{cases}
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c#L353-L445).

```{math}
:label: bh-cold-accretion
\Delta M_{\rm acc}
=\min\!\left[
M_{\rm disk},\,
M_{\rm BH}\left\{
\exp\!\left(\frac{\lambda_{\rm Edd}\Delta t_{\rm eff}}{\eta t_E}\right)-1
\right\}\right],
\qquad
M_{\rm BH}\longleftarrow M_{\rm BH}+(1-\eta)\Delta M_{\rm acc}.
```

*References:* {ref}`Qin et al. (2017b), §2.2 <ref-qin2017x>`.

```{math}
:label: bh-duration-lbol
\begin{gathered}
t_{\rm acc}=\frac{\eta t_E}{\lambda_{\rm Edd}}
\ln\!\left(1+\frac{\Delta M_{\rm acc}}{M_{\rm BH}}\right),
\\
L_{\rm bol}=\frac{\lambda_{\rm Edd}c^2}{t_E}
\sqrt{M_{\rm BH}(M_{\rm BH}+\Delta M_{\rm acc})},
\\
f_{\rm duty}=\operatorname{clip}_{[0,1]}
\left(\frac{t_{\rm acc}}{\Delta t_{\rm eff}}\right).
\end{gathered}
```

*References:* {ref}`Qin et al. (2017b), §2.2.2 <ref-qin2017x>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c).

```{math}
:label: bh-quasar-heating
\Delta M_{\rm heat,Q}
=\epsilon_{\rm Q}\frac{2\eta c^2}{V_{\rm vir}^2}\Delta M_{\rm acc}.
```

*References:* {ref}`Qin et al. (2017b), §2.2 <ref-qin2017x>`.

Luminosities use the pre-accretion black-hole mass. Here $\mu$ is the baryonic merger ratio, $p_Q$ its feeding redshift exponent, and $\eta$ the radiative efficiency.

### Luminosity and escaping photons

For the [Shen et al. (2020)](https://doi.org/10.1093/mnras/staa1381) corrections, $\ell=L_{\rm bol}/(10^{10}L_\odot)$ and $L_{1450}=\nu L_\nu$. Spectral indices describe $L_\nu\propto\nu^{-\alpha}$; $h_P$ is Planck's constant and $\theta_Q$ the full opening angle.

```{math}
:label: bh-bolometric-corrections
\begin{aligned}
k_{1450}(\ell)&=1.862\ell^{-0.361}+4.870\ell^{-0.0063},\\
k_{\rm hard}(\ell)&=4.073\ell^{-0.026}+12.60\ell^{0.278},\\
k_{\rm soft}(\ell)&=5.712\ell^{-0.026}+17.67\ell^{0.278},\\
L_{1450}&=L_{\rm bol}/k_{1450},\qquad
L_{\rm X,hard}=L_{\rm bol}/k_{\rm hard},\qquad
L_{\rm X,soft}=L_{\rm bol}/k_{\rm soft}.
\end{aligned}
```

*References:* {ref}`Shen et al. (2020), Table 1 <ref-shen2020>`.

```{math}
:label: bh-ionizing-rate
\begin{gathered}
L_\nu(912)=\frac{L_{1450}}{\nu_{1450}}
\left(\frac{\nu_{912}}{\nu_{1450}}\right)^{-\alpha_{\rm UV,soft}},
\\
\dot N_{\gamma,\rm int}
=\frac{L_\nu(912)}{h_P\alpha_{\rm UV,hard}}.
\end{gathered}
```

*References:* {ref}`Qin et al. (2017b), §4.1 <ref-qin2017x>`; {ref}`Lusso et al. (2015) <ref-lusso2015>`.

```{math}
:label: bh-uv-escape
\begin{gathered}
f_{\rm open}=1-\cos(\theta_{\rm Q}/2),
\\
f_{\rm esc,BH}=\operatorname{clip}_{[0,1]}
\left[f_{\rm BH,esc,0}
\left(\frac{1+z}{6}\right)^{p_{\rm BH,esc}}\right],
\\
\Delta N_{\gamma,\rm esc}
=f_{\rm open}f_{\rm esc,BH}\dot N_{\gamma,\rm int}t_{\rm acc}.
\end{gathered}
```

*References:* {ref}`Qin et al. (2017b), §§2.3 and 4.1 <ref-qin2017x>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c#L99-L116).

```{math}
:label: bh-equivalent-sources
\begin{gathered}
\Delta M_{\rm BH,eff}
=\frac{m_p\Delta N_{\gamma,\rm esc}}{N_{\gamma,*}},
\\
M_{\rm BH,eff}\longleftarrow M_{\rm BH,eff}+\Delta M_{\rm BH,eff},
\\
\dot M_{\rm BH,eff}^{\rm on}=\frac{\Delta M_{\rm BH,eff}}{t_{\rm acc}}.
\end{gathered}
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c).

```{math}
:label: bh-equivalent-stored-increment
\Delta M_{\rm BH,eff}^{\rm num}
=B\,(8.40925088\times10^{-8})\,
\frac{f_{\rm esc,BH}}{N_{\gamma,*}},
\qquad B=\frac{f_{\rm open}\dot N_{\gamma,\rm int}t_{\rm acc}}{10^{60}}.
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c).

$N_{\gamma,*}$ counts photons per stellar baryon. The equivalent-source increment is expressed numerically in $10^{10}M_\odot$. Response weighting uses the local ionization-response time $t_{\rm resp}$; duty weighting instead multiplies the on-state rate by $f_{\rm duty}$.

```{math}
:label: bh-response-weight
\dot M_{\rm BH,eff}^{\rm response}
=\dot M_{\rm BH,eff}^{\rm on}
\exp\!\left[-\frac{(1-f_{\rm duty})\Delta t_{\rm eff}}{t_{\rm resp}}\right]
\quad(t_{\rm resp}>0).
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c#L353-L445).

```{math}
:label: bh-quasar-magnitude
M_{1450}=-19.07395-2.5\log_{10}
\left[\frac{L_{1450}}{10^{10}L_\odot}\right].
```

*References:* [Meraxes quasar AB-magnitude conversion](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/save.c#L81).

### X-ray absorption

For [Ueda et al. (2014)](https://doi.org/10.1088/0004-637X/786/2/104), $\ell_X=\log_{10}[L_{\rm X,hard}/(\mathrm{erg\,s^{-1}})]$. Columns span $\log_{10}(N_H/\mathrm{cm^{-2}})=[20,21,22,23,24,26]$; transmission uses each bin midpoint. $P_j$ is the bin probability.

```{math}
:label: bh-obscured-fraction
\psi=\operatorname{clip}_{[0.20,0.84]}
\left[0.43\{1+\min(z,2)\}^{0.48}-0.24(\ell_X-43.75)\right],
\qquad \epsilon=1.7,\qquad f_{\rm CTK}=1.
```

*References:* {ref}`Ueda et al. (2014), §2.2 <ref-ueda2014>`.

```{math}
:label: bh-column-distribution
\begin{aligned}
g_0&=\begin{cases}
1-\dfrac{2+\epsilon}{1+\epsilon}\psi,&\psi<\dfrac{1+\epsilon}{3+\epsilon},\\
\dfrac23-\dfrac{3+2\epsilon}{3+3\epsilon}\psi,&\text{otherwise},
\end{cases}\\
g_1&=\begin{cases}
\dfrac{\psi}{1+\epsilon},&\psi<\dfrac{1+\epsilon}{3+\epsilon},\\
\dfrac13-\dfrac{\epsilon}{3+3\epsilon}\psi,&\text{otherwise},
\end{cases}\\
g_2&=\frac{\psi}{1+\epsilon},\qquad
g_3=\frac{\epsilon\psi}{1+\epsilon},\qquad
g_4=\frac{f_{\rm CTK}\psi}{2},\\
A&=g_0+g_1+g_2+g_3+2g_4,\qquad
P_j=g_j/A\ (j<4),\qquad P_4=2g_4/A.
\end{aligned}
```

*References:* {ref}`Ueda et al. (2014), §2.2 <ref-ueda2014>`.

```{math}
:label: bh-xray-transmission
T_b(N_H)=
\frac{\int_{E_{b,\min}}^{E_{b,\max}}
E^{-\gamma_b}\exp[-N_H\sigma_{\rm pe}(E)]\,C_{\rm T}(N_H)\,dE}
{\int_{E_{b,\min}}^{E_{b,\max}} E^{-\gamma_b}\,dE},
\qquad
C_{\rm T}=\begin{cases}
\exp[-1.21\sigma_TN_H],&\log_{10}N_H\ge24,\\
1,&\text{otherwise},
\end{cases}
```

*References:* {ref}`Morrison & McCammon (1983) <ref-morrison1983>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/blackhole_feedback.c#L9-L147).

```{math}
:label: bh-photoelectric-cross-section
\sigma_{\rm pe}(E)
=10^{-24}(C_0+C_1E+C_2E^2)E^{-3}\ {\rm cm^2},\qquad E\text{ in keV}.
```

*References:* {ref}`Morrison & McCammon (1983) <ref-morrison1983>`.

$\gamma_b$ is the band spectral index and $\sigma_T=6.6524\times10^{-25}\,\mathrm{cm^2}$. Transmitted luminosity is $T_bL_{X,b}$. Photoelectric coefficients follow [Morrison & McCammon (1983)](https://doi.org/10.1086/161102):

| Energy (keV) | $C_0$ | $C_1$ | $C_2$ |
|---|---:|---:|---:|
| 0.030–0.100 | 17.3 | 608.1 | −2150.0 |
| 0.100–0.284 | 34.6 | 267.9 | −476.1 |
| 0.284–0.400 | 78.1 | 18.8 | 4.3 |
| 0.400–0.532 | 71.4 | 66.8 | −51.4 |
| 0.532–0.707 | 95.5 | 145.8 | −61.1 |
| 0.707–0.867 | 308.9 | −380.6 | 294.0 |
| 0.867–1.303 | 120.6 | 169.3 | −47.7 |
| 1.303–1.840 | 141.3 | 146.8 | −31.5 |
| 1.840–2.471 | 202.7 | 104.7 | −17.0 |
| 2.471–3.210 | 342.7 | 18.7 | 0.0 |
| 3.210–4.038 | 352.2 | 18.7 | 0.0 |
| 4.038–7.111 | 433.9 | −2.4 | 0.75 |
| 7.111–8.331 | 629.0 | 30.9 | 0.0 |
| 8.331–10.000 | 701.2 | 25.2 | 0.0 |

The absorption cross-section vanishes below 0.03 keV and at or above 10 keV.

## Population III stars and enrichment

### Population assignment and stellar fates

{ref}`Ventura et al. (2024) <ref-ventura2024>` describe this extension. $Z_{\rm crit}$ uses a reference metallicity of $0.01$. IMF masses are in $M_\odot$; $N$ and $F$ denote event numbers per formed solar mass and mass fractions.

```{math}
:label: opt-population-criterion
\text{population}=\begin{cases}
2,& Z_{\rm cold}/0.01>Z_{\rm crit},\\
3,&\text{otherwise},
\end{cases}
\qquad Z_{\rm cold}=\frac{M_{Z,\rm cold}}{M_{\rm cold}}.
```

*References:* {ref}`Ventura et al. (2024), §2 <ref-ventura2024>`.

```{math}
:label: opt-popiii-imf
\phi(m)=\begin{cases}
A m^{-2.35},&\text{Salpeter cases},\\
\dfrac{A}{m}\exp\!\left[-\dfrac{\ln^2(m/m_c)}{2\sigma^2}\right],&\text{lognormal cases},
\end{cases}
\qquad \int_{m_{\min}}^{m_{\max}}m\phi(m)\,dm=1.
```

*References:* {ref}`Ventura et al. (2024) <ref-ventura2024>`; {ref}`Raiter et al. (2010) <ref-raiter2010>`.

| IMF | Mass range ($M_\odot$) | $m_c$ ($M_\odot$) | $\sigma$ | Photons per stellar baryon |
|---|---|---:|---:|---:|
| Salpeter | 1–500 | — | — | 22000 |
| Salpeter | 50–500 | — | — | 72000 |
| Lognormal | 1–500 | 10 | 1 | 47600 |
| Lognormal | 1–500 | 60 | 1 | 71000 |

```{math}
:label: opt-popiii-fate-integrals
\begin{aligned}
N_{\rm CC}&=\int_{[8,40]\cap I}\phi(m)\,dm,&
F_{\rm CC}&=\int_{[8,40]\cap I}m\phi(m)\,dm,\\
N_{\rm PI}&=\int_{[140,260]\cap I}\phi(m)\,dm,&
F_{\rm PI}&=\int_{[140,260]\cap I}m\phi(m)\,dm,\\
F_{\rm BH,rem}&=\int_{([40,140]\cup[260,\infty))\cap I}m\phi(m)\,dm,
&&I=[m_{\min},m_{\max}].
\end{aligned}
```

*References:* {ref}`Ventura et al. (2024), §2 <ref-ventura2024>`.

### Stellar lifetimes and feedback

For $x=\log_{10}(m/M_\odot)$, lifetime coefficients are:

```{math}
:label: opt-popiii-lifetime
\log_{10}\!\left(\frac{t_*}{\rm yr}\right)
=a_0+a_1x+a_2x^2+a_3x^3.
```

*References:* {ref}`Schaerer (2002) <ref-schaerer2002>`.

| Mass loss | $a_0$ | $a_1$ | $a_2$ | $a_3$ |
|---|---:|---:|---:|---:|
| Strong | 8.795 | −1.797 | 0.332 | 0 |
| None | 9.785 | −3.759 | 1.413 | −0.186 |

$m_*(\tau)$ inverts the lifetime. A burst age $\tau_i$ and neighbouring intervals select the progenitor window; $n_i=\int_{W_i}\phi\,dm$ and $f_i=\int_{W_i}m\phi\,dm$.

```{math}
:label: opt-popiii-delay-window
m_{\min,i}=m_*(\tau_i+\Delta t_{\rm before}/2),\qquad
m_{\max,i}=m_*(\tau_i-\Delta t_{\rm after}/2),
\qquad W_i=[m_{\min,i},m_{\max,i}]\cap[8,40]\cap I.
```

*References:* {ref}`Mutch et al. (2016) <ref-mutch2016>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/PopIII.c#L411-L462).

```{math}
:label: opt-popiii-delay-fractions
q_{N,i}=\frac{n_i}{N_{\rm CC}+N_{\rm PI}},\qquad
q_{M,i}=\frac{f_i}{F_{\rm CC}+F_{\rm PI}},\qquad
Y_{k,i}=\frac{f_i}{F_{\rm CC}}\,y_k(m_{\max,i}).
```

*References:* {ref}`Mutch et al. (2016) <ref-mutch2016>`; {ref}`Heger & Woosley (2010) <ref-heger2010>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/PopIII.c#L464-L542).

| Upper progenitor mass ($M_\odot$) | Recycled fraction | Remnant fraction | Metal yield |
|---|---:|---:|---:|
| $\le15$ | 0.88 | 0.12 | 0.05 |
| 15–25 | 0.88 | 0.12 | 0.09 |
| 25–30 | 0.88 | 0.12 | 0.15 |
| $>30$ | 0.60 | 0.40 | 0 |

Core-collapse return factors $r_{\rm CC,i},y_{\rm CC,i},q_{\rm CC,i}$ correspond to these yields. Pair-instability returns are immediate, with $r_{\rm PI}=1$, $y_{\rm PI}=0.5$; $F_{\rm BH}=F_{\rm BH,rem}$.

```{math}
:label: gal-popiii-delayed
\begin{gathered}
\Delta M_{\rm rec,III}=\sum_{i>0}m_{\star,III,i}F_{\rm CC}r_{\rm CC,i},\\
\Delta M_{Z,III}=\sum_{i>0}m_{\star,III,i}F_{\rm CC}y_{\rm CC,i},\\
\Delta M_{\rm rem}=\sum_{i>0}m_{\star,III,i}F_{\rm CC}q_{\rm CC,i}.
\end{gathered}
```

*References:* {ref}`Ventura et al. (2024), §2 <ref-ventura2024>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-popiii-current
\begin{aligned}
\Delta M_{\rm rec,III}&=m_{\star,III}\left(F_{\rm CC}r_{\rm CC,0}+F_{\rm PI}r_{\rm PI}\right),\\
\Delta M_{\rm rem}&=m_{\star,III}\left(F_{\rm BH}+F_{\rm CC}q_{\rm CC,0}\right),\\
\Delta M_{Z,III}&=m_{\star,III}F_{\rm CC}y_{\rm CC,0}
+\left[m_{\star,III}F_{\rm PI}y_{\rm PI}-20M_\odot\right]_+.
\end{aligned}
```

*References:* {ref}`Ventura et al. (2024), §2 <ref-ventura2024>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-popiii-reheat
\Delta M_{\rm rh,III}=\epsilon_{\rm rh,III}m_{\star,III}
\left(\frac{e_{\rm CC,0}}{E_{\rm CC}}+\frac{e_{\rm PI}}{E_{\rm PI}}\right),
```

*References:* {ref}`Ventura et al. (2024), §2 <ref-ventura2024>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

```{math}
:label: gal-popiii-energy-conversion
\begin{aligned}
S_{\rm III}&=\sum_{i>0}m_i e_{\rm CC,i}^{\rm cgs},\\
E_{\rm SN,III}^{\rm delayed}&=\epsilon_{E,III}
\frac{S_{\rm III}N_{\rm CC}\kappa_M}{U_E},\\
E_{\rm SN,III}^{\rm current}&=\epsilon_{E,III}
\frac{m_0(e_{\rm CC,0}^{\rm cgs}+e_{\rm PI}^{\rm cgs})}{U_E}.
\end{aligned}
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c).

$E_{\rm CC},E_{\rm PI}$ are event energies, $e_i$ their population-weighted energies per formed mass. For numerical masses, $\kappa_M=10^{10}/h$ converts to solar masses; $U_E$ converts cgs energy. Delayed energy includes the additional factor $N_{\rm CC}\kappa_M$ shown above.

### Streaming velocities and molecular cooling

$n=0$ or $1$ selects zero or rms streaming speed; $\mu_T$ follows the virial-temperature convention. $J_{\rm LW,21}$ is intensity in $10^{-21}\,\mathrm{erg\,s^{-1}\,Hz^{-1}\,cm^{-2}\,sr^{-1}}$.

```{math}
:label: opt-streaming-cooling
\sigma_{\rm bc}(z)=30\frac{1+z}{1000}\ {\rm km\ s^{-1}},\qquad
v_{\rm bc}=n\sigma_{\rm bc},\quad n\in\{0,1\},\qquad
V_{\rm cool}=\sqrt{(3.714\ {\rm km\ s^{-1}})^2+(4.015v_{\rm bc})^2}.
```

*References:* {ref}`Fialkov et al. (2012) <ref-fialkov2012>`; {ref}`Ventura et al. (2024), Eq. (3) <ref-ventura2024>`.

The molecular cooling temperature is obtained by substituting $V_{\rm cool}$ into Equation {eq}`gal-virial-temperature`. Its threshold mass $M_{\rm cool,0}$ then follows Equation {eq}`gal-temperature-mass`.

```{math}
:label: opt-lw-mass-threshold
M_{\rm crit,MC}(\mathbf{x},z)
=M_{\rm cool,0}(z)
\left[1+6.96\{4\pi J_{\rm LW,21}(\mathbf{x},z)\}^{0.47}\right].
```

*References:* {ref}`Fialkov et al. (2013), Eq. (1) <ref-fialkov2013>`; {ref}`Ventura et al. (2024) <ref-ventura2024>`.

### Metal bubbles

Energy $E$ injected at $t_0$ drives expansion through ambient density $n$. Inside the virial radius, the greater halo/IGM density applies; outside, the IGM density applies. $N_{\rm metal}$ is the number of cells per side.

```{math}
:label: opt-metal-bubble
R_{\rm metal}(t)=\left(\frac{E}{m_p}\right)^{1/5}n^{-1/5}(t-t_0)^{2/5}.
```

*References:* {ref}`Ventura et al. (2024), metal-enrichment model <ref-ventura2024>`.

```{math}
:label: opt-metal-ambient-density
\begin{gathered}
n_{\rm halo}
=\frac{(M_{\rm cold}+M_{\rm hot})U_M/m_p}
{(4\pi/3)(R_{\rm vir}U_L)^3},\\
n_{\rm IGM}
=\frac{M_{\rm gas,cell}U_M/m_p}
{[\Delta x\,U_L/(1+z)]^3},\\
\Delta x=\frac{L_{\rm box}}{N_{\rm metal}}.
\end{gathered}
```

*References:* {ref}`Ventura et al. (2024) <ref-ventura2024>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/supernova_feedback.c#L464-L540).

```{math}
:label: opt-enrichment-probability
P_{\rm enrich,cell}
=\operatorname{clip}_{[0,1]}
\left[\frac{\sum_{g\ \mathrm{in\ cell},\ R_g\ge3R_{{\rm vir},g}}
(4\pi/3)R_{c,g}^3}{(L_{\rm box}/N_{\rm metal})^3}\right].
```

*References:* {ref}`Ventura et al. (2024) <ref-ventura2024>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/metal_evo.c).

```{math}
:label: opt-cell-metallicity
Z_{\rm IGM,cell}=\frac{M_{Z,\rm cell}}{M_{\rm gas,cell}},\qquad
\Delta M_{Z,\rm infall}=Z_{\rm IGM,cell}\Delta M_{\rm infall}
\quad\text{for an externally enriched central galaxy}.
```

*References:* {ref}`Ventura et al. (2024) <ref-ventura2024>`; [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/metal_evo.c).

$R_c=(1+z)R_{\rm metal}$; filling factors sum bubble volumes within each source cell.


## Stellar radiation

Stellar radiation is described by the escaping ionizing output, the continuum spectrum and emission-line luminosities.

### Ionizing escape fraction

For stellar population $p$, the deterministic escape fraction before attenuation is

```{math}
:label: stoch-fesc-relation
f_{\rm esc,p}^{\rm pre}=f_{{\rm norm},p}R_d(z)P_{d,p}^{\alpha},\qquad
R_d(z)=\begin{cases}
1,&d=0,\\
[(1+z)/z_{\rm off}]^{\beta},&1\leq d\leq6.
\end{cases}
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c#L93-L294).

Here $f_{{\rm norm},p}$ is the population normalization, $z_{\rm off}$ the redshift normalization, and $\alpha$ and $\beta$ the property and redshift exponents. The selector $d$ determines the dimensionless property factor:

| $d$ | $P_{d,p}$ | If the required property is non-positive |
|---:|---|---|
| 0, 1 | $1$ | No property required |
| 2 | $M_\star/(10^{10}M_\odot)$ | Set the pre-attenuation fraction to one |
| 3 | $\dot M_{\star,p}/(M_\odot\,\mathrm{yr}^{-1})$ | Set it to zero |
| 4 | $(M_{\rm cold}/R_d^2)/(10M_\odot\,\mathrm{pc}^{-2})$ | Set it to one |
| 5 | $M_{\rm vir}/(10^{10}M_\odot)$ | Set it to one |
| 6 | $(\dot M_{\star,p}/M_\star)/(10\,\mathrm{Gyr}^{-1})$ | Set it to zero |

Masses and radii in this table are physical quantities; $R_d$ is the disk scale radius. Both populations use total stellar mass in selectors 2 and 6. The surface-density proxy in selector 4 contains no disk-area factor.

Circumgalactic attenuation and clipping give

```{math}
:label: stoch-fesc-clamp
f_{\rm esc,p}^{0}=\min\!\left[1,\max\!\left(0,f_{\rm esc,p}^{\rm pre}e^{-\tau_{\rm CGM}}\right)\right].
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c#L93-L294).

The optical depth is zero without attenuation. With positive hot-gas mass and virial radius,

```{math}
:label: stoch-cgm-optical-depth
\tau_{\rm CGM}=A_{\rm CGM}
\left(\frac{M_{\rm hot}}{10^8M_\odot}\right)^a
\left(\frac{10\,\mathrm{kpc}}{R_{\rm vir}}\right)^{2a}S^b.
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c#L1920-L1965).

$A_{\rm CGM}$, $a$ and $b$ set its normalization and response. The environmental driver $S$ is either $10\Gamma_{12}$, the current-snapshot accumulation of $\Gamma_{12}\Delta t/\mathrm{Myr}$, or the local clumping factor. Negative drivers are replaced by zero. Here $\Gamma_{12}$ is the physical photoionization rate in units of $10^{-12}\,\mathrm{s}^{-1}$.

### Scatter and escaped source budgets

A source quantity $q>0$ receives a lognormal draw with width $\sigma$ dex:

```{math}
:label: stoch-lognormal
q^{\rm draw}=q\,10^{\sigma g},\qquad
 g\sim\mathcal N(0,1),\qquad s=(\ln10)\sigma,\qquad
\mathbb E[q^{\rm draw}]=q\,e^{s^2/2}.
```

*References:* [Meraxes lognormal-scatter implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/misc_tools.c).

The input $q$ is the median. Zero sources remain zero. Stellar escape fractions use $q=f_{\rm esc}^{0}$ and are clipped again at one. Their mean is

```{math}
:label: stoch-clipped-mean
\mathbb E[f_{\rm esc}^{\rm draw,clipped}]
=f_{\rm esc}^{0}e^{s^2/2}
\Phi\!\left(\frac{\ln(1/f_{\rm esc}^{0})-s^2}{s}\right)
+1-\Phi\!\left(\frac{\ln(1/f_{\rm esc}^{0})}{s}\right),
\qquad 0<f_{\rm esc}^{0}\leq1,\ s>0.
```

*References:* analytic expectation of Equation {eq}`stoch-lognormal` after clipping at one.

$\Phi$ is the standard-normal cumulative distribution. Clipping can lower the mean relative to a deterministic fraction already near one.

Each star-formation update contributes to cumulative escaped mass $G$ and the snapshot escaped-rate accumulator $W$:

```{math}
:label: stoch-source-accumulators
G\leftarrow G+\Delta M_{\star,j}f_j,\qquad
W\leftarrow W+\dot M_{\star,j}f_j.
```

*References:* [Meraxes source implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c#L93-L294).

$\Delta M_{\star,j}$ is newly formed mass, while $\dot M_{\star,j}$ is the SFR accumulated by update $j$ within the snapshot. Ordinary sources use the deterministic fraction; treated sources use a fresh scattered fraction. Because $W$ sums update contributions, it need not equal the final SFR multiplied by the final escape fraction. $G$ counts formed mass, including mass later returned to the gas.

### Continuum, filters and dust

$\ell_\nu(\tau,Z)$ is luminosity per formed stellar mass. Filter responses use fluxes at 10 pc, wavelengths in Å, and transmission $T$; observer-filter responses require distance conversion for apparent magnitudes.

```{math}
:label: opt-sed-convolution
L_\nu(t)=\int_0^t\dot M_*(t')\ell_\nu(t-t',Z(t'))\,dt'.
```

*References:* {ref}`Qiu et al. (2019), stellar-population synthesis <ref-qiu2019>`.

```{math}
:label: opt-sector-rest-filter
F_b=\frac{3.34\times10^4}{\ln(\lambda_2/\lambda_1)}
\int_{\lambda_1}^{\lambda_2}\lambda f_\lambda(\lambda)\,d\lambda,
\qquad
F_{\beta,b}=\frac{1}{\lambda_2-\lambda_1}
\int_{\lambda_1}^{\lambda_2}f_\lambda(\lambda)\,d\lambda.
```

*References:* [Sector filter integration](https://github.com/meraxes-devs/sector/blob/master/sector/clib/spectra.c#L226-L330).

```{math}
:label: opt-sector-observer-filter
\begin{gathered}
A_b=\left[\int\frac{T(\lambda_{\rm obs})}{\lambda_{\rm obs}}\,d\lambda_{\rm obs}\right]^{-1},
\\
\lambda_{\rm pivot,rest}
=\frac{\left[A_b\int\lambda_{\rm obs}T(\lambda_{\rm obs})\,d\lambda_{\rm obs}\right]^{1/2}}{1+z},
\\
W_b(\lambda_{\rm rest})
=3.34\times10^4 A_b\lambda_{\rm obs}T(\lambda_{\rm obs}),
\quad\lambda_{\rm obs}=(1+z)\lambda_{\rm rest}.
\end{gathered}
```

*References:* [Sector filter integration](https://github.com/meraxes-devs/sector/blob/master/sector/clib/spectra.c#L226-L330).

The {ref}`Qiu et al. (2019) <ref-qiu2019>` dust model uses metallicity exponent $p_Z$, redshift coefficient $a_z$, optical-depth amplitudes $\tau_0$ and spectral slopes $n$. Superscript int denotes numerical mass and length. Birth clouds attenuate young stars; interstellar dust attenuates both components.

```{math}
:label: opt-dust-normalization
D=\left(\frac{Z_{\rm cold}}{0.02}\right)^{p_Z}
M_{\rm cold}^{\rm int}
\left(10^3R_{\rm disk}^{\rm int}\right)^{-2}
\exp(a_z z),\qquad
\tau_{\rm UV,ISM}=\tau_{\rm ISM,0}D,\qquad
\tau_{\rm UV,BC}=\tau_{\rm BC,0}D.
```

*References:* {ref}`Qiu et al. (2019), dust-to-gas model <ref-qiu2019>`.

```{math}
:label: opt-sector-dust-attenuation
\begin{gathered}
\tau_{\rm ISM}(\lambda)=\tau_{\rm UV,ISM}
\left(\frac{\lambda}{1600\ {\rm \mathring A}}\right)^{n_{\rm ISM}},\\
\tau_{\rm BC}(\lambda)=\tau_{\rm UV,BC}
\left(\frac{\lambda}{1600\ {\rm \mathring A}}\right)^{n_{\rm BC}},\\
F_b^{\rm dusty}=e^{-\tau_{\rm ISM}(\lambda_b)}
\left[e^{-\tau_{\rm BC}(\lambda_b)}F_b^{\rm young}+F_b^{\rm old}\right].
\end{gathered}
```

*References:* {ref}`Charlot & Fall (2000) <ref-charlot2000>`; {ref}`Qiu et al. (2019) <ref-qiu2019>`.

```{math}
:label: opt-ab-magnitude
M_b=-2.5\log_{10}F_b^{\rm num}+8.9
-2.5\log_{10}\!\left[
\frac{U_M}{U_T}\frac{{\rm seconds\ per\ year}}{M_\odot}\right].
```

*References:* [Meraxes AB-magnitude conversion](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/magnitudes.c#L742-L795).

### [O III] 5008 Å

Collision strengths use $u=T/(10^4\,\mathrm{K})$; rates are evaluated at $10^4$ K. Transition rates $(A_{31},A_{32},A_{41},A_{43})=(4.57\times10^{-6},3.52\times10^{-5},0.215,1.7)\,\mathrm{s^{-1}}$.

```{math}
:label: opt-oiii-collision
\begin{aligned}
\Omega_{30}&=0.243\,u^{0.120+0.031\ln u},\qquad
\Omega_{40}=0.0321\,u^{0.118+0.057\ln u},\\
\beta_q&=10^6\sqrt{\frac{2\pi\hbar^4}{k_Bm_e^3}},\\
k_{03}&=\beta_q T^{-1/2}\Omega_{30}e^{-29169/T},\qquad
k_{04}=\beta_q T^{-1/2}\Omega_{40}e^{-61207/T},\\
k_{\rm exc}&=k_{03}+k_{04}\frac{A_{43}}{A_{43}+A_{41}},\qquad
b_{32}=\frac{A_{32}}{A_{32}+A_{31}}.
\end{aligned}
```

*References:* {ref}`Draine (2011), §2.3 and Table F.2 <ref-draine2011>`; [Meraxes atomic-rate implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/emission_lines.c#L18-L32).

$M_d=M_*+M_{\rm cold}$, $m_d=M_d/M_{\rm vir}$, $c_s^2=10^{12}\,\mathrm{cm^2\,s^{-2}}$, $E_{\rm SN}=10^{51}$ erg and $\dot M_{\rm cool}=\Delta M_{\rm cool}/\Delta t$.

```{math}
:label: opt-oiii-bubble-radius
r_b^3=(8\times10^{-5}\ {\rm Mpc})^3
\left(\frac{\lambda}{0.05}\right)^4
\left(\frac{m_d}{0.17}\right)^{-2}
\left(\frac{M_{\rm vir}^{\rm int}\,10^2}{h}\right)^{-2/3}
\left(\frac{1+z}{10}\right)^{-4}.
```

*References:* {ref}`Pathak et al. (2025), Eq. (7) <ref-pathak2025>`.

```{math}
:label: opt-oiii-disk-support
\begin{aligned}
\rho_0&=\frac{GM_{\rm cold}M_d}{58.7528\pi c_s^2R_d^4},\\
\dot\Sigma_{\rm SN}&=\frac{0.156\dot M_*}{12.26M_\odot\pi R_d^2},\qquad
s_{\rm SN}=\frac{(2\dot\Sigma_{\rm SN}\,0.03E_{\rm SN}/\rho_0)^{2/3}}{c_s^2},\\
\delta^{-1}&=\frac{M_{\rm vir}(R_d/R_{\rm vir})^3+M_{\rm cold}+M_*}{M_{\rm cold}},\\
Q^2&=0.98\delta^{-2}\frac{c_s^2}{(1.4V_{\max})^2},\qquad
s_{\rm acc}=\frac{(0.6G\dot M_{\rm cool}Q^2/2.94)^{2/3}}{c_s^2},\\
\rho_{\rm eff}&=\frac{\rho_0}{1+s_{\rm SN}+s_{\rm acc}},\qquad
N_b=\frac{3M_d}{4\pi\rho_{\rm eff}r_b^3},\qquad
\dot N_{\gamma,b}=\frac{4000\dot M_*}{m_pN_b}.
\end{aligned}
```

*References:* {ref}`Pathak et al. (2025), §§2.1–2.3 <ref-pathak2025>`; [Meraxes disk-density implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/emission_lines.c).

```{math}
:label: opt-oiii-stromgren
\begin{gathered}
r_S^2=\left[\frac{3\dot N_{\gamma,b}}{4\pi\alpha_B\rho_{\rm eff}^2}\right]^{2/3},
\\
\Delta q_{\rm ion}=
\frac{1.5874\dot N_{\gamma,b}}{4\pi r_S^2\rho_{\rm eff}},
\\ \alpha_B=2.6\times10^{-13}\ {\rm cm^3\ s^{-1}}.
\end{gathered}
```

*References:* [Meraxes ionization-parameter implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/emission_lines.c#L74-L81).

Here $\rho_{\rm eff}$ is a mass-density proxy used directly in the ionization expressions. A physical Strömgren radius requires number density; dimensionless ionization parameter additionally requires division by $c$. Thus $q_{\rm ion}$ is an uncalibrated proxy.

```{math}
:label: opt-oiii-luminosity
\begin{gathered}
L_{5008}=\left[\frac{10^{8.69-12}}{0.0134}Z_{\rm cold}\right]
k_{\rm exc}b_{32}
\frac{\dot N_{\gamma,b}N_b}{\alpha_B}
h_P\nu_{32}\,f_{\rm OIII},
\\ \nu_{32}=\frac{c}{5008\ {\rm \mathring A}},\quad f_{\rm OIII}=0.8.
\end{gathered}
```

*References:* {ref}`Yang & Lidz (2020), §§3.3 and 5.2 <ref-yang2020>`; {ref}`Pathak et al. (2025), §4 <ref-pathak2025>`; [Meraxes line-luminosity implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/physics/emission_lines.c).

```{math}
:label: opt-oiii-dust
L_{5008}^{\rm dusty}=L_{5008}\,10^{0.4(M_b-M_b^{\rm dusty})}.
```

*References:* {ref}`Qiu et al. (2019) <ref-qiu2019>`; [Meraxes line-attenuation implementation](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/save.c).

$L_{5008}$ is in $\mathrm{erg\,s^{-1}}$; dust uses the intrinsic and attenuated magnitudes of a continuum band containing the line.

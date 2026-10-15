# Reionization and the 21-cm signal

Escaped stellar and black-hole radiation ionizes and heats the intergalactic medium (IGM). Meraxes couples these processes to galaxy growth through photoheating suppression of gas accretion. The model follows [Mutch et al. (2016)](https://arxiv.org/abs/1512.00562), [Mesinger et al. (2011)](https://arxiv.org/abs/1003.3878) and [Balu et al. (2023)](https://arxiv.org/abs/2210.08910).

<details>
<summary>On this page</summary>

```{contents}
:local:
:depth: 1
:backlinks: none
```

</details>

## Sources and filtering

### Galaxy source quantities

Ionizing radiation uses cumulative escaped stellar mass and the snapshot escaped-SFR accumulator from Equation {eq}`stoch-source-accumulators`. Stellar histories include newly formed mass and inherited progenitor contributions. Source prescriptions can add scatter or substitute a median SFR while preserving the galaxy model's physical reservoirs and star-formation history.

Thermal sources use instantaneous SFR or a mass-averaged estimate, with $t_H=H^{-1}$ and dimensionless timescale $f_{\rm sfr}$.

```{math}
:label: igm-thermal-sfr
\dot M_{\star,X}=\begin{cases}
\dot M_{\star},&\text{instantaneous source},\\
M_{\star}^{\rm gross}/t_{\rm sfr},&\text{mass-averaged source},
\end{cases}
\qquad t_{\rm sfr}=f_{\rm sfr}\,t_H(z).
```

*References:* {ref}`Balu et al. (2023) <ref-balu2023>`; [Meraxes src/core/reionization.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

### Median source relation

The median relation is constructed separately for centrals, resolved satellites and orphans, and for each stellar population. Its halo-mass coordinate is

```{math}
:label: stoch-mass-grid
x=\log_{10}\!\left(\frac{M_{\rm vir}}{10^{10}h^{-1}M_\odot}\right),\qquad
x_i=-3.50+0.02i,\qquad i=0,\ldots,375.
```

*References:* [Meraxes src/core/Stochasticity.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/Stochasticity.c).

Galaxies enter the lower enclosing mass bin, with out-of-range masses assigned to the nearest endpoint. Each occupied bin uses the median of positive, finite log SFRs across all galaxies in that population and type. For an even sample, the two central logarithms are averaged. Empty interior bins are interpolated in log SFR; empty exterior bins receive a floor of $-30$ in internal log SFR.

At a galaxy's mass, the interpolated source SFR is

```{math}
:label: stoch-sfr-interpolation
\dot M_\star^{\rm src}=U_{\dot M}\,10^{(1-w)y_i+wy_{i+1}},\qquad
w=\frac{x-x_i}{0.02},\qquad
y_i=\log_{10}\!\left(\frac{\dot M_{\star,i}}{U_{\dot M}}\right).
```

*References:* [Meraxes src/core/Stochasticity.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/Stochasticity.c).

$U_{\dot M}$ is the internal mass-rate unit. Endpoint masses use the endpoint value. A floored left node returns the floor; a floored right node holds the valid left value constant. Galaxies with non-positive original SFR retain zero source SFR.

The cumulative source mass advances by

```{math}
:label: stoch-treated-gsm
\Delta M_{\star,g}^{\rm src}=\dot M_{\star,g}^{\rm src}\Delta t_g,\qquad
M_{\star,g}^{\rm src}\leftarrow M_{\star,g}^{\rm src}+\Delta M_{\star,g}^{\rm src}.
```

*References:* [Meraxes src/core/Stochasticity.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/Stochasticity.c).

$\Delta t_g$ is the galaxy evolution interval. Escape-fraction relations involving stellar mass or SFR use these treated quantities; halo and gas properties retain their evolved galaxy values. Cumulative source histories survive population transitions and are added during mergers.

### Source normalization

Let $T_Q$ and $R_Q$ be the ordinary and treated budgets for source quantity $Q$. For a positive treated budget,

```{math}
:label: stoch-recalibration
C_Q=\frac{T_Q}{R_Q},\qquad Q^{\rm grid}=C_QQ^{\rm treated}.
```

*References:* [Meraxes src/core/Stochasticity.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/Stochasticity.c).

The target is the untreated budget in the same evolving galaxy population. Escaped cumulative mass and escaped SFR have independent factors. Escape-fraction scatter uses global population-specific factors; median-SFR sources use factors within each galaxy-type and halo-mass cell. Pop. II X-ray and unweighted Ly-alpha source rates use separate global factors.

Recalibration scales the deposited source field. Per-galaxy cumulative histories retain their original values, and older radiation-history fields retain their earlier normalization. A zero treated budget cannot create a missing source: global escape-fraction recalibration rejects an unmatched positive target, while median-SFR and X-ray/Ly-alpha recalibration leave the zero field unchanged.

### Spatial assignment and filtering

$L$ is the comoving box side, $D$ the grid dimension, and $V_{\rm cell}=(L/D)^3$. Nearest-grid-point assignment sums galaxy properties $q_g$ within each cell $\mathcal C$; density contrast is $\delta$.

```{math}
:label: igm-ngp
q_{ijk}=\sum_{g\in\mathcal C_{ijk}}q_g,
\qquad \delta=\rho/\bar\rho-1.
```

*References:* {ref}`Mutch et al. (2016) <ref-mutch2016>`; [Meraxes src/core/reionization.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

Present-day number densities count hydrogen and helium nuclei; $Y_{\rm He}$ is the helium mass fraction. Number densities use $\mathrm{cm^{-3}}$.

```{math}
:label: igm-number-densities
\bar n_{H,0}=\frac{\Omega_b\rho_{\rm crit,0}(1-Y_{\rm He})}{m_p},
\quad \bar n_{{\rm He},0}=\frac{\Omega_b\rho_{\rm crit,0}Y_{\rm He}}{4m_p},
\quad \bar n_{b,0}=\bar n_{H,0}+\bar n_{{\rm He},0},
\quad f_H=\frac{\bar n_{H,0}}{\bar n_{b,0}},
\quad f_{\rm He}=\frac{\bar n_{{\rm He},0}}{\bar n_{b,0}}.
```

*References:* [Meraxes src/core/reionization.h](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.h).

Fourier filtering smooths sources and density on radius $R$, using the top-hat, sharp-$k$ or Gaussian windows in Equation {eq}`num-filter-windows`.

The transform normalization and smoothed field are defined in Equation {eq}`num-fft-normalisation`.

```{math}
:label: igm-radius-mass
M_R=\begin{cases}
\frac{4\pi}{3}\,\Omega_m\rho_{\rm crit}R^3,&\text{top hat},\\
(2\pi)^{3/2}\,\Omega_m\rho_{\rm crit}R^3,&\text{Gaussian}.
\end{cases}
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

Filtering decreases radii by $f_R$ until the cell scale. $\rho_{\rm crit}$ is the present-day critical density. The evolving maximum-radius prescription is:

```{math}
:label: igm-radius-sequence
R_0=\min(R_{\max},0.620350491L),\qquad
R_{n+1}=R_n/f_R.
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

```{math}
:label: igm-radius-evolution
R_{\max}(z)=\begin{cases}
25.483241248322766\ h^{-1}{\rm cMpc},&z>6,\\
112[(1+z)/5]^{-4.4}\ h^{-1}{\rm cMpc},&z\leq6.
\end{cases}
```

*References:* [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

## Ionization and photoheating feedback

$N_{\gamma,\star}$ counts photons per stellar baryon; escape fractions already enter the source masses. Instantaneous recycling additionally divides $\zeta_\star$ by the adopted recycled fraction.

```{math}
:label: igm-efficiency
\zeta_\star=\frac{N_{\gamma,\star}}
{f_b(1-3Y_{\rm He}/4)},\qquad f_b=\Omega_b/\Omega_m.
```

*References:* {ref}`Mutch et al. (2016) <ref-mutch2016>`; {ref}`Sobacchi & Mesinger (2013, UVB feedback) <ref-sobacchi2013feedback>`; [Meraxes src/core/reionization.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

Filtered cell masses $s_R$, $s_{R,\rm III}$ and $b_R$ represent escaped Pop II, Pop III and stellar-equivalent black-hole sources. $\mathcal B_R$ is the photon budget per atom.

```{math}
:label: igm-photon-budget
\mathcal B_R=
\frac{4\pi R^3}{3V_{\rm cell}M_R}
\frac{\zeta_\star(s_R+b_R)+\zeta_{\rm III}s_{R,\rm III}}
{1+\delta_R}.
```

*References:* {ref}`Mutch et al. (2016) <ref-mutch2016>`; {ref}`Qin et al. (2017b) <ref-qin2017x>`; {ref}`Ventura et al. (2025) <ref-ventura2025>`; [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

A cell becomes ionized when its photon budget exceeds neutral atoms plus recombinations. $x_{e,R}$ is the partial electron fraction; $N_{{\rm rec},R}$ is the cumulative recombination sink. Remaining cells receive the partial-ionization solution.

```{math}
:label: igm-ionization-barrier
\mathcal B_R>(1-x_{e,R})\left[1+
\frac{N_{{\rm rec},R}}{1+\delta_R}\right].
```

*References:* {ref}`Furlanetto et al. (2004) <ref-furlanetto2004>`; {ref}`Sobacchi & Mesinger (2014) <ref-sobacchi2014>`; {ref}`Balu et al. (2023) <ref-balu2023>`.

```{math}
:label: igm-partial-cell
x_{\rm HI}=\operatorname{clip}_{[0,1]}
\left(1-x_{e,\rm cell}-\mathcal B_{\rm cell}\right).
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

The UV photoionization rate uses escaped SFR density; the feedback intensity uses accumulated source density. $R$ is comoving, all lengths are in cm, $h_P$ is Planck's constant, $\alpha$ the UV spectral slope, and $b_\Gamma$ the halo radiation bias.

```{math}
:label: igm-gamma
\Gamma_{\rm HI}=\sigma_{{\rm HI},0}\frac{\alpha}{\alpha+2.75}
(1+z)^2R\,\dot n_{\gamma,\rm esc}^{\rm com},
\qquad
\dot n_{\gamma,\rm esc}^{\rm com}
=\frac{N_\gamma\dot\rho_{\star,\rm esc}^{\rm com}}{m_p}.
```

*References:* {ref}`Sobacchi & Mesinger (2014), equation 4 <ref-sobacchi2014>`.

```{math}
:label: igm-j21
J_{21}=10^{21}\frac{h_P\alpha}{4\pi}\,
b_\Gamma(1+z)^2R\,
\frac{N_\gamma\rho_{\star,\rm esc}^{\rm com}}{m_p t_{\rm sfr}}.
```

*References:* {ref}`Sobacchi & Mesinger (2013, UVB feedback), equations 8–9 <ref-sobacchi2013feedback>`; {ref}`Mutch et al. (2016) <ref-mutch2016>`.

$\Gamma_{\rm HI}$ has units $\mathrm{s^{-1}}$; $J_{21}$ measures intensity in $10^{-21}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}$. Photoheating follows [Sobacchi & Mesinger (2013)](https://arxiv.org/abs/1301.6776), with ionization redshift $z_{\rm ion}$ and $(a,b,c,d)=(0.17,-2.1,2,2.5)$.

```{math}
:label: igm-critical-mass
M_{\rm crit}=M_0J_{21}^{a}
\left(\frac{1+z}{10}\right)^b
\left[1-\left(\frac{1+z}{1+z_{\rm ion}}\right)^c\right]^d,
\quad z<z_{\rm ion}.
```

*References:* {ref}`Sobacchi & Mesinger (2013, gas depletion) <ref-sobacchi2013>`; {ref}`Mutch et al. (2016) <ref-mutch2016>`.

```{math}
:label: igm-baryon-modifier
f_{\rm mod}=2^{-M_{\rm crit}/M_{\rm vir}},\qquad
M_{b,\rm target}=f_bf_{\rm mod}M_{\rm vir}.
```

*References:* {ref}`Sobacchi & Mesinger (2013, gas depletion) <ref-sobacchi2013>`; {ref}`Mutch et al. (2016) <ref-mutch2016>`.

At $M_{\rm vir}=M_{\rm crit}$, a halo retains half the cosmic baryon fraction. Homogeneous alternatives use a transition around $z_{\rm re}$, cooling mass $M_{\rm cool}$, or filtering mass $M_F$. The transition width and offset are $\Delta z_{\rm re}$ and $\Delta z_{\rm sc}$; $M_0(z)$ is the virial mass corresponding to the adopted heated-gas temperature.

```{math}
:label: igm-homogeneous-sobacchi
g(z)=\left[1+\exp\!\left(
\frac{z-(z_{\rm re}-\Delta z_{\rm sc})}{\Delta z_{\rm re}}
\right)\right]^{-1},\qquad
M_{\min}=M_{\rm cool}\left(\frac{M_0(z)}{M_{\rm cool}}\right)^{g(z)}.
```

*References:* {ref}`Sobacchi & Mesinger (2013, UVB feedback), equations 11–12 <ref-sobacchi2013feedback>`.

```{math}
:label: igm-gnedin
M_F=M_J[f(a)]^{3/2},\qquad
M_J=25\,\Omega_m^{-1/2}\,2.21
\quad[10^{10}h^{-1}M_\odot],\qquad
f_{\rm mod}=\left[1+0.26
\frac{\max(M_F,M_{\rm cool})}{M_{\rm vir}}\right]^{-3}.
```

*References:* {ref}`Gnedin (2000) <ref-gnedin2000>`; {ref}`Kravtsov et al. (2004), Appendix B <ref-kravtsov2004>`.

Here $a=(1+z)^{-1}$, $a_0=(1+z_0)^{-1}$, $a_r=(1+z_r)^{-1}$ and $\alpha_G=6$; $z_0$ and $z_r$ delimit the reionization transition.

```{math}
:label: igm-gnedin-filter
f(a)=\begin{cases}
\displaystyle\frac{3a}{(2+\alpha_G)(5+2\alpha_G)}
\left(\frac a{a_0}\right)^{\alpha_G},&a\leq a_0,\\[5pt]
\displaystyle\frac3a\left\{
a_0^2\left[\frac1{2+\alpha_G}-\frac{2(a/a_0)^{-1/2}}{5+2\alpha_G}\right]
+\frac{a^2}{10}-\frac{a_0^2}{10}\left[5-4(a/a_0)^{-1/2}\right]
\right\},&a_0<a<a_r,\\[5pt]
\displaystyle\frac3a\left\{
a_0^2\left[\frac1{2+\alpha_G}-\frac{2(a/a_0)^{-1/2}}{5+2\alpha_G}\right]
+\frac{a_r^2}{10}\left[5-4(a/a_r)^{-1/2}\right]
-\frac{a_0^2}{10}\left[5-4(a/a_0)^{-1/2}\right]
+\frac{aa_r}{3}-\frac{a_r^2}{3}\left[3-2(a/a_r)^{-1/2}\right]
\right\},&a\geq a_r.
\end{cases}
```

*References:* {ref}`Kravtsov et al. (2004), Appendix B <ref-kravtsov2004>`.

## Recombination and self-shielding

Unresolved overdensity $\Delta$ follows a volume-weighted distribution with tabulated $A,C_0,\beta$ ($\beta<0$). Resolved overdensity enters through an effective redshift.

```{math}
:label: igm-mhr-pdf
P_V(\Delta)=A\exp\!\left[
-\frac{(\Delta^{-2/3}-C_0)^2}{2(2\delta_0/3)^2}
\right]\Delta^{\beta},\qquad \delta_0=\frac{7.61}{1+z_{\rm eff}}.
```

*References:* {ref}`Miralda-Escudé et al. (2000) <ref-miralda2000>`; {ref}`Sobacchi & Mesinger (2014) <ref-sobacchi2014>`.

```{math}
:label: igm-effective-redshift
1+z_{\rm eff}=(1+z)(1+\delta)^{1/3}.
```

*References:* {ref}`Sobacchi & Mesinger (2014) <ref-sobacchi2014>`; [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

[Rahmati et al. (2013)](https://arxiv.org/abs/1210.7808) describe attenuation within dense gas; $T_4=T/(10^4\,\mathrm K)$ and $\Gamma_{12}=\Gamma_{\rm bg}/(10^{-12}\,\mathrm{s^{-1}})$.

```{math}
:label: igm-self-shielding
\Delta_{\rm ss}=26.7T_4^{0.17}
\left(\frac{1+z_{\rm eff}}{10}\right)^{-3}
\Gamma_{12}^{2/3},
\qquad
\frac{\Gamma_{\rm ss}}{\Gamma_{\rm bg}}=
0.98\left[1+\left(\frac\Delta{\Delta_{\rm ss}}\right)^{1.64}\right]^{-2.28}
+0.02\left[1+\frac\Delta{\Delta_{\rm ss}}\right]^{-0.84}.
```

*References:* {ref}`Rahmati et al. (2013) <ref-rahmati2013>`; {ref}`Sobacchi & Mesinger (2014), equations 8–9 <ref-sobacchi2014>`; [Meraxes src/core/recombinations.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/recombinations.c).

```{math}
:label: igm-equilibrium-neutral
\Gamma_{\rm ss}\chi=
\alpha_B(T)n_H(1+c_{\rm He})(1-\chi)^2,
\qquad c_{\rm He}=\frac{Y_{\rm He}}{4-3Y_{\rm He}},
\qquad
\alpha_B(T)=\alpha_{B,10^4}\left(\frac T{10^4\,\mathrm K}\right)^{-0.75}.
```

*References:* {ref}`Sobacchi & Mesinger (2014), equation 7 <ref-sobacchi2014>`; [Meraxes src/core/recombinations.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/recombinations.c).

$\chi$ is the subgrid neutral fraction and $\alpha_{B,10^4}=2.59\times10^{-13}\,\mathrm{cm^3\,s^{-1}}$. Density integration gives recombinations $\mathcal R$ ($\mathrm{s^{-1}}$), clumping $C_{\rm HII}$ and residual neutrality.

```{math}
:label: igm-recombination-integrals
\begin{aligned}
\mathcal R&=\alpha_B(T)\bar n_H(z_{\rm eff})
\int_{0.01}^{200}P_V(\Delta)\Delta^2[1-\chi(\Delta)]^2\,d\Delta,\\
C_{\rm HII}&=
\frac{\int_{0.01}^{200}P_V(\Delta)\Delta^2[1-\chi(\Delta)]^2d\Delta}
{\left[\int_{0.01}^{200}P_V(\Delta)\Delta[1-\chi(\Delta)]d\Delta\right]^2},\\
x_{\rm HI,res}^{\rm phys}&=
\int_{0.01}^{200}P_V(\Delta)\chi(\Delta)d\Delta.
\end{aligned}
```

*References:* {ref}`Sobacchi & Mesinger (2014) <ref-sobacchi2014>`; [Meraxes src/core/recombinations.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/recombinations.c); [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

```{math}
:label: igm-recombination-update
N_{\rm rec}^{i+1}=N_{\rm rec}^{i}
+\mathcal R_i\,\left|\frac{dt}{dz}\right|_i
(z_{i-1}-z_i)(1-x_{{\rm HI},i}),
\qquad t_{\rm resp}=\frac1{\Gamma_{\rm HI}+\mathcal R}.
```

*References:* {ref}`Sobacchi & Mesinger (2014) <ref-sobacchi2014>`; [Meraxes src/core/recombinations.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/recombinations.c); [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

## X-ray propagation and deposition

Galaxy X-ray luminosity scales with the stellar thermal source. Each emitting component has band luminosity $L_{[a,b]}$ and spectral index $\alpha_X$.

```{math}
:label: igm-hmxb-luminosity
L_{X,\rm gal}=(L_X/\mathrm{SFR})\dot M_{\star,X},
\qquad L_\nu\propto\nu^{-\alpha_X}.
```

*References:* {ref}`Balu et al. (2023) <ref-balu2023>`; [Meraxes src/core/ComputeTs.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputeTs.c).

The heating-source SFR follows Equation {eq}`igm-thermal-sfr`; median-source models substitute their source SFR or cumulative source mass. For the ordinary luminosity $L_X^0$ in Equation {eq}`igm-hmxb-luminosity`, Pop. II scatter gives

```{math}
:label: stoch-xray-luminosity
L_X^{\rm draw}=L_X^0\,10^{\sigma_Xg},\qquad g\sim\mathcal N(0,1).
```

*References:* [Meraxes src/core/reionization.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

$L_X$ has units of $\mathrm{erg\,s^{-1}}$ and $\sigma_X$ is measured in dex. The draw is unbounded above, with its mean given by Equation {eq}`stoch-lognormal`. Pop. III and AGN luminosities follow their separate prescriptions. Historical heating fields retain the draws made when each field was created.

```{math}
:label: igm-spectral-normalization
L_\nu=A_X\nu^{-\alpha_X},\qquad
A_X=\frac{L_{[a,b]}}{I_\alpha},\qquad
I_\alpha=\begin{cases}
\displaystyle\frac{\nu_b^{1-\alpha_X}-\nu_a^{1-\alpha_X}}{1-\alpha_X},&\alpha_X\ne1,\\
\ln(\nu_b/\nu_a),&\alpha_X=1.
\end{cases}
```

*References:* {ref}`Balu et al. (2023) <ref-balu2023>`; [Meraxes src/core/ComputeTs.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputeTs.c).

Retarded radiation integrates comoving emissivity $\epsilon_\nu^{\rm com}$ over emitting shells. Meraxes approximates attenuation using the frequency at which $\tau_X=1$.

```{math}
:label: igm-radiation-integral
J_\nu(\boldsymbol x,z)=\frac{c(1+z)^3}{4\pi}
\int_z^\infty\left|\frac{dt}{dz'}\right|
\epsilon_{\nu'}^{\rm com}(\boldsymbol x,z')
e^{-\tau_X(\nu,z,z')}dz',
\qquad\nu'=\nu\frac{1+z'}{1+z}.
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; {ref}`Balu et al. (2023) <ref-balu2023>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

Absorption combines H I, He I and He II. Hydrogenic thresholds $h_P\nu_Z$ are 13.60 and 54.40 eV; He I has threshold 24.59 eV. Cross sections vanish below threshold.

```{math}
:label: igm-xray-crosssection
\widetilde\sigma_X(\nu,x_e)=
f_H(1-x_e)\sigma_{\rm HI}(\nu)
+f_{\rm He}(1-x_e)\sigma_{\rm HeI}(\nu)
+f_{\rm He}x_e\sigma_{\rm HeII}(\nu).
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; {ref}`Balu et al. (2023) <ref-balu2023>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-hydrogenic-crosssection
\sigma_Z(\nu)=\frac{6.3\times10^{-18}}{Z^2}
\left(\frac{\nu_Z}{\nu}\right)^4
\frac{\exp[4-4\arctan(\epsilon)/\epsilon]}
{1-\exp[-2\pi/\epsilon]} \mathrm{cm^2},
\qquad \epsilon=\sqrt{\nu/\nu_Z-1},\quad Z=1,2.
```

*References:* {ref}`Osterbrock (1989), p. 14 <ref-osterbrock1989>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-helium-crosssection
x=\frac{h_P\nu}{13.61\,\mathrm{eV}}-0.4434,
\qquad y=\sqrt{x^2+2.136^2},\qquad
\sigma_{\rm HeI}=9.492\times10^{-16}
[(x-1)^2+2.039^2]y^{3.188/2-5.5}
\left(1+\sqrt{y/1.469}\right)^{-3.188}\ \mathrm{cm^2}.
```

*References:* {ref}`Verner et al. (1996) <ref-verner1996>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-xray-optical-depth
\tau_X(\nu,z,z')=
\int_z^{z'}c\left|\frac{dt}{dz''}\right|
\bar n_b(z'')\,Q_{\rm HI}(z'')\,
\widetilde\sigma_X\!\left(\nu\frac{1+z''}{1+z},\bar x_e\right)dz''.
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; {ref}`Balu et al. (2023) <ref-balu2023>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

$Q_{\rm HI}$ is the neutral filling fraction. Species $s$ has weight $a_s$, threshold $\nu_s$ and photoelectron energy $E_s=h_P(\nu-\nu_s)$. Tabulated deposition fractions and secondary counts set the heating, ionization and Ly$\alpha$ kernels.

```{math}
:label: igm-xray-kernels
\begin{aligned}
K_{\rm heat}&=\int_{\nu_{\rm lo}}^{\nu_{\rm hi}}
\left(\frac\nu{\nu_0}\right)^{-\alpha_X-1}
\sum_s a_s\sigma_s(\nu)E_s f_{\rm heat}(E_s,x_e)d\nu,\\
K_{\rm ion}&=\int_{\nu_{\rm lo}}^{\nu_{\rm hi}}
\left(\frac\nu{\nu_0}\right)^{-\alpha_X-1}
\sum_s a_s\sigma_s(\nu)
\left[1+N_{\rm ion,HI}+N_{\rm ion,HeI}+N_{\rm ion,HeII}\right]d\nu,\\
K_{\alpha}&=\frac{c}{4\pi\nu_\alpha H(z)}
\int_{\nu_{\rm lo}}^{\nu_{\rm hi}}
\left(\frac\nu{\nu_0}\right)^{-\alpha_X-1}
\sum_s a_s\sigma_s(\nu)N_\alpha(E_s,x_e)d\nu.
\end{aligned}
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; {ref}`Furlanetto & Stoever (2010) <ref-furlanetto2010>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

For component $b$, $S_{b,j}$ is shell-averaged luminosity density ($\mathrm{erg\,s^{-1}\,cm^{-3}}$); $\nu_0$ is the escape-threshold frequency. Negative $dt/dz'$ and shell increments give positive time weights.

```{math}
:label: igm-xray-shell-sum
\mathcal D_{q,b}(\boldsymbol x,z)=
g_q\mathcal A_b(z)\sum_j
\left(\frac{dt}{dz'}\Delta z'\right)_j
S_{b,j}(\boldsymbol x)(1+z'_j)^{-\alpha_b}
K_{q,b,j}(x_e),
\qquad
\mathcal A_b(z)=\frac{c(1+z)^{\alpha_b+3}}{h_P\nu_0^{\alpha_b+1}I_{\alpha_b}},
\qquad g_{\rm heat}=g_{\rm ion}=1,
\quad g_\alpha=n_b.
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; {ref}`Balu et al. (2023) <ref-balu2023>`; [Meraxes src/core/ComputeTs.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputeTs.c).

## Gas temperature and partial ionization

RECFAST supplies initial mean temperature and residual electrons; $c_T$ adds linear adiabatic fluctuations. Thermal evolution uses the following expansion and growth approximations.

```{math}
:label: igm-thermal-initial
x_e(\boldsymbol x,z)=\bar x_{e,\rm RECFAST}(z),\qquad
T_K(\boldsymbol x,z)=\bar T_{K,\rm RECFAST}(z)[1+c_T(z)\delta(\boldsymbol x,z)],
\qquad c_T(z)=0.58-0.006(z-10).
```

*References:* {ref}`Seager et al. (1999) <ref-seager1999>`; {ref}`Muñoz (2023), equation 48 <ref-munoz2023>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-expansion-helpers
H_X(z)=H_0\sqrt{\Omega_m(1+z)^3+\Omega_r(1+z)^4+\Omega_\Lambda},
\qquad
\left(\frac{dt}{dz}\right)_X=
-\frac{\sqrt{\Omega_m+\Omega_\Lambda}}
{H_0(1+z)\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}}.
```

*References:* {ref}`Peebles (1980) <ref-peebles1980>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-growth-flat
D(z)=\frac{g[\Omega_m(z)]}{g(\Omega_m)(1+z)},
\qquad
\Omega_m(z)=\frac{\Omega_m(1+z)^3}
{\Omega_\Lambda+\Omega_m(1+z)^3+\Omega_r(1+z)^4},
\qquad
g(u)=\frac{2.5u}{1/70+u(209-u)/140+u^{4/7}}.
```

*References:* {ref}`Liddle et al. (1996), equations 6–8 <ref-liddle1996>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

$H_0=100h\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$; the time derivative neglects radiation and curvature. Einstein–de Sitter gives $D=(1+z)^{-1}$; the open, zero-$\Lambda$ growth solution is:

```{math}
:label: igm-growth-open
D(z)=\frac{F[x(z)]}{F(x_0)},\qquad
x_0=\Omega_m^{-1}-1,\quad x(z)=\frac{|\Omega_m^{-1}-1|}{1+z},
\qquad
F(x)=1+\frac3x+
\frac{3\sqrt{1+x}}{x^{3/2}}
\ln[\sqrt{1+x}-\sqrt x].
```

*References:* {ref}`Peebles (1980), equation 11.16 <ref-peebles1980>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

X-rays supply ionization $\mathcal I_{X,b}$ ($\mathrm{s^{-1}}$ per nucleus) and heat $\epsilon_{X,b}$ ($\mathrm{erg\,s^{-1}}$ per nucleus). Temperature also responds to particle creation, adiabatic expansion and Compton exchange.

```{math}
:label: igm-electron-evolution
\frac{dx_e}{dz}=\frac{dt}{dz}
\left[\sum_b\mathcal I_{X,b}
-\alpha_A(T_K)C_Xx_e^2f_Hn_b\right],\qquad C_X=2,
\quad n_b=\bar n_{b,0}(1+z)^3(1+\delta).
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-temperature-evolution
\frac{dT_K}{dz}=
\frac{2}{3k_B(1+x_e)}\frac{dt}{dz}\sum_b\epsilon_{X,b}
-\frac{T_K}{1+x_e}\frac{dx_e}{dz}
+\frac23T_K\left[\frac3{1+z}
+\frac{D'(z)}{1/\delta+D(z)}\right]
+\left.\frac{dT_K}{dz}\right|_{\rm C}.
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-compton
\left.\frac{dT_K}{dt}\right|_{\rm C}
=\frac{8\sigma_Ta_RT_{\rm CMB}^4}{3m_ec}
\frac{x_e}{1+x_e+f_{\rm He}}(T_{\rm CMB}-T_K),
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-compton-fit
\left.\frac{dT_K}{dz}\right|_{\rm C}
=-1.51\times10^{-4}
\frac{x_e}{1+x_e+f_{\rm He}}
\frac{T_{\rm CMB}^4(T_{\rm CMB}-T_K)}
{hE(z)(1+z)},\quad E(z)=H(z)/H_0,
\quad T_{\rm CMB}=T_{\rm CMB,0}(1+z).
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

Temperatures use K. $\sigma_T$, $a_R$, $m_e$ and $k_B$ are the Thomson cross section, radiation constant, electron mass and Boltzmann constant. Case-A recombination uses $u=\ln[T/(1.1604505\times10^4\,\mathrm K)]$.

```{math}
:label: igm-alpha-a
\alpha_A(T)=\exp\!\left(\sum_{j=0}^{9}a_ju^j\right)
\quad[\mathrm{cm^3\,s^{-1}}],
```

*References:* {ref}`Abel et al. (1997) <ref-abel1997>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

| $j$ | $a_j$ | $j$ | $a_j$ |
|---|---:|---|---:|
| 0 | $-28.6130338$ | 5 | $-1.42150291\times10^{-5}$ |
| 1 | $-0.72411256$ | 6 | $4.98910892\times10^{-6}$ |
| 2 | $-2.02604473\times10^{-2}$ | 7 | $5.75561414\times10^{-7}$ |
| 3 | $-2.38086188\times10^{-3}$ | 8 | $-1.85676704\times10^{-8}$ |
| 4 | $-3.21260521\times10^{-4}$ | 9 | $-3.07113524\times10^{-9}$ |

Snapshot evolution uses a forward-Euler redshift step.

```{math}
:label: igm-euler-update
x_e^{\rm new}=x_e^{\rm old}+\left(\frac{dx_e}{dz}\right)_i\Delta z,
\qquad T_K^{\rm new}=T_K^{\rm old}+\left(\frac{dT_K}{dz}\right)_i\Delta z,
\qquad\Delta z=z_i-z_{i-1}<0.
```

*References:* [Meraxes src/core/ComputeTs.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputeTs.c).

## Ly$\alpha$ coupling and spin temperature

Collisions and resonant Ly$\alpha$ scattering couple the spin temperature to gas temperature. $\kappa_{10}$ denotes temperature-dependent collision coefficients.

```{math}
:label: igm-spin-temperature
T_S^{-1}=\frac{T_{\rm CMB}^{-1}
+\widetilde x_\alpha T_{c,\rm eff}^{-1}
+x_cT_K^{-1}}{1+\widetilde x_\alpha+x_c}.
```

*References:* {ref}`Hirata (2006) <ref-hirata2006>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-collision-coupling
x_c=\frac{T_\star}{A_{10}T_{\rm CMB}}
\left[n_{\rm HI}\kappa_{10}^{\rm HH}(T_K)
+n_e\kappa_{10}^{\rm eH}(T_K)
+n_p\kappa_{10}^{\rm pH}(T_K)\right],
\quad T_\star=0.0628\,\mathrm K,
\quad A_{10}=2.85\times10^{-15}\,\mathrm{s}^{-1}.
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; {ref}`Hirata (2006) <ref-hirata2006>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-alpha-coupling
\widetilde x_\alpha=\frac{1.66\times10^{11}}{1+z}
\widetilde S_\alpha J_\alpha,
\qquad J_\alpha=J_{\alpha,\star}+J_{\alpha,X},
```

*References:* {ref}`Hirata (2006) <ref-hirata2006>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

$J_\alpha$ uses $\mathrm{photons\,cm^{-2}\,s^{-1}\,Hz^{-1}\,sr^{-1}}$. Stellar photons redshift into Lyman resonances; $f_{\rm rec}(n)$ is their Ly$\alpha$ cascade probability.

```{math}
:label: igm-stellar-alpha
J_{\alpha,\star}(\boldsymbol x,z)=
\frac{c(1+z)^2}{4\pi}
\sum_{n=2}^{23}f_{\rm rec}(n)
\int_z^{z_{\max}(n)}
\frac{\epsilon_{\nu'_n}^{\rm photon,com}(\boldsymbol x,z')}{H(z')}dz',
\qquad
\nu'_n=\nu_n\frac{1+z'}{1+z}.
```

*References:* {ref}`Pritchard & Furlanetto (2006) <ref-pritchard2006>`; {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

```{math}
:label: igm-lyman-horizon
1+z_{\max}(n)=(1+z)\frac{1-(n+1)^{-2}}{1-n^{-2}},
\qquad \frac{\nu_n}{\nu_\alpha}=\frac{1-n^{-2}}{3/4}.
```

*References:* {ref}`Pritchard & Furlanetto (2006) <ref-pritchard2006>`; {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

Spectral distortion and effective color temperature depend on $T_S$ and are solved iteratively. $\tau_{\rm GP}$ is the Gunn–Peterson optical depth.

```{math}
:label: igm-alpha-correction
\begin{aligned}
\xi&=(10^{-7}\tau_{\rm GP}/T_K^2)^{1/3},\\
\widetilde S_\alpha&=
\frac{1-0.0631789/T_K+0.115995/T_K^2
-0.401403/(T_ST_K)+0.336463/(T_ST_K^2)}
{1+2.98394\xi+1.53583\xi^2+3.85289\xi^3},\\
T_{c,\rm eff}^{-1}&=T_K^{-1}
+\frac{0.405535}{T_K}(T_S^{-1}-T_K^{-1}),\\
\tau_{\rm GP}&=\frac{1.342881\times10^{-7}}{H(z)}
\bar n_{H,0}(1+z)^3(1+\delta)(1-x_e).
\end{aligned}
```

*References:* {ref}`Hirata (2006) <ref-hirata2006>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c).

## Photoionized gas and molecular-cooling feedback

Ionized gas retains memory of its ionization redshift $z_r$, with $T_{\rm re}=10^4\,\mathrm K$. Partly ionized cells use a neutral/ionized temperature mixture.

```{math}
:label: igm-ionized-temperature
\begin{aligned}
\delta_r&=\delta\frac{1+z}{1+z_r},\\
T_{\rm HII}&=\left\{
T_{\rm re}^{1.7}
\left(\frac{1+\delta}{1+\delta_r}\right)^{1.1333}
\left(\frac{1+z}{1+z_r}\right)^{3.4}
\exp\!\left[\left(\frac{1+z}{7.1}\right)^{2.5}
-\left(\frac{1+z_r}{7.1}\right)^{2.5}\right]
+\left[10^4\frac{1+z}{4}\right]^{1.7}(1+\delta)
\right\}^{0.5882}.
\end{aligned}
```

*References:* {ref}`McQuinn & Upton Sanderbeck (2016) <ref-mcquinn2016>`; [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

```{math}
:label: igm-partial-temperature
T_{\rm mix}=x_{\rm HI}T_{\rm HI}+(1-x_{\rm HI})T_{\rm re}.
```

*References:* [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c).

Lyman–Werner photons span 11.2–13.6 eV. Their band integral weights stellar photon spectra by energy; $u_n=\nu_n/\nu_\alpha$.

```{math}
:label: igm-lw-band
\Phi_{{\rm LW},\star}(z,z')=
\sum_{n=2}^{23}\int_{u_{\rm lo}}^{u_{n+1}}
u\,\epsilon_{u,\star}^{\rm photon}\,du,
\qquad u=\nu/\nu_\alpha,\quad
u_{\rm lo}=\max[u_n(1+z')/(1+z),\nu_{\rm LW}/\nu_\alpha],
\qquad z'\leq z_{\max}(n),
```

*References:* {ref}`Ventura et al. (2025) <ref-ventura2025>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c); [Meraxes src/core/ComputeTs.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputeTs.c).

```{math}
:label: igm-lw-shell
F_{{\rm LW},\star}=\frac{c}{4\pi m_p}
(1+z)^2\sum_j\dot\rho_{\star,j}^{\rm com}
(1+z'_j)\left(\frac{dt}{dz'}\Delta z'\right)_j
\Phi_{{\rm LW},\star,j}.
```

*References:* {ref}`Ventura et al. (2025) <ref-ventura2025>`; [Meraxes src/core/XRayHeatingFunctions.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/XRayHeatingFunctions.c); [Meraxes src/core/ComputeTs.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputeTs.c).

Multiplying $F_{{\rm LW},\star}$ by $h_P\nu_\alpha/(\nu_{\rm ion}-\nu_{\rm LW})$ gives band-averaged intensity. $J_{\rm LW}$ uses $10^{-21}$ cgs intensity units. LW radiation and streaming velocity raise the molecular-cooling threshold.

The molecular-cooling threshold including LW radiation is given in Equation {eq}`opt-lw-mass-threshold`.

The streaming-velocity relation is given in Equation {eq}`opt-streaming-cooling`.

## 21-cm brightness and velocities

Positive brightness denotes emission against the CMB; negative brightness denotes absorption. The saturated-temperature approximation sets $1-T_{\rm CMB}/T_S=1$.

```{math}
:label: igm-brightness
\delta T_b(\boldsymbol x,z)=T_0(z)x_{\rm HI}(1+\delta)
\left(1-\frac{T_{\rm CMB}}{T_S}\right),
\qquad
T_0(z)=27\left(\frac{\Omega_bh^2}{0.023}\right)
\left[\frac{0.15}{\Omega_mh^2}\frac{1+z}{10}\right]^{1/2}
\mathrm{mK}.
```

*References:* {ref}`Mesinger et al. (2011), equation 1 <ref-mesinger2011>`; {ref}`Balu et al. (2023) <ref-balu2023>`.

The optically thin velocity correction uses a bounded gradient; finite optical depth uses the unbounded gradient. Temperatures are K except $T_0$ and brightness, which are mK.

```{math}
:label: igm-velocity-thin
\delta T_b^{v}=\frac{\delta T_b}{1+g_v/H(z)},\qquad
g_v=\operatorname{clip}_{[-0.2H,0.2H]}
\left(\frac{\partial v_{\parallel,\rm com}}{\partial r_{\parallel,\rm com}}\right).
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/BrightnessTemperature.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/BrightnessTemperature.c).

```{math}
:label: igm-velocity-thick
\tau_{21}=\frac{T_0x_{\rm HI}(1+\delta)(1+z)}
{1000T_S\,|1+g_v/H|},\qquad
\delta T_b^{v}=\frac{1000(T_S-T_{\rm CMB})}{1+z}
\left(1-e^{-\tau_{21}}\right)\ \mathrm{mK}.
```

*References:* {ref}`Greig & Mesinger (2018), section 2 <ref-greig2018>`; [Meraxes src/core/BrightnessTemperature.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/BrightnessTemperature.c).

Peculiar velocities displace comoving line-of-sight positions into redshift space.

```{math}
:label: igm-rsd-map
s_\parallel=r_\parallel+
\frac{v_{\parallel,\rm pec}}{aH(z)}.
```

*References:* {ref}`Greig & Mesinger (2018), section 2 <ref-greig2018>`; [Meraxes src/core/BrightnessTemperature.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/BrightnessTemperature.c).

## Global quantities, power spectra and lightcones

Volume and mass averages define the global brightness and ionized fraction. The latter supplies the simulated Thomson optical depth, integrated between snapshots.

```{math}
:label: igm-grid-averages
\langle q\rangle_V=\frac1{D^3}\sum_{\rm cells}q_i,
\qquad
\langle q\rangle_M=
\frac{\sum_iq_i(1+\delta_i)}{\sum_i(1+\delta_i)}.
```

*References:* [Meraxes src/core/find_HII_bubbles.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/find_HII_bubbles.c); [Meraxes src/core/BrightnessTemperature.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/BrightnessTemperature.c).

```{math}
:label: igm-thomson-depth
\tau_{e,\rm sim}=c\sigma_T\bar n_{b,0}
\int_{z_{\rm low}}^{z_{\rm high}}
\frac{(1+z)^2}{H(z)}\langle x_{\rm HII}\rangle_M dz,
\qquad \langle x_{\rm HII}\rangle_M=1-\langle x_{\rm HI}\rangle_M.
```

*References:* {ref}`Mutch et al. (2016) <ref-mutch2016>`; [Meraxes src/core/reionization.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

```{math}
:label: igm-thomson-step
\tau_{e,\rm sim}^{i+1}=\tau_{e,\rm sim}^{i}
+\frac{F_i+F_{i-1}}2\,|z_i-z_{i-1}|,
\qquad
F_i=\frac{c\sigma_T\bar n_{b,0}(1+z_i)^2}{H(z_i)}
\langle x_{\rm HII}\rangle_{M,i}.
```

*References:* [Meraxes src/core/reionization.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

Below the final redshift $z_{\rm end}$, hydrogen is fully ionized; helium becomes doubly ionized at $z=4$.

```{math}
:label: igm-thomson-post
\tau_{e,\rm post}=c\sigma_T\int_0^{z_{\rm end}}
\frac{(1+z)^2}{H(z)}
\begin{cases}
\bar n_{H,0}+2\bar n_{{\rm He},0},&z\leq4,\\
\bar n_{H,0}+\bar n_{{\rm He},0},&z>4
\end{cases}dz,
\qquad \tau_{e,\rm total}=\tau_{e,\rm sim}+\tau_{e,\rm post}.
```

*References:* [Meraxes src/core/reionization.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/reionization.c).

For $V=(L/h)^3$ in $\mathrm{cMpc^3}$ and $T'=\delta T_b-\langle\delta T_b\rangle$, $P_{21}$ has units $\mathrm{mK^2\,cMpc^3}$ and $\Delta_{21}^2$ has units $\mathrm{mK^2}$.

```{math}
:label: igm-ps-fourier
\widetilde T(\boldsymbol k)\simeq
\frac{V}{D^3}\sum_jT'(\boldsymbol x_j)
e^{-i\boldsymbol k\cdot\boldsymbol x_j},
\qquad
P_{21}(\boldsymbol k)=\frac{|\widetilde T(\boldsymbol k)|^2}{V},
\qquad
\Delta_{21}^2(k)=\frac{k^3P_{21}(k)}{2\pi^2}.
```

*References:* {ref}`Mesinger et al. (2011) <ref-mesinger2011>`; [Meraxes src/core/ComputePowerSpectrum.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputePowerSpectrum.c).

```{math}
:label: igm-ps-bins
k_b=\frac1{N_b}\sum_{\boldsymbol k\in b}|\boldsymbol k|,
\qquad
\Delta^2_{21,b}=\frac1{N_b}
\sum_{\boldsymbol k\in b}
\frac{k^3|\widetilde T(\boldsymbol k)|^2}{2\pi^2V},
\qquad
\sigma_{\Delta^2,b}=\frac{\Delta^2_{21,b}}{\sqrt{N_b}}.
```

*References:* [Meraxes src/core/ComputePowerSpectrum.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ComputePowerSpectrum.c).

$N_b$ counts sampled Fourier half-grid modes. $\sigma_{\Delta^2,b}$ estimates mode-counting uncertainty. Lightcones interpolate consecutive coeval temperatures in time, with slice width $\Delta\chi=L/(hD)$.

```{math}
:label: igm-lightcone-interpolation
T_{\rm LC}(z_s)=T_1+
\frac{t(z_s)-t_1}{t_2-t_1}(T_2-T_1),
\qquad
\chi(z_{s+1})-\chi(z_s)\simeq\Delta\chi.
```

*References:* {ref}`Greig & Mesinger (2018), section 2 <ref-greig2018>`; [Meraxes src/core/ConstructLightcone.c](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/blob/faa8aafd49e03a870f3e1ba83c22bdacea58257f/src/core/ConstructLightcone.c).


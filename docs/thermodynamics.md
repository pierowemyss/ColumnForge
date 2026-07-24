# Thermodynamic methods

Every phase-equilibrium and energy calculation in ColumnForge is implemented in
[`core/thermodynamics.py`](../src/python/core/thermodynamics.py) (VLE, activity
models, EOS) and [`core/enthalpy.py`](../src/python/core/enthalpy.py). This
document is the equation reference; each equation below is the exact form the
code evaluates, not a textbook idealization. Every function in those modules
also carries a runnable `_demo()` self-check that pins these formulas to known
data (`python -m core.thermodynamics`).

- [The one seam: K-values](#the-one-seam-k-values)
- [vapor pressure](#vapour-pressure)
- [Bubble and dew point](#bubble-and-dew-point)
- [Latent heat](#latent-heat)
- [Activity-coefficient models](#activity-coefficient-models)
  - [NRTL](#nrtl) · [Wilson](#wilson) · [UNIQUAC](#uniquac) · [Margules](#margules-two-suffix) · [UNIFAC](#unifac)
- [vapor-phase equation of state (SRK)](#vapour-phase-equation-of-state-srk)
- [Enthalpy](#enthalpy)
- [Unit conventions](#unit-conventions)
- [Deliberate approximations](#deliberate-approximations)

Symbols: $x_i$ liquid mole fraction, $y_i$ vapor mole fraction, $T$ temperature,
$P$ pressure, $R$ the gas constant, $T_K$ temperature in kelvin, $n$ the number
of components.

---

## The one seam: K-values

Every solver reaches thermodynamics through a single function, `k_values`. The
equilibrium ratio $K_i = y_i/x_i$ is built up in three layers, so swapping a
model never touches a solver:

$$
K_i = \frac{\gamma_i(x,T) P^{sat}_i(T) \phi_i^{sat}}{\phi_i^{V}(y,T,P) P}
$$

| layer                 | what it adds                      | reduces to                   |
| --------------------- | --------------------------------- | ---------------------------- |
| **Ideal (Raoult)**    | nothing — $\gamma_i = \phi_i = 1$ | $K_i = P^{sat}_i/P$          |
| **Activity model**    | non-ideal liquid $\gamma_i(x,T)$  | $K_i = \gamma_i P^{sat}_i/P$ |
| **+ EOS (gamma–phi)** | vapor-phase non-ideality          | full expression above        |

Here $\phi_i^{sat}$ is the fugacity coefficient of **pure** $i$ as a vapor at
$(T, P^{sat}_i)$, and $\phi_i^{V}$ is evaluated for the vapor mixture in
equilibrium with the liquid $x$. The equilibrium vapor is estimated in one shot
as $y \approx K^{Raoult} x$ (normalized); see
[Deliberate approximations](#deliberate-approximations).

---

## Vapor pressure

Dispatched automatically on the width of the coefficient matrix: a 3-column
matrix is regular Antoine, a 7-column matrix is Aspen extended Antoine (PLXANT).

**Antoine** (`antoine_psat`):

$$
\log_{10} P^{sat} = A - \frac{B}{T + C}
$$

$P^{sat}$ comes out in whatever unit the coefficients were fitted to, and the
pressure $P$ passed to the solver must be in that same unit (the bundled
benzene/toluene/xylene fits are mmHg with $T$ in °C).

**Aspen extended Antoine / PLXANT** (`plxant_psat`), with $T$ in kelvin:

$$
\ln P^{sat} = C_1 + \frac{C_2}{C_3 + T} + C_4 T + C_5 \ln T + C_6 T^{C_7}
$$

---

## Bubble and dew point

Both are one-dimensional root finds on temperature (Brent's method, with the
upper bracket auto-widening for very high boilers).

$$
\text{bubble } T: \quad \sum_i K_i(T) x_i = 1
\qquad\qquad
\text{dew } T: \quad \sum_i \frac{y_i}{K_i(T)} = 1
$$

---

## Latent heat

Molar heat of vaporization from the Clausius–Clapeyron slope of the
vapor-pressure fit (`latent_heat`) — no separate latent-heat correlation is
needed, it falls straight out of $P^{sat}(T)$:

$$
\lambda_i = R T_K^{2} \frac{d \ln P^{sat}_i}{dT}
$$

with the slope taken analytically from whichever fit is in use:

$$
\frac{d \ln P^{sat}}{dT} =
\begin{cases}
\dfrac{B \ln 10}{(T + C)^{2}} & \text{(Antoine)} \\
-\dfrac{C_2}{(C_3 + T)^{2}} + C_4 + \dfrac{C_5}{T} + C_6 C_7 T^{C_7 - 1} & \text{(PLXANT)}
\end{cases}
$$

(A degree Celsius and a kelvin are the same size, so the Antoine slope is
numerically valid even though its $T$ is in the fit's unit while the $T_K^2$
prefactor is in kelvin.)

---

## Activity-coefficient models

Each model is a pure function $\gamma(x, T)$ built by a closure (`*_gamma_fn`)
that plugs into the K-value seam. Each raises a user-facing error when its
parameters are missing rather than silently falling back to ideal.
Temperature-dependent parameters use $T_K$ (kelvin).

### NRTL

Interaction energies and the non-randomness weighting:

$$
\tau_{ij} = a_{ij} + \frac{b_{ij}}{T_K},
\qquad
G_{ij} = \exp(-\alpha_{ij} \tau_{ij}),
\qquad
\tau_{ii} = 0,
\qquad
\alpha_{ij} = \alpha_{ji}
$$

$$
\ln \gamma_i =
\frac{\sum_j \tau_{ji} G_{ji} x_j}{\sum_k G_{ki} x_k} +
\sum_j \frac{x_j G_{ij}}{\sum_k G_{kj} x_k}
  \left( \tau_{ij} - \frac{\sum_m x_m \tau_{mj} G_{mj}}{\sum_k G_{kj} x_k} \right)
$$

The implementation evaluates this in vectorized form: with
$S_j = \sum_k x_k G_{kj}$, $r = x/S$ and $C = (x^{T}(\tau \odot G))/S$, it
computes

$$
\ln \gamma = C + (\tau \odot G) r - G (r \odot C)
$$

This is the HYSYS-modified NRTL ($\tau = a + b/T$), matching
`src/native/nifco.f90`'s NRTL subroutine.

### Wilson

$$
\ln \gamma_i = 1 - \ln \left( \sum_j x_j \Lambda_{ij} \right) -
\sum_k \frac{x_k \Lambda_{ki}}{\sum_j x_j \Lambda_{kj}},
\qquad
\ln \Lambda_{ij} = a_{ij} + \frac{b_{ij}}{T_K}
$$

### UNIQUAC

Split into a combinatorial (size/shape) and a residual (energetic) part,
$\ln \gamma_i = \ln \gamma_i^{C} + \ln \gamma_i^{R}$, with structural parameters
$r_i$ (volume), $q_i$ (area), coordination number $z = 10$, and

$$
\Phi_i = \frac{r_i x_i}{\sum_j r_j x_j},
\qquad
\theta_i = \frac{q_i x_i}{\sum_j q_j x_j},
\qquad
l_i = \frac{z}{2}(r_i - q_i) - (r_i - 1)
$$

$$
\ln \gamma_i^{C} = \ln \frac{\Phi_i}{x_i} +
   \frac{z}{2} q_i \ln \frac{\theta_i}{\Phi_i} +
   l_i - \frac{\Phi_i}{x_i} \sum_j x_j l_j
$$

$$
\ln \gamma_i^{R} = q_i \left( 1 - \ln \left( \sum_j \theta_j \tau_{ji} \right) -
   \sum_j \frac{\theta_j \tau_{ij}}{\sum_k \theta_k \tau_{kj}} \right),
\qquad
\tau_{ij} = \exp \left( a_{ij} + \frac{b_{ij}}{T_K} \right)
$$

### Margules (two-suffix)

The one-constant regular-solution teaching model, temperature-independent
($A$ symmetric, $A_{ii} = 0$):

$$
\frac{G^{E}}{RT} = \frac{1}{2} \sum_{i,j} A_{ij} x_i x_j,
\qquad
\ln \gamma_i = \sum_j A_{ij} x_j - \frac{G^{E}}{RT}
$$

Binary limit: $\ln \gamma_1 = A x_2^{2}$.

### UNIFAC

Group contribution — needs no binary parameters; $\gamma$ is assembled from a
group-interaction database (`core/data/unifac_groups.json`). With subgroup
counts $\nu_k^{i}$, subgroup volume/area $R_k, Q_k$, and species parameters
$r_i = \sum_k \nu_k^{i} R_k$, $q_i = \sum_k \nu_k^{i} Q_k$, the combinatorial
term is the Staverman–Guggenheim form (identical in shape to UNIQUAC above,
$z = 10$). The residual term is a difference of group activities between the
mixture and the pure species:

$$
\ln \gamma_i^{R} = \sum_k \nu_k^{i} \left( \ln \Gamma_k - \ln \Gamma_k^{(i)} \right)
$$

$$
\ln \Gamma_k = Q_k \left[ 1 - \ln \left( \sum_m \theta_m \Psi_{mk} \right) -
   \sum_m \frac{\theta_m \Psi_{km}}{\sum_n \theta_n \Psi_{nm}} \right],
\qquad
\Psi_{mn} = \exp \left( -\frac{a_{mn}}{T_K} \right)
$$

where $\theta_m$ is the area fraction of subgroup $m$ and $a_{mn}$ the
main-group interaction energy [K]. $\ln \Gamma_k^{(i)}$ is the same expression
evaluated for pure species $i$.

---

## Vapor-phase equation of state (SRK)

Soave–Redlich–Kwong, used for the $\phi$ layer of gamma–phi VLE (`srk_phi`).
Per component, with critical constants $T_{c,i}, P_{c,i}$ and acentric factor
$\omega_i$:

$$
m_i = 0.480 + 1.574 \omega_i - 0.176 \omega_i^{2},
\qquad
\alpha_i = \left[ 1 + m_i \left( 1 - \sqrt{T/T_{c,i}} \right) \right]^{2}
$$

$$
a_i = 0.42748 \frac{R^{2} T_{c,i}^{2}}{P_{c,i}} \alpha_i,
\qquad
b_i = 0.08664 \frac{R T_{c,i}}{P_{c,i}}
$$

Mixing rules with $k_{ij} = 0$:

$$
a = \left( \sum_i y_i \sqrt{a_i} \right)^{2},
\qquad
b = \sum_i y_i b_i,
\qquad
A = \frac{aP}{(RT)^{2}},
\qquad
B = \frac{bP}{RT}
$$

The compressibility is the largest real root of the cubic (the vapor root):

$$
Z^{3} - Z^{2} + (A - B - B^{2}) Z - AB = 0
$$

and the fugacity coefficient of each component is

$$
\ln \phi_i = \frac{b_i}{b}(Z - 1) - \ln(Z - B) -
   \frac{A}{B} \left( \frac{2 \sqrt{a_i}}{\sum_j y_j \sqrt{a_j}} - \frac{b_i}{b} \right)
     \ln \left( 1 + \frac{B}{Z} \right)
$$

If the cubic has no vapor-like root (e.g. a bubble-point search probing far
below the true bubble point), the code returns $\phi_i = 1$ — an EOS correction
is meaningless where no vapor phase exists.

---

## Enthalpy

The component database stores a **constant** liquid heat capacity
$c_p^{liq}$ and the latent heat at the normal boiling point
$\Delta H_{vap,T_b}$, plus $T_b, T_c$. Enthalpies are referenced to a liquid
datum $T_{ref}$, all in kelvin (`core/enthalpy.py`):

$$
h_L(T) = c_p^{liq} (T - T_{ref}),
\qquad
h_V(T) = h_L(T) + \Delta H_{vap}(T)
$$

The latent heat is carried from $T_b$ to $T$ by the Watson correlation
(exponent $0.38$), vanishing at and above $T_c$:

$$
\Delta H_{vap}(T) = \Delta H_{vap,T_b} \left( \frac{T_c - T}{T_c - T_b} \right)^{0.38}
$$

This is the seam behind the Inside-Out energy balance, enthalpy-based feed
quality $q$, and condenser subcooling.

---

## Unit conventions

Getting units wrong is the single most common way to get nonsense out of the
VLE layer, so they are explicit at every closure:

| quantity                              | unit expected by the code                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| Antoine / PLXANT $P^{sat}$            | whatever the coefficients were fit to; **$P$ must match** (bundled fits: mmHg)             |
| Antoine $T$                           | the fit's unit (bundled fits: °C); PLXANT converts internally to K via `t_to_K`            |
| activity $\tau, \Lambda$ correlations | $T_K$ (kelvin), via `t_to_K` (default °C→K)                                                |
| SRK $T_c$                             | kelvin; $P_c$ in bar (converted to Pa); $T, P$ via `t_to_K` / `p_to_Pa` (default °C, mmHg) |
| enthalpy                              | everything in kelvin                                                                       |
| duties                                | kmol/h × J/mol ≡ kJ/h; display in kW via `KJH_TO_KW`                                       |

---

<!-- ## Deliberate approximations -->
<!---->
<!-- These are intentional shortcuts (marked `ponytail:` in the source), each with a -->
<!-- known scope and an upgrade path — listed so nothing is a silent surprise: -->
<!---->
<!-- - **Dew point** evaluates $\gamma$ at the vapor composition $y$ as a proxy for -->
<!--   the (unknown) equilibrium liquid. Exact for ideal VLE; approximate otherwise. -->
<!--   A rigorous dew point needs an inner liquid-composition solve. -->
<!-- - **gamma–phi vapor** uses a one-shot estimate $y \approx K^{Raoult} x$ -->
<!--   (normalized) for $\phi_i^{V}$, with no inner $y$-iteration — the outer solver -->
<!--   loop refines $x, T$ anyway. The **Poynting factor is neglected**; both are fine -->
<!--   below ~10 bar. -->
<!-- - **SRK mixing uses $k_{ij} = 0$** — adequate for hydrocarbon/hydrocarbon pairs; -->
<!--   add a $k_{ij}$ matrix for polar / light-gas systems. -->
<!-- - **Enthalpy** is constant liquid $c_p$ + Watson latent — the shortcut model. -->
<!--   Ideal-gas $c_p$ polynomials + a residual-enthalpy departure are the upgrade -->
<!--   when the database grows those fields. -->
<!---->
<!-- See also the [component database](../src/python/core/data/components.json) and -->
<!-- its consistency gates (Antoine reproduces $T_b$ within 1 K; $\Delta H_{vap}$ -->
<!-- matches Clausius–Clapeyron within 12%). -->

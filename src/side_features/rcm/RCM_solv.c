//
// Python-callable C script to draw a residue curve: dx/dxi = x - y(x),
// integrated by explicit Euler both ways from a seed composition, with a full
// bubble-point VLE solve (GSL hybrids) at every step.
//
// Author: Piero Wemyss
// Date: January 16, 2025
//
// THERMO: this file computes no thermodynamics of its own. Every property
// comes from src/native/nifco2.f90, and the model parameters arrive as opaque
// column-major (Fortran-ordered) double* blobs that C forwards without ever
// indexing them -- which is what lets all five activity models, all three
// vapor-pressure correlations and SRK fit through one struct, and what keeps
// this engine bit-comparable with core/thermodynamics.py. The Python side is
// src/python/core/rcm.py.
//
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

// nifco2.f90 has no bind(C); these are gfortran's default external names
// (lowercase + one trailing underscore). All temperatures are Celsius.
// NOTE the argument order differs between families: n comes FIRST in the
// vapor-pressure routines and LATE in the activity/EOS ones.
extern void antoine_psat_(const int *n, const double *c, const double *Tcel,
                          double *Psat);
extern void wagner_psat_(const int *n, const double *c, const double *Tcel,
                         double *Psat);
extern void plxant_psat_(const int *n, const double *c, const double *Tcel,
                         double *Psat);
extern void antoine_tsat_(const int *ncoef, const double *c, const double *P,
                          double *Tcel);

extern void nrtl_gamma_(const double *x, const double *Tcel, const double *a,
                        const double *b, const double *c, const int *n,
                        double *gamma);
extern void wilson_gamma_(const double *x, const double *Tcel, const double *a,
                          const double *b, const int *n, double *gamma);
extern void uniquac_gamma_(const double *x, const double *Tcel, const double *r,
                           const double *q, const double *a, const double *b,
                           const int *n, double *gamma);
extern void margules_gamma_(const double *x, const double *Tcel,
                            const double *a, const int *n, double *gamma);
extern void unifac_gamma_(const double *x, const double *Tcel, const double *nu,
                          const double *Rk, const double *Qk, const double *a,
                          const int *n, const int *m, double *gamma);

// ponytail: SRK_phi passes A/B to its minpack callback through a Fortran
// common block, so it is not reentrant. One RCM solve runs at a time on one
// worker thread; give the Fortran a thread-local or an explicit argument
// before ever calling this from two threads.
extern void srk_phi_(const double *y, const double *Tcel, const double *P_Pa,
                     const double *TcCel, const double *Pc_Pa,
                     const double *omega, const int *n, double *phi);

// Activity model ids -- must match _GAMMA_MODELS in core/rcm.py.
enum { G_IDEAL = 0, G_NRTL, G_WILSON, G_UNIQUAC, G_MARGULES, G_UNIFAC };
// EOS ids -- must match _EOS_MODELS in core/rcm.py.
enum { E_IDEAL = 0, E_SRK };

// Passing parameter struct. Mirrored field-for-field by `Params` in
// core/rcm.py; change one and you must change the other.
typedef struct {
  double *x0;
  double P; // in whatever unit this Psat model emits

  const double *psat; // (Ncomps, npsat) column-major
  int npsat;          // 3 = Antoine, 6 = Wagner, 7 = PLXANT

  const double *gp[4]; // activity parameters, forwarded untouched
  int gammaModel;
  int ngroups; // UNIFAC subgroup count (m); unused otherwise

  const double *TcCel; // Celsius
  const double *Pc_Pa; // Pascals
  const double *omega;
  int eosModel;
  double pToPa; // P * pToPa -> Pa, for the SRK call only

  int Ncomps;
  double dxi;
  int n_it;
  int maxiter;
  double ftol;
  double xtol;
} params_t;

// Return structure for RCM curves. `nfail` counts VLE solves that hit their
// iteration budget -- a curve with nfail > 0 is not to be trusted.
typedef struct {
  double *x;
  double *y;
  double *T;
  int nfail;
} curves_t;

// ---------------------------------------------------------------------------
// Property dispatchers: pick the Fortran routine, forward the blob, return.
// ---------------------------------------------------------------------------

static void calc_psat(const params_t *p, const double *T, double *Psat) {
  switch (p->npsat) {
  case 3:
    antoine_psat_(&p->Ncomps, p->psat, T, Psat);
    break;
  case 6:
    wagner_psat_(&p->Ncomps, p->psat, T, Psat);
    break;
  case 7:
    plxant_psat_(&p->Ncomps, p->psat, T, Psat);
    break;
  default:
    for (int i = 0; i < p->Ncomps; i++)
      Psat[i] = 0.0;
  }
}

static void calc_gamma(const params_t *p, const double *x, const double *T,
                       double *gamma) {
  switch (p->gammaModel) {
  case G_NRTL:
    nrtl_gamma_(x, T, p->gp[0], p->gp[1], p->gp[2], &p->Ncomps, gamma);
    break;
  case G_WILSON:
    wilson_gamma_(x, T, p->gp[0], p->gp[1], &p->Ncomps, gamma);
    break;
  case G_UNIQUAC:
    uniquac_gamma_(x, T, p->gp[0], p->gp[1], p->gp[2], p->gp[3], &p->Ncomps,
                   gamma);
    break;
  case G_MARGULES:
    margules_gamma_(x, T, p->gp[0], &p->Ncomps, gamma);
    break;
  case G_UNIFAC:
    unifac_gamma_(x, T, p->gp[0], p->gp[1], p->gp[2], p->gp[3], &p->Ncomps,
                  &p->ngroups, gamma);
    break;
  default: // G_IDEAL
    for (int i = 0; i < p->Ncomps; i++)
      gamma[i] = 1.0;
  }
}

static void calc_phi(const params_t *p, const double *y, const double *T,
                     double *phi) {
  if (p->eosModel == E_SRK) {
    // The VLE residual works in the Psat model's pressure unit; SRK_phi's
    // algebra (R in J/mol/K) needs Pa, so only this call converts.
    const double P_Pa = p->P * p->pToPa;
    srk_phi_(y, T, &P_Pa, p->TcCel, p->Pc_Pa, p->omega, &p->Ncomps, phi);
  } else {
    for (int i = 0; i < p->Ncomps; i++)
      phi[i] = 1.0;
  }
}

// Pure-component saturation fugacity coefficients: phi_i of pure i at
// (T, Psat_i). This is the phi^sat in the gamma-phi K-value of
// core/thermodynamics.py k_values(); without it the compiled RCM would quietly
// disagree with every other solver in the app whenever SRK is selected.
static void calc_phi_sat(const params_t *p, const double *Psat, const double *T,
                         double *phi_sat) {
  const int Ncomps = p->Ncomps;

  if (p->eosModel != E_SRK) {
    for (int i = 0; i < Ncomps; i++)
      phi_sat[i] = 1.0;
    return;
  }
  // ponytail: one one-hot SRK call per component per residual. The vectorised
  // shortcut is _SRKPhi.pure() in Python; port it into nifco2.f90 if SRK+RCM
  // ever becomes a hot path.
  double pure[Ncomps], out[Ncomps];
  for (int i = 0; i < Ncomps; i++) {
    for (int j = 0; j < Ncomps; j++)
      pure[j] = (i == j) ? 1.0 : 0.0;
    const double P_Pa = Psat[i] * p->pToPa;
    srk_phi_(pure, T, &P_Pa, p->TcCel, p->Pc_Pa, p->omega, &p->Ncomps, out);
    phi_sat[i] = out[i];
  }
}

// Composition-weighted mean of the pure-component boiling points: the seed for
// the first VLE solve. Every later step warm-starts from the previous answer.
static double seed_T(const params_t *p) {
  const int Ncomps = p->Ncomps, ncoef = p->npsat;
  double row[16], Tsat, T = 0.0, wsum = 0.0;

  if (ncoef > (int)(sizeof(row) / sizeof(row[0])))
    return 50.0;

  for (int i = 0; i < Ncomps; i++) {
    // Gather component i's coefficients out of the column-major matrix. The
    // only place this file touches a coefficient array, and it is a gather,
    // not an interpretation.
    for (int k = 0; k < ncoef; k++)
      row[k] = p->psat[i + Ncomps * k];
    antoine_tsat_(&ncoef, row, &p->P, &Tsat);
    if (isfinite(Tsat)) {
      const double w = fabs(p->x0[i]);
      T += w * Tsat;
      wsum += w;
    }
  }
  return wsum > 0.0 ? T / wsum : 50.0;
}

// ---------------------------------------------------------------------------
// VLE: x_i gamma_i Psat_i = y_i phi_i P, closed by sum(y) = 1
// ---------------------------------------------------------------------------

static int VLEfunc(const gsl_vector *Y, void *params, gsl_vector *f) {
  params_t *p = (params_t *)params;

  const double *x = p->x0;
  const double P = p->P;
  const int Ncomps = p->Ncomps;

  double y[Ncomps];
  for (int i = 0; i < Ncomps; i++)
    y[i] = gsl_vector_get(Y, i);
  const double T = gsl_vector_get(Y, Ncomps);

  double Psat[Ncomps], gamma[Ncomps], phi[Ncomps], phi_sat[Ncomps];
  double sumy = 0.0;

  calc_psat(p, &T, Psat);
  calc_gamma(p, x, &T, gamma);
  calc_phi(p, y, &T, phi);
  calc_phi_sat(p, Psat, &T, phi_sat);

  // gamma-phi, same form as core/thermodynamics.py k_values():
  //   x_i gamma_i Psat_i phi^sat_i = y_i phi^V_i P
  for (int i = 0; i < Ncomps; i++) {
    gsl_vector_set(f, i,
                   x[i] * gamma[i] * Psat[i] * phi_sat[i] - y[i] * phi[i] * P);
    sumy += fabs(y[i]);
  }
  gsl_vector_set(f, Ncomps, sumy - 1.0);

  return GSL_SUCCESS;
}

static int VLEsolve(gsl_vector *Y_init, void *params, gsl_vector *Y_sol) {
  params_t *p = (params_t *)params;
  const int Ncomps = p->Ncomps;

  gsl_multiroot_fsolver *s =
      gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrids, Ncomps + 1);
  gsl_multiroot_function f = {&VLEfunc, Ncomps + 1, params};
  gsl_multiroot_fsolver_set(s, &f, Y_init);

  int status;
  size_t iter = 0;
  do {
    iter++;
    status = gsl_multiroot_fsolver_iterate(s);
    if (status)
      break;
    status = gsl_multiroot_test_residual(s->f, p->ftol);
  } while (status == GSL_CONTINUE && iter < (size_t)p->maxiter);

  // Take the last iterate either way: a step that ran out of budget is still a
  // better seed for the next one than the previous point, and the caller is
  // told how many did so through curves_t.nfail.
  for (int i = 0; i < Ncomps + 1; i++)
    gsl_vector_set(Y_sol, i, gsl_vector_get(s->x, i));

  gsl_multiroot_fsolver_free(s);
  return status;
}

// ---------------------------------------------------------------------------
// Parent solver
// ---------------------------------------------------------------------------

// Marches `steps` Euler steps from the current p->x0, writing into x/y/T at
// `stride` rows apart (stride = -1 walks backwards through the buffer). dir is
// +1 toward the heavy node, -1 toward the light one.
static int march(params_t *p, gsl_vector *Y_init, gsl_vector *Y_sol, double *x,
                 double *y, double *T, int steps, int stride, double dir) {
  const int Ncomps = p->Ncomps;
  const double dxi = p->dxi;
  int nfail = 0;

  for (int i = 0; i < steps; i++) {
    double *xi = x + (ptrdiff_t)i * stride * Ncomps;
    p->x0 = xi;

    if (VLEsolve(Y_init, p, Y_sol) != GSL_SUCCESS)
      nfail++;

    double *yi = y + (ptrdiff_t)i * stride * Ncomps;
    for (int j = 0; j < Ncomps; j++)
      yi[j] = gsl_vector_get(Y_sol, j);
    T[(ptrdiff_t)i * stride] = gsl_vector_get(Y_sol, Ncomps);
    gsl_vector_memcpy(Y_init, Y_sol);

    if (i + 1 < steps) {
      double *xn = xi + (ptrdiff_t)stride * Ncomps;
      for (int j = 0; j < Ncomps; j++)
        xn[j] = xi[j] + dir * dxi * (xi[j] - yi[j]);
    }
  }
  return nfail;
}

curves_t *RCM(void *params) {
  params_t *p = (params_t *)params;

  const int Ncomps = p->Ncomps;
  const int n_it = p->n_it;
  const int npts = 2 * n_it;
  double *x0 = p->x0;

  curves_t *c = (curves_t *)malloc(sizeof(curves_t));
  if (c == NULL)
    return NULL;
  c->x = (double *)malloc((size_t)npts * Ncomps * sizeof(double));
  c->y = (double *)malloc((size_t)npts * Ncomps * sizeof(double));
  c->T = (double *)malloc((size_t)npts * sizeof(double));
  c->nfail = 0;
  if (c->x == NULL || c->y == NULL || c->T == NULL) {
    free(c->x);
    free(c->y);
    free(c->T);
    free(c);
    return NULL;
  }

  gsl_vector *Y_init = gsl_vector_alloc(Ncomps + 1);
  gsl_vector *Y_sol = gsl_vector_alloc(Ncomps + 1);

  const double T0 = seed_T(p);

  // The seed sits at index n_it-1, so the buffer reads light -> heavy: the
  // backward branch fills [0, n_it) walking down from there, the forward
  // branch fills [n_it-1, 2*n_it). That is the same ordering as
  // gui/plotting.py's residue_curve, so the two engines plot identically.
  double *mid = c->x + (ptrdiff_t)(n_it - 1) * Ncomps;
  for (int i = 0; i < Ncomps; i++) {
    mid[i] = x0[i];
    gsl_vector_set(Y_init, i, x0[i]);
  }
  gsl_vector_set(Y_init, Ncomps, T0);

  c->nfail += march(p, Y_init, Y_sol, mid, c->y + (ptrdiff_t)(n_it - 1) * Ncomps,
                    c->T + (n_it - 1), n_it, -1, -1.0);

  // Restart from the seed for the forward branch.
  for (int i = 0; i < Ncomps; i++)
    gsl_vector_set(Y_init, i, x0[i]);
  gsl_vector_set(Y_init, Ncomps, T0);

  c->nfail += march(p, Y_init, Y_sol, mid, c->y + (ptrdiff_t)(n_it - 1) * Ncomps,
                    c->T + (n_it - 1), n_it + 1, +1, +1.0);

  p->x0 = x0;
  gsl_vector_free(Y_init);
  gsl_vector_free(Y_sol);
  return c;
}

void freeCurveMem(curves_t *c) {
  if (c == NULL)
    return;
  free(c->x);
  free(c->y);
  free(c->T);
  free(c);
}

/* Column Solvers — C kernel behind core/column_solvers.py, called via ctypes.
 *
 * Boundary contract (the thing that breaks if you get it wrong):
 *   - Every float is `double`, every count/flag is `int`. No float, no size_t,
 *     no _Bool — they only change struct padding.
 *   - Every array is a flat, C-contiguous double*. An (N, C) array indexes as
 *     arr[i * C + j]. Bands, compositions and K-values all use that layout.
 *   - **The caller allocates everything, including scratch. Nothing here
 *     mallocs.** Sizes are all derivable from ColParams before the call, so
 *     there is no result to free and no C-owned memory for numpy to alias.
 *   - Structs lay out doubles, then pointers, then ints, so there is no padding
 *     to reason about and no _pack_ on the Python side. Mirror the field order
 *     exactly, and assert cf_sizeof_params()/cf_sizeof_result() at import.
 *
 * Self-check: cc -DCF_MAIN -O2 -o /tmp/cf column_solvers.c && /tmp/cf
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- model ids -------------------------------------------------------------
 * Selected by int, never by string: strcmp chains re-parse on every call and
 * buy nothing. Python maps name -> id once at setup.
 */
enum cf_condenser { CF_COND_TOTAL = 0, CF_COND_PARTIAL = 1, CF_COND_NONE = 2 };
enum cf_gamma { CF_GAMMA_IDEAL = 0, CF_GAMMA_NRTL = 1, CF_GAMMA_UNIFAC = 2 };
enum cf_phi { CF_PHI_IDEAL = 0, CF_PHI_SRK = 1 };

/* Status codes. 0 is the only "trust this answer" value — CF_MAXITER means the
 * run burned its budget, which is exactly the found-vs-converged trap the
 * Python side already has scar tissue for. */
enum cf_status { CF_OK = 0, CF_MAXITER = 1, CF_BAD_INPUT = -1, CF_UNSUPPORTED = -2 };

/* --- input ---------------------------------------------------------------- */

typedef struct {
    /* doubles */
    double R;           /* reflux ratio (resolved operating point) */
    double D;           /* distillate molar rate */
    double subcooling;  /* reflux dT below bubble point; total condenser only */
    double tol;         /* convergence tolerance */

    /* pointers — a NULL optional model block *is* the "model off" flag */
    const double *feed;         /* (N*C) component molar feed per stage */
    const double *liquid_draw;  /* (N)   side-liquid draw */
    const double *vapor_draw;   /* (N)   side-vapor draw */
    const double *duty;         /* (N)   inter-stage heat; CMO ignores it */
    const double *pressure;     /* (N)   stage pressure */
    const double *q;            /* (N)   feed thermal quality; NULL => all 1.0 */
    const double *antoine;      /* (C*n_antoine) psat coefficients, aligned to comps */
    const double *nrtl_a;       /* (C*C) NULL => ideal liquid */
    const double *nrtl_b;       /* (C*C) */
    const double *nrtl_c;       /* (C*C) */
    const double *tc;           /* (C)   NULL => ideal gas */
    const double *pc;           /* (C) */
    const double *omega;        /* (C) */

    /* ints */
    int n_stages;       /* N; stage 0 = distillate/top, N-1 = reboiler */
    int n_comps;        /* C */
    int n_antoine;      /* coefficients per component in `antoine` */
    int max_iter;
    int condenser;      /* enum cf_condenser */
    int gamma_model;    /* enum cf_gamma */
    int phi_model;      /* enum cf_phi */
} ColParams;

/* --- output ----------------------------------------------------------------
 * The array members are filled in by *Python* before the call and written
 * through by C; only the trailing scalars are written back into the struct
 * (which is why it is passed byref, not by value).
 */
typedef struct {
    /* pointers, caller-allocated */
    double *x;          /* (N*C) liquid composition */
    double *y;          /* (N*C) vapor composition */
    double *T;          /* (N)   stage temperature */
    double *L;          /* (N)   liquid leaving each stage */
    double *V;          /* (N)   vapor leaving each stage */

    /* doubles written back */
    double residual;    /* last temperature *step*, not distance to the answer */

    /* ints written back */
    int n_iter;
    int converged;      /* gate results on this, never on n_iter < max_iter */
} ColResult;

/* --- ABI guards ------------------------------------------------------------
 * ctypes.sizeof(ColParams) == cf_sizeof_params() at import catches silent
 * field-order drift, which is otherwise an afternoon of garbage numbers.
 */
int cf_sizeof_params(void) { return (int)sizeof(ColParams); }
int cf_sizeof_result(void) { return (int)sizeof(ColResult); }

/* --- tridiagonal sweep ----------------------------------------------------- */

/* Scratch doubles cf_thomas needs for an (N, nc) solve. Exported so Python has
 * one place to read the size from. */
int cf_thomas_scratch(int N, int nc) { return 2 * N * nc; }

/* Solve a tridiagonal system (sub/diag/sup bands, rhs) into `out`.
 *
 * Bands are (N, nc) row-major. The stage sweep down N is inherently sequential,
 * but every operation across a row is elementwise, so nc independent systems
 * (one per component) share the structure and solve in a single pass — the
 * inner j-loop is the vectorizable one.
 *
 * `scratch` must hold cf_thomas_scratch(N, nc) doubles. `out` holds N*nc.
 * Aliasing `out` with any band is not supported.
 *
 * ponytail: no pivoting, matching the numpy version it has to agree with. A
 * zero pivot yields inf/nan exactly as numpy does. The MESH bands are
 * diagonally dominant by construction; add pivoting only if that stops holding.
 */
void cf_thomas(int N, int nc, const double *sub, const double *diag,
               const double *sup, const double *rhs, double *out,
               double *scratch)
{
    double *cp = scratch;
    double *dp = scratch + N * nc;

    /* forward sweep — row 0 has no sub-diagonal to eliminate */
    for (int j = 0; j < nc; j++) {
        cp[j] = sup[j] / diag[j];
        dp[j] = rhs[j] / diag[j];
    }
    for (int i = 1; i < N; i++) {
        const int r = i * nc;
        const int p = r - nc;
        for (int j = 0; j < nc; j++) {
            const double m = diag[r + j] - sub[r + j] * cp[p + j];
            cp[r + j] = sup[r + j] / m;
            dp[r + j] = (rhs[r + j] - sub[r + j] * dp[p + j]) / m;
        }
    }

    /* back substitution */
    const int last = (N - 1) * nc;
    for (int j = 0; j < nc; j++) {
        out[last + j] = dp[last + j];
    }
    for (int i = N - 2; i >= 0; i--) {
        const int r = i * nc;
        for (int j = 0; j < nc; j++) {
            out[r + j] = dp[r + j] - cp[r + j] * out[r + nc + j];
        }
    }
}

/* --- self-check ------------------------------------------------------------ */

#ifdef CF_MAIN
#include <assert.h>

int main(void)
{
    /* Two components sharing one tridiagonal structure; component 1 has twice
     * the rhs, so its solution must be exactly twice component 0's. Solution is
     * [1,1,1] and [2,2,2]. sub[0] and sup[N-1] are unused (Python zeroes them). */
    const int N = 3, nc = 2;
    const double sub[]  = {0, 0,  1, 1,  1, 1};
    const double diag[] = {4, 4,  4, 4,  4, 4};
    const double sup[]  = {1, 1,  1, 1,  0, 0};
    const double rhs[]  = {5, 10, 6, 12, 5, 10};

    double out[6];
    double scratch[12];
    assert(cf_thomas_scratch(N, nc) == 12);

    cf_thomas(N, nc, sub, diag, sup, rhs, out, scratch);
    for (int i = 0; i < N; i++) {
        assert(fabs(out[i * nc + 0] - 1.0) < 1e-12);
        assert(fabs(out[i * nc + 1] - 2.0) < 1e-12);
    }

    /* N == 1 degenerates to a single division, and must not walk off either end */
    const double s1[] = {0}, d1[] = {2}, u1[] = {0}, r1[] = {8};
    double o1[1], sc1[2];
    cf_thomas(1, 1, s1, d1, u1, r1, o1, sc1);
    assert(fabs(o1[0] - 4.0) < 1e-12);

    printf("cf_thomas ok | sizeof(ColParams)=%d sizeof(ColResult)=%d\n",
           cf_sizeof_params(), cf_sizeof_result());
    return 0;
}
#endif

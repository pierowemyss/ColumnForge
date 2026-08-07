! Author: Piero Wemyss
! Date: July 13, 2026
!
!  ________________________________________________________________________
! |               Non-Ideal Fortran-computed COefficients 2                \
! __________________________________________________________________________
! \  Fortran subroutines for computing coefficients for non-ideal mixtures |
!  ------------------------------------------------------------------------


! Scratch space for the antoine_Tsat root find since common blocks are
! not allocatable
module nifco_tsat_data
    implicit none
    real(8), allocatable :: tsat_c(:)
    real(8) :: tsat_P
end module

! Antoine Equation    | INPUT:  no. of components        |  OUTPUT: saturation pressures
!                     |         Antoine Coefficients (3) |
!                     |         temperature (Celsius)    |
subroutine antoine_psat(n, coeffs, Tcel, Psat)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in) :: Tcel, coeffs(n,3)
    real(8) :: a(n), b(n), c(n)
    real(8), intent(out) :: Psat(n)

    a = coeffs(:,1)
    b = coeffs(:,2)
    c = coeffs(:,3)

    Psat = 10.00d0 ** (a - b / (Tcel+c))

end subroutine

! Extended Antoine    | INPUT:  no. of components        |  OUTPUT: saturation pressures
! Equation (PLXANT)   |         Coefficients (7)         |
!                     |         temperature (Celsius)    |
subroutine PLXANT_psat(n, c, Tcel, Psat)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in) :: Tcel
    real(8), intent(in), dimension(n,0:6) :: c
    real(8) :: Tk, lnP(n)
    real(8), intent(out) :: Psat(n)

    Tk = Tcel + 273.15d0

    lnP = (c(:, 0) + c(:, 1) / (c(:, 2) + Tk) + c(:, 3) * Tk + c(:, 4) * log(Tk) + c(:, 5) * Tk ** c(:, 6))

    Psat = exp(lnP)

end subroutine

! Wagner Equation     | INPUT:  no. of components        |  OUTPUT: saturation pressures
!                     |         Wagner Coefficients (6)  |
!                     |         temperature (Celsius)    |
subroutine wagner_psat(n, c, TCel, Psat)
    implicit none
    
    integer, intent(in) :: n
    real(8), intent(in) :: TCel
    real(8) :: Tc(n), Pc(n), Tr(n), tau(n), f(n)
    real(8), intent(in), dimension(n,0:5) :: c
    real(8), intent(out) :: Psat(n)

    Tc = c(:,4)                      ! already Kelvin, matching the bundled fits
    Pc = c(:,5)

    ! ponytail: tau**1.5 is complex above Tc, so Tr is clamped to 1 and Psat
    ! saturates at Pc there -- a floor, not physics. Mirrors wagner_psat().
    Tr = min((TCel + 273.15d0) / Tc, 1.0d0)
    tau = 1.0d0 - Tr

    f = c(:, 0) * tau + c(:, 1) * tau ** 1.5 + c(:, 2) * tau ** 3 + c(:, 3) * tau ** 6
    Psat = Pc * exp(f / Tr)

end subroutine

! Boiling point of ONE component at pressure P -- mirrors antoine_Tsat() in
! core/thermodynamics.py. The model is dispatched on ncoef: 3 = Antoine (closed
! form, no root find), 6 = Wagner, 7 = PLXANT. Not a mixture bubble point;
! that is bubble_T(), which needs compositions.

! Vapor Pressure      | INPUT:  no. of coefficients      |  OUTPUT: temperature (Celsius)
! T_sat Root Find     |         Model Coefficients       |
!                     |         pressure (bar)           |
subroutine antoine_tsat(ncoef, c, P, TCel)
    use minpack_module, only: hybrd1
    use nifco_tsat_data

    implicit none

    integer, intent(in) :: ncoef
    real(8), intent(in) :: c(ncoef), P
    real(8), intent(out) :: TCel

    integer, parameter :: n = 1
    integer, parameter :: lwa = (n*(3*n+13))/2

    real(8) :: x(n), fvec(n), wa(lwa), tol
    integer :: info
    external :: antoine_tsat_func

    if (ncoef == 3) then            ! log10(P) = A - B/(T+C) inverts directly
        TCel = c(2) / (c(1) - log10(P)) - c(3)
        return
    end if

    tsat_c = c                      ! allocate-on-assignment (F2003)
    tsat_P = P

    x(1) = 20.0d0
    tol = 1.0d-8

    call hybrd1(antoine_tsat_func, n, x, fvec, tol, info, wa, lwa)
    TCel = x(1)

    if (info /= 1) then
        write(*, '(A, i2)') "NIFCO.TSAT ERROR: Unsuccessful root find for saturation temperature, info=", info
    end if

    deallocate(tsat_c)

end subroutine

subroutine antoine_tsat_func(n, TCel, fvec, iflag)
    use nifco_tsat_data

    implicit none

    integer, intent(in) :: n
    real(8), intent(in) :: TCel(n)
    real(8), intent(out) :: fvec(n)
    integer, intent(inout) :: iflag
    !f2py intent(inout) :: iflag

    real(8) :: P1(1), Tsafe
    external :: PLXANT_psat, wagner_psat

    ! ponytail: hybrd1 is unbracketed where the Python side uses brentq over
    ! [-100, 500]. PLXANT has a pole at Tk = -C3 and grows without bound above,
    ! and one Newton step from 20 C routinely overshoots past both. So the fit
    ! is only ever evaluated inside the window; outside it the residual is
    ! continued linearly, steeply enough to point straight back in. Cannot set
    ! iflag < 0 here -- that terminates hybrd1 rather than shortening its step.
    Tsafe = min(max(TCel(1), -100.0d0), 500.0d0)

    select case (size(tsat_c))
    case (7)
        call PLXANT_psat(1, reshape(tsat_c, [1, 7]), Tsafe, P1)
    case (6)
        call wagner_psat(1, reshape(tsat_c, [1, 6]), Tsafe, P1)
    case default
        iflag = -1                  ! unknown fit: nothing to solve, give up
        return
    end select

    fvec(1) = P1(1) - tsat_P + (TCel(1) - Tsafe) * max(tsat_P, 1.0d0)

end subroutine

! HYSYS-modified NRTL | INPUT:  liquid mole fractions  |  OUTPUT: activity coefficients
! (Vectorized)        |         temperature (Celsius)  |
!                     |         binary coefficients    |
!                     |         number of components   |
subroutine NRTL_gamma(x, Tcel, a, b, c, n, gamma)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in)  :: Tcel, x(n), a(n,n), b(n,n), c(n,n)
    real(8), intent(out) :: gamma(n)
    real(8) :: Tk, tau(n,n), G(n,n), tG(n,n), S(n), r(n), Cc(n)

    Tk = Tcel + 273.15d0
    tau = a + b/Tk
    G = exp(-c * tau)
    tG = tau * G
    S = matmul(x, G)
    r = x/S
    Cc = matmul(x, tG)/S
    gamma = exp(Cc + matmul(tG, r) - matmul(G, Cc*r))

end subroutine

! Wilson (HYSYS)      | INPUT:  liquid mole fractions  |  OUTPUT: activity coefficients
! (Vectorized)        |         temperature (Celsius)  |
!                     |         binary coefficients    |
!                     |         number of components   |
subroutine wilson_gamma(x, Tcel, a, b, n, gamma)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in)  :: Tcel, x(n), a(n,n), b(n,n)
    real(8), intent(out) :: gamma(n)
    real(8) :: Tk, Lamb(n,n), S(n)

    Tk = Tcel + 273.15d0
    Lamb = exp(a + b/Tk)
    S = matmul(Lamb, x)
    gamma = exp(1d0 - log(S) - matmul((x/S), Lamb))

end subroutine

! UNIFAC              | INPUT:  liquid mole fractions   |  OUTPUT: activity coefficients
! (classic, VLE)      |         temperature (Celsius)   |
!                     |         subgroup counts nu(n,m) |
!                     |         subgroup volume/area    |
!                     |         subgroup interactions   |
!                     |         no. components, subgrps |
!
! a(m,m) is the SUBGROUP-expanded interaction matrix -- the caller has already
! done a_mn[main_idx, main_idx]. That gather, and every lookup in
! unifac_groups.json behind it, is constant for a given component list and is
! done once per column by unifac_gamma_fn() in core/thermodynamics.py; this
! runs once per Newton step. The parameter table never crosses into Fortran,
! and m is the count of subgroups actually used (~4-10), not the 2862-pair
! published table -- so everything here is an automatic array on the stack.
subroutine UNIFAC_gamma(x, Tcel, nu, Rk, Qk, a, n, m, gamma)
    implicit none

    integer, intent(in) :: n, m
    real(8), intent(in) :: Tcel, x(n), nu(n,m), Rk(m), Qk(m), a(m,m)
    real(8), intent(out) :: gamma(n)

    real(8) :: Tk, xs(n), r(n), q(n), l(n), phi(n), th(n), ln_c(n), ln_r(n)
    real(8) :: Psi(m,m), lnGmix(m), lnGpure(m)
    integer :: i
    external :: unifac_lngroup

    Tk = Tcel + 273.15d0
    xs = max(x, 1d-12)
    xs = xs / sum(xs)

    ! Combinatorial (Staverman-Guggenheim, z = 10)
    r = matmul(nu, Rk)
    q = matmul(nu, Qk)
    phi = r*xs / dot_product(r, xs)
    th  = q*xs / dot_product(q, xs)
    l   = 5d0*(r - q) - (r - 1d0)
    ln_c = log(phi/xs) + 5d0*q*log(th/phi) + l - (phi/xs)*dot_product(xs, l)

    ! Residual: group activity in the mixture minus in each pure component
    Psi = exp(-a/Tk)
    call unifac_lngroup(matmul(xs, nu), Qk, Psi, m, lnGmix)
    do i = 1, n
        call unifac_lngroup(nu(i,:), Qk, Psi, m, lnGpure)
        ln_r(i) = dot_product(nu(i,:), lnGmix - lnGpure)
    end do

    gamma = exp(ln_c + ln_r)

end subroutine

! ln Gamma_k for every subgroup, given a group-weight vector w. Mixture:
! w = sum_i x_i nu_i^k. Pure component i: w = nu(i,:). The group-fraction
! normalization cancels into theta, so w need not be normalized.
subroutine unifac_lngroup(w, Qk, Psi, m, lnG)
    implicit none

    integer, intent(in) :: m
    real(8), intent(in) :: w(m), Qk(m), Psi(m,m)
    real(8), intent(out) :: lnG(m)
    real(8) :: th(m), S(m)

    th = Qk*w / sum(Qk*w)
    S = matmul(th, Psi)             ! S_k = sum_m theta_m Psi_mk
    lnG = Qk * (1d0 - log(S) - matmul(Psi, th/S))

end subroutine

! UNIQUAC (Aspen UNIQ) | INPUT:  liquid mole fractions  |  OUTPUT: activity coefficients
! (Vectorized)         |         temperature (Celsius)  |
!                      |         r, q structural params |
!                      |         binary coefficients    |
!                      |         number of components   |
!
! tau_ij = exp(a_ij + b_ij/Tk), a_ii = b_ii = 0 giving tau_ii = 1, as in
! core/thermodynamics.uniquac_gamma_fn. The ratio forms (Phi_i/x_i = r_i/sum x r)
! keep the pure-component and zero-x limits exact. z = 10 as everywhere else.
subroutine UNIQUAC_gamma(x, Tcel, r, q, a, b, n, gamma)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in) :: x(n), Tcel, r(n), q(n), a(n,n), b(n,n)
    real(8), intent(out) :: gamma(n)

    real(8), parameter :: z = 10d0
    real(8) :: Tk, xr, xq, tau(n,n)
    real(8) :: phi_x(n), theta_phi(n), l(n), ln_gC(n), theta(n), S(n), ln_gR(n)

    Tk = Tcel + 273.15d0
    tau = exp(a + b/Tk)

    xr = dot_product(x, r)
    xq = dot_product(x, q)
    phi_x = r / xr                       ! Phi_i / x_i
    theta_phi = (q / xq) / phi_x         ! theta_i / Phi_i
    l = 0.5d0 * z * (r - q) - (r - 1d0)
    ln_gC = log(phi_x) + 0.5d0 * z * q * log(theta_phi) + l - phi_x * dot_product(x, l)
    theta = q * x / xq
    S = matmul(theta, tau)               ! S_i = sum_j theta_j tau_ji
    ln_gR = q * (1d0 - log(S) - matmul(tau, theta / S))
    gamma = exp(ln_gC + ln_gR)

end subroutine

! Two-suffix Margules | INPUT:  liquid mole fractions    |  OUTPUT: activity coefficients
! (Vectorized)        |         temperature (Celsius)    |
!                     |         a(n,n) symmetric, a_ii=0 |
!                     |         number of components     |
!
! G^E/RT = 1/2 sum_ij a_ij x_i x_j, so ln gamma_i = sum_j a_ij x_j - G^E/RT.
! Temperature-independent: Tcel is unused, and only present so every gamma
! subroutine here has the same calling convention.
subroutine margules_gamma(x, Tcel, a, n, gamma)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in) :: x(n), Tcel, a(n,n)
    real(8), intent(out) :: gamma(n)
    real(8) :: Ax(n)

    Ax = matmul(a, x)
    gamma = exp(Ax - 0.5d0 * dot_product(x, Ax))

end subroutine

! SRK EOS  | INPUT: vapor mole fractions  | OUTPUT: fugacity coefficient
!          |        temperature (celsius) |
!          |        pressure (bar)        |
!          |        crit temp (celsius)   |
!          |        crit press (bar)      |
!          |        acentricity           |
!          |        number of components  |
subroutine SRK_phi(x,Tcel,P,TcCel,Pc,omega,NComps,phi)

    use minpack_module, only: hybrd1

    implicit none

    common /shared_data/ Aa, Bb

    integer, intent(in) :: NComps
    real(8), intent(in) :: P, Tcel
    real(8), intent(in), dimension(0:NComps-1) :: x, TcCel, Pc, omega
    real(8), intent(out), dimension(0:NComps-1) :: phi
    real(8) :: Aa, Bb, Z0, am, bm, Tk
    real(8), dimension(0:NComps-1) :: Tc, Tr, Pr, alph, ai, bi
    integer :: info

    integer,parameter :: n = 1
    integer,parameter :: lwa = (n*(3*n+13))/2

    external :: SRK_Z_resid
    real(8) :: Z(n), fvec(n), wa(lwa)
    real(8) :: tol

    real(8), parameter :: R = 8.314472d0

    Tk = Tcel + 273.15d0
    Tc = TcCel + 273.15d0
    Tr = Tk/Tc
    Pr = P/Pc
    
    alph = (1.0 + (0.480+1.574*omega-0.176*omega**(2.0))*(1.0-Tr**(0.5)))**2.0
    ai = 0.42748*(alph*R**2*Tc**2)/Pc
    bi = 0.08664*R*Tc/Pc

    am = sum(x * ai**(0.5))**2
    bm = sum(x * bi)
    Aa = am * P/(R*Tk)**2
    Bb = bm * P/(R*Tk)
    
    Z0 = 1.0d0
    Z(1) = Z0

    tol = 1.0d-8

    call hybrd1(SRK_Z_resid, n, Z, fvec, tol, info, wa, lwa)

    if (info /= 1) then
        write(*, '(A, i2)') "NIFCO.SRK ERROR: Unsuccessful root find for compressibility."
    end if

    phi = exp((bi/bm)*(Z(1)-1) - log(Z(1)-Bb) - (Aa/Bb)*(2*ai**(0.5)/sum(x*ai**(0.5)) - bi/bm)*log(1+ Bb/Z(1)))

end subroutine

! SRK Z-fact  | INPUT: vapor mole fractions  | OUTPUT: compressibility factor
!             |        temperature (celsius) |
!             |        pressure (bar)        |
!             |        crit temp (celsius)   |
!             |        crit press (bar)      |
!             |        acentricity           |
!             |        number of components  |
subroutine SRK_Z(x,Tcel,P,TcCel,Pc,omega,NComps,Z)

    use minpack_module, only: hybrd1

    implicit none

    common /shared_data/ Aa, Bb

    integer, intent(in) :: NComps
    real(8), intent(in) :: P, Tcel
    real(8), intent(in), dimension(0:NComps-1) :: x, TcCel, Pc, omega
    real(8) :: Aa, Bb, Z0, am, bm, Tk
    real(8), dimension(0:NComps-1) :: Tc, Tr, Pr, alph, ai, bi
    integer :: info

    integer,parameter :: n = 1
    integer,parameter :: lwa = (n*(3*n+13))/2

    external :: SRK_Z_resid
    real(8) :: Z(n), fvec(n), wa(lwa)
    real(8) :: tol

    real(8), parameter :: R = 8.314472d0

    Tk = Tcel + 273.15d0
    Tc = TcCel + 273.15d0
    Tr = Tk/Tc
    Pr = P/Pc
    
    alph = (1.0 + (0.480+1.574*omega-0.176*omega**(2.0))*(1.0-Tr**(0.5)))**2.0
    ai = 0.42748*(alph*R**2*Tc**2)/Pc
    bi = 0.08664*R*Tc/Pc

    am = sum(x * ai**(0.5))**2
    bm = sum(x * bi)
    Aa = am * P/(R*Tk)**2
    Bb = bm * P/(R*Tk)
    
    Z0 = 1.0d0
    Z(1) = Z0

    tol = 1.0d-8

    call hybrd1(SRK_Z_resid, n, Z, fvec, tol, info, wa, lwa)

    if (info /= 1) then
        write(*, '(A, i2)') "NIFCO.SRK ERROR: Unsuccessful root find for compressibility."
    end if

end subroutine

subroutine SRK_Z_resid(n, Z, fvec, iflag)
    
    implicit none

    common /shared_data/ Aa, Bb

    integer, intent(in) :: n
    real(8), intent(in) :: Z(n)
    real(8), intent(out) :: fvec(n)
    integer, intent(inout) :: iflag
    !f2py intent(inout) :: iflag
    real(8) :: Aa, Bb

    fvec(1) = Z(1)**3 - Z(1)**2 + (Aa - Bb - Bb**2)*Z(1) - Aa*Bb

end subroutine

! Author: Piero Wemyss
! Date: July 13, 2026
!
!  ________________________________________________________________________
! |               Non-Ideal Fortran-computed COefficients 2                \
! __________________________________________________________________________
! \  Fortran subroutines for computing coefficients for non-ideal mixtures |
!  ------------------------------------------------------------------------


subroutine antoine_psat(n, coeffs, Tcel, Psat)
   implicit none

   integer, intent(in) :: n
   real(8), intent(in) :: Tcel, coeffs(n,3)
   real(8) :: a(n), b(n), c(n)
   real(8), intent(out) :: Psat(n)
   
   a = coeffs(:,1)
   b = coeffs(:,2)
   c = coeffs(:,3)

   Psat = 10.00d0 ** (a + b / (Tcel+C))

end subroutine

subroutine PLXANT_psat(n, c, Tcel, Psat)
   implicit none

   integer, intent(in) :: n
   real(8), intent(in) :: Tcel
   real(8), intent(in), dimension(0:n,0:7) :: c
   real(8) :: Tk, lnP(n)
   real(8), intent(out) :: Psat(n)

   Tk = Tcel + 273.15d0

   lnP = (c(:, 0) + c(:, 1) / (c(:, 2) + Tk) + c(:, 3) * Tk + c(:, 4) * log(Tk) + c(:, 5) * Tk ** c(:, 6))

   Psat = exp(lnP)

end subroutine

! HYSYS-modified NRTL | INPUT:  liquid mole fractions  |  OUTPUT: activity coefficients
! (Vectorized)        |         temperature (Celsius)  |
!                     |         binary coefficients    |
!                     |         number of components   |
subroutine NRTL(x, Tcel, a, b, c, n, gamma)
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
subroutine Wilson(x, Tcel, a, b, n, gamma)
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

! SRK EOS  | INPUT: vapor mole fractions  | OUTPUT: fugacity coefficient
!          |        temperature (celsius) |
!          |        pressure (bar)        |
!          |        crit temp (celsius)   |
!          |        crit press (bar)      |
!          |        acentricity           |
!          |        number of components  |
subroutine SRK(x,Tcel,P,TcCel,Pc,omega,NComps,phi)

    use minpack_module, only: wp, hybrd1, dpmpar, enorm
    use iso_fortran_env, only: nwrite => output_unit

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

    external :: RKfunc
    real(8) :: Z(n), fvec(n), wa(lwa)
    real(8) :: tol

    real(8), parameter :: R = 8.314472d0

    Tk = Tcel + 273.15d0
    Tc = TcCel + 273.15
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

    call hybrd1(RKfunc, n, Z, fvec, tol, info, wa, lwa)

    if (info /= 1) then
        write(*, '(A, i2)') "NIFCO.SRK ERROR: Unsuccessful root find for compressibility."
    end if

    phi = exp((bi/bm)*(Z(1)-1) - log(Z(1)-Bb) - (Aa/Bb)*(2*ai**(0.5)/sum(x*ai**(0.5)) - bi/bm)*log(1+ Bb/Z(1)))

end subroutine

subroutine RKfunc(n, Z, fvec, iflag)
    
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

// AMReX_DG_BasisFunctions.H

#ifndef BL_DG_BASIS_FUNCTIONS_H_
#define BL_DG_BASIS_FUNCTIONS_H_

#include <AMReX_REAL.H>

#define DG_BASIS_FUNCTIONS_LEGENDRE_P 0
#define DG_BASIS_FUNCTIONS_CHEBYSHEV_T 1

#define DG_BASIS_FUNCTIONS 0

namespace amrex
{

// 1D BASIS FUNCTIONS IN THE INTERVAL [xlo,xhi] #######################
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_BF_1d_Table_NxL(const int& N, const int& L,
                        const T* x, const Real& xlo, const Real& xhi,
                        T* BF)
                        //SX:  N points in x , poly. order <= L, interval [xlo, xhi], BF[N*L] stores the table in a long vector
{
    // LOCAL VARIABLES =====
    Real dx = (xhi-xlo);
    Real xm = 0.5*(xhi+xlo);
    Real J = 2.0/dx;
    // =====================

    // LEGENDRE POLYNOMIALS ===========================================
    // recursion formula
#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    for (int n = 0; n < N; ++n) BF[n] = 1.0;
    if (L > 0) for (int n = 0; n < N; ++n) BF[n+N] = J*(x[n]-xm);
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        BF[n+l*N] = ((2*l-1)*(J*(x[n]-xm))*BF[n+(l-1)*N]-(l-1)*BF[n+(l-2)*N])/l;
#endif
    // ================================================================
}

// ------------------------ __host__ functions ------------------------

template<typename T>
void DG_BF_1d_Table_NxL__host__(const int& N, const int& L,
                                const T* x, const Real& xlo, const Real& xhi,
                                T* BF)
{
    // LOCAL VARIABLES =====
    Real dx = (xhi-xlo);
    Real xm = 0.5*(xhi+xlo);
    Real J = 2.0/dx;
    // =====================

    // LEGENDRE POLYNOMIALS ===========================================
#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    for (int n = 0; n < N; ++n) BF[n] = 1.0;
    if (L > 0) for (int n = 0; n < N; ++n) BF[n+N] = J*(x[n]-xm);
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        BF[n+l*N] = ((2*l-1)*(J*(x[n]-xm))*BF[n+(l-1)*N]-(l-1)*BF[n+(l-2)*N])/l;
#endif
    // ================================================================
}
// ####################################################################


// 1D BASIS FUNCTIONS AND THEIR DERIVATIVES IN THE INTERVAL [xlo,xhi] #
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_BF_1d_Table_NxL(const int& N, const int& L,
                        const T* x, const Real& xlo, const Real& xhi,
                        T* BF, T* dBF)
{
    // LOCAL VARIABLES ========
    Real J = 2.0/(xhi-xlo);
    // ========================

    // LEGENDRE POLYNOMIALS AND THEIR DERIVATIVES =====================
#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF);

    for (int n = 0; n < N; ++n) dBF[n] = 0.0;
    if (L > 0) for (int n = 0; n < N; ++n) dBF[n+N] = 1.0;
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] = (2*l-1)*BF[n+(l-1)*N]+dBF[n+(l-2)*N];

    for (int l = 1; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] *= J;
#endif
    // ================================================================
}

// ------------------------ __host__ functions ------------------------

template<typename T>
void DG_BF_1d_Table_NxL__host__(const int& N, const int& L,
                                const T* x, const Real& xlo, const Real& xhi,
                                T* BF, T* dBF)
{
    // LOCAL VARIABLES ========
    Real J = 2.0/(xhi-xlo);
    // ========================

    // LEGENDRE POLYNOMIALS AND THEIR DERIVATIVES =====================
#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    DG_BF_1d_Table_NxL__host__(N, L, x, xlo, xhi, BF);

    for (int n = 0; n < N; ++n) dBF[n] = 0.0;
    if (L > 0) for (int n = 0; n < N; ++n) dBF[n+N] = 1.0;
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] = (2*l-1)*BF[n+(l-1)*N]+dBF[n+(l-2)*N];

    for (int l = 1; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] *= J;
#endif
    // ================================================================
}
// ####################################################################


// TIME BASIS FUNCTIONS ###############################################
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_time_BF_Table_NxL(const int& N, const int& L,
                          const T* t, const Real& tlo, const Real& thi,
                          T* BF_1d, T* dBF_1d,
                          T* BF, T* dBF)
{
    int l;

    DG_BF_1d_Table_NxL(N, L, t, tlo, thi, BF_1d, dBF_1d);

    l = 0;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]-BF_1d[n+1*N]);
        dBF[n+l*N] = 0.5*(dBF_1d[n+0*N]-dBF_1d[n+1*N]);
    }
    l = L;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]+BF_1d[n+1*N]);
        dBF[n+l*N] = 0.5*(dBF_1d[n+0*N]+dBF_1d[n+1*N]);
    }
    for (l = 1; l < L; ++l)
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = BF_1d[n+(l+1)*N]-BF_1d[n+(l-1)*N];
        dBF[n+l*N] = dBF_1d[n+(l+1)*N]-dBF_1d[n+(l-1)*N];
    }
}

template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_time_BF_Table_NxL(const int& N, const int& L,
                          const T* t, const Real& tlo, const Real& thi,
                          T* BF_1d,
                          T* BF)
{
    int l;

    DG_BF_1d_Table_NxL(N, L, t, tlo, thi, BF_1d);

    l = 0;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]-BF_1d[n+1*N]);
    }
    l = L;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]+BF_1d[n+1*N]);
    }
    for (l = 1; l < L; ++l)
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = BF_1d[n+(l+1)*N]-BF_1d[n+(l-1)*N];
    }
}

// ------------------------ __host__ functions ------------------------

template<typename T>
void DG_time_BF_Table_NxL__host__(const int& N, const int& L,
                                  const T* t, const Real& tlo, const Real& thi,
                                  T* BF_1d, T* dBF_1d,
                                  T* BF, T* dBF)
{
    int l;

    DG_BF_1d_Table_NxL__host__(N, L, t, tlo, thi, BF_1d, dBF_1d);

    l = 0;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]-BF_1d[n+1*N]);
        dBF[n+l*N] = 0.5*(dBF_1d[n+0*N]-dBF_1d[n+1*N]);
    }
    l = L;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]+BF_1d[n+1*N]);
        dBF[n+l*N] = 0.5*(dBF_1d[n+0*N]+dBF_1d[n+1*N]);
    }
    for (l = 1; l < L; ++l)
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = BF_1d[n+(l+1)*N]-BF_1d[n+(l-1)*N];
        dBF[n+l*N] = dBF_1d[n+(l+1)*N]-dBF_1d[n+(l-1)*N];
    }
}

template<typename T>
void DG_time_BF_Table_NxL__host__(const int& N, const int& L,
                                  const T* t, const Real& tlo, const Real& thi,
                                  T* BF_1d,
                                  T* BF)
{
    int l;

    DG_BF_1d_Table_NxL__host__(N, L, t, tlo, thi, BF_1d);

    l = 0;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]-BF_1d[n+1*N]);
    }
    l = L;
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = 0.5*(BF_1d[n+0*N]+BF_1d[n+1*N]);
    }
    for (l = 1; l < L; ++l)
    for (int n = 0; n < N; ++n)
    {
        BF[n+l*N] = BF_1d[n+(l+1)*N]-BF_1d[n+(l-1)*N];
    }
}
// ####################################################################

// SPACE BASIS FUNCTIONS ##############################################
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_1d_BF_Table_NxL(const int& N, const int& L,
                              const T* x, const Real& xlo, const Real& xhi,
                              T* BF_1d,
                              T* BF)
{
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF);
}
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_1d_BF_Table_NxL(const int& N, const int& L,
                              const T* x, const Real& xlo, const Real& xhi,
                              T* BF_1d, T* dBF_1d,
                              T* BF, T* dBF)
{
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF, dBF);
}

template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_2d_BF_Table_NxL(const int& N, const int& L,
                              const T* x1, const T* x2, const Real* xlo, const Real* xhi,
                              T* BF1_1d, T* dBF1_1d, T* BF2_1d, T* dBF2_1d,
                              T* BF, T* dBFdx1, T* dBFdx2)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2];
    }
}
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_2d_BF_Table_NxL(const int& N, const int& L,
                              const T* x1, const T* x2, const Real* xlo, const Real* xhi,
                              T* BF1_1d, T* BF2_1d,
                              T* BF)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
    }
}

template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_3d_BF_Table_NxL(const int& N, const int& L,
                              const T* x1, const T* x2, const T* x3, const Real* xlo, const Real* xhi,
                              T* BF1_1d, T* dBF1_1d, T* BF2_1d, T* dBF2_1d, T* BF3_1d, T* dBF3_1d,
                              T* BF, T* dBFdx1, T* dBFdx2, T* dBFdx3)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);
    DG_BF_1d_Table_NxL(N, L, x3, xlo[2], xhi[2], BF3_1d, dBF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2]*BF3_1d[pos3];
        dBFdx3[pos] = BF1_1d[pos1]*BF2_1d[pos2]*dBF3_1d[pos3];
    }
}
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_3d_BF_Table_NxL(const int& N, const int& L,
                              const T* x1, const T* x2, const T* x3, const Real* xlo, const Real* xhi,
                              T* BF1_1d, T* BF2_1d, T* BF3_1d, T* BF)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d);
    DG_BF_1d_Table_NxL(N, L, x3, xlo[2], xhi[2], BF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
    }
}

// ------------------------ __host__ functions ------------------------

template<typename T>
void DG_space_1d_BF_Table_NxL__host__(const int& N, const int& L,
                                      const T* x, const Real& xlo, const Real& xhi,
                                      T* BF_1d,
                                      T* BF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x, xlo, xhi, BF);
}
template<typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_1d_BF_Table_NxL__host__(const int& N, const int& L,
                                      const T* x, const Real& xlo, const Real& xhi,
                                      T* BF_1d, T* dBF_1d,
                                      T* BF, T* dBF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x, xlo, xhi, BF, dBF);
}

template<typename T>
void DG_space_2d_BF_Table_NxL__host__(const int& N, const int& L,
                                      const T* x1, const T* x2, const Real* xlo, const Real* xhi,
                                      T* BF1_1d, T* dBF1_1d, T* BF2_1d, T* dBF2_1d,
                                      T* BF, T* dBFdx1, T* dBFdx2)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2];
    }
}
template<typename T>
void DG_space_2d_BF_Table_NxL__host__(const int& N, const int& L,
                                      const T* x1, const T* x2, const Real* xlo, const Real* xhi,
                                      T* BF1_1d, T* BF2_1d,
                                      T* BF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
    }
}

template<typename T>
void DG_space_3d_BF_Table_NxL__host__(const int& N, const int& L,
                                      const T* x1, const T* x2, const T* x3, const Real* xlo, const Real* xhi,
                                      T* BF1_1d, T* dBF1_1d, T* BF2_1d, T* dBF2_1d, T* BF3_1d, T* dBF3_1d,
                                      T* BF, T* dBFdx1, T* dBFdx2, T* dBFdx3)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x3, xlo[2], xhi[2], BF3_1d, dBF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2]*BF3_1d[pos3];
        dBFdx3[pos] = BF1_1d[pos1]*BF2_1d[pos2]*dBF3_1d[pos3];
    }
}

template<typename T>
void DG_space_3d_BF_Table_NxL__host__(const int& N, const int& L,
                                      const T* x1, const T* x2, const T* x3, const Real* xlo, const Real* xhi,
                                      T* BF1_1d, T* BF2_1d, T* BF3_1d, T* BF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x3, xlo[2], xhi[2], BF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
    }
}
// ####################################################################
}

#endif
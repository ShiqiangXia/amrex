// AMReX_DG_Base.H

#ifndef BL_DG_BASE_H_
#define BL_DG_BASE_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <algoim_quad.hpp>

#include <AMReX_DG_Quadrature.H>
#include <AMReX_blas.H>
#include <AMReX_lapack.H>
#include <AMReX_DG_utils.H>
#include <AMReX_DG_BasisFunctions.H>


// SOME PARAMETERS ####################################################
// DG REPORT
#ifndef DG_VERBOSITY
#define DG_VERBOSITY 1
#endif

// DG NUMBER OF DISTANCE FUNCTIONS/DOMAINS/UNKNOWN FIELDS PER PATCH
#define DG_MAX_N_PHI_PER_PATCH 4
#define DG_MAX_N_DOM_PER_PATCH 3
#define DG_MAX_N_SOL_PER_PATCH 12

// DG UNIT CELL
#define DG_CELL_N_TIME_BOUNDARIES 2
#define DG_CELL_N_SPACE_BOUNDARIES (2*AMREX_SPACEDIM)

// DG SPACE APPROXIMATION ORDER: DISTANCE FUNCTIONS
#define DG_PHI_MAX_TIME_p 4
#define DG_PHI_MAX_TIME_Np (1+DG_PHI_MAX_TIME_p)

#define DG_PHI_MAX_SPACE_p 3
#if (AMREX_SPACEDIM == 1)
#define DG_PHI_MAX_SPACE_Np (1+DG_PHI_MAX_SPACE_p)
#endif
#if (AMREX_SPACEDIM == 2)
#define DG_PHI_MAX_SPACE_Np ((1+DG_PHI_MAX_SPACE_p)*(1+DG_PHI_MAX_SPACE_p))
#endif
#if (AMREX_SPACEDIM == 3)
#define DG_PHI_MAX_SPACE_Np ((1+DG_PHI_MAX_SPACE_p)*(1+DG_PHI_MAX_SPACE_p)*(1+DG_PHI_MAX_SPACE_p))
#endif

// DG SPACE APPROXIMATION ORDER: SOLUTION
#define DG_SOL_MAX_TIME_p 5
#define DG_SOL_MAX_TIME_Np (1+DG_SOL_MAX_TIME_p)

#define DG_SOL_MAX_SPACE_p 5
#if (AMREX_SPACEDIM == 1)
#define DG_SOL_MAX_SPACE_Np (1+DG_SOL_MAX_SPACE_p)
#endif
#if (AMREX_SPACEDIM == 2)
#define DG_SOL_MAX_SPACE_Np ((1+DG_SOL_MAX_SPACE_p)*(1+DG_SOL_MAX_SPACE_p))
#endif
#if (AMREX_SPACEDIM == 3)
#define DG_SOL_MAX_SPACE_Np ((1+DG_SOL_MAX_SPACE_p)*(1+DG_SOL_MAX_SPACE_p)*(1+DG_SOL_MAX_SPACE_p))
#endif

// DG TIME APPROXIMATION ORDER
#define DG_MAX_RK_ORDER (((DG_PHI_MAX_TIME_p) >= (DG_SOL_MAX_TIME_p)) ? (DG_PHI_MAX_TIME_p) : (DG_SOL_MAX_TIME_p))

// POST-PROCESSING
#define DG_MAX_N_FIELDS 10

// IMPLICIT MESH INFO
#define DG_ELM_TYPE_N_COMP_PER_DOM 1
#define DG_ELM_INFO_N_COMP_PER_DOM 3
#define DG_ELM_BOU_TYPE_N_COMP_PER_DOM 1
#define DG_ELM_DOM_QUAD_N_COMP_PER_DOM 1
#define DG_ELM_BOU_QUAD_N_COMP_PER_DOM 2
#define DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM 1
#define DG_ELM_TRI_N_COMP_PER_DOM 1

#define DG_ELM_TYPE_UNDEFINED -100
#define DG_ELM_TYPE_EMPTY -1
#define DG_ELM_TYPE_ENTIRE 1
#define DG_ELM_TYPE_LARGE 2
#define DG_ELM_TYPE_SMALL 3
#define DG_ELM_TYPE_PARTIAL 5

#define DG_SMALL_ELM_VF_THRESHOLD 0.40

#define DG_ELM_BOU_TYPE_UNDEFINED -100
#define DG_ELM_BOU_TYPE_EMPTY -1
#define DG_ELM_BOU_TYPE_ENTIRE 1
#define DG_ELM_BOU_TYPE_PARTIAL 5

#define DG_INFO_MAX_N_1D_BOU_GRIDS 10
#define DG_INFO_MAX_N_2D_BOU_GRIDS 4
#define DG_INFO_MAX_N_3D_BOU_GRIDS 2
#define DG_INFO_MAX_N_BOU_GRIDS AMREX_D_PICK(DG_INFO_MAX_N_1D_BOU_GRIDS, DG_INFO_MAX_N_2D_BOU_GRIDS, DG_INFO_MAX_N_3D_BOU_GRIDS)

#define DG_DOM_BF_INFO_N_COMP_PER_DOM 1
#define DG_TRI_BF_INFO_N_COMP_PER_DOM 1
#define DG_BOU_BF_INFO_N_COMP_PER_DOM DG_INFO_MAX_N_BOU_GRIDS
#define DG_INT_BOU_BF_INFO_N_COMP_PER_DOM 1
// ####################################################################


namespace amrex
{
namespace DG
{

// TYPEDEFS ###########################################################
typedef FabArray<BaseFab<Real *>> pMultiFab;
typedef FabArray<BaseFab<long>> longMultiFab;

template<int Np>
struct X_
{
    Real data[Np];

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real & operator[](int k)
    {
        return data[k];
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    const Real & operator[](int k) const
    {
        return data[k];
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator=(const X_ & other)
    {
        for (int k = 0; k < Np; ++k)
        {
            this->data[k] = other.data[k];
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator+=(const X_ & other)
    {
        for (int k = 0; k < Np; ++k)
        {
            this->data[k] += other.data[k];
        }
    }
};

typedef FabArray<BaseFab<X_<DG_PHI_MAX_SPACE_Np>>> PhiMultiFab;
typedef FabArray<BaseFab<X_<DG_SOL_MAX_SPACE_Np>>> SolMultiFab;

typedef Algoim::QuadratureRule<AMREX_SPACEDIM> iQuadRule;
typedef Algoim::QuadratureRule<AMREX_SPACEDIM> iTriRule;
// ####################################################################

// ####################################################################
// REFERENCE ELEMENT: HYPERRECTANGLE ##################################
// ####################################################################
template<int DIM>
struct UnitHyperrectangle
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int n_bou = 2*DIM;

    // PARAMETERS
    const int p, Np;
    const int q, dom_Nq, bou_Nq;
    const int g, dom_Ng;

    // VARIABLES
    // Quadrature abscissae and weights
    Gpu::ManagedVector<Real> dom_XI1q;
    Gpu::ManagedVector<Real> dom_XI2q;
    Gpu::ManagedVector<Real> dom_XI3q;
    Gpu::ManagedVector<Real> dom_Wq;

    Gpu::ManagedVector<Real> bou_XI1q;
    Gpu::ManagedVector<Real> bou_XI2q;
    Gpu::ManagedVector<Real> bou_XI3q;
    Gpu::ManagedVector<Real> bou_Wq;

    // Basis functions
    Gpu::ManagedVector<Real> dom_BF;
    Gpu::ManagedVector<Real> dom_dBFdXI1;
    Gpu::ManagedVector<Real> dom_dBFdXI2;
    Gpu::ManagedVector<Real> dom_dBFdXI3;
    
    Gpu::ManagedVector<Real> bou_BF;
    Gpu::ManagedVector<Real> bou_dBFdXI1;
    Gpu::ManagedVector<Real> bou_dBFdXI2;
    Gpu::ManagedVector<Real> bou_dBFdXI3;

    // Mass matrix, Cholesky decomposition, inverse
    Gpu::ManagedVector<Real> MM, MMCh, iMM;

    // Uniformly-spaced grid
    Gpu::ManagedVector<Real> dom_XI1g;
    Gpu::ManagedVector<Real> dom_XI2g;
    Gpu::ManagedVector<Real> dom_XI3g;

    // Basis functions evaluated at the uniformly-spaced grid
    Gpu::ManagedVector<Real> grid_BF;
    // ################################################################

    // CONSTRUCTOR ####################################################
    UnitHyperrectangle(const int& ip, const int& iq):
    p(ip),
    Np((DIM == 1) ? (1+ip) : ((DIM == 2) ? ((1+ip)*(1+ip)) : ((1+ip)*(1+ip)*(1+ip)))),
    q(iq),
    dom_Nq((DIM == 1) ? (1+iq) : ((DIM == 2) ? ((1+iq)*(1+iq)) : ((1+iq)*(1+iq)*(1+iq)))),
    bou_Nq((DIM == 1) ? 1 : ((DIM == 2) ? (1+iq) : ((1+iq)*(1+iq)))),
    g(std::max(2, iq)),
    dom_Ng((DIM == 1) ? (1+this->g) : ((DIM == 2) ? ((1+this->g)*(1+this->g)) : ((1+this->g)*(1+this->g)*(1+this->g))))
    {
        // ALLOCATE MEMORY: QUADRATURE INFORMATION ====================
        // DOMAIN
        if (DIM > 0) this->dom_XI1q.resize(this->dom_Nq);
        if (DIM > 1) this->dom_XI2q.resize(this->dom_Nq);
        if (DIM > 2) this->dom_XI3q.resize(this->dom_Nq);
        this->dom_Wq.resize(this->dom_Nq);

        // BOUNDARIES
        if (DIM > 0) this->bou_XI1q.resize((this->bou_Nq)*(this->n_bou));
        if (DIM > 1) this->bou_XI2q.resize((this->bou_Nq)*(this->n_bou));
        if (DIM > 2) this->bou_XI3q.resize((this->bou_Nq)*(this->n_bou));
        this->bou_Wq.resize(this->bou_Nq);
        // ============================================================

        // POINTERS TO 1D QUADRATURE ABSCISSAE AND WEIGHTS ============
        const Real * Xq_1d;
        const Real * Wq_1d;

        if      (iq == 0) {Xq_1d = gau_leg_1_x; Wq_1d = gau_leg_1_w;}
        else if (iq == 1) {Xq_1d = gau_leg_2_x; Wq_1d = gau_leg_2_w;}
        else if (iq == 2) {Xq_1d = gau_leg_3_x; Wq_1d = gau_leg_3_w;}
        else if (iq == 3) {Xq_1d = gau_leg_4_x; Wq_1d = gau_leg_4_w;}
        else if (iq == 4) {Xq_1d = gau_leg_5_x; Wq_1d = gau_leg_5_w;}
        else if (iq == 5) {Xq_1d = gau_leg_6_x; Wq_1d = gau_leg_6_w;}
        else if (iq == 6) {Xq_1d = gau_leg_7_x; Wq_1d = gau_leg_7_w;}
        else if (iq == 7) {Xq_1d = gau_leg_8_x; Wq_1d = gau_leg_8_w;}
        else if (iq == 8) {Xq_1d = gau_leg_9_x; Wq_1d = gau_leg_9_w;}
        else if (iq == 9) {Xq_1d = gau_leg_10_x; Wq_1d = gau_leg_10_w;}
        else if (iq == 10) {Xq_1d = gau_leg_11_x; Wq_1d = gau_leg_11_w;}
        else if (iq == 11) {Xq_1d = gau_leg_12_x; Wq_1d = gau_leg_12_w;}
        else if (iq == 12) {Xq_1d = gau_leg_13_x; Wq_1d = gau_leg_13_w;}
        else if (iq == 13) {Xq_1d = gau_leg_14_x; Wq_1d = gau_leg_14_w;}
        else if (iq == 14) {Xq_1d = gau_leg_15_x; Wq_1d = gau_leg_15_w;}
        else if (iq == 15) {Xq_1d = gau_leg_16_x; Wq_1d = gau_leg_16_w;}
        else
        {
            Print() << "ERROR: AMReX_DG_Definitions.H - UnitHyperrectangle::UnitHyperrectangle" << std::endl;
            Print() << "| Quadrature order = " << iq << "not implemented yet" << std::endl;
            exit(-1);
        }
        // ============================================================

        // STORE QUADRATURE POINTS, WEIGHTS, UNIT NORMALS =============
        if (DIM == 1)
        {
            // LOCAL VARIABLES
            int q1, pos;
            int b;
            // ---------------

            // DOMAIN -------------------------------------------------
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1;
                this->dom_XI1q[pos] = Xq_1d[q1];
                this->dom_Wq[pos] = Wq_1d[q1];
            }
            // --------------------------------------------------------

            // BOUNDARIES AND UNIT NORMALS ----------------------------
            // Boundary with unit normal {-1}
            b = 0;
            this->bou_XI1q[b] = -1.0;

            // Boundary with unit normal {+1}
            b = 1;
            this->bou_XI1q[b] = +1.0;

            // Weights
            this->bou_Wq[0] = +1.0;
            // --------------------------------------------------------
        }
        else if (DIM == 2)
        {
            // LOCAL VARIABLES
            int q1, q2, pos;
            int b;
            // ---------------

            // DOMAIN -------------------------------------------------
            for (q2 = 0; q2 < (1+iq); ++q2)
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+q2*(1+iq);
                this->dom_XI1q[pos] = Xq_1d[q1];
                this->dom_XI2q[pos] = Xq_1d[q2];
                this->dom_Wq[pos] = Wq_1d[q1]*Wq_1d[q2];
            }
            // --------------------------------------------------------

            // BOUNDARIES AND UNIT NORMALS ----------------------------
            // Boundary with normal {-1,0}
            b = 0;
            for (q2 = 0; q2 < (1+iq); ++q2)
            {
                pos = q2+b*(this->bou_Nq);
                this->bou_XI1q[pos] = -1.0;
                this->bou_XI2q[pos] = Xq_1d[q2];
            }

            // Boundary with normal {+1,0}
            b = 1;
            for (q2 = 0; q2 < (1+iq); ++q2)
            {
                pos = q2+b*(this->bou_Nq);
                this->bou_XI1q[pos] = +1.0;
                this->bou_XI2q[pos] = Xq_1d[q2];
            }

            // Boundary with normal {0,-1}
            b = 2;
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1];
                this->bou_XI2q[pos] = -1.0;
            }

            // Boundary with normal {0,+1}
            b = 3;
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1];
                this->bou_XI2q[pos] = +1.0;
            }

            // Weights
            for (q1 = 0; q1 < (1+iq); ++q1) this->bou_Wq[q1] = Wq_1d[q1];
            // --------------------------------------------------------
            
        }
        else if (DIM == 3)
        {
            // LOCAL VARIABLES -
            int q1, q2, q3, pos;
            int b;
            // -----------------

            // DOMAIN -------------------------------------------------
            for (q3 = 0; q3 < (1+iq); ++q3)
            for (q2 = 0; q2 < (1+iq); ++q2)
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+q2*(1+iq)+q3*(1+iq)*(1+iq);
                this->dom_XI1q[pos] = Xq_1d[q1];
                this->dom_XI2q[pos] = Xq_1d[q2];
                this->dom_XI3q[pos] = Xq_1d[q3];
                this->dom_Wq[pos] = Wq_1d[q1]*Wq_1d[q2]*Wq_1d[q3];
            }
            // --------------------------------------------------------

            // BOUNDARIES AND UNIT NORMALS ----------------------------
            // Boundary with normal {-1,0,0}
            b = 0;
            for (q3 = 0; q3 < (1+iq); ++q3)
            for (q2 = 0; q2 < (1+iq); ++q2)
            {
                pos = q2+q3*(1+iq)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = -1.0;
                this->bou_XI2q[pos] = Xq_1d[q2];
                this->bou_XI3q[pos] = Xq_1d[q3];
            }

            // Boundary with normal {+1,0,0}
            b = 1;
            for (q3 = 0; q3 < (1+iq); ++q3)
            for (q2 = 0; q2 < (1+iq); ++q2)
            {
                pos = q2+q3*(1+iq)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = +1.0;
                this->bou_XI2q[pos] = Xq_1d[q2];
                this->bou_XI3q[pos] = Xq_1d[q3];
            }

            // Boundary with normal {0,-1,0}
            b = 2;
            for (q3 = 0; q3 < (1+iq); ++q3)
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+q3*(1+iq)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1];
                this->bou_XI2q[pos] = -1.0;
                this->bou_XI3q[pos] = Xq_1d[q3];
            }

            // Boundary with normal {0,+1,0}
            b = 3;
            for (q3 = 0; q3 < (1+iq); ++q3)
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+q3*(1+iq)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1];
                this->bou_XI2q[pos] = +1.0;
                this->bou_XI3q[pos] = Xq_1d[q3];
            }

            // Boundary with normal {0,0,-1}
            b = 4;
            for (q2 = 0; q2 < (1+iq); ++q2)
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+q2*(1+iq)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1];
                this->bou_XI2q[pos] = Xq_1d[q2];
                this->bou_XI3q[pos] = -1.0;
            }

            // Boundary with normal {0,0,+1}
            b = 5;
            for (q2 = 0; q2 < (1+iq); ++q2)
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+q2*(1+iq)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1];
                this->bou_XI2q[pos] = Xq_1d[q2];
                this->bou_XI3q[pos] = +1.0;
            }

            // Weights
            for (q2 = 0; q2 < (1+iq); ++q2)
            for (q1 = 0; q1 < (1+iq); ++q1)
            {
                pos = q1+q2*(1+iq);
                this->bou_Wq[pos] = Wq_1d[q1]*Wq_1d[q2];
            }
            // --------------------------------------------------------
        }
        else
        {
            Print() << "ERROR: AMReX_DG_Definitions.H - UnitHyperrectangle::UnitHyperrectangle" << std::endl;
            Print() << "| Dimension = " << DIM << "not implemented yet" << std::endl;
            exit(-1);
        }
        // ============================================================

        // ALLOCATE MEMORY: BASIS FUNCTIONS ===========================
        // DOMAIN
        {
            int size = (this->dom_Nq)*(this->Np);

            this->dom_BF.resize(size);
            if (DIM > 0) this->dom_dBFdXI1.resize(size);
            if (DIM > 1) this->dom_dBFdXI2.resize(size);
            if (DIM > 2) this->dom_dBFdXI3.resize(size);
        }

        // BOUNDARIES
        {
            int size = (this->bou_Nq)*(this->Np)*(this->n_bou);

            this->bou_BF.resize(size);
            if (DIM > 0) this->bou_dBFdXI1.resize(size);
            if (DIM > 1) this->bou_dBFdXI2.resize(size);
            if (DIM > 2) this->bou_dBFdXI3.resize(size);
        }
        // ============================================================

        // STORE BASIS FUNCTIONS ======================================
        {
            // Auxiliary variables
            Vector<Real> XIlo = {-1.0, -1.0, -1.0};
            Vector<Real> XIhi = {+1.0, +1.0, +1.0};
            int size_1d = (this->dom_Nq)*(1+ip);
            Vector<Real> P1(size_1d), P2(size_1d), P3(size_1d);
            Vector<Real> dP1(size_1d), dP2(size_1d), dP3(size_1d);
            int pos, pos2;

            if (DIM == 1)
            {
                // DOMAIN -------------------------------------------------
                DG_space_1d_BF_Table_NxL(this->dom_Nq, ip,
                                         this->dom_XI1q.data(), XIlo[0], XIhi[0],
                                         P1.data(), dP1.data(),
                                         this->dom_BF.data(), this->dom_dBFdXI1.data());
                // --------------------------------------------------------

                // BOUNDARIES ---------------------------------------------
                for (int b = 0; b < (this->n_bou); ++b)
                {
                    pos = b*(this->bou_Nq);
                    pos2 = b*(this->bou_Nq)*(this->Np);

                    DG_space_1d_BF_Table_NxL(this->bou_Nq, ip,
                                             &this->bou_XI1q[pos], XIlo[0], XIhi[0],
                                             P1.data(), dP1.data(),
                                             &this->bou_BF[pos2], &this->bou_dBFdXI1[pos2]);
                }
                // --------------------------------------------------------
            }
            else if (DIM == 2)
            {
                // DOMAIN -------------------------------------------------
                DG_space_2d_BF_Table_NxL(this->dom_Nq, ip,
                                         this->dom_XI1q.data(), this->dom_XI2q.data(), XIlo.data(), XIhi.data(),
                                         P1.data(), dP1.data(), P2.data(), dP2.data(),
                                         this->dom_BF.data(), this->dom_dBFdXI1.data(), this->dom_dBFdXI2.data());
                // --------------------------------------------------------

                // BOUNDARIES ---------------------------------------------
                for (int b = 0; b < (this->n_bou); ++b)
                {
                    pos = b*(this->bou_Nq);
                    pos2 = b*(this->bou_Nq)*(this->Np);

                    DG_space_2d_BF_Table_NxL(this->bou_Nq, ip,
                                             &this->bou_XI1q[pos], &this->bou_XI2q[pos], XIlo.data(), XIhi.data(),
                                             P1.data(), dP1.data(), P2.data(), dP2.data(),
                                             &this->bou_BF[pos2], &this->bou_dBFdXI1[pos2], &this->bou_dBFdXI2[pos2]);
                }
                // --------------------------------------------------------
                
            }
            else if (DIM == 3)
            {
                // DOMAIN -------------------------------------------------
                DG_space_3d_BF_Table_NxL(this->dom_Nq, ip,
                                         this->dom_XI1q.data(), this->dom_XI2q.data(), this->dom_XI3q.data(), XIlo.data(), XIhi.data(),
                                         P1.data(), dP1.data(), P2.data(), dP2.data(), P3.data(), dP3.data(),
                                         this->dom_BF.data(), this->dom_dBFdXI1.data(), this->dom_dBFdXI2.data(), this->dom_dBFdXI3.data());
                // --------------------------------------------------------

                // BOUNDARIES ---------------------------------------------
                for (int b = 0; b < (this->n_bou); ++b)
                {
                    pos = b*(this->bou_Nq);
                    pos2 = b*(this->bou_Nq)*(this->Np);

                    DG_space_3d_BF_Table_NxL(this->bou_Nq, ip,
                                             &this->bou_XI1q[pos], &this->bou_XI2q[pos], &this->bou_XI3q[pos], XIlo.data(), XIhi.data(),
                                             P1.data(), dP1.data(), P2.data(), dP2.data(), P3.data(), dP3.data(),
                                             &this->bou_BF[pos2], &this->bou_dBFdXI1[pos2], &this->bou_dBFdXI2[pos2], &this->bou_dBFdXI3[pos2]);
                }
                // --------------------------------------------------------
            }
            else
            {
                Print() << "ERROR: AMReX_DG_Definitions.H - UnitHyperrectangle::UnitHyperrectangle" << std::endl;
                Print() << "| Dimension = " << DIM << "not implemented yet" << std::endl;
                exit(-1);
            }
        }
        // ============================================================

        // MASS MATRIX AND ITS CHOLESKY DECOMPOSITION AND INVERSE =====
        {
            int size = (this->Np)*(this->Np);
            int pos, posr, posc;

            this->MM.resize(size);
            this->MMCh.resize(size);
            this->iMM.resize(size);

            std::fill(this->MM.begin(), this->MM.end(), 0.0);
            std::fill(this->MMCh.begin(), this->MMCh.end(), 0.0);
            std::fill(this->iMM.begin(), this->iMM.end(), 0.0);

            // MASS MATRIX
            for (int c = 0; c < (this->Np); ++c)
            for (int r = 0; r < (this->Np); ++r)
            for (int k = 0; k < (this->dom_Nq); ++k)
            {
                pos = r+c*(this->Np);
                posr = k+r*(this->dom_Nq);
                posc = k+c*(this->dom_Nq);

                this->MM[pos] += this->dom_BF[posr]*this->dom_BF[posc]*this->dom_Wq[k];
            }

            // CHOLESKY DECOMPOSITION
            DG_utils::Cholesky(this->Np, this->MM.data(), this->MMCh.data());

            // INVERSE
            DG_utils::matinv(this->Np, this->MM.data(), this->iMM.data());

        }

        // DEBUG ------------------------------------------------------
        if (false)
        {
            Print() << "#######################################################################" << std::endl;
            Print() << "# REPORT ON REFERENCE SQUARE ELEMENT: " << std::endl;
            Print() << "# DIM: " << DIM << std::endl;
            Print() << "# p  : " << this->p << std::endl;
            Print() << "# Np : " << this->Np << std::endl;
            Print() << "# Mass Matrix: " << std::endl;
            DG_utils::PrintRealArray2D(this->Np, this->Np, this->MM.data());
            Print() << "# Cholesky decomposition of the mass matrix: " << std::endl;
            DG_utils::PrintRealArray2D(this->Np, this->Np, this->MMCh.data());
            Print() << "# Inverse of the mass matrix: " << std::endl;
            DG_utils::PrintRealArray2D(this->Np, this->Np, this->iMM.data());
            Print() << "#######################################################################" << std::endl;
        }
        // ------------------------------------------------------------
        // ============================================================

        // ALLOCATE MEMORY: UNIFORMLY-SPACE GRID ======================
        // DOMAIN
        if (DIM > 0) this->dom_XI1g.resize(this->dom_Ng);
        if (DIM > 1) this->dom_XI2g.resize(this->dom_Ng);
        if (DIM > 2) this->dom_XI3g.resize(this->dom_Ng);
        // ============================================================

        // POINTERS TO 1D GRID POINTS =================================
        const Real * Xg_1d;

        if      (this->g == 0) {Xg_1d = gau_lob_1_x;}
        else if (this->g == 1) {Xg_1d = gau_lob_2_x;}
        else if (this->g == 2) {Xg_1d = gau_lob_3_x;}
        else if (this->g == 3) {Xg_1d = gau_lob_4_x;}
        else if (this->g == 4) {Xg_1d = gau_lob_5_x;}
        else if (this->g == 5) {Xg_1d = gau_lob_6_x;}
        else if (this->g == 6) {Xg_1d = gau_lob_7_x;}
        else if (this->g == 7) {Xg_1d = gau_lob_8_x;}
        else if (this->g == 8) {Xg_1d = gau_lob_9_x;}
        else if (this->g == 9) {Xg_1d = gau_lob_10_x;}
        else if (this->g == 10) {Xg_1d = gau_lob_11_x;}
        else if (this->g == 11) {Xg_1d = gau_lob_12_x;}
        else if (this->g == 12) {Xg_1d = gau_lob_13_x;}
        else if (this->g == 13) {Xg_1d = gau_lob_14_x;}
        else if (this->g == 14) {Xg_1d = gau_lob_15_x;}
        else if (this->g == 15) {Xg_1d = gau_lob_16_x;}
        else
        {
            Print() << "ERROR: AMReX_DG_Definitions.H - UnitHyperrectangle::UnitHyperrectangle" << std::endl;
            Print() << "| Grid order = " << this->g << "not implemented yet" << std::endl;
            exit(-1);
        }
        // ============================================================

        // STORE GRID POINTS ==========================================
        if (DIM == 1)
        {
            // LOCAL VARIABLES
            int g1, pos;
            // ---------------

            // DOMAIN -------------------------------------------------
            for (g1 = 0; g1 < (1+this->g); ++g1)
            {
                pos = g1;
                this->dom_XI1g[pos] = Xg_1d[g1];
            }
            // --------------------------------------------------------
        }
        else if (DIM == 2)
        {
            // LOCAL VARIABLES
            int g1, g2, pos;
            // ---------------

            // DOMAIN -------------------------------------------------
            for (g2 = 0; g2 < (1+this->g); ++g2)
            for (g1 = 0; g1 < (1+this->g); ++g1)
            {
                pos = g1+g2*(1+this->g);
                this->dom_XI1g[pos] = Xg_1d[g1];
                this->dom_XI2g[pos] = Xg_1d[g2];
            }
            // --------------------------------------------------------
            
        }
        else if (DIM == 3)
        {
            // LOCAL VARIABLES -
            int g1, g2, g3, pos;
            // -----------------

            // DOMAIN -------------------------------------------------
            for (g3 = 0; g3 < (1+this->g); ++g3)
            for (g2 = 0; g2 < (1+this->g); ++g2)
            for (g1 = 0; g1 < (1+this->g); ++g1)
            {
                pos = g1+g2*(1+this->g)+g3*(1+this->g)*(1+this->g);
                this->dom_XI1g[pos] = Xg_1d[g1];
                this->dom_XI2g[pos] = Xg_1d[g2];
                this->dom_XI3g[pos] = Xg_1d[g3];
            }
            // --------------------------------------------------------
        }
        else
        {
            Print() << "ERROR: AMReX_DG_Definitions.H - UnitHyperrectangle::UnitHyperrectangle" << std::endl;
            Print() << "| Grid order = " << this->g << "not implemented yet" << std::endl;
            exit(-1);
        }
        // ============================================================

        // ALLOCATE MEMORY: BASIS FUNCTIONS ===========================
        // DOMAIN
        {
            int size = (this->dom_Ng)*(this->Np);

            this->grid_BF.resize(size);
        }
        // ============================================================

        // STORE BASIS FUNCTIONS ======================================
        {
            // Auxiliary variables
            Vector<Real> XIlo = {-1.0, -1.0, -1.0};
            Vector<Real> XIhi = {+1.0, +1.0, +1.0};
            int size_1d = (this->dom_Ng)*(1+ip);
            Vector<Real> P1(size_1d), P2(size_1d), P3(size_1d);

            if (DIM == 1)
            {
                // DOMAIN -------------------------------------------------
                DG_space_1d_BF_Table_NxL(this->dom_Ng, ip,
                                         this->dom_XI1g.data(), XIlo[0], XIhi[0],
                                         P1.data(),
                                         this->grid_BF.data());
                // --------------------------------------------------------

            }
            else if (DIM == 2)
            {
                // DOMAIN -------------------------------------------------
                DG_space_2d_BF_Table_NxL(this->dom_Ng, ip,
                                         this->dom_XI1g.data(), this->dom_XI2g.data(), XIlo.data(), XIhi.data(),
                                         P1.data(), P2.data(),
                                         this->grid_BF.data());
                // --------------------------------------------------------
                
            }
            else if (DIM == 3)
            {
                // DOMAIN -------------------------------------------------
                DG_space_3d_BF_Table_NxL(this->dom_Ng, ip,
                                         this->dom_XI1g.data(), this->dom_XI2g.data(), this->dom_XI3g.data(), XIlo.data(), XIhi.data(),
                                         P1.data(), P2.data(), P3.data(),
                                         this->grid_BF.data());
                // --------------------------------------------------------
            }
            else
            {
                Print() << "ERROR: AMReX_DG_Definitions.H - UnitHyperrectangle::UnitHyperrectangle" << std::endl;
                Print() << "| Dimension = " << DIM << "not implemented yet" << std::endl;
                exit(-1);
            }
        }
        // ============================================================
    }
    // ################################################################
};
// ####################################################################
// ####################################################################



// ####################################################################
// DATA STRUCTURES USED TO EVALUATE THE DG SOLUTION IN SPACE AND TIME #
// ####################################################################
// FOR ALGOIM
template<int N, int DG_MAX_p, int DG_MAX_Np>
class _algoim_DG_space_BFX_
{
private:
    // VARIABLES
    const Real * lo;
    const Real * hi;
    Array4<Real const> const & fab;
    const int p, Np;
    int i, j, k, u;
    Real sgn;

public:
    // CONSTRUCTOR ====================================================
    _algoim_DG_space_BFX_(const Real * lo_, const Real * hi_,
                          Array4<Real const> const & fab_,
                          const int & p_,
                          const int & i_, const int & j_, const int & k_, const int & u_):
    lo(lo_), hi(hi_),
    fab(fab_),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))),
    i(i_), j(j_), k(k_), u(u_),
    sgn(-1.0)
    {}
    // ================================================================

    // MISC ===========================================================
    void set_fab_ptrs(const int & i_, const int & j_, const int & k_, const int & u_)
    {
        this->i = i_;
        this->j = j_;
        this->k = k_;
        this->u = u_;
    }

    void set_sign(const int & sgn_)
    {
        this->sgn = 1.0*sgn_;
    }
    // ================================================================

    // EVAL THE SOLUTION AT LOCATION x ================================
    Real eval(const Real * x)
    {
        GpuArray<Real, N> x_array;
        AMREX_D_TERM
        (
            x_array[0] = x[0];,
            x_array[1] = x[1];,
            x_array[2] = x[2];
        )
        return this->operator()(x_array);
    }
    // ================================================================

    // EVAL THE SOLUTION'S GRADIENT AT LOCATION x =====================
    void eval_grad(const Real * x, Real * res)
    {
        GpuArray<Real, N> x_array, res_array;
        AMREX_D_TERM
        (
            x_array[0] = x[0];,
            x_array[1] = x[1];,
            x_array[2] = x[2];
        )
        res_array = this->grad(x_array);
        AMREX_D_TERM
        (
            res[0] = res_array[0];,
            res[1] = res_array[1];,
            res[2] = res_array[2];
        )
    }
    // ================================================================

    // METHODS REQUIRED BY ALGOIM =====================================
    template<typename T>
    T operator() (const GpuArray<T, N> & x) const
    {
        T T_BF[DG_MAX_Np];
        AMREX_D_TERM
        (
            T T_P1[1+DG_MAX_p];,
            T T_P2[1+DG_MAX_p];,
            T T_P3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            T T_dP1[1+DG_MAX_p];,
            T T_dP2[1+DG_MAX_p];,
            T T_dP3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            T T_dBFdx1[DG_MAX_Np];,
            T T_dBFdx2[DG_MAX_Np];,
            T T_dBFdx3[DG_MAX_Np];
        )
        T res;
        
        AMREX_D_PICK
        (
            DG_space_1d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], this->lo[0], this->hi[0],
                                             T_P1, T_dP1,
                                             T_BF, T_dBFdx1);,
            DG_space_2d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2,
                                             T_BF, T_dBFdx1, T_dBFdx2);,
            DG_space_3d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], &x[2], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2, T_P3, T_dP3,
                                             T_BF, T_dBFdx1, T_dBFdx2, T_dBFdx3);
        )

        res = 0.0;
        for (int c = 0; c < (this->Np); ++c)
            res += T_BF[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));

        res *= -this->sgn;

        return res;
    }

    template<typename T>
    GpuArray<T, N> grad(const GpuArray<T, N> & x) const
    {
        T T_BF[DG_MAX_Np];
        AMREX_D_TERM
        (
            T T_P1[1+DG_MAX_p];,
            T T_P2[1+DG_MAX_p];,
            T T_P3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            T T_dP1[1+DG_MAX_p];,
            T T_dP2[1+DG_MAX_p];,
            T T_dP3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            T T_dBFdx1[DG_MAX_Np];,
            T T_dBFdx2[DG_MAX_Np];,
            T T_dBFdx3[DG_MAX_Np];
        )
        GpuArray<T, N> res;
        
        AMREX_D_PICK
        (
            DG_space_1d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], this->lo[0], this->hi[0],
                                             T_P1, T_dP1,
                                             T_BF, T_dBFdx1);,
            DG_space_2d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2,
                                             T_BF, T_dBFdx1, T_dBFdx2);,
            DG_space_3d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], &x[2], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2, T_P3, T_dP3,
                                             T_BF, T_dBFdx1, T_dBFdx2, T_dBFdx3);
        )

        AMREX_D_TERM
        (
            res[0] = 0.0;,
            res[1] = 0.0;,
            res[2] = 0.0;
        )

        for (int c = 0; c < (this->Np); ++c)
        {
            AMREX_D_TERM
            (
                res[0] += T_dBFdx1[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[1] += T_dBFdx2[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[2] += T_dBFdx3[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));
            )
        }

        AMREX_D_TERM
        (
            res[0] *= -this->sgn;,
            res[1] *= -this->sgn;,
            res[2] *= -this->sgn;
        )

        return res;
    }
    // ================================================================
};

typedef _algoim_DG_space_BFX_<AMREX_SPACEDIM, DG_PHI_MAX_SPACE_p, DG_PHI_MAX_SPACE_Np> algoim_DG_phi_space_BFX;


template<int N, int DG_MAX_p, int DG_MAX_Np>
class _DG_space_BF_
{
private:
    // VARIABLES
    const Real * lo;
    const Real * hi;
    const int p, Np;

public:
    Real BF[DG_MAX_Np];
    AMREX_D_TERM
    (
        Real P1[1+DG_MAX_p];,
        Real P2[1+DG_MAX_p];,
        Real P3[1+DG_MAX_p];
    )
    AMREX_D_TERM
    (
        Real dP1[1+DG_MAX_p];,
        Real dP2[1+DG_MAX_p];,
        Real dP3[1+DG_MAX_p];
    )
    AMREX_D_TERM
    (
        Real dBFdx1[DG_MAX_Np];,
        Real dBFdx2[DG_MAX_Np];,
        Real dBFdx3[DG_MAX_Np];
    )

public:
    // CONSTRUCTOR ====================================================
    AMREX_GPU_HOST_DEVICE
    _DG_space_BF_(const Real * lo_, const Real * hi_, const int & p_):
    lo(lo_), hi(hi_),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_)))
    {}

    AMREX_GPU_HOST_DEVICE
    _DG_space_BF_(const int & p_):
    lo(nullptr), hi(nullptr),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_)))
    {}
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE
    void set_support(const Real * lo_, const Real * hi_)
    {
        this->lo = lo_;
        this->hi = hi_;
    }
    // ================================================================

    // EVAL THE BASIS FUNCTIONS =======================================
    AMREX_GPU_HOST_DEVICE
    void _eval_BF_table_(const Real * x)
    {
        AMREX_D_PICK
        (
            DG_space_1d_BF_Table_NxL(1, this->p,
                                     x, this->lo[0], this->hi[0],
                                     this->P1, this->dP1,
                                     this->BF, this->dBFdx1);,
            DG_space_2d_BF_Table_NxL(1, this->p,
                                     &x[0], &x[1], this->lo, this->hi,
                                     this->P1, this->dP1, this->P2, this->dP2,
                                     this->BF, this->dBFdx1, this->dBFdx2);,
            DG_space_3d_BF_Table_NxL(1, this->p,
                                     &x[0], &x[1], &x[2], this->lo, this->hi,
                                     this->P1, this->dP1, this->P2, this->dP2, this->P3, this->dP3,
                                     this->BF, this->dBFdx1, this->dBFdx2, this->dBFdx3);
        )
    }
    // ================================================================
};

typedef _DG_space_BF_<AMREX_SPACEDIM, DG_PHI_MAX_SPACE_p, DG_PHI_MAX_SPACE_Np> DG_phi_space_BF;
typedef _DG_space_BF_<AMREX_SPACEDIM, DG_SOL_MAX_SPACE_p, DG_SOL_MAX_SPACE_Np> DG_sol_space_BF;


template<int N, int DG_MAX_p, int DG_MAX_Np>
class _DG_space_BFX_
{
private:
    // VARIABLES
    const Real * lo;
    const Real * hi;
    Array4<Real const> const & fab;
    const int p, Np;
    int i, j, k, u;
    Real sgn;

public:
    Real BF[DG_MAX_Np];
    AMREX_D_TERM
    (
        Real P1[1+DG_MAX_p];,
        Real P2[1+DG_MAX_p];,
        Real P3[1+DG_MAX_p];
    )
    AMREX_D_TERM
    (
        Real dP1[1+DG_MAX_p];,
        Real dP2[1+DG_MAX_p];,
        Real dP3[1+DG_MAX_p];
    )
    AMREX_D_TERM
    (
        Real dBFdx1[DG_MAX_Np];,
        Real dBFdx2[DG_MAX_Np];,
        Real dBFdx3[DG_MAX_Np];
    )

public:
    // CONSTRUCTOR ====================================================
    AMREX_GPU_HOST_DEVICE
    _DG_space_BFX_(const Real * lo_, const Real * hi_,
                   Array4<Real const> const & fab_,
                   const int & p_,
                   const int & i_, const int & j_, const int & k_, const int & u_):
    lo(lo_), hi(hi_),
    fab(fab_),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))),
    i(i_), j(j_), k(k_), u(u_),
    sgn(-1.0)
    {}
    
    AMREX_GPU_HOST_DEVICE
    _DG_space_BFX_(Array4<Real const> const & fab_, const int & p_):
    lo(nullptr), hi(nullptr),
    fab(fab_),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))),
    i(-1), j(-1), k(-1), u(-1),
    sgn(-1.0)
    {}
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE
    void set_support(const Real * lo_, const Real * hi_)
    {
        this->lo = lo_;
        this->hi = hi_;
    }
    
    AMREX_GPU_HOST_DEVICE
    void set_fab_ptrs(const int & i_, const int & j_, const int & k_, const int & u_)
    {
        this->i = i_;
        this->j = j_;
        this->k = k_;
        this->u = u_;
    }

    AMREX_GPU_HOST_DEVICE
    void set_sign(const int & sgn_)
    {
        this->sgn = 1.0*sgn_;
    }
    // ================================================================

    // EVAL THE SOLUTION AT LOCATION x ================================
    AMREX_GPU_HOST_DEVICE
    Real eval(const Real * x)
    {
        this->_eval_BF_table_(x);

        return this->_eval_unsafe_();
    }

    AMREX_GPU_HOST_DEVICE
    Real _eval_unsafe_()
    {
        Real res;

        res = 0.0;
        for (int c = 0; c < (this->Np); ++c)
            res += this->BF[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));

        res *= -this->sgn;

        return res;
    }
    // ================================================================

    // EVAL THE SOLUTION'S GRADIENT AT LOCATION x =====================
    AMREX_GPU_HOST_DEVICE
    void eval_grad(const Real * x, Real * res)
    {
        this->_eval_BF_table_(x);

        this->_eval_grad_unsafe_(&res[0]);
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_grad_unsafe_(Real * res)
    {
        AMREX_D_TERM
        (
            res[0] = 0.0;,
            res[1] = 0.0;,
            res[2] = 0.0;
        )

        for (int c = 0; c < (this->Np); ++c)
        {
            AMREX_D_TERM
            (
                res[0] += this->dBFdx1[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[1] += this->dBFdx2[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[2] += this->dBFdx3[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));
            )
        }

        AMREX_D_TERM
        (
            res[0] *= -this->sgn;,
            res[1] *= -this->sgn;,
            res[2] *= -this->sgn;
        )
    }
    // ================================================================

    // EVAL THE BASIS FUNCTIONS =======================================
    AMREX_GPU_HOST_DEVICE
    void _eval_BF_table_(const Real * x)
    {
        AMREX_D_PICK
        (
            DG_space_1d_BF_Table_NxL(1, this->p,
                                     x, this->lo[0], this->hi[0],
                                     this->P1, this->dP1,
                                     this->BF, this->dBFdx1);,
            DG_space_2d_BF_Table_NxL(1, this->p,
                                     &x[0], &x[1], this->lo, this->hi,
                                     this->P1, this->dP1, this->P2, this->dP2,
                                     this->BF, this->dBFdx1, this->dBFdx2);,
            DG_space_3d_BF_Table_NxL(1, this->p,
                                     &x[0], &x[1], &x[2], this->lo, this->hi,
                                     this->P1, this->dP1, this->P2, this->dP2, this->P3, this->dP3,
                                     this->BF, this->dBFdx1, this->dBFdx2, this->dBFdx3);
        )
    }
    // ================================================================
};

typedef _DG_space_BFX_<AMREX_SPACEDIM, DG_PHI_MAX_SPACE_p, DG_PHI_MAX_SPACE_Np> DG_phi_space_BFX;
typedef _DG_space_BFX_<AMREX_SPACEDIM, DG_SOL_MAX_SPACE_p, DG_SOL_MAX_SPACE_Np> DG_sol_space_BFX;


template<int N, int DG_MAX_p, int DG_MAX_Np>
class _DG_MEM_space_BF_
{
private:
    // VARIABLES
    const int p, Np;
    const Real * BF_ptr;
    const Real * dBFdXI1_ptr;
    const Real * dBFdXI2_ptr;
    const Real * dBFdXI3_ptr;
    const Real * iJ;

public:
    Real BF[DG_MAX_Np];
    AMREX_D_TERM
    (
        Real dBFdx1[DG_MAX_Np];,
        Real dBFdx2[DG_MAX_Np];,
        Real dBFdx3[DG_MAX_Np];
    )

public:
    // CONSTRUCTOR ====================================================
    AMREX_GPU_HOST_DEVICE
    _DG_MEM_space_BF_(const int & p_,
                      const Real * BF_ptr_, AMREX_D_DECL(const Real * dBFdXI1_ptr_, const Real * dBFdXI2_ptr_, const Real * dBFdXI3_ptr_),
                      const Real * iJ_):
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))),
    BF_ptr(BF_ptr_),
    AMREX_D_DECL
    (
        dBFdXI1_ptr(dBFdXI1_ptr_),
        dBFdXI2_ptr(dBFdXI2_ptr_),
        dBFdXI3_ptr(dBFdXI3_ptr_)
    ),
    iJ(iJ_)
    {}
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE
    void set_BF_ptrs(const Real * BF_ptr_)
    {
        this->BF_ptr = BF_ptr_;
        AMREX_D_TERM
        (
            this->dBFdXI1_ptr = nullptr;,
            this->dBFdXI2_ptr = nullptr;,
            this->dBFdXI3_ptr = nullptr;
        )
    }
    
    AMREX_GPU_HOST_DEVICE
    void set_BF_ptrs(const Real * BF_ptr_, AMREX_D_DECL(const Real * dBFdXI1_ptr_, const Real * dBFdXI2_ptr_, const Real * dBFdXI3_ptr_))
    {
        this->BF_ptr = BF_ptr_;
        AMREX_D_TERM
        (
            this->dBFdXI1_ptr = dBFdXI1_ptr_;,
            this->dBFdXI2_ptr = dBFdXI2_ptr_;,
            this->dBFdXI3_ptr = dBFdXI3_ptr_;
        )
    }
    // ================================================================

    // EVAL THE BASIS FUNCTIONS =======================================
    AMREX_GPU_HOST_DEVICE
    void _eval_BF_table_(const int & q)
    {
        for (int c = 0; c < (this->Np); ++c)
        {
            this->BF[c] = BF_ptr[c+q*(this->Np)];
            AMREX_D_TERM
            (
                this->dBFdx1[c] = dBFdXI1_ptr[c+q*(this->Np)]*(this->iJ[0]);,
                this->dBFdx2[c] = dBFdXI2_ptr[c+q*(this->Np)]*(this->iJ[1]);,
                this->dBFdx3[c] = dBFdXI3_ptr[c+q*(this->Np)]*(this->iJ[2]);
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_BF_only_table_(const int & q)
    {
        for (int c = 0; c < (this->Np); ++c)
        {
            this->BF[c] = BF_ptr[c+q*(this->Np)];
        }
    }
    // ================================================================
};

typedef _DG_MEM_space_BF_<AMREX_SPACEDIM, DG_SOL_MAX_SPACE_p, DG_SOL_MAX_SPACE_Np> DG_mem_sol_space_BF;


template<int N, int DG_MAX_p, int DG_MAX_Np>
class _DG_MEM_space_BFX_
{
private:
    // VARIABLES
    Array4<Real const> const & fab;
    const int p, Np;
    int i, j, k, u;
    const Real * BF_ptr;
    const Real * dBFdXI1_ptr;
    const Real * dBFdXI2_ptr;
    const Real * dBFdXI3_ptr;
    const Real * iJ;

public:
    Real BF[DG_MAX_Np];
    AMREX_D_TERM
    (
        Real dBFdx1[DG_MAX_Np];,
        Real dBFdx2[DG_MAX_Np];,
        Real dBFdx3[DG_MAX_Np];
    )

public:
    // CONSTRUCTOR ====================================================
    AMREX_GPU_HOST_DEVICE
    _DG_MEM_space_BFX_(Array4<Real const> const & fab_,
                       const int & p_,
                       const int & i_, const int & j_, const int & k_, const int & u_,
                       const Real * BF_ptr_, AMREX_D_DECL(const Real * dBFdXI1_ptr_, const Real * dBFdXI2_ptr_, const Real * dBFdXI3_ptr_),
                       const Real * iJ_):
    fab(fab_),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))),
    i(i_), j(j_), k(k_), u(u_),
    BF_ptr(BF_ptr_),
    AMREX_D_DECL
    (
        dBFdXI1_ptr(dBFdXI1_ptr_),
        dBFdXI2_ptr(dBFdXI2_ptr_),
        dBFdXI3_ptr(dBFdXI3_ptr_)
    ),
    iJ(iJ_)
    {}

    AMREX_GPU_HOST_DEVICE
    _DG_MEM_space_BFX_(Array4<Real const> const & fab_,
                       const int & p_,
                       const int & i_, const int & j_, const int & k_, const int & u_,
                       const Real * iJ_):
    fab(fab_),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))),
    i(i_), j(j_), k(k_), u(u_),
    BF_ptr(nullptr),
    AMREX_D_DECL
    (
        dBFdXI1_ptr(nullptr),
        dBFdXI2_ptr(nullptr),
        dBFdXI3_ptr(nullptr)
    ),
    iJ(iJ_)
    {}
    
    AMREX_GPU_HOST_DEVICE
    _DG_MEM_space_BFX_(Array4<Real const> const & fab_,
                       const int & p_,
                       const Real * iJ_):
    fab(fab_),
    p(p_),
    Np(AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))),
    i(-1), j(-1), k(-1), u(-1),
    BF_ptr(nullptr),
    AMREX_D_DECL
    (
        dBFdXI1_ptr(nullptr),
        dBFdXI2_ptr(nullptr),
        dBFdXI3_ptr(nullptr)
    ),
    iJ(iJ_)
    {}
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE
    void set_fab_ptrs(const int & i_, const int & j_, const int & k_, const int & u_)
    {
        this->i = i_;
        this->j = j_;
        this->k = k_;
        this->u = u_;
    }

    AMREX_GPU_HOST_DEVICE
    void set_BF_ptrs(const Real * BF_ptr_)
    {
        this->BF_ptr = BF_ptr_;
        AMREX_D_TERM
        (
            this->dBFdXI1_ptr = nullptr;,
            this->dBFdXI2_ptr = nullptr;,
            this->dBFdXI3_ptr = nullptr;
        )
    }
    
    AMREX_GPU_HOST_DEVICE
    void set_BF_ptrs(const Real * BF_ptr_, AMREX_D_DECL(const Real * dBFdXI1_ptr_, const Real * dBFdXI2_ptr_, const Real * dBFdXI3_ptr_))
    {
        this->BF_ptr = BF_ptr_;
        AMREX_D_TERM
        (
            this->dBFdXI1_ptr = dBFdXI1_ptr_;,
            this->dBFdXI2_ptr = dBFdXI2_ptr_;,
            this->dBFdXI3_ptr = dBFdXI3_ptr_;
        )
    }
    // ================================================================

    // EVAL THE SOLUTION AT LOCATION x ================================
    AMREX_GPU_HOST_DEVICE
    Real eval(const int & q)
    {
        this->_eval_BF_table_(q);

        return this->_eval_unsafe_();
    }

    AMREX_GPU_HOST_DEVICE
    Real _eval_unsafe_()
    {
        Real res;

        res = 0.0;
        for (int c = 0; c < (this->Np); ++c)
            res += this->BF[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));

        return res;
    }
    // ================================================================

    // EVAL THE SOLUTION'S GRADIENT AT LOCATION x =====================
    AMREX_GPU_HOST_DEVICE
    void eval_grad(const int & q, Real * res)
    {
        this->_eval_BF_table_(q);

        this->_eval_grad_unsafe_(&res[0]);
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_grad_unsafe_(Real * res)
    {
        AMREX_D_TERM
        (
            res[0] = 0.0;,
            res[1] = 0.0;,
            res[2] = 0.0;
        )

        for (int c = 0; c < (this->Np); ++c)
        {
            AMREX_D_TERM
            (
                res[0] += this->dBFdx1[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[1] += this->dBFdx2[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[2] += this->dBFdx3[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));
            )
        }
    }
    // ================================================================

    // EVAL THE BASIS FUNCTIONS =======================================
    AMREX_GPU_HOST_DEVICE
    void _eval_BF_table_(const int & q)
    {
        for (int c = 0; c < (this->Np); ++c)
        {
            this->BF[c] = BF_ptr[c+q*(this->Np)];
            AMREX_D_TERM
            (
                this->dBFdx1[c] = dBFdXI1_ptr[c+q*(this->Np)]*(this->iJ[0]);,
                this->dBFdx2[c] = dBFdXI2_ptr[c+q*(this->Np)]*(this->iJ[1]);,
                this->dBFdx3[c] = dBFdXI3_ptr[c+q*(this->Np)]*(this->iJ[2]);
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_BF_only_table_(const int & q)
    {
        for (int c = 0; c < (this->Np); ++c)
        {
            this->BF[c] = BF_ptr[c+q*(this->Np)];
        }
    }
    // ================================================================
};

typedef _DG_MEM_space_BFX_<AMREX_SPACEDIM, DG_SOL_MAX_SPACE_p, DG_SOL_MAX_SPACE_Np> DG_mem_sol_space_BFX;

// ####################################################################
// ####################################################################



// ####################################################################
// EXPORT ROUTINES ####################################################
// ####################################################################
std::string GetPlotFilepath(const std::string & dst_folder,
                            const std::string & filename_root,
                            const int & time_id,
                            const int & time_id_max);

std::string GetOutputFilepath(const std::string & dst_folder,
                              const std::string & filename_root,
                              const int & time_id,
                              const int & time_id_max);

Gpu::ManagedVector<int> Get_VTK_BaseCell_Connectivity(const int & ne);
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif
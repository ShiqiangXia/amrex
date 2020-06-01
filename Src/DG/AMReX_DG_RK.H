// AMReX_DG_RK.H

#ifndef BL_DG_RK_H_
#define BL_DG_RK_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_MatrixFactory.H>
#include <AMReX_DG_ImplicitGeometry.H>

namespace amrex
{
namespace DG
{

// ####################################################################
// RUNGE-KUTTA CLASS ##################################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
struct RungeKutta
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;

    // PARAMETERS

    // VARIABLES
    Array<MultiFab, DG_MAX_RK_ORDER> X, dXdt;
    // ################################################################

    // CONSTRUCTOR ####################################################
    // ################################################################

    // DESTRUCTOR #####################################################
    // ################################################################

    // INITIALIZATION #################################################
    void InitData(const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                  const MatrixFactory<N_PHI, N_DOM> & MatFactory)
    {
        // PARAMETERS ===========================
        const int space_Np = MatFactory.space_Np;
        const int X_n_comp = space_Np*N_U;
        // ======================================

        // MULTIFAB CONTAINING RUNGE-KUTTA MULTISTEP SOLUTIONS ========
        for (int k = 0; k <= MatFactory.time_p; ++k)
        {
            this->X[k].define(iGeom.cc_ba, iGeom.dm, X_n_comp, this->dom_data_ngr);
            this->dXdt[k].define(iGeom.cc_ba, iGeom.dm, X_n_comp, this->dom_data_ngr);

            this->X[k] = 0.0;
            this->dXdt[k] = 0.0;
        }
        // ============================================================
    }
    // ################################################################

    // TIME-STEP: HYPERBOLIC ##########################################
    template<typename IBVP_CLASS>
    void TakeTimeStep_Hyperbolic(const int & RK_order,
                                 const Real & dt,
                                 const Real & t,
                                 const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                                 const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                                 const MultiFab & X_old,
                                 MultiFab & X_new,
                                 const IBVP_CLASS & IBVP)
    {
        // PARAMETERS =================================================
        const int ngr = 1;
        const int space_Np = MatFactory.space_Np;
        const int X_n_comp = space_Np*N_U;
        // ============================================================

        // UPDATE ACCORDING TO SELECTED RUNGE-KUTTA ORDER =============
        if (RK_order == 1)
        {
            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, X_old, this->dXdt[0], IBVP);

            MultiFab::Saxpy(X_new, dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
        }
        else if (RK_order == 2)
        {
            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, X_old, this->dXdt[0], IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[1], this->dXdt[1], IBVP);

            MultiFab::Saxpy(X_new, 0.5*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, 0.5*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);
        }
        else if (RK_order == 3)
        {
            /*
            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, X_old, this->dXdt[0], IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[1], this->dXdt[1], IBVP);

            MultiFab::Copy(this->X[2], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.25*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.25*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[2], this->dXdt[2], IBVP);

            MultiFab::Saxpy(X_new,     dt/6.0, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,     dt/6.0, this->dXdt[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, 2.0*dt/3.0, this->dXdt[2], 0, 0, X_n_comp, ngr);
            */

            // SX++++++++++++++++++
            // SSP RK
            // X_new = X_old   u0
            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, X_old, this->dXdt[0], IBVP); 
            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr); // u1

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[1], this->dXdt[1], IBVP);
            MultiFab::Copy(this->X[2], this->X[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], dt, this->dXdt[1], 0, 0, X_n_comp, ngr); // u2


            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[2], this->dXdt[2], IBVP);
            
             MultiFab::Copy(this->X[0], this->X[2], 0, 0, X_n_comp, ngr);
             MultiFab::Saxpy(this->X[0], dt, this->dXdt[2], 0, 0, X_n_comp, ngr); // u3

            // u_new = 1/3 u0 + 1/2 u1 + 1/6 u3
            
             MultiFab::Saxpy(X_new,    (1.0/3.0 - 1.0), X_old, 0, 0, X_n_comp, ngr);
             MultiFab::Saxpy(X_new,    1.0/2.0, this->X[1], 0, 0, X_n_comp, ngr);
             MultiFab::Saxpy(X_new,    1.0/6.0, this->X[0], 0, 0, X_n_comp, ngr);

             
             

            /*
            MultiFab::Saxpy(X_new,    2.0*dt/3.0, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,     dt/6.0, this->dXdt[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,  dt/6.0, this->dXdt[2], 0, 0, X_n_comp, ngr);
            */

            //+++++++++++++++++++++++++
        }
        else if (RK_order == 4)
        {
            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, X_old, this->dXdt[0], IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], 0.5*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[1], this->dXdt[1], IBVP);

            MultiFab::Copy(this->X[2], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.5*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[2], this->dXdt[2], IBVP);

            MultiFab::Copy(this->X[3], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[3], dt, this->dXdt[2], 0, 0, X_n_comp, ngr);

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[3], this->dXdt[3], IBVP);

            MultiFab::Saxpy(X_new, dt/6.0, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/3.0, this->dXdt[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/3.0, this->dXdt[2], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/6.0, this->dXdt[3], 0, 0, X_n_comp, ngr);
        }
        else if (RK_order == 5)
        {
            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, X_old, this->dXdt[0], IBVP); 
            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr); // u1

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[1], this->dXdt[1], IBVP);
            MultiFab::Copy(this->X[2], this->X[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], dt, this->dXdt[1], 0, 0, X_n_comp, ngr); // u2


            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[2], this->dXdt[2], IBVP);
            MultiFab::Copy(this->X[3], this->X[2], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[3], dt, this->dXdt[2], 0, 0, X_n_comp, ngr); // u3

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[3], this->dXdt[3], IBVP);
            MultiFab::Copy(this->X[4], this->X[3], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[4], dt, this->dXdt[3], 0, 0, X_n_comp, ngr); // u4

            this->Eval_dXdt_Hyperbolic(t, iGeom, MatFactory, this->X[4], this->dXdt[4], IBVP);
            MultiFab::Copy(this->X[0], this->X[4], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[0], dt, this->dXdt[4], 0, 0, X_n_comp, ngr); // u5

            // u_new = 11/30 u0 + 3/8 u1 + 1/6 u2 + 1/12 u3 + 1/120 u5

            MultiFab::Saxpy(X_new,    (11.0/30.0 - 1.0), X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,    3.0/8.0, this->X[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,    1.0/6.0, this->X[2], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,    1.0/12.0, this->X[3], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,    1.0/120.0, this->X[0], 0, 0, X_n_comp, ngr);


        }
        else
        {
            Print() << "ERROR: AMReX_DG_RK.H - RungeKutta::TakeTimeStep_Hyperbolic" << std::endl;
            Print() << "| Unexpected order: " << RK_order << std::endl;
            exit(-1);
        }
        // ============================================================
    }
    // ################################################################

    // TIME-DERIVATIVE EVALUATION: HYPERBOLIC #########################
    template<typename IBVP_CLASS>
    void Eval_dXdt_Hyperbolic(const Real & t,
                              const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                              const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                              const MultiFab & _X_,
                              MultiFab & _dXdt_,
                              const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("RungeKutta::Eval_dXdt_Hyperbolic(const Real &, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const Real iaJ = iGeom._iaJ;

        // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
        const Real * space_MMCh_mem_ptr = MatFactory.space_MMCh_device_mem.data();
#else
        const Real * space_MMCh_mem_ptr = MatFactory.space_MMCh_host_mem.data();
#endif

        // DG
        const int space_Np = MatFactory.space_Np;
        // ============================================================

        // INITIALIZATION =============================================
        _dXdt_ = 0.0;
        // ============================================================

        // DOMAIN AND BOUNDARY CONTRIBUTION ===========================
        this->Add_DomContrib_Hyperbolic(t, iGeom, MatFactory, _X_, _dXdt_, IBVP);
        this->Add_BouContrib_Hyperbolic(t, iGeom, MatFactory, _X_, _dXdt_, IBVP);
        // ============================================================

        // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX
        for (MFIter mfi(_X_); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<long const> const & space_MMCh_pos_fab = MatFactory.space_MMCh_pos.array(mfi);
            Array4<Real> const & dXdt_fab = _dXdt_.array(mfi);

            ParallelFor(bx, N_U,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const int dom = IBVP.F_U2DOM(ru);
                const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
                const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                const bool elm_is_valid = elm_is_entire || elm_is_large;
                const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

                // MASS MATRIX
                const long pos = space_MMCh_pos_fab(i,j,k,dom);
                const Real * space_MMCh_ptr = &space_MMCh_mem_ptr[pos];

                // LOCAL VARIABLES
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_etype, nbr_merged_b;
                bool nbr_is_small;
                Real dXhdt[DG_SOL_MAX_SPACE_Np];

                // EXTENDED ELEMENTS
                if (elm_is_extended)
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    if (nbr_merged_b == nbr_b)
                    {
                        for (int rs = 0; rs < space_Np; ++rs)
                        {
                            dXdt_fab(i,j,k,rs+ru*space_Np) += dXdt_fab(nbr_i,nbr_j,nbr_k,rs+ru*space_Np);
                        }
                    }
                }
                // IF VALID, MULTIPLY BY THE INVERSE OF THE MASS MATRIX
                if (elm_is_valid)
                {
                    // Copy from memory. Remember that the mass matrix
                    // is that of an element defined over the
                    // [-1,1]^AMREX_SPACEDIM domain.
                    for (int rs = 0; rs < space_Np; ++rs)
                        dXhdt[rs] = dXdt_fab(i,j,k,rs+ru*space_Np)*iaJ;

                    // Actual matrix operation (using mass matrix'
                    // Cholesky decomposition)
                    DG_utils::Cholesky_solve_overwrite(space_Np, 1, space_MMCh_ptr, dXhdt);

                    // Copy back to memory
                    for (int rs = 0; rs < space_Np; ++rs)
                        dXdt_fab(i,j,k,rs+ru*space_Np) = dXhdt[rs];
                }
            });
            Gpu::synchronize();
        }
        _dXdt_.FillBoundary(iGeom.geom.periodicity());
        // ============================================================
    }
    // ################################################################

    // TIME-DERIVATIVE EVALUATION: HYPERBOLIC (DOMAIN CONTRIBUTION) ###
    template<typename IBVP_CLASS>
    void Add_DomContrib_Hyperbolic(const Real & t,
                                   const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                                   const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                                   const MultiFab & _X_,
                                   MultiFab & _dXdt_,
                                   const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("RungeKutta::Add_DomContrib_Hyperbolic(const Real &, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = iGeom._dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = iGeom._prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = iGeom._J;
        const GpuArray<Real, AMREX_SPACEDIM> iJ = iGeom._iJ;
        const Real aJ = iGeom._aJ;

        // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
        const Real * eDomQuad_mem_ptr = iGeom.eDomQuad_device_mem.data();
        const Real * space_Dom_BF_mem_ptr = MatFactory.space_Dom_BF_device_mem.data();
        AMREX_D_TERM
        (
            const Real * space_Dom_dBFdXI1_mem_ptr = MatFactory.space_Dom_dBFdXI1_device_mem.data();,
            const Real * space_Dom_dBFdXI2_mem_ptr = MatFactory.space_Dom_dBFdXI2_device_mem.data();,
            const Real * space_Dom_dBFdXI3_mem_ptr = MatFactory.space_Dom_dBFdXI3_device_mem.data();
        )
#else
        const Real * eDomQuad_mem_ptr = iGeom.eDomQuad_host_mem.data();
        const Real * space_Dom_BF_mem_ptr = MatFactory.space_Dom_BF_host_mem.data();
        AMREX_D_TERM
        (
            const Real * space_Dom_dBFdXI1_mem_ptr = MatFactory.space_Dom_dBFdXI1_host_mem.data();,
            const Real * space_Dom_dBFdXI2_mem_ptr = MatFactory.space_Dom_dBFdXI2_host_mem.data();,
            const Real * space_Dom_dBFdXI3_mem_ptr = MatFactory.space_Dom_dBFdXI3_host_mem.data();
        )
#endif

        // DG
        const int sp = MatFactory.space_p;
        const int sNp = MatFactory.space_Np;
        // ============================================================

        // ============================================================
        //
        // int_{Vh} V_{,i}.T F_i
        //
        // ============================================================
        for (MFIter mfi(_X_); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<int const> const & eDomQuad_Nq_fab = iGeom.eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = iGeom.eDomQuad_pos.array(mfi);
            Array4<long const> const & space_Dom_BF_pos_fab = MatFactory.space_Dom_BF_pos.array(mfi);
            Array4<Real const> const & X_fab = _X_.array(mfi);
            Array4<Real> const & dXdt_fab = _dXdt_.array(mfi);

            ParallelFor(bx, N_U,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const int dom = IBVP.F_U2DOM(ru);
                const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};

                // QUADRATURE INFO
                const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                const long eDomQuad_pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                const Real * xptr = &eDomQuad_mem_ptr[eDomQuad_pos];

                // BASIS FUNCTIONS INFO
                const long space_BF_pos = space_Dom_BF_pos_fab(i,j,k,DG_DOM_BF_INFO_N_COMP_PER_DOM*dom);
                const Real * BF_ptr = &space_Dom_BF_mem_ptr[space_BF_pos];
                AMREX_D_TERM
                (
                    const Real * dBFdXI1_ptr = &space_Dom_dBFdXI1_mem_ptr[space_BF_pos];,
                    const Real * dBFdXI2_ptr = &space_Dom_dBFdXI2_mem_ptr[space_BF_pos];,
                    const Real * dBFdXI3_ptr = &space_Dom_dBFdXI3_mem_ptr[space_BF_pos];
                )

                // LOCAL VARIABLES
                long qi_pos;
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM], w, grad[AMREX_SPACEDIM];
                Real Uh[N_U];
                Real AMREX_D_DECL(dUhdx1[N_U], dUhdx2[N_U], dUhdx3[N_U]);
                Real AMREX_D_DECL(F1h[N_U], F2h[N_U], F3h[N_U]);
                Real integrand;

                // Get the support of the basis functions
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // Data structures used to compute the basis functions
                // and the unknown fields
                DG_mem_sol_space_BFX sol(X_fab, sp, BF_i, BF_j, BF_k, -1, BF_ptr, AMREX_D_DECL(dBFdXI1_ptr, dBFdXI2_ptr, dBFdXI3_ptr), &iJ[0]);

                // COMPUTE THE INTEGRAL
                for (int qi = 0; qi < dom_Nq; ++qi)
                {
                    // QUADRATURE POINT AND WEIGHT
                    qi_pos = (AMREX_SPACEDIM+1)*qi;
                    AMREX_D_TERM
                    (
                        x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                        x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                        x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                    )
                    w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;
                    
                    // SOLUTION
                    sol._eval_BF_table_(qi);
                    for (int cu = 0; cu < N_U; ++cu)
                    {
                        sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);
                        
                        Uh[cu] = sol._eval_unsafe_();

                        sol._eval_grad_unsafe_(grad);
                        AMREX_D_TERM
                        (
                            dUhdx1[cu] = grad[0];,
                            dUhdx2[cu] = grad[1];,
                            dUhdx3[cu] = grad[2];
                        );
                    }
                    
                    // FLUX
                    IBVP.F_F(t, x, Uh, AMREX_D_DECL(F1h, F2h, F3h));
                    
                    // DG CONTRIBUTION
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        AMREX_D_TERM
                        (
                            integrand  = sol.dBFdx1[rs]*F1h[ru];,
                            integrand += sol.dBFdx2[rs]*F2h[ru];,
                            integrand += sol.dBFdx3[rs]*F3h[ru];
                        )

                        dXdt_fab(i,j,k,rs+ru*sNp) += integrand*w;
                    }
                }
            });
            Gpu::synchronize();
        }
        // IMPORTANT NOTE
        // We are calling FillBoundary after we add the contribution of
        // the boundary integrals.
        //
        //_dXdt_.FillBoundary(iGeom.geom.periodicity());
        // ============================================================
    }
    // ################################################################

    // TIME-DERIVATIVE EVALUATION: HYPERBOLIC (BOUNDARY CONTRIBUTION) #
    template<typename IBVP_CLASS>
    void Add_BouContrib_Hyperbolic(const Real & t,
                                   const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                                   const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                                   const MultiFab & _X_,
                                   MultiFab & _dXdt_,
                                   const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("RungeKutta::Add_BouContrib_Hyperbolic(const Real &, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const Box & domain = iGeom.geom.Domain();
        const GpuArray<Real, AMREX_SPACEDIM> dx = iGeom._dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = iGeom._prob_lo;
        const GpuArray<int, AMREX_SPACEDIM> is_periodic = iGeom._is_periodic;
        const GpuArray<Real, AMREX_SPACEDIM> J = iGeom._J;
        const GpuArray<Real, AMREX_SPACEDIM> iJ = iGeom._iJ;
        const GpuArray<Real, AMREX_SPACEDIM> bou_aJ = iGeom._bou_aJ;
        
        // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
        const Real * eBouQuad_mem_ptr = iGeom.eBouQuad_device_mem.data();
        const Real * eIntBouQuad_mem_ptr = iGeom.eIntBouQuad_device_mem.data();
        const Real * space_Bou_BF_mem_ptr = MatFactory.space_Bou_BF_device_mem.data();
        AMREX_D_TERM
        (
            const Real * space_Bou_dBFdXI1_mem_ptr = MatFactory.space_Bou_dBFdXI1_device_mem.data();,
            const Real * space_Bou_dBFdXI2_mem_ptr = MatFactory.space_Bou_dBFdXI2_device_mem.data();,
            const Real * space_Bou_dBFdXI3_mem_ptr = MatFactory.space_Bou_dBFdXI3_device_mem.data();
        )
        const Real * space_IntBou_BF_mem_ptr = MatFactory.space_IntBou_BF_device_mem.data();
        AMREX_D_TERM
        (
            const Real * space_IntBou_dBFdXI1_mem_ptr = MatFactory.space_IntBou_dBFdXI1_device_mem.data();,
            const Real * space_IntBou_dBFdXI2_mem_ptr = MatFactory.space_IntBou_dBFdXI2_device_mem.data();,
            const Real * space_IntBou_dBFdXI3_mem_ptr = MatFactory.space_IntBou_dBFdXI3_device_mem.data();
        )
#else
        const Real * eBouQuad_mem_ptr = iGeom.eBouQuad_host_mem.data();
        const Real * eIntBouQuad_mem_ptr = iGeom.eIntBouQuad_host_mem.data();
        const Real * space_Bou_BF_mem_ptr = MatFactory.space_Bou_BF_host_mem.data();
        AMREX_D_TERM
        (
            const Real * space_Bou_dBFdXI1_mem_ptr = MatFactory.space_Bou_dBFdXI1_host_mem.data();,
            const Real * space_Bou_dBFdXI2_mem_ptr = MatFactory.space_Bou_dBFdXI2_host_mem.data();,
            const Real * space_Bou_dBFdXI3_mem_ptr = MatFactory.space_Bou_dBFdXI3_host_mem.data();
        )
        const Real * space_IntBou_BF_mem_ptr = MatFactory.space_IntBou_BF_host_mem.data();
        AMREX_D_TERM
        (
            const Real * space_IntBou_dBFdXI1_mem_ptr = MatFactory.space_IntBou_dBFdXI1_host_mem.data();,
            const Real * space_IntBou_dBFdXI2_mem_ptr = MatFactory.space_IntBou_dBFdXI2_host_mem.data();,
            const Real * space_IntBou_dBFdXI3_mem_ptr = MatFactory.space_IntBou_dBFdXI3_host_mem.data();
        )
#endif

        // DG
        const int sp = MatFactory.space_p;
        const int sNp = MatFactory.space_Np;
        // ============================================================

        // ============================================================
        //
        // int_{dVh} [[V.T]]_{i} NF_i
        //
        // ============================================================
        for (MFIter mfi(_X_); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            AMREX_D_TERM
            (
                Array4<int const> const & eBouQuad_Nq_fab_0 = iGeom.eBouQuad_Nq[0].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_0 = iGeom.eBouQuad_pos[0].array(mfi);,
                Array4<int const> const & eBouQuad_Nq_fab_1 = iGeom.eBouQuad_Nq[1].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_1 = iGeom.eBouQuad_pos[1].array(mfi);,
                Array4<int const> const & eBouQuad_Nq_fab_2 = iGeom.eBouQuad_Nq[2].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_2 = iGeom.eBouQuad_pos[2].array(mfi);
            )
            Array4<int const> const & eIntBouQuad_Nq_fab = iGeom.eIntBouQuad_Nq.array(mfi);
            Array4<long const> const & eIntBouQuad_pos_fab = iGeom.eIntBouQuad_pos.array(mfi);

            AMREX_D_TERM
            (
                Array4<long const> const & space_Bou_BF_pos_fab_0 = MatFactory.space_Bou_BF_pos[0].array(mfi);,
                Array4<long const> const & space_Bou_BF_pos_fab_1 = MatFactory.space_Bou_BF_pos[1].array(mfi);,
                Array4<long const> const & space_Bou_BF_pos_fab_2 = MatFactory.space_Bou_BF_pos[2].array(mfi);
            )
            Array4<long const> const & space_IntBou_BF_pos_fab = MatFactory.space_IntBou_BF_pos.array(mfi);

            Array4<Real const> const & X_fab = _X_.array(mfi);
            Array4<Real> const & dXdt_fab = _dXdt_.array(mfi);
            
            ParallelFor(bx, N_U,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const int dom = IBVP.F_U2DOM(ru);
                const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};
                const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM], grad[AMREX_SPACEDIM];
                Real Uh[N_U];
                Real AMREX_D_DECL(dUhdx1[N_U], dUhdx2[N_U], dUhdx3[N_U]);
                Real AMREX_D_DECL(NF1[N_U], NF2[N_U], NF3[N_U]);
                Real integrand;

                // LOCAL VARIABLES (NEIGHBORING CELL)
                bool intraphase;
                int nbr_etype;
                bool nbr_is_small, cells_are_merged, nbr_is_empty;
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
                int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                Real nbr_Uh[N_U];
                Real AMREX_D_DECL(nbr_dUhdx1[N_U], nbr_dUhdx2[N_U], nbr_dUhdx3[N_U]);

                // QUADRATURE INFO
                int fi, fj, fk;
                int bou_Nq;
                long eBouQuad_pos, qi_pos;
                const Real * xptr;

                // BASIS FUNCTIONS INFO
                long space_BF_pos, nbr_space_BF_pos;
                const Real * BF_ptr;
                AMREX_D_TERM
                (
                    const Real * dBFdXI1_ptr;,
                    const Real * dBFdXI2_ptr;,
                    const Real * dBFdXI3_ptr;
                )
                const Real * nbr_BF_ptr;
                AMREX_D_TERM
                (
                    const Real * nbr_dBFdXI1_ptr;,
                    const Real * nbr_dBFdXI2_ptr;,
                    const Real * nbr_dBFdXI3_ptr;
                )

                // Get the support of the basis functions (for the 
                // current cell)
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // Data structures used to compute the basis functions
                // and the unknown fields for the current cell and the
                // neighboring cell
                DG_mem_sol_space_BFX sol(X_fab, sp, BF_i, BF_j, BF_k, -1, &iJ[0]);
                DG_mem_sol_space_BFX nbr_sol(X_fab, sp, &iJ[0]);

                // CELL BOUNDARIES
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    const int dir = b/2;

                    // Neighboring cell info
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    // Conditions for NOT computing the flux
                    // contribution:
                    // i) the current cell extends to the neighboring
                    //    cell;
                    // ii) the current cell is small and the
                    //     neighboring cell extends into it;
                    // iii) the neighboring cell is empty (it means
                    //      that the cells must be separated by the
                    //      zero level set of a distance function;
                    //      this case is handled in the INTERPHASE
                    //      case
                    //
                    cells_are_merged = (nbr_merged_b == nbr_b);
                    cells_are_merged = cells_are_merged || (merged_b == b);
                    nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
                    if (cells_are_merged || nbr_is_empty) continue;

                    // Get the support of the basis functions (for the
                    // neighboring cell)
                    BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                    // QUADRATURE AND BASIS FUNCTIONS INFO
                    GRID_FACE(i, j, k, b, fi, fj, fk);
                    if (dir == 0)
                    {
                        bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        eBouQuad_pos = eBouQuad_pos_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

                        space_BF_pos = space_Bou_BF_pos_fab_0(fi,fj,fk,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+1-(b%2));
                        BF_ptr = &space_Bou_BF_mem_ptr[space_BF_pos];
                        AMREX_D_TERM
                        (
                            dBFdXI1_ptr = &space_Bou_dBFdXI1_mem_ptr[space_BF_pos];,
                            dBFdXI2_ptr = &space_Bou_dBFdXI2_mem_ptr[space_BF_pos];,
                            dBFdXI3_ptr = &space_Bou_dBFdXI3_mem_ptr[space_BF_pos];
                        )
                        nbr_space_BF_pos = space_Bou_BF_pos_fab_0(fi,fj,fk,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+(b%2));
                        nbr_BF_ptr = &space_Bou_BF_mem_ptr[nbr_space_BF_pos];
                        AMREX_D_TERM
                        (
                            nbr_dBFdXI1_ptr = &space_Bou_dBFdXI1_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI2_ptr = &space_Bou_dBFdXI2_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI3_ptr = &space_Bou_dBFdXI3_mem_ptr[nbr_space_BF_pos];
                        )

                        if ((fi == domain.smallEnd(dir) || fi == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
                        else intraphase = true;
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (dir == 1)
                    {
                        bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        eBouQuad_pos = eBouQuad_pos_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

                        space_BF_pos = space_Bou_BF_pos_fab_1(fi,fj,fk,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+1-(b%2));
                        BF_ptr = &space_Bou_BF_mem_ptr[space_BF_pos];
                        AMREX_D_TERM
                        (
                            dBFdXI1_ptr = &space_Bou_dBFdXI1_mem_ptr[space_BF_pos];,
                            dBFdXI2_ptr = &space_Bou_dBFdXI2_mem_ptr[space_BF_pos];,
                            dBFdXI3_ptr = &space_Bou_dBFdXI3_mem_ptr[space_BF_pos];
                        )
                        nbr_space_BF_pos = space_Bou_BF_pos_fab_1(fi,fj,fk,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+(b%2));
                        nbr_BF_ptr = &space_Bou_BF_mem_ptr[nbr_space_BF_pos];
                        AMREX_D_TERM
                        (
                            nbr_dBFdXI1_ptr = &space_Bou_dBFdXI1_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI2_ptr = &space_Bou_dBFdXI2_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI3_ptr = &space_Bou_dBFdXI3_mem_ptr[nbr_space_BF_pos];
                        )

                        if ((fj == domain.smallEnd(dir) || fj == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
                        else intraphase = true;
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else
                    {
                        bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        eBouQuad_pos = eBouQuad_pos_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

                        space_BF_pos = space_Bou_BF_pos_fab_2(fi,fj,fk,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+1-(b%2));
                        BF_ptr = &space_Bou_BF_mem_ptr[space_BF_pos];
                        AMREX_D_TERM
                        (
                            dBFdXI1_ptr = &space_Bou_dBFdXI1_mem_ptr[space_BF_pos];,
                            dBFdXI2_ptr = &space_Bou_dBFdXI2_mem_ptr[space_BF_pos];,
                            dBFdXI3_ptr = &space_Bou_dBFdXI3_mem_ptr[space_BF_pos];
                        )
                        nbr_space_BF_pos = space_Bou_BF_pos_fab_2(fi,fj,fk,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+(b%2));
                        nbr_BF_ptr = &space_Bou_BF_mem_ptr[nbr_space_BF_pos];
                        AMREX_D_TERM
                        (
                            nbr_dBFdXI1_ptr = &space_Bou_dBFdXI1_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI2_ptr = &space_Bou_dBFdXI2_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI3_ptr = &space_Bou_dBFdXI3_mem_ptr[nbr_space_BF_pos];
                        )

                        if ((fk == domain.smallEnd(dir) || fk == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
                        else intraphase = true;
                    }
#endif
                    xptr = &eBouQuad_mem_ptr[eBouQuad_pos];

                    // UNIT NORMAL
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                    un[dir] = 2.0*(b%2)-1.0;

                    // BASIS FUNCTIONS POINTERS
                    sol.set_BF_ptrs(BF_ptr, AMREX_D_DECL(dBFdXI1_ptr, dBFdXI2_ptr, dBFdXI3_ptr));
                    nbr_sol.set_BF_ptrs(nbr_BF_ptr, AMREX_D_DECL(nbr_dBFdXI1_ptr, nbr_dBFdXI2_ptr, nbr_dBFdXI3_ptr));

                    // INTER-ELEMENT BOUNDARY OR PERIODIC GRID BOUNDARY
                    // INTRAPHASE CONDITIONS TO BE ASSIGNED
                    if (intraphase)
                    {
                        // COMPUTE THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            qi_pos = (AMREX_SPACEDIM+1)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                            )
                            w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                            // SOLUTION
                            sol._eval_BF_table_(qi);
                            nbr_sol._eval_BF_table_(qi);
                            for (int cu = 0; cu < N_U; ++cu)
                            {
                                sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                Uh[cu] = sol._eval_unsafe_();
                                
                                sol._eval_grad_unsafe_(grad);
                                AMREX_D_TERM
                                (
                                    dUhdx1[cu] = grad[0];,
                                    dUhdx2[cu] = grad[1];,
                                    dUhdx3[cu] = grad[2];
                                );

                                nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);
                                
                                nbr_Uh[cu] = nbr_sol._eval_unsafe_();

                                nbr_sol._eval_grad_unsafe_(grad);
                                AMREX_D_TERM
                                (
                                    nbr_dUhdx1[cu] = grad[0];,
                                    nbr_dUhdx2[cu] = grad[1];,
                                    nbr_dUhdx3[cu] = grad[2];
                                );
                            }

                            // NUMERICAL FLUX FOR INTRAPHASE
                            IBVP.F_NF_ICS(t, x, un, Uh, nbr_Uh, AMREX_D_DECL(NF1, NF2, NF3));

                            // DG CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                AMREX_D_TERM
                                (
                                    integrand  = sol.BF[rs]*NF1[ru]*un[0];,
                                    integrand += sol.BF[rs]*NF2[ru]*un[1];,
                                    integrand += sol.BF[rs]*NF3[ru]*un[2];
                                )

                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                    // GRID BOUNDARY
                    // BOUNDARY CONDITIONS TO BE ASSIGNED
                    else
                    {
                        // COMPUTE THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            qi_pos = (AMREX_SPACEDIM+1)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                            )
                            w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                            // SOLUTION
                            sol._eval_BF_table_(qi);
                            for (int cu = 0; cu < N_U; ++cu)
                            {
                                sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                Uh[cu] = sol._eval_unsafe_();
                                
                                sol._eval_grad_unsafe_(grad);
                                AMREX_D_TERM
                                (
                                    dUhdx1[cu] = grad[0];,
                                    dUhdx2[cu] = grad[1];,
                                    dUhdx3[cu] = grad[2];
                                );
                            }

                            // NUMERICAL FLUX FOR INTRAPHASE  
                            IBVP.F_NF_BCS(t, x, un, Uh, AMREX_D_DECL(NF1, NF2, NF3));

                            // DG CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                AMREX_D_TERM
                                (
                                    integrand  = sol.BF[rs]*NF1[ru]*un[0];,
                                    integrand += sol.BF[rs]*NF2[ru]*un[1];,
                                    integrand += sol.BF[rs]*NF3[ru]*un[2];
                                )

                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                }

                // INTERNAL BOUNDARY
                {
                    // Neighboring domain info
                    const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

                    // QUADRATURE INFO
                    bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                    eBouQuad_pos = eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                    xptr = &eIntBouQuad_mem_ptr[eBouQuad_pos];

                    // BASIS FUNCTIONS INFO
                    space_BF_pos = space_IntBou_BF_pos_fab(i,j,k,DG_INT_BOU_BF_INFO_N_COMP_PER_DOM*dom);
                    BF_ptr = &space_IntBou_BF_mem_ptr[space_BF_pos];
                    AMREX_D_TERM
                    (
                        dBFdXI1_ptr = &space_IntBou_dBFdXI1_mem_ptr[space_BF_pos];,
                        dBFdXI2_ptr = &space_IntBou_dBFdXI2_mem_ptr[space_BF_pos];,
                        dBFdXI3_ptr = &space_IntBou_dBFdXI3_mem_ptr[space_BF_pos];
                    )

                    // BASIS FUNCTIONS POINTERS
                    sol.set_BF_ptrs(BF_ptr, AMREX_D_DECL(dBFdXI1_ptr, dBFdXI2_ptr, dBFdXI3_ptr));

                    // NO NEIGHBORING DOMAIN
                    // BOUNDARY CONDITIONS TO BE ASSIGNED
                    if (nbr_dom == -1)
                    {
                        // COMPUTE THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0];,
                                x[1] = xptr[qi_pos+1];,
                                x[2] = xptr[qi_pos+2];
                            )
                            w = xptr[qi_pos+AMREX_SPACEDIM];

                            // UNIT NORMAL
                            AMREX_D_TERM
                            (
                                un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                                un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                                un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                            )

                            // SOLUTION
                            sol._eval_BF_table_(qi);
                            for (int cu = 0; cu < N_U; ++cu)
                            {
                                sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                Uh[cu] = sol._eval_unsafe_();
                                
                                sol._eval_grad_unsafe_(grad);
                                AMREX_D_TERM
                                (
                                    dUhdx1[cu] = grad[0];,
                                    dUhdx2[cu] = grad[1];,
                                    dUhdx3[cu] = grad[2];
                                );
                            }

                            // NUMERICAL FLUX FOR INTERNAL BOUNDARY
                            IBVP.F_NF_PHI_BCS(dom, t, x, un, Uh, AMREX_D_DECL(NF1, NF2, NF3));

                            // DG CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                AMREX_D_TERM
                                (
                                    integrand  = sol.BF[rs]*NF1[ru]*un[0];,
                                    integrand += sol.BF[rs]*NF2[ru]*un[1];,
                                    integrand += sol.BF[rs]*NF3[ru]*un[2];
                                )

                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                    // VALID NEIGHBORING DOMAIN
                    // INTERFACE CONDITIONS TO BE ASSIGNED
                    else
                    {
                        // Neighboring cell info
                        nbr_etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*nbr_dom);

                        // Get the support of the basis functions (for
                        // the neighboring cell)
                        BF_CELL(i, j, k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // BASIS FUNCTIONS INFO
                        nbr_space_BF_pos = space_IntBou_BF_pos_fab(i,j,k,DG_INT_BOU_BF_INFO_N_COMP_PER_DOM*nbr_dom);
                        nbr_BF_ptr = &space_IntBou_BF_mem_ptr[nbr_space_BF_pos];
                        AMREX_D_TERM
                        (
                            nbr_dBFdXI1_ptr = &space_IntBou_dBFdXI1_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI2_ptr = &space_IntBou_dBFdXI2_mem_ptr[nbr_space_BF_pos];,
                            nbr_dBFdXI3_ptr = &space_IntBou_dBFdXI3_mem_ptr[nbr_space_BF_pos];
                        )

                        // BASIS FUNCTIONS POINTERS
                        nbr_sol.set_BF_ptrs(nbr_BF_ptr, AMREX_D_DECL(nbr_dBFdXI1_ptr, nbr_dBFdXI2_ptr, nbr_dBFdXI3_ptr));

                        // COMPUTE THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0];,
                                x[1] = xptr[qi_pos+1];,
                                x[2] = xptr[qi_pos+2];
                            )
                            w = xptr[qi_pos+AMREX_SPACEDIM];

                            // UNIT NORMAL
                            AMREX_D_TERM
                            (
                                un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                                un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                                un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                            )

                            // SOLUTION
                            sol._eval_BF_table_(qi);
                            nbr_sol._eval_BF_table_(qi);
                            for (int cu = 0; cu < N_U; ++cu)
                            {
                                sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                Uh[cu] = sol._eval_unsafe_();
                                
                                sol._eval_grad_unsafe_(grad);
                                AMREX_D_TERM
                                (
                                    dUhdx1[cu] = grad[0];,
                                    dUhdx2[cu] = grad[1];,
                                    dUhdx3[cu] = grad[2];
                                );

                                nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);

                                nbr_Uh[cu] = nbr_sol._eval_unsafe_();

                                nbr_sol._eval_grad_unsafe_(grad);
                                AMREX_D_TERM
                                (
                                    nbr_dUhdx1[cu] = grad[0];,
                                    nbr_dUhdx2[cu] = grad[1];,
                                    nbr_dUhdx3[cu] = grad[2];
                                );
                            }

                            // NUMERICAL FLUX FOR INTERNAL INTERFACE
                            IBVP.F_NF_PHI_ICS(dom, t, x, un, Uh, nbr_Uh, AMREX_D_DECL(NF1, NF2, NF3));

                            // DG CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                AMREX_D_TERM
                                (
                                    integrand  = sol.BF[rs]*NF1[ru]*un[0];,
                                    integrand += sol.BF[rs]*NF2[ru]*un[1];,
                                    integrand += sol.BF[rs]*NF3[ru]*un[2];
                                )

                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        _dXdt_.FillBoundary(iGeom.geom.periodicity());
        // ============================================================
    }
    // ################################################################

};
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif
// AMREX_DG.H

#ifndef BL_DG_H_
#define BL_DG_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_MatrixFactory.H>
#include <AMReX_DG_ImplicitGeometry.H>
#include <AMReX_DG_RK.H>
#include <AMReX_DG_PostProcessByConvolution.H>

namespace amrex
{
namespace DG
{
// ####################################################################
// DG BASE CLASS ######################################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
class DG
{
private:
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;

    // PARAMETERS
    const std::string PDEs_info, time_stepping_info;

    // VARIABLES
    // Time-stepping data structures
    RungeKutta<N_PHI, N_DOM, N_U> RK;

    // Solution
        
    MultiFab X_old, X_new ;
    // MultiFab X_star_DG;
    MultiFab X_star;
    // ################################################################

public:
    // CONSTRUCTOR ####################################################
    DG(const std::string& PDEs_info_,
       const std::string& time_stepping_info_)
    :
    PDEs_info(PDEs_info_),
    time_stepping_info(time_stepping_info_)
    {}
    // ################################################################

    // DESTRUCTOR #####################################################
    ~DG()
    {}
    // ################################################################

    // MISC ###########################################################
    bool isPDEsSystemHyperbolic() const
    {
        return (this->PDEs_info.compare("Hyperbolic") == 0);
    }

    bool isTimeSteppingRungeKutta() const
    {
        return (this->time_stepping_info.compare("Runge-Kutta") == 0);
    }
    // ################################################################

    // INITIALIZATION #################################################
    void InitData(const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                  const MatrixFactory<N_PHI, N_DOM> & MatFactory)
    {
        // PARAMETERS ===========================
        const int space_Np = MatFactory.space_Np;
        const int U_n_comp = space_Np*N_U;
        // ======================================

        // INIT THE DATA STRUCTURE FOR TIME-STEPPING ==================
        if (this->isTimeSteppingRungeKutta())
        {
            this->RK.InitData(iGeom, MatFactory);
        }
        else
        {
            std::cout << std::endl;
            std::cout << "ERROR: AMReX_DG.H - DG::InitData" << std::endl;
            std::cout << "| Unexpected time-stepping scheme: " << this->time_stepping_info << std::endl;
            std::cout << std::endl;
            exit(-1);
        }
        // ============================================================

        // STORE MEMORY FOR THE MULTIFABS CONTAINING DG SOLUTION ======
        this->X_old.define(iGeom.cc_ba, iGeom.dm, U_n_comp, this->dom_data_ngr);
        this->X_new.define(iGeom.cc_ba, iGeom.dm, U_n_comp, this->dom_data_ngr);

        this->X_old = 0.0;
        this->X_new = 0.0;
        // ============================================================
    }
    // ################################################################

    // INITIAL CONDITIONS #############################################
    template<typename IBVP_CLASS>
    void SetICs(const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("DG::SetICs(const ImplicitGeometry<N_PHI, N_DOM> &, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = iGeom._dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = iGeom._prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = iGeom._J;
        const GpuArray<Real, AMREX_SPACEDIM> iJ = iGeom._iJ;
        const Real aJ = iGeom._aJ, iaJ = iGeom._iaJ;

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
        const Real * space_MMCh_mem_ptr = MatFactory.space_MMCh_device_mem.data();
#else
        const Real * eDomQuad_mem_ptr = iGeom.eDomQuad_host_mem.data();
        const Real * space_Dom_BF_mem_ptr = MatFactory.space_Dom_BF_host_mem.data();
        AMREX_D_TERM
        (
            const Real * space_Dom_dBFdXI1_mem_ptr = MatFactory.space_Dom_dBFdXI1_host_mem.data();,
            const Real * space_Dom_dBFdXI2_mem_ptr = MatFactory.space_Dom_dBFdXI2_host_mem.data();,
            const Real * space_Dom_dBFdXI3_mem_ptr = MatFactory.space_Dom_dBFdXI3_host_mem.data();
        )
        const Real * space_MMCh_mem_ptr = MatFactory.space_MMCh_host_mem.data();
#endif

        // DG
        const int sp = MatFactory.space_p;
        const int sNp = MatFactory.space_Np;
        // ============================================================

        // INITIALIZATION
        this->X_new = 0.0;
        // ===============

        // EVAL THE PROJECTION ========================================
        for (MFIter mfi(this->X_new); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            //Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<int const> const & eDomQuad_Nq_fab = iGeom.eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = iGeom.eDomQuad_pos.array(mfi);
            Array4<long const> const & space_Dom_BF_pos_fab = MatFactory.space_Dom_BF_pos.array(mfi);
            
            Array4<Real> const & X_fab = this->X_new.array(mfi);

            ParallelFor(bx, N_U,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const int dom = IBVP.F_U2DOM(ru);
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
                Real x[AMREX_SPACEDIM], w;
                Real integrand;

                // Data structures used to compute the basis functions
                DG_mem_sol_space_BF bf(sp, BF_ptr, AMREX_D_DECL(dBFdXI1_ptr, dBFdXI2_ptr, dBFdXI3_ptr), &iJ[0]);

                // PROJECTION
                for (int qi = 0; qi < dom_Nq; ++qi)
                {
                    qi_pos = (AMREX_SPACEDIM+1)*qi;
                    AMREX_D_TERM
                    (
                        x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                        x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                        x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                    )
                    w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;

                    bf._eval_BF_table_(qi);
                    
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        integrand = bf.BF[rs]*IBVP.F_U0(ru, x);
                        X_fab(i,j,k,rs+ru*sNp) += integrand*w;
                    }
                }
            });
            Gpu::synchronize();
        }
        // IMPORTANT NOTE
        // We are looping within the fabbox. Therefore we do not need
        // to share information among the processors.
        // 
        //this->X_new.FillBoundary(geom.periodicity());
        // ============================================================

        // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX
        for (MFIter mfi(this->X_new); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<long const> const & space_MMCh_pos_fab = MatFactory.space_MMCh_pos.array(mfi);
            Array4<Real> const & X_fab = this->X_new.array(mfi);

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
                Real Xh[DG_SOL_MAX_SPACE_Np];

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
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            X_fab(i,j,k,rs+ru*sNp) += X_fab(nbr_i,nbr_j,nbr_k,rs+ru*sNp);
                        }
                    }
                }
                // IF VALID, MULTIPLY BY THE INVERSE OF THE MASS MATRIX
                if (elm_is_valid)
                {
                    // Copy from memory. Remember that the mass matrix
                    // is that of an element defined over the
                    // [-1,1]^AMREX_SPACEDIM domain.
                    for (int rs = 0; rs < sNp; ++rs)
                        Xh[rs] = X_fab(i,j,k,rs+ru*sNp)*iaJ;

                    // Actual matrix operation (using mass matrix'
                    // Cholesky decomposition)
                    DG_utils::Cholesky_solve_overwrite(sNp, 1, space_MMCh_ptr, Xh);

                    // Copy back to memory
                    for (int rs = 0; rs < sNp; ++rs)
                        X_fab(i,j,k,rs+ru*sNp) = Xh[rs];
                }

                /*
                 Print()<<X_fab(i,j,k,0)<<std::endl;
                Print()<<X_fab(i,j,k,1)<<std::endl;
                Print()<<X_fab(i,j,k,2)<<std::endl;
                Print()<<X_fab(i,j,k,3)<<std::endl;
                Print()<<"i: "<<i<<"j: "<<j<<"k: "<<k<<std::endl;
                */

            });
            Gpu::synchronize();
        }
        this->X_new.FillBoundary(iGeom.geom.periodicity());
        // ============================================================
    }
    // ################################################################

    // TIME STEP: CFL CONDITION #######################################
    Real Estimate_CFL(const MatrixFactory<N_PHI, N_DOM> & MatFactory) const
    {
        Real CFL;
        if (this->isPDEsSystemHyperbolic() && this->isTimeSteppingRungeKutta())
        {
            CFL = 1.0/(2.0*MatFactory.space_p+1.0);
        }
        else
        {
            Print() << "ERROR: AMReX_DG.H - DG::EstimateCFL" << std::endl;
            Print() << "| Unexpected type of PDEs               : " << this->PDEs_info << std::endl;
            Print() << "| and/or unexpected time-stepping scheme: " << this->time_stepping_info << std::endl;
            exit(-1);
        }

        return CFL;
    }
    // ################################################################

    // TIME STEP ######################################################
    template<typename IBVP_CLASS>
    Real Compute_dt(const Real & t,
                    const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                    const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                    const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("DG::Compute_dt(const Real &, const ImplicitGeometry<N_PHI, N_DOM> &, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = iGeom._dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = iGeom._prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = iGeom._J;
        const GpuArray<Real, AMREX_SPACEDIM> iJ = iGeom._iJ;

        // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
        const Real * eTri_mem_ptr = iGeom.eTri_device_mem.data();
        const Real * space_Tri_BF_mem_ptr = MatFactory.space_Tri_BF_device_mem.data();
#else
        const Real * eTri_mem_ptr = iGeom.eTri_host_mem.data();
        const Real * space_Tri_BF_mem_ptr = MatFactory.space_Tri_BF_host_mem.data();
#endif

        // DG
        const int sp = MatFactory.space_p;

        // CFL NUMBER
        const Real CFL = this->Estimate_CFL(MatFactory);
        // ============================================================

        // VARIABLES ==================================================
        MultiFab dt(iGeom.cc_ba, iGeom.dm, 1, 0);
        dt = 1.0e12;
        Real dt_min;
        // ============================================================

        // EVAL THE MINIMUM TIME STEP =================================
        for (MFIter mfi(dt); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            
            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<Real const> const & eInfo_fab = iGeom.eInfo.array(mfi);
            Array4<int const> const & eTri_Npt_fab = iGeom.eTri_Npt.array(mfi);
            Array4<long const> const & eTri_pos_fab = iGeom.eTri_pos.array(mfi);
            Array4<long const> const & space_Tri_BF_pos_fab = MatFactory.space_Tri_BF_pos.array(mfi);
            Array4<Real const> const & X_fab = this->X_new.array(mfi);
            Array4<Real> const & dt_fab = dt.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};

                // LOCAL VARIABLES
                int etype;
                bool elm_is_empty, elm_is_large, elm_is_small, elm_is_partial;
                Real vf;
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM];
                Real Uh[N_U];
                Real elm_dt, x_dt;

                // QUERY INFO
                int dom_Npt;
                long eTri_pos, qi_pos, space_BF_pos;
                const Real * xptr;
                const Real * BF_ptr;

                // Data structures used to compute the basis functions
                // and the unknown fields
                DG_mem_sol_space_BFX sol(X_fab, sp, &iJ[0]);

                // CYCLE OVER THE DOMAINS
                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    // Element info
                    etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    elm_is_empty = (etype == DG_ELM_TYPE_EMPTY);
                    elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                    elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
                    elm_is_partial = (elm_is_large || elm_is_small);
                    vf = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom);

                    if (!elm_is_empty)
                    {
                        // Get the support of the basis functions
                        BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                        // Query info
                        dom_Npt = eTri_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                        eTri_pos = eTri_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                        xptr = &eTri_mem_ptr[eTri_pos];

                        // Basis functions info
                        space_BF_pos = space_Tri_BF_pos_fab(i,j,k,DG_TRI_BF_INFO_N_COMP_PER_DOM*dom);
                        BF_ptr = &space_Tri_BF_mem_ptr[space_BF_pos];
                        sol.set_BF_ptrs(BF_ptr);

                        // CYCLE OVER THE QUERY POINTS
                        elm_dt = 1.0e12;
                        for (int qi = 0; qi < dom_Npt; ++qi)
                        {
                            qi_pos = AMREX_SPACEDIM*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                            )

                            sol._eval_BF_only_table_(qi);
                            for (int ru = 0; ru < N_U; ++ru)
                            {
                                sol.set_fab_ptrs(BF_i, BF_j, BF_k, ru);

                                Uh[ru] = sol._eval_unsafe_();
                            }

                            x_dt = IBVP.F_DT(&dx[0], t, x, Uh);
                            elm_dt = fmin(elm_dt, x_dt);
                        }
                        elm_dt = (elm_is_large) ? (vf*elm_dt) : elm_dt;
                        dt_fab(i,j,k,0) = fmin(dt_fab(i,j,k,0), elm_dt);
                    }
                }
            });
            Gpu::synchronize();
        }
        // ============================================================

        // GET THE MINUMUM 
        dt_min = dt.min(0);
        dt_min *= CFL;
        // ================

        return dt_min;
    }
    // ################################################################

    // TIME-STEP: HYPERBOLIC ##########################################
    template<typename IBVP_CLASS>
    void TakeTimeStep_Hyperbolic(const Real & dt,
                                 const Real & t,
                                 const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                                 const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                                 const IBVP_CLASS & IBVP)
    {
        // PARAMETERS =================================================
        const int space_Np = MatFactory.space_Np;
        const int U_n_comp = space_Np*N_U;
        // ============================================================

        // CONSISTENCY CHECK ==========================================
        if (!(this->isPDEsSystemHyperbolic()))
        {
            Print() << "ERROR: AMReX_DG.H - DG::TakeTimeStep_Hyperbolic" << std::endl;
            Print() << "| Unexpected type of PDEs: " << this->PDEs_info << std::endl;
            exit(-1);
        }
        // ============================================================

        // COPY NEW TO OLD ============================================
        MultiFab::Copy(this->X_old, this->X_new, 0, 0, U_n_comp, this->dom_data_ngr);
        // ============================================================

        // CALL SELECTED TIME-STEPPING SCHEME =========================
        if (this->isTimeSteppingRungeKutta())
        {
            this->RK.TakeTimeStep_Hyperbolic(MatFactory.time_Np, dt, t, iGeom, MatFactory, this->X_old, this->X_new, IBVP);
        }
        else
        {
            Print() << "ERROR: AMReX_DG.H - DG::TakeTimeStep_Hyperbolic" << std::endl;
            Print() << "| Unexpected time-stepping scheme: " << this->time_stepping_info << std::endl;
            exit(-1);
        }
        // ============================================================
    }
    // ################################################################

    // ERROR EVALUATION ###############################################
    template<typename IBVP_CLASS>
    Real EvalErrorNorm(const Real & t,
                       const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                       const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                       const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("DG::EvalErrorNorm(const Real &, const ImplicitGeometry<N_PHI, N_DOM> &, ....)");
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
#else
        const Real * eDomQuad_mem_ptr = iGeom.eDomQuad_host_mem.data();
        const Real * space_Dom_BF_mem_ptr = MatFactory.space_Dom_BF_host_mem.data();
#endif

        // DG
        const int sp = MatFactory.space_p;
        // ============================================================

        // VARIABLES ==================================================
        MultiFab error(iGeom.cc_ba, iGeom.dm, 2, 0);
        Real err;
        // ============================================================

        // EVAL THE ERROR =============================================
        for (MFIter mfi(error); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            
            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<int const> const & eDomQuad_Nq_fab = iGeom.eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = iGeom.eDomQuad_pos.array(mfi);
            Array4<long const> const & space_Dom_BF_pos_fab = MatFactory.space_Dom_BF_pos.array(mfi);
            Array4<Real const> const & X_fab = this->X_new.array(mfi);
            Array4<Real> const & err_fab = error.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // INITIALIZATION
                err_fab(i,j,k,0) = 0.0;
                err_fab(i,j,k,1) = 0.0;

                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};

                // LOCAL VARIABLES
                int dom, etype;
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM], w;
                Real Uh[N_U], Ue[N_U];
                Real integrand;

                // QUADRATURE INFO
                int dom_Nq;
                long eDomQuad_pos, qi_pos, space_BF_pos;
                const Real * xptr;
                const Real * BF_ptr;

                // Data structures used to compute the basis functions
                // and the unknown fields
                DG_mem_sol_space_BFX sol(X_fab, sp, &iJ[0]);

                // CYCLE OVER THE UNKNOWN FIELDS
                for (int ru = 0; ru < N_U; ++ru)
                //for (int ru = 0; ru < 1; ++ru)
                {
                    dom = IBVP.F_U2DOM(ru);
                    etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);

                    // Get the support of the basis functions
                    BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                    // Quadrature info
                    dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    eDomQuad_pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    xptr =  &eDomQuad_mem_ptr[eDomQuad_pos];

                    // Basis functions info
                    space_BF_pos = space_Dom_BF_pos_fab(i,j,k,DG_DOM_BF_INFO_N_COMP_PER_DOM*dom);
                    BF_ptr = &space_Dom_BF_mem_ptr[space_BF_pos];
                    sol.set_BF_ptrs(BF_ptr);

                    // COMPUTE THE INTEGRALS
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
                        sol._eval_BF_only_table_(qi);
                        for (int cu = 0; cu < N_U; ++cu)
                        {
                            sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);
                            
                            Uh[cu] = sol._eval_unsafe_();
                        }
                        
                        // EXACT SOLUTION
                        IBVP.F_EXACT(t, x, Ue);
                        // Print()<<"index i: "<<i<<",j: "<<j<<",k: "<<k<<",point_x: "<<xptr[qi_pos+0]<<",point_y: "<<xptr[qi_pos+1]<<", "<<"u exact: "<<Ue[ru]<<"uh : "<<Uh[ru]<<std::endl;

                        integrand = std::pow(Ue[ru],2.0);
                        err_fab(i,j,k,0) += integrand*w;
                        
                        Ue[ru] -= Uh[ru];
                        integrand = std::pow(Ue[ru],2.0);
                        err_fab(i,j,k,1) += integrand*w;
                    }
                }
            });


            Gpu::synchronize();
        }
        // ============================================================

        // SUM =============================
Print() << "(error.sum(0)): " << std::sqrt(error.sum(0)) << std::endl;
Print() << "(error.sum(1)): " << std::sqrt(error.sum(1)) << std::endl;
        err = std::sqrt(error.sum(1))/std::sqrt(error.sum(0));
        // err = std::sqrt(error.sum(1));
        // =================================

        return err;
    }
    // ################################################################

    // EXPORT DG SOLUTION FOR VTK #####################################
    template<typename IBVP_CLASS>
    void Export_VTK(const std::string & dst_folder,
                    const std::string & filename_root,
                    const int & time_id,
                    const int & time_id_max,
                    const std::vector<int> & field_domains,
                    const std::vector<std::string> & field_names,
                    const Real & t,
                    const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                    const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                    const IBVP_CLASS & IBVP) const
    {
        // PARAMETERS =================================================
        const bool vtk_legacy_format = false;
        const bool vtk_xml_format = true;

        // FILEPATH
        std::string filepath = GetOutputFilepath(dst_folder, filename_root, time_id, time_id_max)+".vtk";

        // DATE AND TIME
        const time_t now = time(0);
        const char * date_time = ctime(&now);

        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = iGeom._dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = iGeom._prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = iGeom._J;

        // STANDARD ELEMENT
        const int cell_dom_Ng = std::max(iGeom.uHyperrect.dom_Ng, MatFactory.uHyperrect.dom_Ng);
        const int cell_e = std::max(iGeom.uHyperrect.g, MatFactory.uHyperrect.g);
        const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
        const int cell_conn_len = AMREX_D_PICK(2, 4, 8)*cell_dom_Ne;
        const Gpu::ManagedVector<int> cell_base_conn = Get_VTK_BaseCell_Connectivity(cell_e);
        const int * cell_base_conn_ptr = cell_base_conn.data();

        AMREX_D_TERM
        (
            Real const * dom_XI1g_ptr = (iGeom.uHyperrect.g > MatFactory.uHyperrect.g) ? iGeom.uHyperrect.dom_XI1g.data() : MatFactory.uHyperrect.dom_XI1g.data();,
            Real const * dom_XI2g_ptr = (iGeom.uHyperrect.g > MatFactory.uHyperrect.g) ? iGeom.uHyperrect.dom_XI2g.data() : MatFactory.uHyperrect.dom_XI2g.data();,
            Real const * dom_XI3g_ptr = (iGeom.uHyperrect.g > MatFactory.uHyperrect.g) ? iGeom.uHyperrect.dom_XI3g.data() : MatFactory.uHyperrect.dom_XI3g.data();
        )

        // DG
        const int phi_p = iGeom.uHyperrect.p;
        const int space_sol_p = MatFactory.space_p;

        // DOMAINS AND FIELDS
        const int n_requested_fields = field_domains.size();
        const int n_total_fields = n_requested_fields+N_PHI;
        Gpu::ManagedVector<int> Gpu_field_domains;
        Gpu_field_domains.resize(n_requested_fields);
        for (int f = 0; f < n_requested_fields; ++f) Gpu_field_domains[f] = field_domains[f];
        const int * field_domains_ptr = Gpu_field_domains.data();

        // VTK
        const int VTK_cell_id = AMREX_D_PICK(3, 9, 12);
        // ============================================================

        // VARIABLES ==================================================
        FILE * fp;

        long n_dG_cells;
        Gpu::ManagedVector<Real> nodes;
        Gpu::ManagedVector<long> conn;

        long nodes_pos, conn_pos, conn_offset;
        longMultiFab grid_info(iGeom.cc_ba, iGeom.dm, 3, iGeom.dom_data_ngr);

        // FIELDS
        Gpu::ManagedVector<Real> fields;

        long fields_pos;
        longMultiFab fields_info(iGeom.cc_ba, iGeom.dm, 1, iGeom.dom_data_ngr);
        // ============================================================

        // SOME CHECKS ================================================
        if (n_total_fields > DG_MAX_N_FIELDS)
        {
            std::cout << std::endl;
            std::cout << "ERROR: AMReX_DG.H - DG::Export_VTK" << std::endl;
            std::cout << "| The number of requested fields exceeds the number of allowed fields:" << std::endl;
            std::cout << "| DG_MAX_N_FIELDS = " << DG_MAX_N_FIELDS << std::endl;
            std::cout << "| n_total_fields = " << n_total_fields << std::endl;
            std::cout << std::endl;
            exit(-1);
        }
        // ============================================================

        // ALIGN PROCESSORS ==========
        ParallelDescriptor::Barrier();
        // ===========================

        // COLLECT INFORMATION ========================================
        n_dG_cells = 0L;
        nodes_pos = 0L;
        conn_pos = 0L;
        conn_offset = 0L;

        fields_pos = 0L;

        for (MFIter mfi(this->X_new); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);
            const auto len = length(bx);

            Array4<long> const & grid_info_fab = grid_info.array(mfi);
            Array4<long> const & fields_info_fab = fields_info.array(mfi);

            n_dG_cells += len.x*len.y*len.z;

            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // UPDATE INFO
                grid_info_fab(i,j,k,0) = nodes_pos;
                grid_info_fab(i,j,k,1) = conn_pos;
                grid_info_fab(i,j,k,2) = conn_offset;

                nodes_pos += cell_dom_Ng*AMREX_SPACEDIM;
                conn_pos += cell_conn_len;
                conn_offset += cell_dom_Ng;

                fields_info_fab(i,j,k,0) = fields_pos;

                fields_pos += cell_dom_Ng*n_total_fields;
            }
        }
        // ============================================================

        // COMPUTE OUTPUT DATA ========================================
        nodes.resize(n_dG_cells*cell_dom_Ng*AMREX_SPACEDIM);
        conn.resize(n_dG_cells*cell_conn_len);
        fields.resize(n_dG_cells*cell_dom_Ng*n_total_fields);

        Real * nodes_ptr = nodes.data();
        long * conn_ptr = conn.data();
        Real * fields_ptr = fields.data();

        for (MFIter mfi(this->X_new); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();

            Array4<Real const> const & PHI_fab = iGeom.PHI.array(mfi);
            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<Real const> const & X_fab = this->X_new.array(mfi);

            Array4<long const> const & grid_info_fab = grid_info.array(mfi);
            Array4<long const> const & fields_info_fab = fields_info.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0], prob_lo[1]+j*dx[1], prob_lo[2]+k*dx[2])};
                const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0], prob_lo[1]+(j+1)*dx[1], prob_lo[2]+(k+1)*dx[2])};
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};
                const long cell_nodes_pos = grid_info_fab(i,j,k,0);
                const long cell_conn_pos = grid_info_fab(i,j,k,1);
                const long cell_conn_offset = grid_info_fab(i,j,k,2);
                const long cell_fields_pos = fields_info_fab(i,j,k,0);

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], grad[AMREX_SPACEDIM];
                Real PHIh[N_PHI], Uh[N_U];
                Real AMREX_D_DECL(dPHIhdx1[N_PHI], dPHIhdx2[N_PHI], dPHIhdx3[N_PHI]);
                Real AMREX_D_DECL(dUhdx1[N_U], dUhdx2[N_U], dUhdx3[N_U]);
                Real Fh[DG_MAX_N_FIELDS];
                int dom, etype;
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

                // Data structures used to compute the distance
                // functions and the dG solution
                DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, phi_p, i, j, k, -1);
                DG_sol_space_BFX sol(X_fab, space_sol_p);

                // UPDATE INFO: NODES AND FIELDS
                for (int n = 0; n < cell_dom_Ng; ++n)
                {
                    // NODES
                    AMREX_D_TERM
                    (
                        x[0] = J[0]*dom_XI1g_ptr[n]+cell_m[0];,
                        x[1] = J[1]*dom_XI2g_ptr[n]+cell_m[1];,
                        x[2] = J[2]*dom_XI3g_ptr[n]+cell_m[2];
                    )
                    AMREX_D_TERM
                    (
                        nodes_ptr[cell_nodes_pos+AMREX_SPACEDIM*n+0] = x[0];,
                        nodes_ptr[cell_nodes_pos+AMREX_SPACEDIM*n+1] = x[1];,
                        nodes_ptr[cell_nodes_pos+AMREX_SPACEDIM*n+2] = x[2];
                    )

                    // FIELDS (DISTANCE FUNCTIONS)
                    phi._eval_BF_table_(x);
                    for (int rph = 0; rph < N_PHI; ++rph)
                    {
                        phi.set_fab_ptrs(i, j, k, rph);

                        PHIh[rph] = phi._eval_unsafe_();

                        phi._eval_grad_unsafe_(grad);
                        AMREX_D_TERM
                        (
                            dPHIhdx1[rph] = grad[0];,
                            dPHIhdx2[rph] = grad[1];,
                            dPHIhdx3[rph] = grad[2];
                        );

                        fields_ptr[cell_fields_pos+n_total_fields*n+rph] = PHIh[rph];
                    }

                    // FIELDS
                    for (int rf = 0; rf < n_requested_fields; ++rf)
                    {
                        dom = field_domains_ptr[rf];
                        etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                        BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                        sol.set_support(BF_lo, BF_hi);
                        sol._eval_BF_table_(x);

                        for (int ru = 0; ru < N_U; ++ru)
                        {
                            sol.set_fab_ptrs(BF_i, BF_j, BF_k, ru);

                            Uh[ru] = sol._eval_unsafe_();
                    
                            sol._eval_grad_unsafe_(grad);
                            AMREX_D_TERM
                            (
                                dUhdx1[ru] = grad[0];,
                                dUhdx2[ru] = grad[1];,
                                dUhdx3[ru] = grad[2];
                            );
                        }
                        
                        IBVP.F_SOL(t, x,
                                   PHIh, AMREX_D_DECL(dPHIhdx1, dPHIhdx2, dPHIhdx3),
                                   Uh, AMREX_D_DECL(dUhdx1, dUhdx2, dUhdx3),
                                   Fh);

                        fields_ptr[cell_fields_pos+n_total_fields*n+N_PHI+rf] = Fh[rf];
                    }
                }

                // UPDATE INFO: CONNECTIVITY
                for (int c = 0; c < cell_conn_len; ++c)
                {
                    conn_ptr[cell_conn_pos+c] = cell_base_conn_ptr[c]+cell_conn_offset;
                }
            });
            Gpu::synchronize();   
        }
        // ============================================================

        // WRITE OUTPUT DATA: LEGACY FORMAT ===========================
        if (vtk_legacy_format)
        {
            // OPEN FILE FOR WRITING --------------------------------------
            fp = fopen(filepath.c_str(), "w");
            if (fp != nullptr)
            {
                fprintf(fp, "# vtk DataFile Version 2.0\n");
                fprintf(fp, "VTK mesh file created on %s", date_time);
                fprintf(fp, "ASCII\n");
            }
            else
            {
                Print() << "ERROR: AMReX_DG.H - DG::Export_VTK" << std::endl;
                Print() << "| Unable to open and write: " << filepath << "file" << std::endl;
                exit(-1);
            }
            // ------------------------------------------------------------
            
            // WRITE TO FILE: GRID INFORMATION ----------------------------
            const long grid_n_nodes = n_dG_cells*cell_dom_Ng;
            const long grid_n_cells = conn.size()/AMREX_D_PICK(2, 4, 8);
            
            fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    #if (AMREX_SPACEDIM == 1)
            // NODES
            fprintf(fp, "POINTS %lu float\n", grid_n_nodes);
            for (int n = 0; n < grid_n_nodes; ++n)
                fprintf(fp, "%+.12f 0.0 0.0\n", nodes[AMREX_SPACEDIM*n]);
            
            // CELLS
            fprintf(fp, "CELLS %lu %lu\n", grid_n_cells, (1+2)*grid_n_cells);
            for (int n = 0; n < grid_n_cells; ++n)
                fprintf(fp, "2 %lu %lu\n", conn[2*n], conn[2*n+1]);
    #endif
    #if (AMREX_SPACEDIM == 2)
            // NODES
            fprintf(fp, "POINTS %lu float\n", grid_n_nodes);
            for (int n = 0; n < grid_n_nodes; ++n)
                fprintf(fp, "%+.12f %+.12f 0.0\n", nodes[AMREX_SPACEDIM*n], nodes[AMREX_SPACEDIM*n+1]);
            
            // CELLS
            fprintf(fp, "CELLS %lu %lu\n", grid_n_cells, (1+4)*grid_n_cells);
            for (int n = 0; n < grid_n_cells; ++n)
                fprintf(fp, "4 %lu %lu %lu %lu\n", conn[4*n], conn[4*n+1], conn[4*n+2], conn[4*n+3]);
    #endif
    #if (AMREX_SPACEDIM == 3)
            // NODES
            fprintf(fp, "POINTS %lu float\n", grid_n_nodes);
            for (int n = 0; n < grid_n_nodes; ++n)
                fprintf(fp, "%+.12f %+.12f %+.12f\n", nodes[AMREX_SPACEDIM*n], nodes[AMREX_SPACEDIM*n+1], nodes[AMREX_SPACEDIM*n+2]);
            
            // CELLS
            fprintf(fp, "CELLS %lu %lu\n", grid_n_cells, (1+8)*grid_n_cells);
            for (int n = 0; n < grid_n_cells; ++n)
                fprintf(fp, "8 %lu %lu %lu %lu %lu %lu %lu %lu\n", conn[8*n], conn[8*n+1], conn[8*n+2], conn[8*n+3],
                                                                   conn[8*n+4], conn[8*n+5], conn[8*n+6], conn[8*n+7]);
    #endif

            // CELL TYPES
            fprintf(fp, "CELL_TYPES %lu\n", grid_n_cells);
            for (int n = 0; n < grid_n_cells; ++n)
                fprintf(fp, "%d\n", VTK_cell_id);
            // ------------------------------------------------------------

            // WRITE TO FILE: FIELDS INFORMATION --------------------------
            fprintf(fp, "POINT_DATA %lu\n", grid_n_nodes);

            for (int rf = 0; rf < N_PHI; ++rf)
            {
                const std::string field_name = "Grid_PHI_"+std::to_string(rf);
                
                fprintf(fp, "SCALARS %s float 1\n", field_name.c_str());
                fprintf(fp, "LOOKUP_TABLE default\n");
                for (int n = 0; n < grid_n_nodes; ++n)
                {
                    fprintf(fp, "%+.12f\n", fields[n_total_fields*n+rf]);
                }
            }
            for (int rf = 0; rf < n_requested_fields; ++rf)
            {
                fprintf(fp, "SCALARS %s float 1\n", field_names[rf].c_str());
                fprintf(fp, "LOOKUP_TABLE default\n");
                for (int n = 0; n < grid_n_nodes; ++n)
                {
                    fprintf(fp, "%+.12f\n", fields[n_total_fields*n+N_PHI+rf]);
                }
            }
            // ------------------------------------------------------------

            // CLOSE FILE
            fclose(fp);
            // ----------   
        }
        // WRITE OUTPUT DATA: XML FORMAT ==============================
        else if (vtk_xml_format)
        {
            // VARIABLES ----------------------------------------------
            const long n_nodes = n_dG_cells*cell_dom_Ng;
            const long n_cells = conn.size()/AMREX_D_PICK(2, 4, 8);
            const int n_nodal_fields = n_total_fields;

            Vector<VTK::Float_t> VTK_nodes;
            Vector<VTK::Cell_conn_t> VTK_cell_conn;
            Vector<VTK::Cell_offs_t> VTK_cell_offset;
            Vector<VTK::Cell_type_t> VTK_cell_type;
            Vector<Vector<VTK::Float_t>> VTK_nodal_field(n_nodal_fields);
            Vector<std::string> VTK_nodal_field_name(n_nodal_fields);
            // --------------------------------------------------------

            // RESIZE MEMORY ------------------------------------------
            VTK_nodes.resize(3*n_nodes);
            VTK_cell_conn.resize(conn.size());
            VTK_cell_offset.resize(n_cells+1);
            VTK_cell_type.resize(n_cells);
            for (int f = 0; f < n_nodal_fields; ++f)
            {
                VTK_nodal_field[f].resize(n_nodes);
            }
            // --------------------------------------------------------

            // FILL IN DATA -------------------------------------------
            for (int n = 0; n < n_nodes; ++n)
            {
#if (AMREX_SPACEDIM == 2)
                VTK_nodes[3*n+0] = nodes[AMREX_SPACEDIM*n+0];
                VTK_nodes[3*n+1] = nodes[AMREX_SPACEDIM*n+1];
                VTK_nodes[3*n+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                VTK_nodes[3*n+0] = nodes[AMREX_SPACEDIM*n+0];
                VTK_nodes[3*n+1] = nodes[AMREX_SPACEDIM*n+1];
                VTK_nodes[3*n+2] = nodes[AMREX_SPACEDIM*n+2];
#endif
            }

            for (int k = 0; k < conn.size(); ++k)
            {
                VTK_cell_conn[k] = conn[k];
            }
            
            VTK_cell_offset[0];
            for (int k = 1; k < (n_cells+1); ++k)
            {
                VTK_cell_offset[k] = VTK_cell_offset[k-1]+AMREX_D_PICK(2, 4, 8);
            }

            for (int k = 0; k < n_cells; ++k)
            {
                VTK_cell_type[k] = VTK_cell_id;
            }

            for (int rph = 0; rph < N_PHI; ++rph) VTK_nodal_field_name[rph] = "PHI_"+std::to_string(rph);
            for (int rf = 0; rf < n_requested_fields; ++rf) VTK_nodal_field_name[N_PHI+rf] = field_names[rf];

            for (int f = 0; f < n_nodal_fields; ++f)
            {
                for (int n = 0; n < n_nodes; ++n)
                {
                    VTK_nodal_field[f][n] = fields[n_nodal_fields*n+f];
                }
            }
            // --------------------------------------------------------

            // DUMP HEADER FILE ---------------------------------------
            if (amrex::ParallelDescriptor::IOProcessor())
            {
                VTK::PrintHeaderFile_VTU(dst_folder, filename_root, time_id, time_id_max, {}, VTK_nodal_field_name);
            }
            // --------------------------------------------------------

            // DUMP DATA FILE -----------------------------------------
            filepath = GetOutputFilepath(dst_folder, filename_root, time_id, time_id_max)+".vtu";

            VTK::PrintUnstructuredGridData_VTU(filepath,
                                               n_nodes,
                                               n_cells,
                                               VTK_nodes,
                                               VTK_cell_conn,
                                               VTK_cell_offset,
                                               VTK_cell_type,
                                               {},
                                               {},
                                               VTK_nodal_field,
                                               VTK_nodal_field_name,
                                               "binary");
            // --------------------------------------------------------
        }
        // ============================================================

        // ALIGN PROCESSORS ==========
        ParallelDescriptor::Barrier();
        // ===========================
    }
    // ################################################################

    // SX
    // Post-process by convolution
    void Convolution_Postprocessing(const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                                const MatrixFactory<N_PHI, N_DOM> & MatFactory)
    {
        // step 1: load and pre-compute necessary data for the kernel

        // DG info
        const int sp = MatFactory.space_p;
        const int sNp_1D = sp+1;
        const int sNp = MatFactory.space_Np;

        const int sp_star = 2*sp + 1;

        // 1D Quadrature points on the reference unithyperrectangle
        // make sure sq >= sp_star
        const int sq = iGeom.uHyperrect.q ;  

        UnitHyperrectangle<1> uHyperrect(sp_star,sq) ; 
        Gpu::ManagedVector<Real> quad_points = uHyperrect.dom_XI1q ;  // sq+1 points
        Gpu::ManagedVector<Real> quad_weights = uHyperrect.dom_Wq ;  // sq+1 weights

        // points of interests
        Gpu::ManagedVector<Real> points = quad_points;
        const int Npoints_1D = points.size();
        

        // compute the convolution matrix
        Gpu::ManagedVector<Real> Conv_matrix ;
        get_convolution_matrix(Conv_matrix,points,sp,quad_points,quad_weights);
       

        // step 2: Convolve with the DG solution
            // 1. go through all elements
            // 2. for each element, find necessary neighbors
            // 3. do the convolution

        // store data for the post-processed solution
        // MultiFab X_star;
        MultiFab X_DG_temp;
        const int one_side_support = 2*sp;
        const int Nj_1D = 4*sp+1 ;

#if (AMREX_SPACEDIM == 1)
        const int Npoints = Npoints_1D;
        const int Nj = Nj_1D;
#endif
#if (AMREX_SPACEDIM == 2)
        const int Npoints = Npoints_1D * Npoints_1D;
        const int Nj = (Nj_1D)*(Nj_1D);
#endif
#if (AMREX_SPACEDIM == 3)
        const int Npoints = Npoints_1D * Npoints_1D * Npoints_1D;
        const int Nj = (Nj_1D)*(Nj_1D)*(Nj_1D);
#endif

        // Multifab to store post-processed data
        X_star.define(iGeom.cc_ba, iGeom.dm, Npoints * N_U, 0 ); // Npoints x N_u components; creat 0 ghoast cells

        X_DG_temp.define(iGeom.cc_ba,iGeom.dm, sNp*N_U, one_side_support); //MultiFab with 2k ghoast cells

        // copy from the DG solution
        MultiFab::Copy(X_DG_temp,this->X_new, 0, 0, sNp*N_U, 0);
        X_DG_temp.FillBoundary(iGeom.geom.periodicity());

        // loop all boxes
        for (MFIter mfi(X_star); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            Array4<Real const> const & X_fab = X_DG_temp.array(mfi);
            Array4<Real> const & X_star_fab = X_star.array(mfi);
            
            // loop all cells in a box
            ParallelFor(bx,N_U,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // loop all points in a cell (i,j,k)
                for (int point_iter=0; point_iter < Npoints; ++point_iter)
                {
                    X_star_fab(i,j,k,point_iter + ru*Npoints) = 0.0;

                    // loop all neighbors of a cell
                    for(int jj=0; jj < Nj; ++jj)
                    { 
                        // loop all basis
                        for(int mm = 0; mm < sNp; ++mm)
                        {
                            int nn_dir[AMREX_SPACEDIM];                      
                            int jj_dir[AMREX_SPACEDIM];
                            int mm_dir[AMREX_SPACEDIM];
                            int temp_i=0;
                            int temp_j=0;
                            int temp_k=0;

                            AMREX_D_TERM
                        (
                            nn_dir[0] = point_iter%Npoints_1D;
                            jj_dir[0] = jj%(Nj_1D);
                            mm_dir[0] = mm%(sNp_1D);
                            temp_i = i - one_side_support + jj_dir[0];,

                            nn_dir[1] = ( (point_iter - nn_dir[0])/Npoints_1D )%Npoints_1D;
                            jj_dir[1] = ( (jj - jj_dir[0])/Nj_1D )%Nj_1D;
                            mm_dir[1] = ((mm - mm_dir[0])/sNp_1D) % (sNp_1D);
                            temp_j = j - one_side_support + jj_dir[1];,

                            nn_dir[2] = ( point_iter - nn_dir[0] - nn_dir[1] * Npoints_1D) /( Npoints_1D * Npoints_1D );
                            jj_dir[2] = (jj - jj_dir[0] - jj_dir[1]*(Nj_1D)) / ( Nj_1D * Nj_1D);
                            mm_dir[2] = ( mm - mm_dir[0] - mm_dir[1]*(sNp_1D) ) / ( (sNp_1D)*(sNp_1D) );
                            temp_k = k - one_side_support + jj_dir[2];

                        )
                        
                             Real temp_var = X_fab(temp_i,temp_j,temp_k,mm+ru*sNp);
                            // Real temp_var = 1.0;
                            
                            for (int tt = 0 ; tt < AMREX_SPACEDIM; ++tt)
                            {
                                temp_var *= Conv_matrix[jj_dir[tt]+mm_dir[tt]*Nj_1D + nn_dir[tt]*Nj_1D*(sNp_1D)];

                            }
                            X_star_fab(i, j, k, point_iter + ru * Npoints) += temp_var;

                        }
                        
                    }  
                    
                    

                     // Print()<<"index i: "<<i<<",j: "<<j<<",k: "<<k<<",point_x: "<<points[point_iter%Npoints_1D]<<",point_y: "<<points[ ( (point_iter - point_iter%Npoints_1D)/Npoints_1D )%Npoints_1D ] <<", "<<"X_star_fab: "<<X_star_fab(i,j,k, point_iter + ru * Npoints)<<std::endl;
                    
                }
                

                //Print()<<"index i: "<<i<<",j: "<<j<<",k: "<<k<<",ru: "<<ru<<", "<<"DG: "<<X_fab_new(i,j,k,ru)<<std::endl;
                //Print()<<"index temp_i: "<<temp_i<<",j: "<<temp_j<<",k: "<<temp_k<<",ru: "<<0<<", "<<"DG: "<<X_fab(temp_i,temp_j,temp_k,0)<<std::endl;

            });
            Gpu::synchronize();
            

        }

        /*

        // step 3: project the post-processed solution 
        UnitHyperrectangle<AMREX_SPACEDIM> dHyperreect(sp_star,sq) ; 
        const int Np = dHyperreect.Np;
        const int dom_Nq = dHyperreect.dom_Nq;
        Real const * dom_Wq_ptr = dHyperreect.dom_Wq.data();
        Real const * dom_BF_ptr = dHyperreect.dom_BF.data(); 
        Real const * iMM_ptr = dHyperreect.iMM.data();
        // MultiFab X_star_DG;

       this->X_star_DG.define(iGeom.cc_ba, iGeom.dm, Np * N_U, 0 );

        for (MFIter mfi(X_star_DG); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            Array4<Real const> const & X_star_fab = X_star.array(mfi);
            Array4<Real> const & X_star_DG_fab = this->X_star_DG.array(mfi);

            // loop over cells
            // do the projection
            ParallelFor(bx,N_U,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL VARIABLES
                Real integrand;
                // PROJECTION
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Eval the integral
                    for (int rs = 0; rs < Np; ++rs)
                    {
                        integrand = 0.0;
                        X_star_DG_fab(i,j,k,rs+ru*Np) = 0.0;
                        //inverse mass matrix times basis_function(quadrature point)

                        for (int cs = 0; cs < Np; ++cs) integrand += iMM_ptr[rs+cs*Np]*dom_BF_ptr[q+cs*dom_Nq];

                        integrand *= X_star_fab(i,j,k,q+ru*dom_Nq);
                        // Sum integral contribution
                        X_star_DG_fab(i,j,k,rs+ru*Np) += integrand*dom_Wq_ptr[q];
                    }

                }

            // Print()<<X_star_DG_fab(i,j,k,0)<<std::endl;
            });
            Gpu::synchronize();

            

            


        }
        */




    }



    void Convolution_Postprocessing_OneSideKernel(const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                                const MatrixFactory<N_PHI, N_DOM> & MatFactory)
    {
        // step 1: load and pre-compute necessary data for the kernel

        // DG info
        const int sp = MatFactory.space_p; //k
        const int sNp_1D = sp+1;  // k+1
        const int sNp = MatFactory.space_Np;  // (k+1)^DIM

        const int sp_star = 2*sp + 1;

        // 1D Quadrature points on the reference unithyperrectangle
        // make sure sq >= sp_star
        const int sq = iGeom.uHyperrect.q ;  

        UnitHyperrectangle<1> uHyperrect(sp_star,sq) ; 
        Gpu::ManagedVector<Real> quad_points = uHyperrect.dom_XI1q ;  // sq+1 points
        Gpu::ManagedVector<Real> quad_weights = uHyperrect.dom_Wq ;  // sq+1 weights

        // points of interests
        Gpu::ManagedVector<Real> points = quad_points;
        const int Npoints_1D = points.size();

        // compute the convolution matrix
        Gpu::ManagedVector<Real> Conv_matrix_sym ;
        get_convolution_matrix(Conv_matrix_sym,points,sp,quad_points,quad_weights);

        // compute one side kernel matrix
        Gpu::ManagedVector<Real> Conv_matrix_Left ;
        Gpu::ManagedVector<Real> Conv_matrix_Right ;
        get_convolution_matrix_one_side_kernel(Conv_matrix_Right,Conv_matrix_Left,points,sp,quad_points,quad_weights)

        // step 2: Convolve with the DG solution
            // 1. go through all elements
            // 2. for each element, find necessary neighbors
            // 3. do the convolution
            //    1) interior element: use symmetric kernel
            //    2) boundary element / interface element: use one side kernel

        MultiFab X_DG_temp;
        const int symmetric_kernl_one_side_support = 2*sp;  // 2k
        const int oneside_kernl_one_side_support = 3*sp;   // 3k
        const int Nj_1D = 4*sp+1 ;

#if (AMREX_SPACEDIM == 1)
        const int Npoints = Npoints_1D;
        // const int Nj = Nj_1D;
#endif
#if (AMREX_SPACEDIM == 2)
        const int Npoints = Npoints_1D * Npoints_1D;
        // const int Nj = (Nj_1D)*(Nj_1D);
#endif
#if (AMREX_SPACEDIM == 3)
        const int Npoints = Npoints_1D * Npoints_1D * Npoints_1D;
        // const int Nj = (Nj_1D)*(Nj_1D)*(Nj_1D);
#endif

        // Multifab to store post-processed data
        X_star.define(iGeom.cc_ba, iGeom.dm, Npoints * N_U, 0 ); // Npoints x N_u components; creat 0 ghoast cells

        //X_DG_temp.define(iGeom.cc_ba,iGeom.dm, sNp*N_U, one_side_support); //MultiFab with 2k ghoast cells
        // copy from the DG solution
        // MultiFab::Copy(X_DG_temp,this->X_new, 0, 0, sNp*N_U, 0);
       // X_DG_temp.FillBoundary(iGeom.geom.periodicity());

       // loop all boxes
       const Box & domain = iGeom.geom.Domain();
        for (MFIter mfi(X_star); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            Array4<Real const> const & X_fab = this->X_new.array(mfi);
            Array4<Real> const & X_star_fab = X_star.array(mfi);
            
            // loop all cells in a box
            ParallelFor(bx,N_U,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // loop all points in a cell (i,j,k)

                // check the neighbors of cell (i,j,k)
                // set how many neighbor cells are avaiable
               
                int fi, fj, fk;
                bool kernel_flag[DG_CELL_N_SPACE_BOUNDARIES];
                    
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    kernel_flag[b] = true;

                    const int dir = b/2;

                    GRID_FACE(i, j, k, b, fi, fj, fk);

                    if (b==0)
                    {
                        if ( (fi - domain.smallEnd(dir)) < symmetric_kernl_one_side_support ) kernel_flag[b] = false;
                    }
                    else if (b == 1)
                    {
                        if ( (domain.bigEnd(dir)+1) - fi < symmetric_kernl_one_side_support  ) kernel_flag[b] = false;   
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (b==2)
                    {
                        if ( (fj - domain.smallEnd(dir)) < symmetric_kernl_one_side_support) kernel_flag[b] = false;
                    }
                    else if (b==3)
                    {
                        if ( (domain.bigEnd(dir)+1) - fj < symmetric_kernl_one_side_support) kernel_flag[b] = false;
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else if (b==4)
                    {
                        if ( (fk - domain.smallEnd(dir)) < symmetric_kernl_one_side_support) kernel_flag[b] = false;
                    }
                    else
                    {
                        if ( (domain.bigEnd(dir)+1) - fk < symmetric_kernl_one_side_support) kernel_flag[b] = false;
                    }
                        
#endif 
                }

                 // determine to use symmetric kernel or one-side kernel
                 int j_low[AMREX_SPACEDIM], j_up[AMREX_SPACEDIM],Nj[AMREX_SPACEDIM];
                 Real * conv_matrix_1;

                 if (kernel_flag[0] && kernel_flag[1])
                 {
                     j_low[0] = - symmetric_kernl_one_side_support;
                     j_up[0] = symmetric_kernl_one_side_support;
                     Nj[0] = Nj_1D;
                     conv_matrix_1 = Conv_matrix_sym;

                 }
                 else if (kernel_flag[0])
                 {
                     j_low[0] = -oneside_kernl_one_side_support;
                     j_up[0] = 0;
                     Nj[0] = oneside_kernl_one_side_support + 1;
                     conv_matrix_1 = Conv_matrix_Left;
                 }
                 else if (kernel_flag[1])
                 {
                     j_low[0]= 0;
                     j_up[0] = oneside_kernl_one_side_support;
                     Nj[0] = oneside_kernl_one_side_support + 1;
                     conv_matrix_1 = Conv_matrix_Right
                 }
                 else
                 {
                     Print()<<"ERROR: no enough elements to post-process in x directions! Refine the mesh"<<std::endl;
                 }


#if (AMREX_SPACEDIM > 1)
                Real * conv_matrix_2;
                if (kernel_flag[2] && kernel_flag[3])
                 {
                     j_low[1] = - symmetric_kernl_one_side_support;
                     j_up[1] = symmetric_kernl_one_side_support;
                     Nj[1] = Nj_1D;
                     conv_matrix_2 = Conv_matrix_sym;

                 }
                 else if (kernel_flag[2])
                 {
                     j_low[1] = -oneside_kernl_one_side_support;
                     j_up[1] = 0;
                     Nj[1] = oneside_kernl_one_side_support + 1;
                     conv_matrix_2 = Conv_matrix_Left;
                 }
                 else if (kernel_flag[3])
                 {
                     j_low[1]= 0;
                     j_up[1] = oneside_kernl_one_side_support;
                     Nj[1] = oneside_kernl_one_side_support + 1;
                     conv_matrix_2 = Conv_matrix_Right
                 }
                 else
                 {
                     Print()<<"ERROR: no enough elements to post-process in y directions! Refine the mesh"<<std::endl;
                 }

#endif
#if (AMREX_SPACEDIM > 2)
                Real * conv_matrix_3;
                if (kernel_flag[4] && kernel_flag[5])
                 {
                     j_low[2] = - symmetric_kernl_one_side_support;
                     j_up[2] = symmetric_kernl_one_side_support;
                     Nj[2] = Nj_1D;
                     conv_matrix_3 = Conv_matrix_sym;

                 }
                 else if (kernel_flag[4])
                 {
                     j_low[2] = -oneside_kernl_one_side_support;
                     j_up[2] = 0;
                     Nj[2] = oneside_kernl_one_side_support + 1;
                     conv_matrix_3 = Conv_matrix_Left;
                 }
                 else if (kernel_flag[5])
                 {
                     j_low[2]= 0;
                     j_up[2] = oneside_kernl_one_side_support;
                     Nj[2] = oneside_kernl_one_side_support + 1;
                     conv_matrix_3 = Conv_matrix_Right
                 }
                 else
                 {
                     Print()<<"ERROR: no enough elements to post-process in z directions! Refine the mesh"<<std::endl;
                 }
#endif
                
                
                for (int point_iter=0; point_iter < Npoints; ++point_iter)
                {
                    for(int mm = 0; mm < sNp; ++mm)
                    {
                        int nn_dir[AMREX_SPACEDIM];                     
                        int mm_dir[AMREX_SPACEDIM];
                        int temp_i=0;
                        int temp_j=0;
                        int temp_k=0;
#if (AMREX_SPACEDIM == 1)
                        for (int j1 = j_low[0]; j1 <= j_up[0]; ++j1 )
                        {
                            nn_dir[0] = point_iter%Npoints_1D;
                            mm_dir[0] = mm%(sNp_1D);
                            temp_i = i + j1;

                            Real temp_var = X_fab(temp_i,temp_j,temp_k,mm+ru*sNp);
                            temp_var *= conv_matrix_1[ j1-j_low[0] + mm_dir[0]*Nj[0] + nn_dir[0]*Nj[0]*(sNp_1D)];

                            X_star_fab(i, j, k, point_iter + ru * Npoints) += temp_var;
                             
                            //temp_var *= Conv_matrix[jj_dir[tt]+mm_dir[tt]*Nj_1D + nn_dir[tt]*Nj_1D*(sNp_1D)];

                            

                        }
                    
#endif
#if (AMREX_SPACEDIM == 2)
                        for (int j1 = j_low[0]; j1 <= j_up[0]; ++j1 )
                        {
                            for (int j2 = j_low[1]; j2 <= j_up[1]; ++j2 )
                            {
                                nn_dir[0] = point_iter%Npoints_1D;
                                mm_dir[0] = mm%(sNp_1D);
                                temp_i = i + j1;

                                nn_dir[1] = ( (point_iter - nn_dir[0])/Npoints_1D )%Npoints_1D;
                                mm_dir[1] = ((mm - mm_dir[0])/sNp_1D) % (sNp_1D);
                                temp_j = j  + j2;

                                Real temp_var = X_fab(temp_i,temp_j,temp_k,mm+ru*sNp);
                                temp_var *= conv_matrix_1[ j1-j_low[0] + mm_dir[0]*Nj[0] + nn_dir[0]*Nj[0]*(sNp_1D)];
                                temp_var *= conv_matrix_2[ j2-j_low[1] + mm_dir[1]*Nj[1] + nn_dir[1]*Nj[1]*(sNp_1D)];

                                X_star_fab(i, j, k, point_iter + ru * Npoints) += temp_var;

                            }
                            
                        }
#endif
#if (AMREX_SPACEDIM == 3)
                        for (int j1 = j_low[0]; j1 <= j_up[0]; ++j1 )
                        {
                            for (int j2 = j_low[1]; j2 <= j_up[1]; ++j2 )
                            {
                                for (int j3 = j_low[2]; j3 <= j_up[2]; ++j3)
                                {
                                    nn_dir[0] = point_iter%Npoints_1D;
                                    mm_dir[0] = mm%(sNp_1D);
                                    temp_i = i + j1;

                                    nn_dir[1] = ( (point_iter - nn_dir[0])/Npoints_1D )%Npoints_1D;
                                    mm_dir[1] = ((mm - mm_dir[0])/sNp_1D) % (sNp_1D);
                                    temp_j = j  + j2;

                                    nn_dir[2] = ( point_iter - nn_dir[0] - nn_dir[1] * Npoints_1D) /( Npoints_1D * Npoints_1D );
                                    mm_dir[2] = ( mm - mm_dir[0] - mm_dir[1]*(sNp_1D) ) / ( (sNp_1D)*(sNp_1D) );
                                    temp_k = k + j3;

                                    Real temp_var = X_fab(temp_i,temp_j,temp_k,mm+ru*sNp);
                                    temp_var *= conv_matrix_1[ j1-j_low[0] + mm_dir[0]*Nj[0] + nn_dir[0]*Nj[0]*(sNp_1D)];
                                    temp_var *= conv_matrix_2[ j2-j_low[1] + mm_dir[1]*Nj[1] + nn_dir[1]*Nj[1]*(sNp_1D)];
                                    temp_var *= conv_matrix_3[ j3-j_low[2] + mm_dir[2]*Nj[2] + nn_dir[2]*Nj[2]*(sNp_1D)];

                                    X_star_fab(i, j, k, point_iter + ru * Npoints) += temp_var;

                                }
                                
                            }
                            
                        }
#endif

                    } // endfor m

                }// endfor point_iter

            }); //end paranell for
            Gpu::synchronize();
        }//endfor MFIter

    } // end func  Convolution_Postprocessing_OneSideKernel

// ####################################################################

// ERROR EVALUATION ###############################################
    template<typename IBVP_CLASS>
    Real PostProcessedEvalErrorNorm(const Real & t,
                       const ImplicitGeometry<N_PHI, N_DOM> & iGeom,
                       const MatrixFactory<N_PHI, N_DOM> & MatFactory,
                       const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("DG::EvalErrorNorm(const Real &, const ImplicitGeometry<N_PHI, N_DOM> &, ....)");
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
        // const Real * space_Dom_BF_mem_ptr = MatFactory.space_Dom_BF_device_mem.data();
#else
        const Real * eDomQuad_mem_ptr = iGeom.eDomQuad_host_mem.data();
        // const Real * space_Dom_BF_mem_ptr = MatFactory.space_Dom_BF_host_mem.data();
#endif

        // DG
        // const int sp = MatFactory.space_p;
        // ============================================================

        // VARIABLES ==================================================
        MultiFab error(iGeom.cc_ba, iGeom.dm, 2, 0);
        Real err;
        // ============================================================

        // EVAL THE ERROR =============================================
        for (MFIter mfi(error); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            
            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<int const> const & eDomQuad_Nq_fab = iGeom.eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = iGeom.eDomQuad_pos.array(mfi);
            // Array4<long const> const & space_Dom_BF_pos_fab = MatFactory.space_Dom_BF_pos.array(mfi);
            Array4<Real const> const & X_fab = this->X_star.array(mfi);
            Array4<Real> const & err_fab = error.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // INITIALIZATION
                err_fab(i,j,k,0) = 0.0;
                err_fab(i,j,k,1) = 0.0;
                // Print()<<X_fab(i,j,k,0)<<std::endl;

                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};

                // LOCAL VARIABLES
                int dom, etype;
                // int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM], w;
                Real Uh[N_U], Ue[N_U];
                Real integrand;

                // QUADRATURE INFO
                int dom_Nq;
                long eDomQuad_pos, qi_pos;// space_BF_pos;
                const Real * xptr;
                // const Real * BF_ptr;

                // Data structures used to compute the basis functions
                // and the unknown fields

                //SX
                // DG_mem_sol_space_BFX sol(X_fab, sp, &iJ[0]);

                // CYCLE OVER THE UNKNOWN FIELDS
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dom = IBVP.F_U2DOM(ru);
                    etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);

                    // Get the support of the basis functions
                    // BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                    // Quadrature info
                    dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    eDomQuad_pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    xptr =  &eDomQuad_mem_ptr[eDomQuad_pos];

                    // Basis functions info
                    //space_BF_pos = space_Dom_BF_pos_fab(i,j,k,DG_DOM_BF_INFO_N_COMP_PER_DOM*dom);
                    //BF_ptr = &space_Dom_BF_mem_ptr[space_BF_pos];

                     //SX
                    // sol.set_BF_ptrs(BF_ptr);

                    // COMPUTE THE INTEGRALS
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
                        //SX
                        // sol._eval_BF_only_table_(qi);
                        for (int cu = 0; cu < N_U; ++cu)
                        {
                            //SX
                            //sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                            
                            
                            //Uh[cu] = sol._eval_unsafe_();
                            Uh[cu] = X_fab(i,j,k, qi + cu*dom_Nq);
                        }
                        
                        // EXACT SOLUTION
                        IBVP.F_EXACT(t, x, Ue);

                        //integrand = std::abs(Ue[ru]);
                        integrand = std::pow(Ue[ru],2.0);
                        err_fab(i,j,k,0) += integrand*w;
                        
                        Ue[ru] -= Uh[ru];
                        //integrand = std::abs(Ue[ru]);
                        integrand = std::pow(Ue[ru],2.0);
                        err_fab(i,j,k,1) += integrand*w;
                    }
                }
            });
            Gpu::synchronize();
        }
        // ============================================================

        // SUM =============================
        Print() << "(error.sum(0)): " << std::sqrt(error.sum(0)) << std::endl;
        Print() << "(error.sum(1)): " << std::sqrt(error.sum(1)) << std::endl;
        
        err = std::sqrt(error.sum(1))/std::sqrt(error.sum(0));
        // err = std::sqrt(error.sum(1));
        // =================================

        return err;
    }
    // ################################################################


};
// ####################################################################

} // namespace DG
} // namespace amrex


#endif
// AMReX_DG_ImplicitGeometry.H

#ifndef BL_DG_IMPLICIT_GEOMETRY_H_
#define BL_DG_IMPLICIT_GEOMETRY_H_

#include <AMReX_DG_Base.H>
#include <AMReX_VTK_utils.H>

namespace amrex
{
namespace DG
{

// AUXILIARY FUNCTIONS ################################################
// ELEMENT INFORMATION ================================================
void PrintElementInfo(const Real * prob_lo, const Real * dx,
                      const int & i, const int & j, const int & k, const int & dom,
                      Array4<int const> const & etype_fab);
void PrintElementInfo(const Real * prob_lo, const Real * dx,
                      const int & i, const int & j, const int & k, const int & dom,
                      Array4<int const> const & etype_fab, Array4<Real const> const & einfo_fab);
//PrintElementInfo(&prob_lo[0], &dx[0], i, j, k, dom, eType_fab, eInfo_fab);
// ====================================================================

// GRID INFORMATION ===================================================
AMREX_GPU_HOST_DEVICE
void NBR_CELL(const int & i, const int & j, const int & k, const int & b,
              int & nbr_i, int & nbr_j, int & nbr_k, int & nbr_b);

AMREX_GPU_HOST_DEVICE
void GRID_FACE(const int & i, const int & j, const int & k, const int & b,
               int & fi, int & fj, int & fk);

AMREX_GPU_HOST_DEVICE
void FACE_2_NBRS(const int & i, const int & j, const int & k, const int & dir,
                 int & min_i, int & min_j, int & min_k,
                 int & pls_i, int & pls_j, int & pls_k);

AMREX_GPU_HOST_DEVICE
void BF_CELL(const int & i, const int & j, const int & k, const int & etype,
             int & BF_i, int & BF_j, int & BF_k);

AMREX_GPU_HOST_DEVICE
void BF_CELL(const Real* prob_lo, const Real* dx,
             const int & i, const int & j, const int & k, const int & etype,
             int & BF_i, int & BF_j, int & BF_k,
             Real * BF_lo, Real * BF_hi);

AMREX_GPU_HOST_DEVICE
void BF_CELL(const Real * prob_lo, const Real * dx,
             const int & i, const int & j, const int & k, const int & etype,
             Real * BF_lo, Real * BF_hi);
// ====================================================================

// ALGOIM RELATED FUNCTIONS ===========================================
template<typename F>
iQuadRule iQuadGen(const F & phi, const Real * lo, const Real * hi, const int & dim, const int & side, const int & qo)
{
    return Algoim::quadGen<AMREX_SPACEDIM>(phi, Algoim::BoundingBox<Real,AMREX_SPACEDIM>(lo, hi), dim, side, qo);
}

template<typename F>
iTriRule iTriGen(const F & phi, const Real * lo, const Real * hi, const int & dim, const int & side, const int & qo)
{
    return Algoim::quadGen<AMREX_SPACEDIM>(phi, Algoim::BoundingBox<Real,AMREX_SPACEDIM>(lo, hi), dim, side, qo, [] (int p, int i) {return 1.0e-12+i*(1.0-2.0e-12)/(p-1);}, [] (int p, int i) {return (1.0-2.0e-12)/(p-1);});
}
// ====================================================================
// ####################################################################


// ####################################################################
// IMPLICIT GEOMETRY CLASS ############################################
// ####################################################################
template<int N_PHI, int N_DOM>
struct ImplicitGeometry
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;
    static const int bou_data_ngr = 0;

    static const int eType_n_comp = DG_ELM_TYPE_N_COMP_PER_DOM*N_DOM;
    static const int eInfo_n_comp = DG_ELM_INFO_N_COMP_PER_DOM*N_DOM;
    static const int eBouType_n_comp = DG_ELM_BOU_TYPE_N_COMP_PER_DOM*N_DOM;
    static const int eDomQuad_Nq_n_comp = DG_ELM_DOM_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eBouQuad_Nq_n_comp = DG_ELM_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eIntBouQuad_Nq_n_comp = DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eTri_Npt_n_comp = DG_ELM_TRI_N_COMP_PER_DOM*N_DOM;
    static const int eDomQuad_pos_n_comp = DG_ELM_DOM_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eBouQuad_pos_n_comp = DG_ELM_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eIntBouQuad_pos_n_comp = DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eTri_pos_n_comp = DG_ELM_TRI_N_COMP_PER_DOM*N_DOM;

    // PARAMETERS
    const Box cc_ibx;
    const RealBox rbx;
    const BoxArray cc_ba;
    const Array<BoxArray, AMREX_SPACEDIM> fc_ba;
    const DistributionMapping dm;
    const Geometry geom;

    const GpuArray<Real, AMREX_SPACEDIM> _dx, _prob_lo;
    const GpuArray<int, AMREX_SPACEDIM> _is_periodic;
    const GpuArray<Real, AMREX_SPACEDIM> _J, _iJ;
    const Real _aJ, _iaJ;
    const GpuArray<Real, AMREX_SPACEDIM> _bou_aJ;
    const Real _tol;

    /*const*/ UnitHyperrectangle<AMREX_SPACEDIM> uHyperrect;

    const int PHI_n_comp;

    // VARIABLES
    MultiFab PHI;

    iMultiFab eType;
    MultiFab eInfo;
    iMultiFab eDomQuad_Nq, eTri_Npt;
    longMultiFab eDomQuad_pos, eTri_pos;
    
    Array<iMultiFab, AMREX_SPACEDIM> eBouType;
    Array<iMultiFab, AMREX_SPACEDIM> eBouQuad_Nq;
    Array<longMultiFab, AMREX_SPACEDIM> eBouQuad_pos;

    iMultiFab eIntBouQuad_Nq;
    longMultiFab eIntBouQuad_pos;

    int n_cut_elms, n_cut_elm_bou, n_entire_extended_elms;
    long mem_for_eDomQuad, mem_for_eBouQuad, mem_for_eIntBouQuad, mem_for_eTri;

    Gpu::HostVector<Real> eDomQuad_host_mem;
    Gpu::HostVector<Real> eBouQuad_host_mem;
    Gpu::HostVector<Real> eIntBouQuad_host_mem;
    Gpu::HostVector<Real> eTri_host_mem;
    
    Gpu::DeviceVector<Real> eDomQuad_device_mem;
    Gpu::DeviceVector<Real> eBouQuad_device_mem;
    Gpu::DeviceVector<Real> eIntBouQuad_device_mem;
    Gpu::DeviceVector<Real> eTri_device_mem;

    // POST-PROCESSING
    const int pp_ratio;
    const Box pp_cc_ibx;
    BoxArray pp_cc_ba;
    DistributionMapping pp_dm;
    Geometry pp_geom;

    const UnitHyperrectangle<AMREX_SPACEDIM> pp_uHyperrect;

    MultiFab pp_PHI;
    // ################################################################

    // CONSTRUCTOR ####################################################
    ImplicitGeometry(const Box & cc_ibx_,
                     const RealBox & rbx_,
                     const BoxArray & cc_ba_,
                     const DistributionMapping & dm_,
                     const Geometry & geom_,
                     const int & p_,
                     const int & q_)
    :
    cc_ibx(cc_ibx_),
    rbx(rbx_),
    cc_ba(cc_ba_),
#if (AMREX_SPACEDIM == 1)
    fc_ba({convert(cc_ba_, IntVect(1))}),
#endif
#if (AMREX_SPACEDIM == 2)
    fc_ba({convert(cc_ba_, IntVect(1,0)), convert(cc_ba_, IntVect(0,1))}),
#endif
#if (AMREX_SPACEDIM == 3)
    fc_ba({convert(cc_ba_, IntVect(1,0,0)), convert(cc_ba_, IntVect(0,1,0)), convert(cc_ba_, IntVect(0,0,1))}),
#endif
    dm(dm_),
    geom(geom_),
    _dx(geom_.CellSizeArray()),
    _prob_lo(geom_.ProbLoArray()),
    _is_periodic(geom_.isPeriodicArray()),
    _J({AMREX_D_DECL(0.5*this->_dx[0], 0.5*this->_dx[1], 0.5*this->_dx[2])}),
    _iJ({AMREX_D_DECL(2.0/this->_dx[0], 2.0/this->_dx[1], 2.0/this->_dx[2])}),
    _aJ(AMREX_D_PICK(this->_J[0], (this->_J[0])*(this->_J[1]), (this->_J[0])*(this->_J[1])*(this->_J[2]))),
    _iaJ(1.0/this->_aJ),
#if (AMREX_SPACEDIM == 1)
    _bou_aJ({1.0}),
    _tol(1.0e-12*this->_dx[0]),
#endif
#if (AMREX_SPACEDIM == 2)
    _bou_aJ({this->_J[1], this->_J[0]}),
    _tol(1.0e-12*fmin(this->_dx[0], this->_dx[1])),
#endif
#if (AMREX_SPACEDIM == 3)
    _bou_aJ({this->_J[1]*this->_J[2], this->_J[0]*this->_J[2], this->_J[0]*this->_J[1]}),
    _tol(1.0e-12*fmin(fmin(this->_dx[0], this->_dx[1]), this->_dx[2])),
#endif
    uHyperrect(p_, q_),
    PHI_n_comp((this->uHyperrect.Np)*N_PHI),
    n_cut_elms(0),
    n_cut_elm_bou(0),
    n_entire_extended_elms(0),
    mem_for_eDomQuad(0L),
    mem_for_eBouQuad(0L),
    mem_for_eIntBouQuad(0L),
    mem_for_eTri(0L),
    pp_ratio(2*p_+1),
    pp_cc_ibx(refine(cc_ibx, this->pp_ratio)),
    pp_cc_ba(cc_ba),
    pp_uHyperrect(0, 1)
    {
        // SOME CHECKS ================================================
        BL_ASSERT(p_ >= 1);
        BL_ASSERT(p_ <= DG_PHI_MAX_SPACE_p);
        if (p_ != DG_PHI_MAX_SPACE_p)
        {
            Print() << std::endl;
            Print() << "WARNING: AMReX_DG_ImplicitGeometry.H - ImplicitGeometry::ImplicitGeometry" << std::endl;
            Print() << "| DG_PHI_MAX_SPACE_p = " << DG_PHI_MAX_SPACE_p << std::endl;
            Print() << "| input phi space p = " << p_ << std::endl;
            Print() << "| Consider setting DG_PHI_MAX_SPACE_p = " << p_ << " and recompiling the code." <<std::endl;
            Print() << std::endl;
        }
        // ============================================================

        // STORE MEMORY FOR THE MULTIFABS =============================
        // DISTANCE FUNCTIONS -----------------------------------------
        this->PHI.define(this->cc_ba, this->dm, this->PHI_n_comp, this->dom_data_ngr);
        
        this->PHI = 0.0;
        // ------------------------------------------------------------

        // IMPLICIT QUADRATURE INFORMATION ----------------------------
        // WARNING: Remember that on the MultiFabs
        //          - this->eDomQuad_pos
        //          - this->eBouQuad_pos[dim]
        //          - this->eTri_pos
        //          the FillBoundary method must NEVER be called.
        // ------------------------------------------------------------
        // DOMAIN (CELL-CENTERED) INFO
        this->eType.define(this->cc_ba, this->dm, this->eType_n_comp, this->dom_data_ngr);
        this->eInfo.define(this->cc_ba, this->dm, this->eInfo_n_comp, this->dom_data_ngr);
        this->eDomQuad_Nq.define(this->cc_ba, this->dm, this->eDomQuad_Nq_n_comp, this->dom_data_ngr);
        this->eTri_Npt.define(this->cc_ba, this->dm, this->eTri_Npt_n_comp, this->dom_data_ngr);
        this->eDomQuad_pos.define(this->cc_ba, this->dm, this->eDomQuad_pos_n_comp, this->dom_data_ngr);
        this->eTri_pos.define(this->cc_ba, this->dm, this->eTri_pos_n_comp, this->dom_data_ngr);

        this->eType = DG_ELM_TYPE_UNDEFINED;
        this->eInfo = 0.0;
        this->eDomQuad_Nq = -1;
        this->eTri_Npt = -1;
        this->eDomQuad_pos = -1L;
        this->eTri_pos = -1L;

        this->eIntBouQuad_Nq.define(this->cc_ba, this->dm, this->eIntBouQuad_Nq_n_comp, this->dom_data_ngr);
        this->eIntBouQuad_pos.define(this->cc_ba, this->dm, this->eIntBouQuad_pos_n_comp, this->dom_data_ngr);

        this->eIntBouQuad_Nq = -1;
        this->eIntBouQuad_pos = -1L;

        // BOUNDARY (FACE-CENTERED) INFO
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->eBouType[dim].define(this->fc_ba[dim], this->dm, this->eBouType_n_comp, this->bou_data_ngr);
            this->eBouType[dim] = DG_ELM_BOU_TYPE_UNDEFINED;

            this->eBouQuad_Nq[dim].define(this->fc_ba[dim], this->dm, this->eBouQuad_Nq_n_comp, this->bou_data_ngr);
            this->eBouQuad_pos[dim].define(this->fc_ba[dim], this->dm, this->eBouQuad_pos_n_comp, this->bou_data_ngr);

            this->eBouQuad_Nq[dim] = -1;
            this->eBouQuad_pos[dim] = -1L;
        }
        // ------------------------------------------------------------
        // ============================================================

        // POST-PROCESSING ============================================
        this->pp_cc_ba.refine(this->pp_ratio);
        this->pp_dm.define(this->pp_cc_ba);
        this->pp_geom.define(this->pp_cc_ibx, &rbx_, this->geom.CoordInt(), this->geom.isPeriodic().data());

        this->pp_PHI.define(this->pp_cc_ba, this->pp_dm, N_PHI, this->dom_data_ngr);

        this->pp_PHI = 0.0;
        // ============================================================
    }
    // ################################################################

    // DESTRUCTOR #####################################################
    ~ImplicitGeometry()
    {}
    // ################################################################

    // PROJECT DISTANCE FUNCTIONS #####################################
    template<typename IBVP_CLASS>
    void ProjectDistanceFunctions(const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("ImplicitGeometry::ProjectDistanceFunctions(const IBVP_CLASS &)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = this->_dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = this->_prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = this->_J;

        // STANDARD ELEMENT
        const int Np = this->uHyperrect.Np;
        const int dom_Nq = this->uHyperrect.dom_Nq;

        AMREX_D_TERM
        (
            Real const * dom_XI1q_ptr = this->uHyperrect.dom_XI1q.data();,
            Real const * dom_XI2q_ptr = this->uHyperrect.dom_XI2q.data();,
            Real const * dom_XI3q_ptr = this->uHyperrect.dom_XI3q.data();
        )
        Real const * dom_Wq_ptr = this->uHyperrect.dom_Wq.data();
        Real const * dom_BF_ptr = this->uHyperrect.dom_BF.data(); 
        
        Real const * iMM_ptr = this->uHyperrect.iMM.data();
        // ============================================================

        // INITIALIZATION =============================================
        this->PHI = 0.0;
        // ============================================================

        // IMPORTANT NOTE ---------------------------------------------
        // Because we will be working with pointers, we have to store
        // quadrature info in the ghost rows. This is why we will be
        // looping over the fabbox instead of the validbox.
        // ------------------------------------------------------------

        // EVAL THE PROJECTION ========================================
        for (MFIter mfi(this->PHI); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            Array4<Real> const & PHI_fab = this->PHI.array(mfi);
                
            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};
                
                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM];
                Real integrand;

                // PROJECTION
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Quadrature point
                    AMREX_D_TERM
                    (
                        x[0] = J[0]*dom_XI1q_ptr[q]+cell_m[0];,
                        x[1] = J[1]*dom_XI2q_ptr[q]+cell_m[1];,
                        x[2] = J[2]*dom_XI3q_ptr[q]+cell_m[2];
                    )

                    // Eval the integral
                    for (int ru = 0; ru < N_PHI; ++ru)
                    for (int rs = 0; rs < Np; ++rs)
                    {
                        integrand = 0.0;
                        for (int cs = 0; cs < Np; ++cs) integrand += iMM_ptr[rs+cs*Np]*dom_BF_ptr[q+cs*dom_Nq];
                        integrand *= IBVP.F_PHI(ru, 0.0, x);

                        // Sum integral contribution
                        PHI_fab(i,j,k,rs+ru*Np) += integrand*dom_Wq_ptr[q];
                    }
                }

            });
            Gpu::synchronize();
            
        }
        this->PHI.FillBoundary(this->geom.periodicity());
        // ============================================================
    }
    // ################################################################

    // IMPLICIT MESH EVALUATION #######################################
    template<typename IBVP_CLASS>
    void EvalImplicitMesh(const IBVP_CLASS & IBVP, const bool & im_flag = true)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("ImplicitGeometry::EvalImplicitMesh(const IBVP_CLASS & IBVP)");
        // ------------------------------------------------------------

#if (AMREX_SPACEDIM == 1)
        printf("\n");
        printf("ERROR: AMReX_DG_ImplicitGeometry.H - ImplicitGeometry::EvalImplicitMesh\n");
        printf("| The method EvalImplicitMesh must be suitably modified to handle AMREX_SPACEDIM = 1.\n");
        printf("\n");
        exit(-1);
#endif

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = this->_dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = this->_prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = this->_J;
        const Real aJ = this->_aJ;
        const GpuArray<Real, AMREX_SPACEDIM> bou_aJ = this->_bou_aJ;
        const Real tol = this->_tol;

        // STANDARD ELEMENT
        const int p = this->uHyperrect.p;
        const int qo = this->uHyperrect.q+1;
        // ============================================================

        // VARIABLES ==================================================
        long eDomQuad_mem_pos, eTri_mem_pos;
        long eBouQuad_mem_pos;
        long eIntBouQuad_mem_pos;

        // QUADRATURE RULES CHECKING
        Real volume[N_DOM], volume_from_bou_quadrature[N_DOM], surface[N_DOM];
        // ============================================================

        // INITIALIZATION =============================================
        this->n_cut_elms = 0;
        eDomQuad_mem_pos = 0L;
        eTri_mem_pos = 0L;

        this->n_cut_elm_bou = 0;
        eBouQuad_mem_pos = 0L;

        eIntBouQuad_mem_pos = 0L;

        this->n_entire_extended_elms = 0;
        // ============================================================

        // IMPORTANT NOTE ---------------------------------------------
        // Because we are working with pointers, we have to store
        // quadrature info in the ghost rows. This is why we will be
        // looping over the fabbox instead of the validbox.
        // ------------------------------------------------------------

        // STANDARD ELEMENT QUADRATURE INFO ===========================
        // DOMAIN: QUADRATURE
        {
            // QUADRATURE INFO
            const int dom_Nq = this->uHyperrect.dom_Nq;

            // Resize memory
            this->eDomQuad_host_mem.resize((AMREX_SPACEDIM+1)*dom_Nq);

            // Store data
            for (int qi = 0; qi < dom_Nq; ++qi)
            {
                AMREX_D_TERM
                (
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+0] = this->uHyperrect.dom_XI1q[qi];,
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+1] = this->uHyperrect.dom_XI2q[qi];,
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+2] = this->uHyperrect.dom_XI3q[qi];
                )
                this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+AMREX_SPACEDIM] = this->uHyperrect.dom_Wq[qi];
            }

            // Update memory (nodes coordinates + weights)
            eDomQuad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
        }

        // DOMAIN: QUERY POINTS
        {
            // GRID INFO
            const int dom_Ng = this->uHyperrect.dom_Ng;

            // Resize memory
            this->eTri_host_mem.resize(AMREX_SPACEDIM*dom_Ng);

            // Store data
            for (int qi = 0; qi < dom_Ng; ++qi)
            {
                AMREX_D_TERM
                (
                    this->eTri_host_mem[AMREX_SPACEDIM*qi+0] = this->uHyperrect.dom_XI1g[qi];,
                    this->eTri_host_mem[AMREX_SPACEDIM*qi+1] = this->uHyperrect.dom_XI2g[qi];,
                    this->eTri_host_mem[AMREX_SPACEDIM*qi+2] = this->uHyperrect.dom_XI3g[qi];
                )
            }

            // Update memory (nodes coordinates)
            eTri_mem_pos += AMREX_SPACEDIM*dom_Ng;
        }

        // BOUNDARIES
        {
            // QUADRATURE INFO
            const int bou_Nq = this->uHyperrect.bou_Nq;

            // Resize memory
            this->eBouQuad_host_mem.resize((AMREX_SPACEDIM+1)*bou_Nq*DG_CELL_N_SPACE_BOUNDARIES);

            // Store data
            for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                AMREX_D_TERM
                (
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+0] = this->uHyperrect.bou_XI1q[bou_Nq*b+qi];,
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+1] = this->uHyperrect.bou_XI2q[bou_Nq*b+qi];,
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+2] = this->uHyperrect.bou_XI3q[bou_Nq*b+qi];
                )
                this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+AMREX_SPACEDIM] = this->uHyperrect.bou_Wq[qi];
            }

            // Update memory (nodes coordinates + weights)
            eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq*DG_CELL_N_SPACE_BOUNDARIES;
        }
        // ============================================================

        // STORE THE IMPLICIT MESH INFO: DOMAIN (CELL-CENTERED) DATA ==
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            Array4<Real const> const & PHI_fab = this->PHI.array(mfi);
            Array4<int> const & eType_fab = this->eType.array(mfi);
            Array4<Real> const & eInfo_fab = this->eInfo.array(mfi);
            Array4<int> const & eDomQuad_Nq_fab = this->eDomQuad_Nq.array(mfi);
            Array4<long> const & eDomQuad_pos_fab = this->eDomQuad_pos.array(mfi);
            Array4<int> const & eTri_Npt_fab = this->eTri_Npt.array(mfi);
            Array4<long> const & eTri_pos_fab = this->eTri_pos.array(mfi);
            Array4<int> const & eIntBouQuad_Nq_fab = this->eIntBouQuad_Nq.array(mfi);
            Array4<long> const & eIntBouQuad_pos_fab = this->eIntBouQuad_pos.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0], prob_lo[1]+j*dx[1], prob_lo[2]+k*dx[2])};
                const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0], prob_lo[1]+(j+1)*dx[1], prob_lo[2]+(k+1)*dx[2])};
                const Real entire_cell_volume = AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);

                // LOCAL VARIABLES
                bool is_cell_empty, is_cell_entire, is_cell_cut;
                Real cell_volume, volume_fraction;
                int phi_info[2];
                algoim_DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, p, i, j, k, -1);

                // Get info about the distance function associated to
                // the domain
                IBVP.F_DOM2PHI(dom, phi_info);
                phi.set_fab_ptrs(i, j, k, phi_info[0]);
                phi.set_sign(phi_info[1]);
                
                // DOMAIN: QUADRATURE
                {
                    // Algoim data structure
                    iQuadRule quad;
                    if (im_flag)
                    {
                        quad = iQuadGen(phi, cell_lo, cell_hi, -1, -1, qo);

                        // Compute the volume fraction
                        cell_volume = quad([](const auto& x) {return 1.0;});
                        volume_fraction = cell_volume/entire_cell_volume;
                    }
                    else
                    {
                        volume_fraction = 1.0;
                    } 

                    // EMPTY ELEMENTS ---------------------------------
                    if (volume_fraction < tol)
                    {
                        is_cell_empty = true;
                        is_cell_entire = false;
                        is_cell_cut = false;

                        // ELEMENT TYPE
                        eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_EMPTY;
                        
                        // ELEMENT INFO
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom) = 0.0;
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1) = cell_volume;

                        // QUADRATURE INFO
                        eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = 0;
                        eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = -1L;
                    }
                    // ------------------------------------------------
                    // ENTIRE ELEMENTS --------------------------------
                    else if (volume_fraction > 1.0-tol)
                    {
                        is_cell_empty = false;
                        is_cell_entire = true;
                        is_cell_cut = false;

                        // ELEMENT INFO
                        eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_ENTIRE;
                        
                        // ELEMENT INFO
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom) = 1.0;
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1) = cell_volume;

                        // QUADRATURE INFO
                        eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = this->uHyperrect.dom_Nq;
                        eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = 0L;
                    }
                    // ------------------------------------------------
                    // PARTIAL ELEMENTS -------------------------------
                    else
                    {
                        is_cell_empty = false;
                        is_cell_entire = false;
                        is_cell_cut = true;

                        this->n_cut_elms += 1;
                         
                        // ELEMENT TYPE
                        eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_PARTIAL;
                        
                        // ELEMENT INFO
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom) = volume_fraction;
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1) = cell_volume;

                        // QUADRATURE INFO
                        const int dom_Nq = quad.nodes.size();
                        
                        // Store memory offset
                        eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = dom_Nq;
                        eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = eDomQuad_mem_pos;

                        // Expand memory
                        this->eDomQuad_host_mem.resize(eDomQuad_mem_pos+(AMREX_SPACEDIM+1)*dom_Nq);

                        // Store data: remember that we store the
                        // quadrature abscissae and weights as if they
                        // were computed on a [-1,1]^AMREX_SPACEDIM
                        // cell
                        for (int qi = 0; qi < dom_Nq; ++qi)
                        {
                            long pos = eDomQuad_mem_pos+(AMREX_SPACEDIM+1)*qi;

                            AMREX_D_TERM
                            (
                                this->eDomQuad_host_mem[pos+0] = (quad.nodes[qi].x[0]-cell_m[0])/J[0];,
                                this->eDomQuad_host_mem[pos+1] = (quad.nodes[qi].x[1]-cell_m[1])/J[1];,
                                this->eDomQuad_host_mem[pos+2] = (quad.nodes[qi].x[2]-cell_m[2])/J[2];
                            )
                            this->eDomQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w/aJ;
                        }

                        // Update memory (nodes coordinates + weights)
                        eDomQuad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
                    }
                    // ------------------------------------------------
                }

                // DOMAIN: QUERY POINTS
                {
                    // EMPTY ELEMENTS ---------------------------------
                    if (is_cell_empty)
                    {
                        // GRID INFO
                        eTri_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = 0;
                        eTri_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = -1L;
                    }
                    // ------------------------------------------------
                    // ENTIRE ELEMENTS --------------------------------
                    else if (is_cell_entire)
                    {
                        // GRID INFO
                        eTri_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = this->uHyperrect.dom_Ng;
                        eTri_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = 0L;
                    }
                    // ------------------------------------------------
                    // PARTIAL ELEMENTS -------------------------------
                    else
                    {
                        iTriRule tri = iTriGen(phi, cell_lo, cell_hi, -1, -1, qo);

                        // GRID INFO
                        const int dom_Npt = tri.nodes.size();

                        // Store memory offset
                        eTri_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = dom_Npt;
                        eTri_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = eTri_mem_pos;

                        // Expand memory
                        this->eTri_host_mem.resize(eTri_mem_pos+AMREX_SPACEDIM*dom_Npt);

                        // Store data: remember that we store the
                        // query points as if they  were computed on a
                        // [-1,1]^AMREX_SPACEDIM cell
                        for (int qi = 0; qi < dom_Npt; ++qi)
                        {
                            long pos = eTri_mem_pos+AMREX_SPACEDIM*qi;

                            AMREX_D_TERM
                            (
                                this->eTri_host_mem[pos+0] = (tri.nodes[qi].x[0]-cell_m[0])/J[0];,
                                this->eTri_host_mem[pos+1] = (tri.nodes[qi].x[1]-cell_m[1])/J[1];,
                                this->eTri_host_mem[pos+2] = (tri.nodes[qi].x[2]-cell_m[2])/J[2];
                            )
                        }

                        // Update memory (nodes coordinates)
                        eTri_mem_pos += AMREX_SPACEDIM*dom_Npt;
                    }
                    // ------------------------------------------------
                }

                // INTERNAL BOUNDARY (IF ANY)
                if (is_cell_cut)
                {
                    iQuadRule quad = iQuadGen(phi, cell_lo, cell_hi, AMREX_SPACEDIM, -1, qo);

                    // QUADRATURE INFO
                    const int bou_Nq = quad.nodes.size();
                    Real un[AMREX_SPACEDIM], tmp;

                    // Store memory offset
                    eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom) = bou_Nq;
                    eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom) = eIntBouQuad_mem_pos;

                    // Expand memory
                    this->eIntBouQuad_host_mem.resize(eIntBouQuad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq);

                    // Store data
                    for (int qi = 0; qi < bou_Nq; ++qi)
                    {
                        long pos = eIntBouQuad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;

                        AMREX_D_TERM
                        (
                            this->eIntBouQuad_host_mem[pos+0] = quad.nodes[qi].x[0];,
                            this->eIntBouQuad_host_mem[pos+1] = quad.nodes[qi].x[1];,
                            this->eIntBouQuad_host_mem[pos+2] = quad.nodes[qi].x[2];
                        )
                        this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w;

                        phi.eval_grad(&quad.nodes[qi].x[0], un);
                        tmp = 1.0/std::sqrt(AMREX_D_PICK(un[0]*un[0], un[0]*un[0]+un[1]*un[1], un[0]*un[0]+un[1]*un[1]+un[2]*un[2]));
                        AMREX_D_TERM
                        (
                            un[0] *= tmp;,
                            un[1] *= tmp;,
                            un[2] *= tmp;
                        )

                        AMREX_D_TERM
                        (
                            this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+0] = un[0];,
                            this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+1] = un[1];,
                            this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+2] = un[2];
                        )
                    }

                    // Update memory (nodes coordinates + weights +
                    // unit normal components)
                    eIntBouQuad_mem_pos += (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq;
                }
            }
        }

#if (DG_VERBOSITY == 1)
        // PRINT A BRIEF REPORT ---------------------------------------
        {
            ParallelDescriptor::Barrier();
            Print() << "REPORT:" << std::endl;

            for (int rank = 0; rank < ParallelDescriptor::NProcs(); ++rank)
            {
                if (ParallelDescriptor::MyProc() == rank)
                {
                    printf("| Hello, I am processor %d and I have allocated:\n", ParallelDescriptor::MyProc());
                    printf("| %lu doubles (%f MB) for implicit domain quadrature;\n", eDomQuad_mem_pos, eDomQuad_mem_pos*sizeof(Real)/(1024.0*1024.0));
                    printf("| %lu doubles (%f MB) for implicit internal boundary quadrature;\n", eIntBouQuad_mem_pos, eIntBouQuad_mem_pos*sizeof(Real)/(1024.0*1024.0));
                    printf("| %lu doubles (%f MB) for query points.\n", eTri_mem_pos, eTri_mem_pos*sizeof(Real)/(1024.0*1024.0));
                }

                ParallelDescriptor::Barrier();
            }

            ParallelDescriptor::Barrier();
            Print() << std::endl;
        }
        // ------------------------------------------------------------
#endif
        // STORE MEMORY INFO
        this->mem_for_eDomQuad = eDomQuad_mem_pos;
        this->mem_for_eIntBouQuad = eIntBouQuad_mem_pos;
        this->mem_for_eTri = eTri_mem_pos;
        // ============================================================

        // STORE THE IMPLICIT MESH INFO: BOUNDARY (FACE-CENTERED) DATA
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            const int flag[AMREX_SPACEDIM] = {AMREX_D_DECL((int) (dir == 0), (int) (dir == 1), (int) (dir == 2))};
            const Real delta[AMREX_SPACEDIM] = {AMREX_D_DECL(dx[0]*flag[0], dx[1]*flag[1], dx[2]*flag[2])};

            for (MFIter mfi(this->eBouQuad_Nq[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                Array4<Real const> const & PHI_fab = this->PHI.array(mfi);
                Array4<int> const & eBouType_fab = this->eBouType[dir].array(mfi);
                Array4<int> const & eBouQuad_Nq_fab = this->eBouQuad_Nq[dir].array(mfi);
                Array4<long> const & eBouQuad_pos_fab = this->eBouQuad_pos[dir].array(mfi);

                for (int dom = 0; dom < N_DOM; ++dom)
                for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    // IMPORTANT NOTE ---------------------------------
                    // Please, consider that we are using the indices
                    // defined from the face-centered box to access the
                    // data stored in the cell-centered PHI_fab.
                    // ------------------------------------------------

                    // LOCAL PARAMETERS
                    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0], prob_lo[1]+j*dx[1], prob_lo[2]+k*dx[2])};
                    const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0], prob_lo[1]+(j+1)*dx[1], prob_lo[2]+(k+1)*dx[2])};
                    const Real entire_boundary_surface = AMREX_D_PICK(1.0, dx[1], dx[1]*dx[2]);

                    // LOCAL VARIABLES
                    Real boundary_surface, surface_fraction;
                    int phi_info[2];
                    algoim_DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, p, i, j, k, -1);

                    // Get info about the distance function associated to
                    // the domain
                    IBVP.F_DOM2PHI(dom, phi_info);
                    phi.set_fab_ptrs(i, j, k, phi_info[0]);
                    phi.set_sign(phi_info[1]);

                    // BOUNDARY QUADRATURE
                    const int b = 2*dir;

                    // Algoim data structure
                    iQuadRule quad;
                    if (im_flag)
                    {
                        quad = iQuadGen(phi, cell_lo, cell_hi, b/2, b%2, qo);

                        // Compute the surface fraction
                        boundary_surface = quad([](const auto& x) {return 1.0;});
                        surface_fraction = boundary_surface/entire_boundary_surface;
                    }
                    else
                    {
                        surface_fraction = 1.0;
                    }

                    if (surface_fraction < tol)
                    {
                        // ELEMENT BOUNDARY TYPE
                        eBouType_fab(i,j,k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_BOU_TYPE_EMPTY;

                        // QUADRATURE INFO
                        // BOUNDARY^-
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = 0;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = -1L;

                        // BOUNDARY^+
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = 0;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = -1L;
                    }
                    else if (surface_fraction > 1.0-tol)
                    {
                        // ELEMENT BOUNDARY TYPE
                        eBouType_fab(i,j,k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_BOU_TYPE_ENTIRE;

                        // QUADRATURE INFO
                        const int bou_Nq = this->uHyperrect.bou_Nq;
                        
                        // BOUNDARY^-
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = (AMREX_SPACEDIM+1)*bou_Nq*(b+1);

                        // BOUNDARY^+
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = (AMREX_SPACEDIM+1)*bou_Nq*b;
                    }
                    else
                    {
                        // ELEMENT BOUNDARY TYPE
                        eBouType_fab(i,j,k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_BOU_TYPE_PARTIAL;

                        // QUADRATURE INFO
                        const int bou_Nq = quad.nodes.size();

                        // BOUNDARY^-
                        // Store memory offset
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = eBouQuad_mem_pos;

                        // Expand memory
                        this->eBouQuad_host_mem.resize(eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*bou_Nq);

                        // Store data: remember that we store the
                        // quadrature abscissae and weights as if they
                        // were computed on a [-1,1]^AMREX_SPACEDIM
                        // cell
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            long pos = eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*qi;

                            AMREX_D_TERM
                            (
                                this->eBouQuad_host_mem[pos+0] = (quad.nodes[qi].x[0]+delta[0]-cell_m[0])/J[0];,
                                this->eBouQuad_host_mem[pos+1] = (quad.nodes[qi].x[1]+delta[1]-cell_m[1])/J[1];,
                                this->eBouQuad_host_mem[pos+2] = (quad.nodes[qi].x[2]+delta[2]-cell_m[2])/J[2];
                            )
                            this->eBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w/bou_aJ[dir];
                        }

                        // Update memory (nodes coordinates + weights)
                        eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq;

                        // BOUNDARY^+
                        // Store memory offset
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = eBouQuad_mem_pos;

                        // Expand memory
                        this->eBouQuad_host_mem.resize(eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*bou_Nq);

                        // Store data: remember that we store the
                        // quadrature abscissae and weights as if they
                        // were computed on a [-1,1]^AMREX_SPACEDIM
                        // cell
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            long pos = eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*qi;

                            AMREX_D_TERM
                            (
                                this->eBouQuad_host_mem[pos+0] = (quad.nodes[qi].x[0]-cell_m[0])/J[0];,
                                this->eBouQuad_host_mem[pos+1] = (quad.nodes[qi].x[1]-cell_m[1])/J[1];,
                                this->eBouQuad_host_mem[pos+2] = (quad.nodes[qi].x[2]-cell_m[2])/J[2];
                            )
                            this->eBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w/bou_aJ[dir];
                        }

                        // Update memory (nodes coordinates + weights)
                        eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq;
                    }
                }
            }
        }

#if (DG_VERBOSITY == 1)
        // PRINT A BRIEF REPORT ---------------------------------------
        {
            ParallelDescriptor::Barrier();
            Print() << "REPORT:" << std::endl;

            for (int rank = 0; rank < ParallelDescriptor::NProcs(); ++rank)
            {
                if (ParallelDescriptor::MyProc() == rank)
                {
                    printf("| Hello, I am processor %d and I have allocated:\n", ParallelDescriptor::MyProc());
                    printf("| %lu doubles (%f MB) for implicit boundary quadrature.\n", eBouQuad_mem_pos, eBouQuad_mem_pos*sizeof(Real)/(1024.0*1024.0));
                }

                ParallelDescriptor::Barrier();
            }

            ParallelDescriptor::Barrier();
            Print() << std::endl;
        }
        // ------------------------------------------------------------
#endif

        // STORE MEMORY INFO
        this->mem_for_eBouQuad = eBouQuad_mem_pos;
        // ============================================================

        // SEARCH FOR SMALL ELEMENTS AND SETUP THE MERGING INFO =======
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int> const & eType_fab = this->eType.array(mfi);
            Array4<Real const> const & eInfo_fab = this->eInfo.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const bool elm_is_partial = (eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) == DG_ELM_TYPE_PARTIAL);
                const Real vf = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom);

                // LOCAL VARIABLES
                int nbr_i, nbr_j, nbr_k, nbr_b;
                Real nbr_vf, best_nbr_vf;
                bool elm_has_been_merged;

                if (elm_is_partial && (vf < DG_SMALL_ELM_VF_THRESHOLD))
                {
                    elm_has_been_merged = false;
                    best_nbr_vf = DG_SMALL_ELM_VF_THRESHOLD;

                    // Search among the neighbors. We currently limit
                    // ourselves to the neighboring cells that share a 
                    // boundary with the current cell.
                    for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                    {
                        // Neighboring cell id and boundary
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom);
                        if (nbr_vf > best_nbr_vf)
                        {
                            eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_SMALL+b*10;
                            elm_has_been_merged = true;
                            best_nbr_vf = nbr_vf;
                        }
                    }

                    if (elm_has_been_merged)
                    {
                        // I would like to mark an element as extended
                        // without falling into any raise condition. The
                        // conflict can be raised because an element can
                        // be extended to multiple small elements.
                    }
                    else
                    {
                        printf("\n");
                        printf("ERROR: AMReX_DG_ImplicitGeometry.H - ImplicitGeometry::EvalImplicitMesh\n");
                        printf("| Could not find a neighboring cell suitable for merging:\n");
                        printf("| i,j,k: %d,%d,%d\n", i, j, k);
                        printf("| Volume fraction: %f\n", vf);
                        for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                        {
                            // Neighboring cell id and boundary
                            NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                            nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom);
                            printf("| Volume fraction (nbr = %d): %f\n", b, nbr_vf);
                        }
                        printf("\n");
#if (!AMREX_USE_CUDA)
                        exit(-1);
#endif
                    }
                }
                else if (elm_is_partial && (vf >= DG_SMALL_ELM_VF_THRESHOLD))
                {
                    eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_LARGE;
                }
            });
            Gpu::synchronize();
        }
        this->eType.FillBoundary(this->geom.periodicity());
        // ============================================================

        // MARK THE EXTENDED ELEMENTS =================================
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();

            Array4<int> const & eType_fab = this->eType.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
                const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                const bool elm_is_valid = elm_is_entire || elm_is_large;

                // LOCAL VARIABLES
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_etype, nbr_merged_b;
                bool nbr_is_small;

                // Cycle over the boundaries, check whether the valid
                // element is also extended and mark it
                if (elm_is_valid)
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    if (nbr_merged_b == nbr_b)
                    {
                        eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = etype+10;
                    }
                }
            });
            Gpu::synchronize();
        }
        this->eType.FillBoundary(this->geom.periodicity());
        // ============================================================

        // COUNT THE NUMBER OF ENTIRE EXTENDED ELEMENTS ===============
        this->n_entire_extended_elms = 0;

        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.fabbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            Array4<int> const & eType_fab = this->eType.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
                const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                const bool elm_is_valid = elm_is_entire || elm_is_large;
                const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

                if (elm_is_entire && elm_is_extended) this->n_entire_extended_elms += 1;
            }
        }

#if (DG_VERBOSITY == 1)
        // PRINT A BRIEF REPORT ---------------------------------------
        {
            ParallelDescriptor::Barrier();
            Print() << "REPORT:" << std::endl;

            for (int rank = 0; rank < ParallelDescriptor::NProcs(); ++rank)
            {
                if (ParallelDescriptor::MyProc() == rank)
                {
                    printf("| Hello, I am processor %d and I have counted:\n", ParallelDescriptor::MyProc());
                    printf("| %d cut elements\n", this->n_cut_elms);
                    printf("| %d entire extended elements\n", this->n_entire_extended_elms);
                }

                ParallelDescriptor::Barrier();
            }

            ParallelDescriptor::Barrier();
            Print() << std::endl;
        }
        // ------------------------------------------------------------
#endif
        // ============================================================

        // COPY MEMORY TO THE DEVICE ==================================
#ifdef AMREX_USE_CUDA
        this->eDomQuad_device_mem.resize(this->eDomQuad_host_mem.size());
        this->eBouQuad_device_mem.resize(this->eBouQuad_host_mem.size());
        this->eIntBouQuad_device_mem.resize(this->eIntBouQuad_host_mem.size());
        this->eTri_device_mem.resize(this->eTri_host_mem.size());

        Gpu::copy(Gpu::hostToDevice, this->eDomQuad_host_mem.begin(), this->eDomQuad_host_mem.end(), this->eDomQuad_device_mem.begin());
        Gpu::copy(Gpu::hostToDevice, this->eBouQuad_host_mem.begin(), this->eBouQuad_host_mem.end(), this->eBouQuad_device_mem.begin());
        Gpu::copy(Gpu::hostToDevice, this->eIntBouQuad_host_mem.begin(), this->eIntBouQuad_host_mem.end(), this->eIntBouQuad_device_mem.begin());
        Gpu::copy(Gpu::hostToDevice, this->eTri_host_mem.begin(), this->eTri_host_mem.end(), this->eTri_device_mem.begin());
#endif
        // ============================================================

        // CHECK THE QUADRATURE RULES =================================
        if (true)
        {
            const int QUAD_CHECK_N_COMP_PER_DOM = 3;
            MultiFab quad_check(this->cc_ba, this->dm, QUAD_CHECK_N_COMP_PER_DOM*N_DOM, this->dom_data_ngr);
#ifdef AMREX_USE_CUDA
            Real * eDomQuad_mem_ptr = this->eDomQuad_device_mem.data();
            Real * eBouQuad_mem_ptr = this->eBouQuad_device_mem.data();
            Real * eIntBouQuad_mem_ptr = this->eIntBouQuad_device_mem.data();
#else
            Real * eDomQuad_mem_ptr = this->eDomQuad_host_mem.data();
            Real * eBouQuad_mem_ptr = this->eBouQuad_host_mem.data();
            Real * eIntBouQuad_mem_ptr = this->eIntBouQuad_host_mem.data();
#endif
            
            quad_check = 0.0;

            for (MFIter mfi(quad_check); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();
                Array4<int const> const & eDomQuad_Nq_fab = this->eDomQuad_Nq.array(mfi);
                Array4<long const> const & eDomQuad_pos_fab = this->eDomQuad_pos.array(mfi);
                AMREX_D_TERM
                (
                    Array4<int const> const & eBouQuad_Nq_fab_0 = this->eBouQuad_Nq[0].array(mfi);
                    Array4<long const> const & eBouQuad_pos_fab_0 = this->eBouQuad_pos[0].array(mfi);,
                    Array4<int const> const & eBouQuad_Nq_fab_1 = this->eBouQuad_Nq[1].array(mfi);
                    Array4<long const> const & eBouQuad_pos_fab_1 = this->eBouQuad_pos[1].array(mfi);,
                    Array4<int const> const & eBouQuad_Nq_fab_2 = this->eBouQuad_Nq[2].array(mfi);
                    Array4<long const> const & eBouQuad_pos_fab_2 = this->eBouQuad_pos[2].array(mfi);
                )
                Array4<int const> const & eIntBouQuad_Nq_fab = this->eIntBouQuad_Nq.array(mfi);
                Array4<long const> const & eIntBouQuad_pos_fab = this->eIntBouQuad_pos.array(mfi);
                Array4<Real> const & dst_fab = quad_check.array(mfi);

                ParallelFor(bx, N_DOM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
                {
                    // LOCAL PARAMETERS
                    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};

                    // LOCAL VARIABLES
                    int dom_Nq, bou_Nq;
                    int fi, fj, fk;
                    long pos, qi_pos;
                    const Real * xptr;
                    Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                    Real integrand;
                    Real cell_volume, cell_volume_from_bou_quadrature, cell_surface;

                    // INITIALIZATION
                    dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom) = 0.0;
                    dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+1) = 0.0;
                    dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+2) = 0.0;
                    cell_volume = 0.0;
                    cell_volume_from_bou_quadrature = 0.0;
                    cell_surface = 0.0;

                    // DOMAIN INTEGRAL
                    {
                        // QUADRATURE INFO
                        dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                        pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                        xptr = &eDomQuad_mem_ptr[pos];

                        // EVAL THE INTEGRAL
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

                            integrand = 1.0;
                            dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom) += integrand*w;

                            integrand = 1.0;
                            cell_volume += integrand*w;
                        }
                    }

                    // BOUNDARY INTEGRAL
                    for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                    {
                        GRID_FACE(i, j, k, b, fi, fj, fk);

                        // QUADRATURE INFO
                        if (b/2 == 0)
                        {
                            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                            pos = eBouQuad_pos_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (b/2 == 1)
                        {
                            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                            pos = eBouQuad_pos_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        }
#endif
#if (AMREX_SPACEDIM > 2)
                        else
                        {
                            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                            pos = eBouQuad_pos_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        }
#endif
                        xptr = &eBouQuad_mem_ptr[pos];

                        // UNIT NORMAL
                        AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                        un[b/2] = 2.0*(b%2)-1.0;

                        // EVAL THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            qi_pos = (AMREX_SPACEDIM+1)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                            )
                            w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[b/2];

                            integrand = x[1]*un[1];
                            dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+1) += integrand*w;

                            integrand = 0.0;
                            dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+2) += integrand*w;

                            integrand = x[1]*un[1];
                            cell_volume_from_bou_quadrature += integrand*w;
                            
                            integrand = 0.0;
                            cell_surface += integrand*w;
                        }
                    }

                    // INTERNAL BOUNDARY INTEGRAL
                    {
                        // QUADRATURE INFO
                        bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                        pos = eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                        xptr = &eIntBouQuad_mem_ptr[pos];

                        // COMPUTE THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
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

                            integrand = x[1]*un[1];
                            dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+1) += integrand*w;

                            integrand = 1.0;
                            dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+2) += integrand*w;

                            integrand = x[1]*un[1];
                            cell_volume_from_bou_quadrature += integrand*w;

                            integrand = 1.0;
                            cell_surface += integrand*w;
                        }
                    }
                });
                Gpu::synchronize();
            }

            for (int dom = 0; dom < N_DOM; ++dom)
            {
                volume[dom] = quad_check.sum(QUAD_CHECK_N_COMP_PER_DOM*dom);
                volume_from_bou_quadrature[dom] = quad_check.sum(QUAD_CHECK_N_COMP_PER_DOM*dom+1);
                surface[dom] = quad_check.sum(QUAD_CHECK_N_COMP_PER_DOM*dom+2);
            }
        }

#if (DG_VERBOSITY == 1)
        // PRINT A BRIEF REPORT ---------------------------------------
        {
            Real exact_volume[N_DOM] = {400.0};
            Real exact_surface[N_DOM] = {M_PI*8.0};
            
            //Real exact_volume[N_DOM] = {400.0-M_PI*16.0, M_PI*16.0};
            //Real exact_surface[N_DOM] = {M_PI*8.0, M_PI*8.0};

            //Real exact_volume[N_DOM] = {8000.0-4.0/3.0*M_PI*64.0, 4.0/3.0*M_PI*64.0};
            //Real exact_surface[N_DOM] = {4.0*M_PI*16.0, 4.0*M_PI*16.0};

            Real error_in_volume[N_DOM];
            Real error_in_volume_from_bou_quadrature[N_DOM];
            Real error_in_surface[N_DOM];

            for (int dom = 0; dom < N_DOM; ++dom)
            {
                error_in_volume[dom] = std::abs(volume[dom]-exact_volume[dom])/exact_volume[dom]*100.0;
                error_in_volume_from_bou_quadrature[dom] = std::abs(volume_from_bou_quadrature[dom]-exact_volume[dom])/exact_volume[dom]*100.0;
                error_in_surface[dom] = std::abs(surface[dom]-exact_surface[dom])/exact_surface[dom]*100.0;
            }
            
            Print() << "REPORT:" << std::endl;
            if (ParallelDescriptor::IOProcessor())
            for (int dom = 0; dom < N_DOM; ++dom)
            {
                Print() << "| Domain " << dom << std::endl;
                printf("| - volume (from domain quadrature)  : %.5f ( %.5f %% error)\n", volume[dom], error_in_volume[dom]);
                printf("| - volume (from boundary quadrature): %.5f ( %.5f %% error)\n", volume_from_bou_quadrature[dom], error_in_volume_from_bou_quadrature[dom]);
                printf("| - surface                          : %.5f ( %.5f %% error)\n", surface[dom], error_in_surface[dom]);
            }

            ParallelDescriptor::Barrier();
            Print() << std::endl;
        }
        // ------------------------------------------------------------
#endif
        // ============================================================
    }
    // ################################################################

    // EXPORT DOMAINS MESH FOR VTK ####################################
    void Export_VTK_Mesh(const std::string & dst_folder,
                                const std::string & filename_root,
                                const int & time_id,
                                const int & time_id_max) const
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("ImplicitGeometry::Export_VTK_Mesh(const std::string &, const std::string &, const int &, const int &)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        const bool vtk_legacy_format = false;
        const bool vtk_xml_format = true;

        // FILEPATH
        std::string filepath = GetOutputFilepath(dst_folder, filename_root, time_id, time_id_max)+".vtk";

        // CURRENT PROCESSOR
        const int rank = ParallelDescriptor::MyProc();

        // DATE AND TIME
        const time_t now = time(0);
        const char * date_time = ctime(&now);

        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = this->_dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = this->_prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = this->_J;

        // STANDARD ELEMENT
        const int cell_dom_Ng = this->uHyperrect.dom_Ng;
        const int cell_e = this->uHyperrect.g;
        const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
        const int cell_conn_len = AMREX_D_PICK(2, 4, 8)*cell_dom_Ne;
        const Gpu::ManagedVector<int> cell_base_conn = Get_VTK_BaseCell_Connectivity(cell_e);
        const int * cell_base_conn_ptr = cell_base_conn.data();

        AMREX_D_TERM
        (
            Real const * dom_XI1g_ptr = this->uHyperrect.dom_XI1g.data();,
            Real const * dom_XI2g_ptr = this->uHyperrect.dom_XI2g.data();,
            Real const * dom_XI3g_ptr = this->uHyperrect.dom_XI3g.data();
        )

        // DG
        const int p = this->uHyperrect.p;
        
        // CELL-DATA
        // Processor rank
        // Box id
        // Element type (per each domain)
        // Element id (per each domain)
        const int n_cell_fields = 1+1+N_DOM+N_DOM;

        // NODAL-DATA
        // Distance functions
        const int n_nodal_fields = N_PHI;

        // VTK
        const int VTK_cell_id = AMREX_D_PICK(3, 9, 12);
        // ============================================================

        // VARIABLES ==================================================
        FILE * fp;

        long n_base_cells, n_nodes, n_cells;
        long nodes_pos, conn_pos, cell_fields_pos, nodal_fields_pos;
        longMultiFab output_data_info(this->cc_ba, this->dm, 4, this->dom_data_ngr);
        longMultiFab elm_id(this->cc_ba, this->dm, N_DOM, this->dom_data_ngr);

        Gpu::ManagedVector<long> conn;
        Gpu::ManagedVector<Real> nodes, cell_fields, nodal_fields;
        // ============================================================

        // COUNT THE AMOUNT OF MEMORY REQUIRED BY THE OUTPUT DATA =====
        n_base_cells = 0L;
        
        nodes_pos = 0L;
        conn_pos = 0L;
        cell_fields_pos = 0L;
        nodal_fields_pos = 0L;

        for (MFIter mfi(output_data_info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);
            const auto len = length(bx);

            Array4<long> const & output_data_info_fab = output_data_info.array(mfi);

            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // UPDATE INFO
                output_data_info_fab(i,j,k,0) = nodes_pos;
                output_data_info_fab(i,j,k,1) = conn_pos;
                output_data_info_fab(i,j,k,2) = cell_fields_pos;
                output_data_info_fab(i,j,k,3) = nodal_fields_pos;

                nodes_pos += cell_dom_Ng*AMREX_SPACEDIM;
                conn_pos += cell_conn_len;
                cell_fields_pos += cell_dom_Ne*n_cell_fields;
                nodal_fields_pos += cell_dom_Ng*n_nodal_fields;
            }

            n_base_cells += len.x*len.y*len.z;
        }
        
        n_nodes = nodes_pos/AMREX_SPACEDIM;
        nodes.resize(nodes_pos);

        n_cells = n_base_cells*cell_dom_Ne;
        conn.resize(conn_pos);

        cell_fields.resize(cell_fields_pos);

        nodal_fields.resize(nodal_fields_pos);

        // SET POINTERS TO MEMORY
        Real * nodes_ptr = nodes.data();
        long * conn_ptr = conn.data();
        Real * cell_fields_ptr = cell_fields.data();
        Real * nodal_fields_ptr = nodal_fields.data();
        // ============================================================

        // POPULATE THE ELEMENT ID (RANDOM COLORING) ==================
        for (MFIter mfi(output_data_info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            Array4<long> const & elm_id_fab = elm_id.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                elm_id_fab(i,j,k,dom) = Random_int(N_DOM*n_cells);
            }
        }
        elm_id.FillBoundary(this->geom.periodicity());
        // ============================================================

        // POPULATE OUTPUT DATA =======================================
        long bx_id;
        bx_id = 0L;

        for (MFIter mfi(output_data_info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();

            Array4<long const> const & output_data_info_fab = output_data_info.array(mfi);
            Array4<long const> const & elm_id_fab = elm_id.array(mfi);
            Array4<int const> const & eType_fab = this->eType.array(mfi);
            Array4<Real const> const & PHI_fab = this->PHI.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0], prob_lo[1]+j*dx[1], prob_lo[2]+k*dx[2])};
                const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0], prob_lo[1]+(j+1)*dx[1], prob_lo[2]+(k+1)*dx[2])};
                const long nodes_offset = output_data_info_fab(i,j,k,0);
                const long conn_offset = output_data_info_fab(i,j,k,1);
                const long cell_fields_offset = output_data_info_fab(i,j,k,2);
                const long nodal_fields_offset = output_data_info_fab(i,j,k,3);

                // LOCAL VARIABLES
                int etype;
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM];

                // Data structures used to compute the distance
                // functions
                DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, p, i, j, k, -1);

                // UPDATE INFO: NODES AND NODAL FIELDS
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
                        nodes_ptr[nodes_offset+AMREX_SPACEDIM*n+0] = x[0];,
                        nodes_ptr[nodes_offset+AMREX_SPACEDIM*n+1] = x[1];,
                        nodes_ptr[nodes_offset+AMREX_SPACEDIM*n+2] = x[2];
                    )

                    phi._eval_BF_table_(x);
                    for (int rph = 0; rph < N_PHI; ++rph)
                    {
                        phi.set_fab_ptrs(i, j, k, rph);

                        nodal_fields_ptr[nodal_fields_offset+n_nodal_fields*n+rph] = phi._eval_unsafe_();
                    }
                }

                // UPDATE INFO: CONNECTIVITY
                for (int c = 0; c < cell_conn_len; ++c)
                {
                    conn_ptr[conn_offset+c] = cell_base_conn_ptr[c]+nodes_offset/AMREX_SPACEDIM;
                }

                // UPDATE INFO: CELL FIELDS
                for (int c = 0; c < cell_dom_Ne; ++c)
                {
                    cell_fields_ptr[cell_fields_offset+c*n_cell_fields+0] = rank*1.0;
                    cell_fields_ptr[cell_fields_offset+c*n_cell_fields+1] = bx_id;

                    for (int dom = 0; dom < N_DOM; ++dom)
                    {
                        etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                        BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                        cell_fields_ptr[cell_fields_offset+c*n_cell_fields+1+1+dom] = (etype%10)*1.0;
                        cell_fields_ptr[cell_fields_offset+c*n_cell_fields+1+1+N_DOM+dom] = elm_id_fab(BF_i, BF_j, BF_k, dom);
                    }
                }
            });
            Gpu::synchronize();

            bx_id += 1;   
        }
        // ============================================================

        // ALIGN PROCESSORS ==========
        ParallelDescriptor::Barrier();
        // ===========================

        // OUTPUT FILES: LEGACY FORMAT ================================
        if (vtk_legacy_format)
        {
            // OPEN FILE FOR WRITING ----------------------------------
            fp = fopen(filepath.c_str(), "w");
            if (fp != nullptr)
            {
                fprintf(fp, "# vtk DataFile Version 2.0\n");
                fprintf(fp, "VTK mesh file created on %s", date_time);
                fprintf(fp, "ASCII\n");
            }
            else
            {
                std::cout << "ERROR: AMReX_DG_ImplicitGeometry.H - ImplicitGeometry::Export_VTK_Mesh" << std::endl;
                std::cout << "| Unable to open and write: " << filepath << " file" << std::endl;
                exit(-1);
            }
            // --------------------------------------------------------

            // WRITE TO FILE: GRID ------------------------------------
            fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    #if (AMREX_SPACEDIM == 1)
            // NODES
            fprintf(fp, "POINTS %lu float\n", n_nodes);
            for (int n = 0; n < n_nodes; ++n)
                fprintf(fp, "%+.12f 0.0 0.0\n", nodes[AMREX_SPACEDIM*n]);
            
            // CELLS
            fprintf(fp, "CELLS %lu %lu\n", n_cells, (1+2)*n_cells);
            for (int n = 0; n < n_cells; ++n)
                fprintf(fp, "2 %lu %lu\n", conn[2*n], conn[2*n+1]);
    #endif
    #if (AMREX_SPACEDIM == 2)
            // NODES
            fprintf(fp, "POINTS %lu float\n", n_nodes);
            for (int n = 0; n < n_nodes; ++n)
                fprintf(fp, "%+.12f %+.12f 0.0\n", nodes[AMREX_SPACEDIM*n], nodes[AMREX_SPACEDIM*n+1]);

            // CELLS
            fprintf(fp, "CELLS %lu %lu\n", n_cells, (1+4)*n_cells);
            for (int n = 0; n < n_cells; ++n)
                fprintf(fp, "4 %lu %lu %lu %lu\n", conn[4*n], conn[4*n+1], conn[4*n+2], conn[4*n+3]);
    #endif
    #if (AMREX_SPACEDIM == 3)
            // NODES
            fprintf(fp, "POINTS %lu float\n", n_nodes);
            for (int n = 0; n < n_nodes; ++n)
                fprintf(fp, "%+.12f %+.12f %+.12f\n", nodes[AMREX_SPACEDIM*n], nodes[AMREX_SPACEDIM*n+1], nodes[AMREX_SPACEDIM*n+2]);

            // CELLS
            fprintf(fp, "CELLS %lu %lu\n", n_cells, (1+8)*n_cells);
            for (int n = 0; n < n_cells; ++n)
                fprintf(fp, "8 %lu %lu %lu %lu %lu %lu %lu %lu\n", conn[8*n], conn[8*n+1], conn[8*n+2], conn[8*n+3],
                                                                   conn[8*n+4], conn[8*n+5], conn[8*n+6], conn[8*n+7]);
    #endif
            // CELL TYPES
            fprintf(fp, "CELL_TYPES %lu\n", n_cells);
            for (int n = 0; n < n_cells; ++n)
                fprintf(fp, "%d\n", VTK_cell_id);
            // --------------------------------------------------------

            // WRITE TO FILE: CELL FIELDS -----------------------------
            Vector<std::string> cell_field_names;
            cell_field_names.push_back("RANK");
            cell_field_names.push_back("BOX");
            for (int dom = 0; dom < N_DOM; ++dom) cell_field_names.push_back("ELM_TYPE_DOM_"+std::to_string(dom));
            for (int dom = 0; dom < N_DOM; ++dom) cell_field_names.push_back("ELM_ID_DOM_"+std::to_string(dom));

            fprintf(fp, "CELL_DATA %lu\n", n_cells);
            for (int f = 0; f < n_cell_fields; ++f)
            {
                fprintf(fp, "SCALARS %s float 1\n", cell_field_names[f].c_str());
                fprintf(fp, "LOOKUP_TABLE default\n");
                for (int n = 0; n < n_cells; ++n)
                {
                    fprintf(fp, "%+.12f\n", cell_fields[n_cell_fields*n+f]);
                }
            }
            // --------------------------------------------------------
            
            // WRITE TO FILE: NODAL FIELDS ----------------------------
            Vector<std::string> nodal_field_names;
            for (int rph = 0; rph < N_PHI; ++rph) nodal_field_names.push_back("PHI_"+std::to_string(rph));

            fprintf(fp, "POINT_DATA %lu\n", n_nodes);
            for (int f = 0; f < n_nodal_fields; ++f)
            {
                fprintf(fp, "SCALARS %s float 1\n", nodal_field_names[f].c_str());
                fprintf(fp, "LOOKUP_TABLE default\n");
                for (int n = 0; n < n_nodes; ++n)
                {
                    fprintf(fp, "%+.12f\n", nodal_fields[n_nodal_fields*n+f]);
                }
            }
            // --------------------------------------------------------

            // CLOSE FILE
            fclose(fp);
            // ----------  
        }
        // OUTPUT FILES: XML FORMAT ===================================
        else if (vtk_xml_format)
        {
            // VARIABLES ----------------------------------------------
            Vector<VTK::Float_t> VTK_nodes;
            Vector<VTK::Cell_conn_t> VTK_cell_conn;
            Vector<VTK::Cell_offs_t> VTK_cell_offset;
            Vector<VTK::Cell_type_t> VTK_cell_type;
            Vector<Vector<VTK::Int_t>> VTK_cell_field(n_cell_fields);
            Vector<std::string> VTK_cell_field_name(n_cell_fields);
            Vector<Vector<VTK::Float_t>> VTK_nodal_field(n_nodal_fields);
            Vector<std::string> VTK_nodal_field_name(n_nodal_fields);
            // --------------------------------------------------------

            // RESIZE MEMORY ------------------------------------------
            VTK_nodes.resize(3*n_nodes);
            VTK_cell_conn.resize(AMREX_D_PICK(2, 4, 8)*n_cells);
            VTK_cell_offset.resize(n_cells+1);
            VTK_cell_type.resize(n_cells);
            for (int f = 0; f < n_cell_fields; ++f)
            {
                VTK_cell_field[f].resize(n_cells);
            }
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

            VTK_cell_field_name[0] = "RANK";
            VTK_cell_field_name[1] = "BOX";
            for (int dom = 0; dom < N_DOM; ++dom) VTK_cell_field_name[2+dom] = "ELM_TYPE_DOM_"+std::to_string(dom);
            for (int dom = 0; dom < N_DOM; ++dom) VTK_cell_field_name[2+N_DOM+dom] = "ELM_ID_DOM_"+std::to_string(dom);
            for (int f = 0; f < n_cell_fields; ++f)
            {
                for (int n = 0; n < n_cells; ++n)
                {
                    VTK_cell_field[f][n] = (VTK::Int_t) cell_fields[n_cell_fields*n+f];
                }
            }

            for (int rph = 0; rph < N_PHI; ++rph) VTK_nodal_field_name[rph] = "PHI_"+std::to_string(rph);

            for (int f = 0; f < n_nodal_fields; ++f)
            {
                for (int n = 0; n < n_nodes; ++n)
                {
                    VTK_nodal_field[f][n] = nodal_fields[n_nodal_fields*n+f];
                }
            }
            // --------------------------------------------------------

            // DUMP HEADER FILE ---------------------------------------
            if (amrex::ParallelDescriptor::IOProcessor())
            {
                VTK::PrintHeaderFile_VTU(dst_folder, filename_root, time_id, time_id_max, VTK_cell_field_name, VTK_nodal_field_name);
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
                                               VTK_cell_field,
                                               VTK_cell_field_name,
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

    // PLOT ###########################################################
    void Plot_Mesh(const std::string & dst_folder,
                   const std::string & filename_root,
                   const Real & time,
                   const int & time_id,
                   const int & time_id_max)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("ImplicitGeometry::EvalImplicitMesh(const std::string &, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        int n_digits;
        Vector<std::string> varnames;
        std::string filename;
        // ============================================================

        // VARIABLES ==================================================
        // ============================================================

        // INITIALIZATION =============================================
        this->pp_PHI = 0.0;
        // ============================================================

        // PROJECT THE DISTANCE FUNCTIONS INTO THE FINER GRID =========
        {
            // FINE GRID PARAMETERS -----------------------------------
            // GRID
            const int rr = this->pp_ratio;
            const GpuArray<Real, AMREX_SPACEDIM> dst_dx = this->pp_geom.CellSizeArray();
            const GpuArray<Real, AMREX_SPACEDIM> src_dx = this->_dx;
            const GpuArray<Real, AMREX_SPACEDIM> prob_lo = this->pp_geom.ProbLoArray();
            const GpuArray<Real, AMREX_SPACEDIM> dst_J = {AMREX_D_DECL(0.5*dst_dx[0], 0.5*dst_dx[1], 0.5*dst_dx[2])};

            // STANDARD ELEMENT
            const int src_p = this->uHyperrect.p;
            const int Np = this->pp_uHyperrect.Np;
            const int dom_Nq = this->pp_uHyperrect.dom_Nq;

            AMREX_D_TERM
            (
                Real const * dom_XI1q_ptr = this->pp_uHyperrect.dom_XI1q.data();,
                Real const * dom_XI2q_ptr = this->pp_uHyperrect.dom_XI2q.data();,
                Real const * dom_XI3q_ptr = this->pp_uHyperrect.dom_XI3q.data();
            )
            Real const * dom_Wq_ptr = this->pp_uHyperrect.dom_Wq.data();
            Real const * dom_BF_ptr = this->pp_uHyperrect.dom_BF.data(); 
            
            Real const * iMM_ptr = this->pp_uHyperrect.iMM.data();
            // --------------------------------------------------------

            // EVAL THE PROJECTION ------------------------------------
            for (MFIter mfi(this->pp_PHI); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                Array4<Real const> const & PHI_fab = this->PHI.array(mfi);
                Array4<Real> const & pp_PHI_fab = this->pp_PHI.array(mfi);
                    
                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // LOCAL PARAMETERS
                    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dst_dx[0], prob_lo[1]+(j+0.5)*dst_dx[1], prob_lo[2]+(k+0.5)*dst_dx[2])};

                    // LOCAL PARAMETERS (COARSE GRID)
                    const int ci = i/rr, cj = j/rr, ck = k/rr;
                    const Real src_cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+ci*src_dx[0], prob_lo[1]+cj*src_dx[1], prob_lo[2]+ck*src_dx[2])};
                    const Real src_cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(ci+1)*src_dx[0], prob_lo[1]+(cj+1)*src_dx[1], prob_lo[2]+(ck+1)*src_dx[2])};

                    // LOCAL VARIABLES
                    Real x[AMREX_SPACEDIM];
                    Real integrand;

                    DG_phi_space_BFX phi(src_cell_lo, src_cell_hi, PHI_fab, src_p, ci, cj, ck, -1);

                    // PROJECTION
                    for (int q = 0; q < dom_Nq; ++q)
                    {
                        // Quadrature point
                        AMREX_D_TERM
                        (
                            x[0] = dst_J[0]*dom_XI1q_ptr[q]+cell_m[0];,
                            x[1] = dst_J[1]*dom_XI2q_ptr[q]+cell_m[1];,
                            x[2] = dst_J[2]*dom_XI3q_ptr[q]+cell_m[2];
                        )

                        phi._eval_BF_table_(x);

                        // Eval the integral
                        for (int rph = 0; rph < N_PHI; ++rph)
                        {
                            phi.set_fab_ptrs(ci, cj, ck, rph);

                            for (int rs = 0; rs < Np; ++rs)
                            {
                                integrand = 0.0;
                                for (int cs = 0; cs < Np; ++cs) integrand += iMM_ptr[rs+cs*Np]*dom_BF_ptr[q+cs*dom_Nq];
                                integrand *= phi._eval_unsafe_();

                                // Sum integral contribution
                                pp_PHI_fab(i,j,k,rs+rph*Np) += integrand*dom_Wq_ptr[q];
                            }
                        }
                    }
                });
                Gpu::synchronize();

            }
            this->pp_PHI.FillBoundary(this->geom.periodicity());
            // --------------------------------------------------------
        }
        // ============================================================
    }
    // ################################################################
};
// ####################################################################
// ####################################################################


} // namespace DG
} // namespace amrex

#endif
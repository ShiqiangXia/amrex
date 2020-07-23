// AMReX_DG_MatrixFactory.H

#ifndef BL_DG_MATRIX_FACTORY_H_
#define BL_DG_MATRIX_FACTORY_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_ImplicitGeometry.H>

namespace amrex
{
namespace DG
{


// ####################################################################
// DG MATRIX FACTORY CLASS ############################################
// ####################################################################
template<int N_PHI, int N_DOM>
struct MatrixFactory
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 3;
    static const int bou_data_ngr = 0;
    
    static const int Dom_BF_pos_n_comp = DG_DOM_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int Tri_BF_pos_n_comp = DG_TRI_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int IntBou_BF_pos_n_comp = DG_INT_BOU_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int Bou_BF_pos_n_comp = DG_BOU_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int MMCh_pos_n_comp = N_DOM;

    // PARAMETERS
    const Box cc_ibx;
    const RealBox rbx;
    const BoxArray cc_ba;
    const Array<BoxArray, AMREX_SPACEDIM> fc_ba;
    const DistributionMapping dm;
    const Geometry geom;

    const int space_p, space_Np, time_p, time_Np;
    const int space_q, time_q;
    
    /*const*/ UnitHyperrectangle<AMREX_SPACEDIM> uHyperrect;

    // VARIABLES
    longMultiFab space_Dom_BF_pos, space_Tri_BF_pos, space_IntBou_BF_pos;
    Array<longMultiFab, AMREX_SPACEDIM> space_Bou_BF_pos;
    longMultiFab space_MMCh_pos;

    int n_cut_elms, n_special_elms;
    long mem_for_space_Dom_BF, mem_for_space_Tri_BF, mem_for_space_IntBou_BF, mem_for_space_Bou_BF;
    long mem_for_space_MMCh;

    Gpu::HostVector<Real> space_Dom_BF_host_mem;
    Gpu::HostVector<Real> space_Tri_BF_host_mem;
    Gpu::HostVector<Real> space_Dom_dBFdXI1_host_mem;
    Gpu::HostVector<Real> space_Dom_dBFdXI2_host_mem;
    Gpu::HostVector<Real> space_Dom_dBFdXI3_host_mem;
    Gpu::HostVector<Real> space_Bou_BF_host_mem;
    Gpu::HostVector<Real> space_Bou_dBFdXI1_host_mem;
    Gpu::HostVector<Real> space_Bou_dBFdXI2_host_mem;
    Gpu::HostVector<Real> space_Bou_dBFdXI3_host_mem;
    Gpu::HostVector<Real> space_IntBou_BF_host_mem;
    Gpu::HostVector<Real> space_IntBou_dBFdXI1_host_mem;
    Gpu::HostVector<Real> space_IntBou_dBFdXI2_host_mem;
    Gpu::HostVector<Real> space_IntBou_dBFdXI3_host_mem;
    Gpu::HostVector<Real> space_MMCh_host_mem;
    
    Gpu::DeviceVector<Real> space_Dom_BF_device_mem;
    Gpu::DeviceVector<Real> space_Tri_BF_device_mem;
    Gpu::DeviceVector<Real> space_Dom_dBFdXI1_device_mem;
    Gpu::DeviceVector<Real> space_Dom_dBFdXI2_device_mem;
    Gpu::DeviceVector<Real> space_Dom_dBFdXI3_device_mem;
    Gpu::DeviceVector<Real> space_Bou_BF_device_mem;
    Gpu::DeviceVector<Real> space_Bou_dBFdXI1_device_mem;
    Gpu::DeviceVector<Real> space_Bou_dBFdXI2_device_mem;
    Gpu::DeviceVector<Real> space_Bou_dBFdXI3_device_mem;
    Gpu::DeviceVector<Real> space_IntBou_BF_device_mem;
    Gpu::DeviceVector<Real> space_IntBou_dBFdXI1_device_mem;
    Gpu::DeviceVector<Real> space_IntBou_dBFdXI2_device_mem;
    Gpu::DeviceVector<Real> space_IntBou_dBFdXI3_device_mem;
    Gpu::DeviceVector<Real> space_MMCh_device_mem;
    // ################################################################

    // CONSTRUCTOR ####################################################
    MatrixFactory(const Box & cc_ibx_,
                  const RealBox & rbx_,
                  const BoxArray & cc_ba_,
                  const DistributionMapping & dm_,
                  const Geometry & geom_,
                  const int & space_p_,
                  const int & time_p_,
                  const int & space_q_,
                  const int & time_q_)
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
    space_p(space_p_),
    space_Np(AMREX_D_PICK(1+space_p_, (1+space_p_)*(1+space_p_), (1+space_p_)*(1+space_p_)*(1+space_p_))),
    time_p(time_p_),
    time_Np(time_p_+1),
    space_q(space_q_),
    time_q(time_q_),
    uHyperrect(space_p, space_q),
    n_cut_elms(0),
    n_special_elms(0),
    mem_for_space_Dom_BF(0L),
    mem_for_space_Tri_BF(0L),
    mem_for_space_IntBou_BF(0L),
    mem_for_space_Bou_BF(0L),
    mem_for_space_MMCh(0L)
    {
        // SOME CHECKS ================================================
        BL_ASSERT(space_p_ >= 0);
        BL_ASSERT(space_p_ <= DG_SOL_MAX_SPACE_p);

        if (space_p_ != DG_SOL_MAX_SPACE_p)
        {
            Print() << std::endl;
            Print() << "WARNING: AMReX_DG_MatrixFactory.H - MatrixFactory::MatrixFactory" << std::endl;
            Print() << "| DG_SOL_MAX_SPACE_p = " << DG_SOL_MAX_SPACE_p << std::endl;
            Print() << "| input space p = " << space_p_ << std::endl;
            Print() << "| Consider setting DG_SOL_MAX_SPACE_p = " << space_p_ << " and recompiling the code." <<std::endl;
            Print() << std::endl;
        }

        BL_ASSERT(time_p_ >= 0);
        BL_ASSERT(time_p_ <= DG_SOL_MAX_TIME_p);

        if (time_p_ != DG_SOL_MAX_TIME_p)
        {
            Print() << std::endl;
            Print() << "WARNING: AMReX_DG_MatrixFactory.H - MatrixFactory::MatrixFactory" << std::endl;
            Print() << "| DG_SOL_MAX_TIME_p = " << DG_SOL_MAX_TIME_p << std::endl;
            Print() << "| input time p = " << time_p_ << std::endl;
            Print() << "| Consider setting DG_SOL_MAX_TIME_p = " << time_p_ << " and recompiling the code." <<std::endl;
            Print() << std::endl;
        }
        // ============================================================

        // STORE MEMORY FOR THE MULTIFABS =============================
        // WARNING: Remember that on these MultiFabs, the FillBoundary
        //          method must NEVER be called.
        // ============================================================
        // DOMAIN (CELL-CENTERED) INFO
        this->space_Dom_BF_pos.define(this->cc_ba, this->dm, this->Dom_BF_pos_n_comp, this->dom_data_ngr);
        this->space_Tri_BF_pos.define(this->cc_ba, this->dm, this->Tri_BF_pos_n_comp, this->dom_data_ngr);
        this->space_IntBou_BF_pos.define(this->cc_ba, this->dm, this->IntBou_BF_pos_n_comp, this->dom_data_ngr);
        this->space_MMCh_pos.define(this->cc_ba, this->dm, this->MMCh_pos_n_comp, this->dom_data_ngr);

        this->space_Dom_BF_pos = -1L;
        this->space_Tri_BF_pos = -1L;
        this->space_IntBou_BF_pos = -1L;
        this->space_MMCh_pos = -1L;

        // BOUNDARY (FACE-CENTERED) INFO
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->space_Bou_BF_pos[dim].define(this->fc_ba[dim], this->dm, this->Bou_BF_pos_n_comp, this->bou_data_ngr);

            this->space_Bou_BF_pos[dim] = -1L;
        }
        // ============================================================
    }
    // ################################################################

    // DESTRUCTOR #####################################################
    ~MatrixFactory()
    {}
    // ################################################################

    // EVAL DG BASIS FUNCTIONS AND MATRICES ###########################
    void Eval(const ImplicitGeometry<N_PHI, N_DOM> & iGeom)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("MatrixFactory::Eval(const ImplicitGeometry<N_PHI, N_DOM> &)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = iGeom._dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = iGeom._prob_lo;
        const GpuArray<Real, AMREX_SPACEDIM> J = iGeom._J;
        const Real aJ = iGeom._aJ, iaJ = iGeom._iaJ;

        // IMPLICIT MESH
        const Real * eDomQuad_mem_ptr = iGeom.eDomQuad_host_mem.data();
        const Real * eTri_mem_ptr = iGeom.eTri_host_mem.data();
        const Real * eBouQuad_mem_ptr = iGeom.eBouQuad_host_mem.data();
        const Real * eIntBouQuad_mem_ptr = iGeom.eIntBouQuad_host_mem.data();

        // DG
        const int sp = this->space_p;
        const int sNp = this->space_Np;
        // ============================================================

        // VARIABLES ==================================================
        long space_Dom_BF_mem_pos, space_Tri_BF_mem_pos, space_IntBou_BF_mem_pos, space_Bou_BF_mem_pos;
        long space_MMCh_mem_pos;
        // ============================================================

        // INITIALIZATION =============================================
        this->n_cut_elms = 0;
        this->n_special_elms = 0;

        space_Dom_BF_mem_pos = 0L;
        space_Tri_BF_mem_pos = 0L;
        space_IntBou_BF_mem_pos = 0L;
        space_Bou_BF_mem_pos = 0L;
        space_MMCh_mem_pos = 0L;
        // ============================================================

        // CHECK THAT THE STANDARD ELEMENTS OF THE GEOMETRY AND OF THE
        // MATRIX FACTORY ARE CONSISTENT ==============================
        if (iGeom.uHyperrect.dom_Nq != this->uHyperrect.dom_Nq)
        {
            Print() << std::endl;
            Print() << "ERROR: AMReX_DG_MatrixFactory.H - MatrixFactory::Eval" << std::endl;
            Print() << "| The standard elements of the input ImplicitGeometry and the MatrixFactory are not consistent." << std::endl;
            Print() << "| ImplicitGeometry.uHyperrect.dom_Nq: " << iGeom.uHyperrect.dom_Nq << std::endl;
            Print() << "| MatrixFactory.uHyperrect.dom_Nq   : " << this->uHyperrect.dom_Nq << std::endl;
            Print() << std::endl;
            exit(-1);
        }
        // ============================================================

        // STANDARD ELEMENT INFO ======================================
        // DOMAIN
        {
            // QUADRATURE INFO
            const int dom_Nq = iGeom.uHyperrect.dom_Nq;

            // Resize memory
            this->space_Dom_BF_host_mem.resize(dom_Nq*sNp);
            AMREX_D_TERM
            (
                this->space_Dom_dBFdXI1_host_mem.resize(dom_Nq*sNp);,
                this->space_Dom_dBFdXI2_host_mem.resize(dom_Nq*sNp);,
                this->space_Dom_dBFdXI3_host_mem.resize(dom_Nq*sNp);
            )

            // Store data
            for (int qi = 0; qi < dom_Nq; ++qi)
            for (int rs = 0; rs < sNp; ++rs)
            {
                this->space_Dom_BF_host_mem[rs+qi*sNp] = this->uHyperrect.dom_BF[qi+rs*dom_Nq];
                AMREX_D_TERM
                (
                    this->space_Dom_dBFdXI1_host_mem[rs+qi*sNp] = this->uHyperrect.dom_dBFdXI1[qi+rs*dom_Nq];,
                    this->space_Dom_dBFdXI2_host_mem[rs+qi*sNp] = this->uHyperrect.dom_dBFdXI2[qi+rs*dom_Nq];,
                    this->space_Dom_dBFdXI3_host_mem[rs+qi*sNp] = this->uHyperrect.dom_dBFdXI3[qi+rs*dom_Nq];
                )
            }

            // Update memory
            space_Dom_BF_mem_pos += dom_Nq*sNp;
        }

        // DOMAIN: QUERY POINTS
        {
            // GRID POINTS INFO
            const int dom_Ng = iGeom.uHyperrect.dom_Ng;

            // Resize memory
            this->space_Tri_BF_host_mem.resize(dom_Ng*sNp);

            // Store data
            for (int qi = 0; qi < dom_Ng; ++qi)
            for (int rs = 0; rs < sNp; ++rs)
            {
                this->space_Tri_BF_host_mem[rs+qi*sNp] = this->uHyperrect.grid_BF[qi+rs*dom_Ng];
            }

            // Update memory
            space_Tri_BF_mem_pos += dom_Ng*sNp;
        }

        // BOUNDARIES
        {
            // QUADRATURE INFO
            const int bou_Nq = iGeom.uHyperrect.bou_Nq;

            // Resize memory
            this->space_Bou_BF_host_mem.resize(bou_Nq*sNp*DG_CELL_N_SPACE_BOUNDARIES);
            AMREX_D_TERM
            (
                this->space_Bou_dBFdXI1_host_mem.resize(bou_Nq*sNp*DG_CELL_N_SPACE_BOUNDARIES);,
                this->space_Bou_dBFdXI2_host_mem.resize(bou_Nq*sNp*DG_CELL_N_SPACE_BOUNDARIES);,
                this->space_Bou_dBFdXI3_host_mem.resize(bou_Nq*sNp*DG_CELL_N_SPACE_BOUNDARIES);
            )

            // Store data
            for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
            {
                int pos = b*bou_Nq*sNp;
                
                for (int qi = 0; qi < bou_Nq; ++qi)
                for (int rs = 0; rs < sNp; ++rs)
                {
                    this->space_Bou_BF_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = this->uHyperrect.bou_BF[pos+qi+rs*bou_Nq];
                    AMREX_D_TERM
                    (
                        this->space_Bou_dBFdXI1_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = this->uHyperrect.bou_dBFdXI1[pos+qi+rs*bou_Nq];,
                        this->space_Bou_dBFdXI2_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = this->uHyperrect.bou_dBFdXI2[pos+qi+rs*bou_Nq];,
                        this->space_Bou_dBFdXI3_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = this->uHyperrect.bou_dBFdXI3[pos+qi+rs*bou_Nq];
                    )
                }

                // Update memory
                space_Bou_BF_mem_pos += bou_Nq*sNp;
            }
        }

        // CHOLESKY DECOMPOSITION OF THE MASS MATRIX
        {
            // Resize memory
            this->space_MMCh_host_mem.resize(sNp*sNp);

            // Store data
            std::copy(this->uHyperrect.MMCh.begin(), this->uHyperrect.MMCh.end(), this->space_MMCh_host_mem.data());

            // Update memory
            space_MMCh_mem_pos += sNp*sNp;
        }
        // ============================================================

        // STORE THE IMPLICIT MESH INFO: DOMAIN (CELL-CENTERED) DATA ==
        for (MFIter mfi(iGeom.eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);
            
            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);

            Array4<int const> const & eDomQuad_Nq_fab = iGeom.eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = iGeom.eDomQuad_pos.array(mfi);
            Array4<int const> const & eTri_Npt_fab = iGeom.eTri_Npt.array(mfi);
            Array4<long const> const & eTri_pos_fab = iGeom.eTri_pos.array(mfi);
            Array4<int const> const & eIntBouQuad_Nq_fab = iGeom.eIntBouQuad_Nq.array(mfi);
            Array4<long const> const & eIntBouQuad_pos_fab = iGeom.eIntBouQuad_pos.array(mfi);

            Array4<long> const & space_Dom_BF_pos_fab = this->space_Dom_BF_pos.array(mfi);
            Array4<long> const & space_Tri_BF_pos_fab = this->space_Tri_BF_pos.array(mfi);
            Array4<long> const & space_IntBou_BF_pos_fab = this->space_IntBou_BF_pos.array(mfi);
            Array4<long> const & space_MMCh_pos_fab = this->space_MMCh_pos.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0], prob_lo[1]+(j+0.5)*dx[1], prob_lo[2]+(k+0.5)*dx[2])};
                const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const bool elm_is_empty = (etype == DG_ELM_TYPE_EMPTY);
                const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
                const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
                const bool elm_is_partial = (elm_is_large || elm_is_small);
                const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;
                Real MM[DG_SOL_MAX_SPACE_Np*DG_SOL_MAX_SPACE_Np];
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

                // Get the support of the basis functions. It is
                // recalled that if the element is small, the support
                // of the basis functions is given by the element that
                // is merging into the small current element.
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_lo, BF_hi);

                // Data structures used to compute the basis functions
                DG_sol_space_BF bf(BF_lo, BF_hi, sp);

                // EMPTY ELEMENTS -------------------------------------
                if (elm_is_empty)
                {
                    // DOMAIN
                    space_Dom_BF_pos_fab(i,j,k,DG_DOM_BF_INFO_N_COMP_PER_DOM*dom) = -1L;

                    // DOMAIN: QUERY POINTS
                    space_Tri_BF_pos_fab(i,j,k,DG_TRI_BF_INFO_N_COMP_PER_DOM*dom) = -1L;

                    // INTERNAL BOUNDARY
                    space_IntBou_BF_pos_fab(i,j,k,DG_INT_BOU_BF_INFO_N_COMP_PER_DOM*dom) = -1L;

                    // CHOLESKY DECOMPOSITION OF THE MASS MATRIX
                    space_MMCh_pos_fab(i,j,k,dom) = -1L;
                }
                // ----------------------------------------------------
                // ENTIRE ELEMENTS ------------------------------------
                else if (elm_is_entire)
                {
                    // DOMAIN
                    space_Dom_BF_pos_fab(i,j,k,DG_DOM_BF_INFO_N_COMP_PER_DOM*dom) = 0L;

                    // DOMAIN: QUERY POINTS
                    space_Tri_BF_pos_fab(i,j,k,DG_TRI_BF_INFO_N_COMP_PER_DOM*dom) = 0L;

                    // INTERNAL BOUNDARY
                    space_IntBou_BF_pos_fab(i,j,k,DG_INT_BOU_BF_INFO_N_COMP_PER_DOM*dom) = -1L;

                    // MASS MATRIX
                    // At the moment we store the mass matrix of the
                    // standard element which will be subsequently
                    // combined with the mass matrix of the small
                    // element the current element extends to.
                    if (elm_is_entire_extended)
                    {
                        // Store memory offset
                        space_MMCh_pos_fab(i,j,k,dom) = space_MMCh_mem_pos;

                        // Expand memory
                        this->space_MMCh_host_mem.resize(space_MMCh_mem_pos+sNp*sNp);

                        // Store data
                        std::copy(this->uHyperrect.MM.begin(), this->uHyperrect.MM.end(), &this->space_MMCh_host_mem[space_MMCh_mem_pos]);

                        // Update memory
                        space_MMCh_mem_pos += sNp*sNp;
                    }
                    else
                    {
                        space_MMCh_pos_fab(i,j,k,dom) = 0L;
                    }
                }
                // ----------------------------------------------------
                // PARTIAL ELEMENTS -----------------------------------
                else if (elm_is_partial)
                {
                    // DOMAIN
                    {
                        // QUADRATURE INFO
                        const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                        const long pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                        const Real * xptr = &eDomQuad_mem_ptr[pos];
                        long qi_pos;

                        // Store memory offset
                        space_Dom_BF_pos_fab(i,j,k,DG_DOM_BF_INFO_N_COMP_PER_DOM*dom) = space_Dom_BF_mem_pos;

                        // Expand memory
                        this->space_Dom_BF_host_mem.resize(space_Dom_BF_mem_pos+dom_Nq*sNp);
                        AMREX_D_TERM
                        (
                            this->space_Dom_dBFdXI1_host_mem.resize(space_Dom_BF_mem_pos+dom_Nq*sNp);,
                            this->space_Dom_dBFdXI2_host_mem.resize(space_Dom_BF_mem_pos+dom_Nq*sNp);,
                            this->space_Dom_dBFdXI3_host_mem.resize(space_Dom_BF_mem_pos+dom_Nq*sNp);
                        )

                        // Store data
                        for (int qi = 0; qi < dom_Nq; ++qi)
                        {
                            qi_pos = (AMREX_SPACEDIM+1)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                            )

                            bf._eval_BF_table_(x);

                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                this->space_Dom_BF_host_mem[space_Dom_BF_mem_pos+rs+qi*sNp] = bf.BF[rs];
                                AMREX_D_TERM
                                (
                                    this->space_Dom_dBFdXI1_host_mem[space_Dom_BF_mem_pos+rs+qi*sNp] = bf.dBFdx1[rs]*J[0];,
                                    this->space_Dom_dBFdXI2_host_mem[space_Dom_BF_mem_pos+rs+qi*sNp] = bf.dBFdx2[rs]*J[1];,
                                    this->space_Dom_dBFdXI3_host_mem[space_Dom_BF_mem_pos+rs+qi*sNp] = bf.dBFdx3[rs]*J[3];
                                )
                            }
                        }

                        // Update memory
                        space_Dom_BF_mem_pos += dom_Nq*sNp;
                    }

                    // DOMAIN: QUERY POINTS
                    {
                        // GRID POINTS INFO
                        const int dom_Npt = eTri_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                        const long pos = eTri_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                        const Real * xptr = &eTri_mem_ptr[pos];
                        long qi_pos;

                        // Store memory offset
                        space_Tri_BF_pos_fab(i,j,k,DG_TRI_BF_INFO_N_COMP_PER_DOM*dom) = space_Tri_BF_mem_pos;

                        // Expand memory
                        this->space_Tri_BF_host_mem.resize(space_Tri_BF_mem_pos+dom_Npt*sNp);

                        // Store data
                        for (int qi = 0; qi < dom_Npt; ++qi)
                        {
                            qi_pos = AMREX_SPACEDIM*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                            )

                            bf._eval_BF_table_(x);

                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                this->space_Tri_BF_host_mem[space_Tri_BF_mem_pos+rs+qi*sNp] = bf.BF[rs];
                            }
                        }

                        // Update memory
                        space_Tri_BF_mem_pos += dom_Npt*sNp;
                    }

                    // INTERNAL BOUNDARY
                    {
                        // QUADRATURE INFO
                        const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                        const long pos = eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                        const Real * xptr = &eIntBouQuad_mem_ptr[pos];
                        long qi_pos;

                        // Store memory offset
                        space_IntBou_BF_pos_fab(i,j,k,DG_INT_BOU_BF_INFO_N_COMP_PER_DOM*dom) = space_IntBou_BF_mem_pos;

                        // Expand memory
                        this->space_IntBou_BF_host_mem.resize(space_IntBou_BF_mem_pos+bou_Nq*sNp);
                        AMREX_D_TERM
                        (
                            this->space_IntBou_dBFdXI1_host_mem.resize(space_IntBou_BF_mem_pos+bou_Nq*sNp);,
                            this->space_IntBou_dBFdXI2_host_mem.resize(space_IntBou_BF_mem_pos+bou_Nq*sNp);,
                            this->space_IntBou_dBFdXI3_host_mem.resize(space_IntBou_BF_mem_pos+bou_Nq*sNp);
                        )

                        // Store data
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0];,
                                x[1] = xptr[qi_pos+1];,
                                x[2] = xptr[qi_pos+2];
                            )

                            bf._eval_BF_table_(x);

                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                this->space_IntBou_BF_host_mem[space_IntBou_BF_mem_pos+rs+qi*sNp] = bf.BF[rs];
                                AMREX_D_TERM
                                (
                                    this->space_IntBou_dBFdXI1_host_mem[space_IntBou_BF_mem_pos+rs+qi*sNp] = bf.dBFdx1[rs];,
                                    this->space_IntBou_dBFdXI2_host_mem[space_IntBou_BF_mem_pos+rs+qi*sNp] = bf.dBFdx2[rs];,
                                    this->space_IntBou_dBFdXI3_host_mem[space_IntBou_BF_mem_pos+rs+qi*sNp] = bf.dBFdx3[rs];
                                )
                            }
                        }

                        // Update memory
                        space_IntBou_BF_mem_pos += bou_Nq*sNp;
                    }

                    // MASS MATRIX
                    // At the moment we compute and store the mass
                    // matrix of the element.
                    {
                        // QUADRATURE INFO
                        const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                        const long pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                        const Real * xptr = &eDomQuad_mem_ptr[pos];
                        long qi_pos;

                        // Store memory offset
                        space_MMCh_pos_fab(i,j,k,dom) = space_MMCh_mem_pos;

                        // Expand memory
                        this->space_MMCh_host_mem.resize(space_MMCh_mem_pos+sNp*sNp);

                        // Store data
                        std::fill(MM, MM+sNp*sNp, 0.0);

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

                            bf._eval_BF_table_(x);

                            for (int cs = 0; cs < sNp; ++cs)
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                MM[rs+cs*sNp] += bf.BF[rs]*bf.BF[cs]*w;
                            }
                        }

                        // Note: consistently with the storing of the 
                        //       quadrature points, we are storing the
                        //       mass matrix as if it were computed on
                        //       a [-1,1]^AMREX_SPACEDIM domain.
                        for (int rs = 0; rs < (sNp*sNp); ++rs)
                        {
                            this->space_MMCh_host_mem[space_MMCh_mem_pos+rs] = MM[rs]*iaJ;
                        }

                        // Update memory
                        space_MMCh_mem_pos += sNp*sNp;
                    }
                }
                // ----------------------------------------------------
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
                    printf("| %lu doubles (%f MB) for the basis functions evaluated at the domain quadrature points;\n", (AMREX_SPACEDIM+1)*space_Dom_BF_mem_pos, (AMREX_SPACEDIM+1)*space_Dom_BF_mem_pos*sizeof(Real)/(1024.0*1024.0));
                    printf("| %lu doubles (%f MB) for the basis functions evaluated at the domain query points;\n", space_Tri_BF_mem_pos, space_Tri_BF_mem_pos*sizeof(Real)/(1024.0*1024.0));
                    printf("| %lu doubles (%f MB) for the basis functions evaluated at the internal boundary quadrature points;\n", (AMREX_SPACEDIM+1)*space_IntBou_BF_mem_pos, (AMREX_SPACEDIM+1)*space_IntBou_BF_mem_pos*sizeof(Real)/(1024.0*1024.0));
                    printf("| %lu doubles (%f MB) for the mass matrices.\n", space_MMCh_mem_pos, space_MMCh_mem_pos*sizeof(Real)/(1024.0*1024.0));
                }

                ParallelDescriptor::Barrier();
            }

            ParallelDescriptor::Barrier();
            Print() << std::endl;
        }
        // ------------------------------------------------------------
#endif
        // STORE MEMORY INFO
        this->mem_for_space_Dom_BF = space_Dom_BF_mem_pos;
        this->mem_for_space_Tri_BF = space_Tri_BF_mem_pos;
        this->mem_for_space_IntBou_BF = space_IntBou_BF_mem_pos;
        this->mem_for_space_MMCh = space_MMCh_mem_pos;
        // ============================================================

        // STORE THE IMPLICIT MESH INFO: BOUNDARY (FACE-CENTERED) DATA
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            for (MFIter mfi(iGeom.eBouType[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
                Array4<int const> const & eBouType_fab = iGeom.eBouType[dir].array(mfi);

                Array4<int const> const & eBouQuad_Nq_fab = iGeom.eBouQuad_Nq[dir].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab = iGeom.eBouQuad_pos[dir].array(mfi);

                Array4<long> const & space_Bou_BF_pos_fab = this->space_Bou_BF_pos[dir].array(mfi);

                for (int dom = 0; dom < N_DOM; ++dom)
                for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    // LOCAL PARAMETERS
                    const int etype = eBouType_fab(i,j,k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom);
                    const bool elm_bou_is_empty = (etype == DG_ELM_BOU_TYPE_EMPTY);
                    const bool elm_bou_is_entire = (etype == DG_ELM_BOU_TYPE_ENTIRE);
                    const bool elm_bou_is_partial = (etype == DG_ELM_BOU_TYPE_PARTIAL);

                    // LOCAL VARIABLES
                    int b;
                    int min_i, min_j, min_k, pls_i, pls_j, pls_k;
                    int min_etype, pls_etype;
                    Real x[AMREX_SPACEDIM];
                    Real min_BF_lo[AMREX_SPACEDIM], min_BF_hi[AMREX_SPACEDIM];
                    Real pls_BF_lo[AMREX_SPACEDIM], pls_BF_hi[AMREX_SPACEDIM];

                    // Get the indices, the type and the midpoint of the
                    // "plus" and "minus" elements
                    FACE_2_NBRS(i, j, k, dir, min_i, min_j, min_k, pls_i, pls_j, pls_k);
                    min_etype = eType_fab(min_i,min_j,min_k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom);
                    pls_etype = eType_fab(pls_i,pls_j,pls_k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom);
                    Real min_cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(min_i+0.5)*dx[0], prob_lo[1]+(min_j+0.5)*dx[1], prob_lo[2]+(min_k+0.5)*dx[2])};
                    Real pls_cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(pls_i+0.5)*dx[0], prob_lo[1]+(pls_j+0.5)*dx[1], prob_lo[2]+(pls_k+0.5)*dx[2])};
                    
                    // Get the support of the basis functions for both
                    // the "plus" and the "minus" elements
                    BF_CELL(&prob_lo[0], &dx[0], min_i, min_j, min_k, min_etype, min_BF_lo, min_BF_hi);
                    BF_CELL(&prob_lo[0], &dx[0], pls_i, pls_j, pls_k, pls_etype, pls_BF_lo, pls_BF_hi);

                    // Data structures used to compute the basis functions
                    DG_sol_space_BF min_bf(min_BF_lo, min_BF_hi, sp);
                    DG_sol_space_BF pls_bf(pls_BF_lo, pls_BF_hi, sp);

                    // EMPTY ELEMENT BOUNDARY -------------------------
                    if (elm_bou_is_empty)
                    {
                        // BOUNDARY^-
                        // Store memory offset
                        space_Bou_BF_pos_fab(i,j,k,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom) = -1L;

                        // BOUNDARY^+
                        // Store memory offset
                        space_Bou_BF_pos_fab(i,j,k,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+1) = -1L;
                    }
                    // ------------------------------------------------
                    // ENTIRE ELEMENT BOUNDARY ------------------------
                    else if (elm_bou_is_entire)
                    {
                        // QUADRATURE INFO
                        const int bou_Nq = iGeom.uHyperrect.bou_Nq;

                        // BOUNDARY^-
                        b = 2*dir+1;

                        // Store memory offset
                        space_Bou_BF_pos_fab(i,j,k,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom) = b*bou_Nq*sNp;

                        // BOUNDARY^+
                        b = 2*dir;

                        // Store memory offset
                        space_Bou_BF_pos_fab(i,j,k,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+1) = b*bou_Nq*sNp;
                    }
                    // ------------------------------------------------
                    // PARTIAL ELEMENT BOUNDARY -----------------------
                    else if (elm_bou_is_partial)
                    {
                        // QUADRATURE INFO
                        int bou_Nq;
                        long pos;
                        const Real * xptr;
                        long qi_pos;

                        // BOUNDARY^-
                        bou_Nq = eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom);
                        pos = eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom);
                        xptr = &eBouQuad_mem_ptr[pos];

                        // Store memory offset
                        space_Bou_BF_pos_fab(i,j,k,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom) = space_Bou_BF_mem_pos;

                        // Expand memory
                        this->space_Bou_BF_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);
                        AMREX_D_TERM
                        (
                            this->space_Bou_dBFdXI1_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);,
                            this->space_Bou_dBFdXI2_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);,
                            this->space_Bou_dBFdXI3_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);
                        )

                        // Store data
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            qi_pos = (AMREX_SPACEDIM+1)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+min_cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+min_cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+min_cell_m[2];
                            )

                            min_bf._eval_BF_table_(x);

                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                this->space_Bou_BF_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = min_bf.BF[rs];
                                AMREX_D_TERM
                                (
                                    this->space_Bou_dBFdXI1_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = min_bf.dBFdx1[rs]*J[0];,
                                    this->space_Bou_dBFdXI2_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = min_bf.dBFdx2[rs]*J[1];,
                                    this->space_Bou_dBFdXI3_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = min_bf.dBFdx3[rs]*J[2];
                                )
                            }
                        }

                        // Update memory
                        space_Bou_BF_mem_pos += bou_Nq*sNp;

                        // BOUNDARY^+
                        bou_Nq = eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1);
                        pos = eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1);
                        xptr = &eBouQuad_mem_ptr[pos];

                        // Store memory offset
                        space_Bou_BF_pos_fab(i,j,k,DG_BOU_BF_INFO_N_COMP_PER_DOM*dom+1) = space_Bou_BF_mem_pos;

                        // Expand memory
                        this->space_Bou_BF_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);
                        AMREX_D_TERM
                        (
                            this->space_Bou_dBFdXI1_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);,
                            this->space_Bou_dBFdXI2_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);,
                            this->space_Bou_dBFdXI3_host_mem.resize(space_Bou_BF_mem_pos+bou_Nq*sNp);
                        )

                        // Store data
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            qi_pos = (AMREX_SPACEDIM+1)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+pls_cell_m[0];,
                                x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+pls_cell_m[1];,
                                x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+pls_cell_m[2];
                            )

                            pls_bf._eval_BF_table_(x);

                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                this->space_Bou_BF_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = pls_bf.BF[rs];
                                AMREX_D_TERM
                                (
                                    this->space_Bou_dBFdXI1_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = pls_bf.dBFdx1[rs]*J[0];,
                                    this->space_Bou_dBFdXI2_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = pls_bf.dBFdx2[rs]*J[1];,
                                    this->space_Bou_dBFdXI3_host_mem[space_Bou_BF_mem_pos+rs+qi*sNp] = pls_bf.dBFdx3[rs]*J[2];
                                )
                            }
                        }

                        // Update memory
                        space_Bou_BF_mem_pos += bou_Nq*sNp;
                    }
                    // ------------------------------------------------
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
                    printf("| %lu doubles (%f MB) for the basis functions evaluated at the boundary quadrature points.\n", (AMREX_SPACEDIM+1)*space_Bou_BF_mem_pos, (AMREX_SPACEDIM+1)*space_Bou_BF_mem_pos*sizeof(Real)/(1024.0*1024.0));
                }

                ParallelDescriptor::Barrier();
            }

            ParallelDescriptor::Barrier();
            Print() << std::endl;
        }
        // ------------------------------------------------------------
#endif
        // STORE MEMORY INFO
        this->mem_for_space_Bou_BF = space_Bou_BF_mem_pos;
        // ============================================================

        // COPY MEMORY TO THE DEVICE ==================================
#ifdef AMREX_USE_CUDA
        this->space_Dom_BF_device_mem.resize(this->space_Dom_BF_host_mem.size());
        this->space_Tri_BF_device_mem.resize(this->space_Tri_BF_host_mem.size());
        AMREX_D_TERM
        (
            this->space_Dom_dBFdXI1_device_mem.resize(this->space_Dom_dBFdXI1_host_mem.size());,
            this->space_Dom_dBFdXI2_device_mem.resize(this->space_Dom_dBFdXI2_host_mem.size());,
            this->space_Dom_dBFdXI3_device_mem.resize(this->space_Dom_dBFdXI3_host_mem.size());
        )
        this->space_Bou_BF_device_mem.resize(this->space_Bou_BF_host_mem.size());
        AMREX_D_TERM
        (
            this->space_Bou_dBFdXI1_device_mem.resize(this->space_Bou_dBFdXI1_host_mem.size());,
            this->space_Bou_dBFdXI2_device_mem.resize(this->space_Bou_dBFdXI2_host_mem.size());,
            this->space_Bou_dBFdXI3_device_mem.resize(this->space_Bou_dBFdXI3_host_mem.size());
        )
        this->space_IntBou_BF_device_mem.resize(this->space_IntBou_BF_host_mem.size());
        AMREX_D_TERM
        (
            this->space_IntBou_dBFdXI1_device_mem.resize(this->space_IntBou_dBFdXI1_host_mem.size());,
            this->space_IntBou_dBFdXI2_device_mem.resize(this->space_IntBou_dBFdXI2_host_mem.size());,
            this->space_IntBou_dBFdXI3_device_mem.resize(this->space_IntBou_dBFdXI3_host_mem.size());
        )
        this->space_MMCh_device_mem.resize(this->space_MMCh_host_mem.size());

        Gpu::copy(Gpu::hostToDevice, this->space_Dom_BF_host_mem.begin(), this->space_Dom_BF_host_mem.end(), this->space_Dom_BF_device_mem.begin());
        Gpu::copy(Gpu::hostToDevice, this->space_Tri_BF_host_mem.begin(), this->space_Tri_BF_host_mem.end(), this->space_Tri_BF_device_mem.begin());
        AMREX_D_TERM
        (
            Gpu::copy(Gpu::hostToDevice, this->space_Dom_dBFdXI1_host_mem.begin(), this->space_Dom_dBFdXI1_host_mem.end(), this->space_Dom_dBFdXI1_device_mem.begin());,
            Gpu::copy(Gpu::hostToDevice, this->space_Dom_dBFdXI2_host_mem.begin(), this->space_Dom_dBFdXI2_host_mem.end(), this->space_Dom_dBFdXI2_device_mem.begin());,
            Gpu::copy(Gpu::hostToDevice, this->space_Dom_dBFdXI3_host_mem.begin(), this->space_Dom_dBFdXI3_host_mem.end(), this->space_Dom_dBFdXI3_device_mem.begin());;
        )
        Gpu::copy(Gpu::hostToDevice, this->space_Bou_BF_host_mem.begin(), this->space_Bou_BF_host_mem.end(), this->space_Bou_BF_device_mem.begin());
        AMREX_D_TERM
        (
            Gpu::copy(Gpu::hostToDevice, this->space_Bou_dBFdXI1_host_mem.begin(), this->space_Bou_dBFdXI1_host_mem.end(), this->space_Bou_dBFdXI1_device_mem.begin());,
            Gpu::copy(Gpu::hostToDevice, this->space_Bou_dBFdXI2_host_mem.begin(), this->space_Bou_dBFdXI2_host_mem.end(), this->space_Bou_dBFdXI2_device_mem.begin());,
            Gpu::copy(Gpu::hostToDevice, this->space_Bou_dBFdXI3_host_mem.begin(), this->space_Bou_dBFdXI3_host_mem.end(), this->space_Bou_dBFdXI3_device_mem.begin());;
        )
        Gpu::copy(Gpu::hostToDevice, this->space_IntBou_BF_host_mem.begin(), this->space_IntBou_BF_host_mem.end(), this->space_IntBou_BF_device_mem.begin());
        AMREX_D_TERM
        (
            Gpu::copy(Gpu::hostToDevice, this->space_IntBou_dBFdXI1_host_mem.begin(), this->space_IntBou_dBFdXI1_host_mem.end(), this->space_IntBou_dBFdXI1_device_mem.begin());,
            Gpu::copy(Gpu::hostToDevice, this->space_IntBou_dBFdXI2_host_mem.begin(), this->space_IntBou_dBFdXI2_host_mem.end(), this->space_IntBou_dBFdXI2_device_mem.begin());,
            Gpu::copy(Gpu::hostToDevice, this->space_IntBou_dBFdXI3_host_mem.begin(), this->space_IntBou_dBFdXI3_host_mem.end(), this->space_IntBou_dBFdXI3_device_mem.begin());;
        )
        Gpu::copy(Gpu::hostToDevice, this->space_MMCh_host_mem.begin(), this->space_MMCh_host_mem.end(), this->space_MMCh_device_mem.begin());
#endif
        // ============================================================
        
        // ADD THE CONTRIBUTION FOR THE EXTENDED ELEMENTS =============
#ifdef AMREX_USE_CUDA
        Real * space_MMCh_mem_ptr = this->space_MMCh_device_mem.data();
#else
        Real * space_MMCh_mem_ptr = this->space_MMCh_host_mem.data();
#endif

        for (MFIter mfi(iGeom.eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & eType_fab = iGeom.eType.array(mfi);
            Array4<long const> const & space_MMCh_pos_fab = this->space_MMCh_pos.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
                const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);
                const bool elm_is_special = elm_is_large || elm_is_entire_extended;
                const bool elm_is_valid = elm_is_entire || elm_is_large;
                const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

                const long pos = space_MMCh_pos_fab(i,j,k,dom);

                // LOCAL VARIABLES
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_etype, nbr_merged_b;
                bool nbr_is_small;
                long nbr_pos;
                Real MM[DG_SOL_MAX_SPACE_Np*DG_SOL_MAX_SPACE_Np];
                Real work[DG_SOL_MAX_SPACE_Np*DG_SOL_MAX_SPACE_Np];

                // RETRIEVE THE DG MATRICES FOR VALID SPECIAL ELEMENTS
                if (elm_is_special)
                {
                    for (int rs = 0; rs < (sNp*sNp); ++rs)
                    {
                        // Mass matrix
                        MM[rs] = space_MMCh_mem_ptr[pos+rs];
                    }
                }

                // Cycle over the boundaries, check whether the element
                // is extended and add the contribution
                if (elm_is_extended)
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    if (nbr_merged_b == nbr_b)
                    {
                        nbr_pos = space_MMCh_pos_fab(nbr_i,nbr_j,nbr_k,dom);

                        for (int rs = 0; rs < (sNp*sNp); ++rs)
                        {
                            // Mass matrix
                            MM[rs] += space_MMCh_mem_ptr[nbr_pos+rs];
                        }
                    }
                }

                // CHOLESKY DECOMPOSITION OF MASS MATRIX FOR VALID
                // SPECIAL ELEMENTS
                if (elm_is_special)
                {
                    DG_utils::Cholesky(sNp, MM, &space_MMCh_mem_ptr[pos], work);
                }
            });
            Gpu::synchronize();
        }
        // ============================================================
        
        // CHECK THE COMPUTED BASIS FUNCTIONS =========================
        if (true)
        {

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
// AMReX_DG_InputReader.H

#ifndef BL_DG_INPUT_READER_H_
#define BL_DG_INPUT_READER_H_

#include <AMReX_DG_Base.H>

namespace amrex
{
namespace DG
{

// INPUT DATA STRUCTURES ##############################################
struct SpaceInput
{
    Vector<int> is_periodic;
    int coord_sys;
    Vector<Real> prob_lo, prob_hi;
    Array<Vector<int>, 2*AMREX_SPACEDIM> nbr, nbr_bou, nbr_bou_nbr;

    SpaceInput()
    :
    is_periodic({AMREX_D_DECL(0, 0, 0)}),
    coord_sys(0),
    prob_lo({AMREX_D_DECL( 0.0,  0.0,  0.0)}),
    prob_hi({AMREX_D_DECL(+1.0, +1.0, +1.0)})
    {}
};

struct TimeInput
{
    Real T;
    int n_steps;

    TimeInput()
    :
    T(0.0),
    n_steps(0)
    {}
};

struct MeshInput
{
    IntVect n_cells, max_grid_size;

    MeshInput()
    :
    n_cells(AMREX_D_DECL(0, 0, 0)),
    max_grid_size(AMREX_D_DECL(0, 0, 0))
    {}
};

struct DGInput
{
    int phi_space_p, space_p, time_p;
    int space_q, time_q;
    

    DGInput()
    :
    phi_space_p(-1),
    space_p(-1),
    time_p(-1),
    space_q(-1),
    time_q(-1)
    {}
};
// ####################################################################



// ####################################################################
// INPUT READER CLASS #################################################
// ####################################################################
struct InputReader
{
    // DATA MEMBERS ###################################################
    int n_patches;

    Vector<SpaceInput> space;

    TimeInput time;

    Vector<MeshInput> mesh;

    Vector<DGInput> dG;

    int plot_int;
    //SX ========
    std::string post_processing_by_convolution;
    bool post_processing_by_convolution_flag;
    //SX ========
    // ################################################################

    // CONSTRUCTOR ####################################################
    InputReader()
    :
    n_patches(1),
    plot_int(-1),
    post_processing_by_convolution_flag(false)
    {
        // VARIABLES ==================================================
        amrex::ParmParse pp;
        amrex::ParmParse pp_time("time");
        // ============================================================

        // GET THE NUMBER OF PATCHES ==================================
        pp.query("n_patches", this->n_patches);

        // SET MEMORY
        if (this->n_patches > 0)
        {
            // SPACE
            this->space.resize(this->n_patches);

            // MESH
            this->mesh.resize(this->n_patches);

            // DISCONTINUOUS GALERKIN
            this->dG.resize(this->n_patches);
        }
        else
        {
            Print() << std::endl;
            Print() << "ERROR: AMReX_DG_InputReader.H - InputReader::InputReader" << std::endl;
            Print() << "| Unexpected number of patches in the input file:" << std::endl;
            Print() << "| n_patches: " << this->n_patches << std::endl;
            exit(-1);
        }
        // ============================================================

        // READ GEOMETRY INFO =========================================
        for (int p = 0; p < this->n_patches; ++p)
        {
            amrex::ParmParse pp_space("space["+std::to_string(p)+"]");

            // PERIODICITY --------------------------------------------
            pp_space.queryarr("is_periodic", this->space[p].is_periodic);
            // --------------------------------------------------------

            // COORDINATE SYSTEM --------------------------------------
            pp_space.query("coord_sys", this->space[p].coord_sys);
            // --------------------------------------------------------

            // PATCH REAL BOX -----------------------------------------
            pp_space.queryarr("prob_lo", this->space[p].prob_lo);
            pp_space.queryarr("prob_hi", this->space[p].prob_hi);
            // --------------------------------------------------------

            // NEIGHBORING DOMAIN INFORMATION -------------------------
            for (int b = 0; b < (2*AMREX_SPACEDIM); ++b)
            {
                // DEFAULT VALUES
                // In case the neighbor information is not specified,
                // the domain is assumed to have no neighbors.
                this->space[p].nbr[b] = {this->n_patches};

                // READ FROM FILE
                const std::string s_nbr = "nbr["+std::to_string(b)+"]";
                
                pp_space.queryarr(s_nbr.c_str(), this->space[p].nbr[b]);
            }
            // --------------------------------------------------------
        }

        for (int p = 0; p < this->n_patches; ++p)
        {
            amrex::ParmParse pp_space("space["+std::to_string(p)+"]");

            // NEIGHBORING DOMAIN BOUNDARY INFORMATION ----------------
            for (int b = 0; b < (2*AMREX_SPACEDIM); ++b)
            {
                const int n_nbr = this->space[p].nbr[b].size();

                // ALLOCATE MEMORY
                if (n_nbr > 0)
                {
                    this->space[p].nbr_bou[b].resize(n_nbr);
                    this->space[p].nbr_bou_nbr[b].resize(n_nbr);
                }

                // DEFAULT VALUES
                // In case the neighbor boundary information is not
                // specified, determine it.
                for (int n = 0; n < n_nbr; ++n)
                {
                    if (std::abs(this->space[p].nbr[b][n]) >= this->n_patches)
                    {
                        this->space[p].nbr_bou[b][n] = this->space[p].nbr[b][n];
                        this->space[p].nbr_bou_nbr[b][n] = this->space[p].nbr[b][n];
                    }
                    else
                    {
                        const int nbr_p = this->space[p].nbr[b][n];

                        bool found;
                        int nbr_b, nbr_n;

                        found = false;
                        nbr_b = 0;
                        while ((!found) && (nbr_b < (2*AMREX_SPACEDIM)))
                        {
                            const int nbr_n_nbr = this->space[nbr_p].nbr[nbr_b].size();

                            nbr_n = 0;
                            while ((!found) && (nbr_n < nbr_n_nbr))
                            {
                                if (this->space[nbr_p].nbr[nbr_b][nbr_n] == p)
                                {
                                    found = true;
                                    this->space[p].nbr_bou[b][n] = nbr_b;
                                    this->space[p].nbr_bou_nbr[b][n] = nbr_n;
                                }
                                else if (this->space[nbr_p].nbr[nbr_b][nbr_n] == -p)
                                {
                                    // periodic neighbor to be implemented
                                }
                                else
                                {
                                    nbr_n += 1;
                                }
                            }

                            if (!found) nbr_b += 1;
                        }
                    }
                }

                // READ FROM FILE
                const std::string s_nbr = "nbr_bou["+std::to_string(b)+"]";
                
                pp_space.queryarr(s_nbr.c_str(), this->space[p].nbr_bou[b]);
            }
            // --------------------------------------------------------
        }
        // ============================================================

        // READ TIME INFO =============================================
        pp_time.get("T", this->time.T);
        pp_time.get("n_steps", this->time.n_steps);
        // ============================================================

        // READ MESH INFO =============================================
        for (int p = 0; p < this->n_patches; ++p)
        {
            Vector<int> aux;
            amrex::ParmParse pp_mesh("mesh["+std::to_string(p)+"]");

            // NUMBER OF CELLS ----------------------------------------
            pp_mesh.getarr("n_cells", aux);
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) this->mesh[p].n_cells[dim] = aux[dim];
            // --------------------------------------------------------

            // MAX GRID SIZE ------------------------------------------
            pp_mesh.getarr("max_grid_size", aux);
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) this->mesh[p].max_grid_size[dim] = aux[dim];
            // --------------------------------------------------------
        }
        // ============================================================

        // DISCONTINUOUS GALERKIN INFO ================================
        for (int p = 0; p < this->n_patches; ++p)
        {
            amrex::ParmParse pp_dG("dG["+std::to_string(p)+"]");

            pp_dG.get("phi_space_p", this->dG[p].phi_space_p);
            pp_dG.get("space_p", this->dG[p].space_p);
            pp_dG.get("time_p", this->dG[p].time_p);
            

            this->dG[p].space_q = std::max(this->dG[p].phi_space_p+2, this->dG[p].space_p+4);
            //this->dG[p].space_q = ?
        }
        // ============================================================

        // A FEW CONSISTENCY CHECKS (AND SOME POST-PROCESSING) ========
        for (int p = 0; p < this->n_patches; ++p)
        {
            // GEOMETRY INFO
            // PATCH REAL BOX -----------------------------------------
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
            {
                if (this->space[p].prob_lo[dim] >= this->space[p].prob_hi[dim])
                {
                    Print() << std::endl;
                    Print() << "ERROR: AMReX_DG_InputReader.H - InputReader::InputReader" << std::endl;
                    Print() << "| The real box limits must be strictly ascending:" << std::endl;
                    Print() << "| PATCH: " << p << std::endl;
                    Print() << "|  - prob_lo[" << dim << "]: " << this->space[p].prob_lo[dim] << std::endl;
                    Print() << "|  - prob_hi[" << dim << "]: " << this->space[p].prob_hi[dim] << std::endl;
                    exit(-1);
                }
            }
            // --------------------------------------------------------

            // NEIGHBORING DOMAIN INFORMATION -------------------------
            for (int b = 0; b < (2*AMREX_SPACEDIM); ++b)
            {
                const int n_nbr = this->space[p].nbr[b].size();

                for (int n = 0; n < n_nbr; ++n)
                {
                    if (std::abs(this->space[p].nbr[b][n]) < this->n_patches)
                    {
                        const int nbr_p = this->space[p].nbr[b][n];
                        const int nbr_b = this->space[p].nbr_bou[b][n];

                        bool found;
                        int nbr_n;

                        found = false;
                        nbr_n = 0;
                        while ((!found) && (nbr_n < n_nbr))
                        {
                            if (this->space[nbr_p].nbr[nbr_b][nbr_n] == p)
                            {
                                found = true;
                                this->space[p].nbr_bou_nbr[b][n] = nbr_n;
                            }
                            else if (this->space[nbr_p].nbr[nbr_b][nbr_n] == -p)
                            {
                                // periodic neighbor to implemented
                            }
                            else
                            {
                                nbr_n += 1;
                            }
                        }

                        if (!found)
                        {
                            Print() << std::endl;
                            Print() << "ERROR: AMReX_DG_InputReader.H - InputReader::InputReader" << std::endl;
                            Print() << "| Neighbors information is not consistent:" << std::endl;
                            Print() << "| PATCH: " << p << std::endl;
                            Print() << "|  - nbr[" << b << "]: ";
                            if (std::abs(this->space[p].nbr[b][0]) >= this->n_patches) Print() << "None";
                            else Print() << this->space[p].nbr[b][0];
                            for (int i = 1; i < n_nbr; ++i)
                            if (std::abs(this->space[p].nbr[b][i]) >= this->n_patches) Print() << ", None";
                            else Print() << ", " << this->space[p].nbr[b][i];
                            Print() << std::endl;
                            Print() << "| PATCH: " << nbr_p << std::endl;
                            Print() << "|  - nbr[" << nbr_b << "]: ";
                            if (std::abs(this->space[nbr_p].nbr[nbr_b][0]) >= this->n_patches) Print() << "None";
                            else Print() << this->space[nbr_p].nbr[nbr_b][0];
                            for (int i = 1; i < n_nbr; ++i)
                            if (std::abs(this->space[nbr_p].nbr[nbr_b][i]) >= this->n_patches) Print() << ", None";
                            else Print() << ", " << this->space[nbr_p].nbr[nbr_b][i];
                            Print() << std::endl;
                            exit(-1);
                        }
                    }
                }
            }
            // --------------------------------------------------------
        }
        // ============================================================

        // OUTPUT =====================================================
        pp.query("plot_int", this->plot_int);
        pp.query("post_processing_by_convolution", this->post_processing_by_convolution);
        if (this->post_processing_by_convolution == "true") this->post_processing_by_convolution_flag = true;
        else this->post_processing_by_convolution_flag  = false;

        // ============================================================

        // REPORT TO SCREEN ===========================================
#if (DG_VERBOSITY > 0)
        Print() << "# ANALYSIS INPUT                                                       " << std::endl;
        Print() << "# MODEL ===============================================================" << std::endl;
        Print() << "# Number of patches: " << this->n_patches << std::endl;
        Print() << "# =====================================================================" << std::endl;
        Print() << "# SPACE ===============================================================" << std::endl;
        for (int p = 0; p < this->n_patches; ++p)
        {
            Print() << "# PATCH: " << p << std::endl;
            Print() << "#  - is_periodic: " << this->space[p].is_periodic[0];
            for (int dim = 1; dim < AMREX_SPACEDIM; ++dim) Print() << ", " << this->space[p].is_periodic[dim];
            Print() << std::endl;

            Print() << "#  - coord_sys: " << this->space[p].coord_sys << std::endl;
            
            Print() << "#  - prob_lo: " << this->space[p].prob_lo[0];
            for (int dim = 1; dim < AMREX_SPACEDIM; ++dim) Print() << ", " << this->space[p].prob_lo[dim];
            Print() << std::endl;
            Print() << "#  - prob_hi: " << this->space[p].prob_hi[0];
            for (int dim = 1; dim < AMREX_SPACEDIM; ++dim) Print() << ", " << this->space[p].prob_hi[dim];
            Print() << std::endl;
            
            for (int b = 0; b < (2*AMREX_SPACEDIM); ++b)
            {
                const int n_nbr = this->space[p].nbr[b].size();

                Print() << "#  - nbr[" << b << "]: ";
                if (std::abs(this->space[p].nbr[b][0]) >= this->n_patches)
                {
                    Print() << "None";
                }
                else
                {
                    Print() << this->space[p].nbr[b][0] << "(" << this->space[p].nbr_bou[b][0] << ")";
                }
                for (int n = 1; n < n_nbr; ++n)
                {
                    if (std::abs(this->space[p].nbr[b][n]) >= this->n_patches)
                    {
                        Print() << ", None";
                    }
                    else
                    {
                        Print() << ", " << this->space[p].nbr[b][n] << "(" << this->space[p].nbr_bou[b][n] << ")";
                    }
                }
                Print() << std::endl;
            }
        }
        Print() << "# =====================================================================" << std::endl;
        Print() << "# TIME ================================================================" << std::endl;
        Print() << "# time.T: " << this->time.T << std::endl;
        Print() << "# time.n_steps: " << this->time.n_steps << std::endl;
        Print() << "# =====================================================================" << std::endl;
        Print() << "# SPACE AND TIME DISCRETIZATION =======================================" << std::endl;
        Print() << "# MESH GRID -----------------------------------------------------------" << std::endl;
        for (int p = 0; p < this->n_patches; ++p)
        {
            Print() << "# PATCH: " << p << std::endl;
            Print() << "#  - n_cells      : " << this->mesh[p].n_cells[0];
            for (int dim = 1; dim < AMREX_SPACEDIM; ++dim) Print() << ", " << this->mesh[p].n_cells[dim];
            Print() << std::endl;
            Print() << "#  - max_grid_size: " << this->mesh[p].max_grid_size[0];
            for (int dim = 1; dim < AMREX_SPACEDIM; ++dim) Print() << ", " << this->mesh[p].max_grid_size[dim];
            Print() << std::endl;
        }
        Print() << "# ---------------------------------------------------------------------" << std::endl;
        Print() << "# DISCONTINUOUS GALERKIN DISCRETIZATION -------------------------------" << std::endl;
        for (int p = 0; p < this->n_patches; ++p)
        {
            Print() << "# PATCH: " << p << std::endl;
            Print() << "#  - phi_space_p: " << this->dG[p].phi_space_p << std::endl;
            Print() << "#  - space_p    : " << this->dG[p].space_p << std::endl;
            Print() << "#  - time_p     : " << this->dG[p].time_p << std::endl;
            Print() << "#  - space_q    : " << this->dG[p].space_q << std::endl;
            Print() << "#  - time_q     : " << this->dG[p].time_q << std::endl;
        }
        Print() << "# ---------------------------------------------------------------------" << std::endl;
        Print() << "# =====================================================================" << std::endl;
        Print() << "# POST-PROCESSING INFO ================================================" << std::endl;
        Print() << "# plot_int = " << this->plot_int << std::endl;
        Print() << "# =====================================================================" << std::endl;
        Print() << "#######################################################################" << std::endl;
#endif
        // ============================================================
    }
    // ################################################################
};
// ####################################################################
// ####################################################################


} // namespace DG
} // namespace amrex

#endif
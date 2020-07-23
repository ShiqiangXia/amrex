#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>

#include <AMReX_BC_TYPES.H>

#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we are solving an advection equation of the form
//
// phi_{,t}+(u_i phi)_{,i} = 0.
//
// where phi is the advected field and u = {u_i} is a divergence-free
// velocity field.
// The problem is subjected to periodic boundary conditions and the
// following initial conditions:
//
// phi(0,x) = phi0(x)
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include <IBVP_Linear_Advection.H>
// ====================================================================
// ####################################################################

// USEFUL DATA STRUCTURES #############################################
// SPACE --------------------------------------------------------------
struct space_struct
{
    amrex::Vector<int> is_periodic;
    int coord_sys;
    amrex::Vector<amrex::Real> prob_lo, prob_hi;
    amrex::Vector<int> bc_lo, bc_hi;

    public:
    space_struct()
    :
    is_periodic({AMREX_D_DECL(1, 1, 1)}),
    coord_sys(0),
    prob_lo({AMREX_D_DECL( 0.0,  0.0,  0.0)}),
    prob_hi({AMREX_D_DECL(-1.0, -1.0, -1.0)}),
    bc_lo({AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir)}),
    bc_hi({AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir)})
    {}
};
// --------------------------------------------------------------------

// TIME ---------------------------------------------------------------
struct time_struct
{
    amrex::Real T;

    public:
    time_struct()
    :
    T(0.0)
    {}
};
// --------------------------------------------------------------------

// SPACE AND TIME DISCRETIZATION --------------------------------------
struct mesh_struct
{
    amrex::IntVect n_cells;
    int max_grid_size, n_time_steps;

    public:
    mesh_struct():
    n_cells(AMREX_D_DECL(0, 0, 0)),
    max_grid_size(0),
    n_time_steps(0)
    {}
};
// --------------------------------------------------------------------

// DISCONTINOUS GALERKIN METHOD ---------------------------------------
struct dG_struct
{
    int phi_space_p, space_p, time_p;
    int space_q, time_q;
    std::string use_slope_limiter;
    bool use_slope_limiter_flag;
    //SX ========
    std::string post_processing_by_convolution;
    bool post_processing_by_convolution_flag;
    //SX ========

    public:
    dG_struct()
    :
    phi_space_p(-1),
    space_p(-1),
    time_p(-1),
    use_slope_limiter_flag(false),
    //SX ========
    post_processing_by_convolution_flag(false)
    //SX ========
    {}
};
// --------------------------------------------------------------------

// INPUTS -------------------------------------------------------------
struct inputs_struct
{
    space_struct space;
    time_struct time;
    mesh_struct mesh;
    dG_struct dG;
    int plot_int;

    public:
    inputs_struct()
    :
    plot_int(-1)
    {}
};
// --------------------------------------------------------------------
// ####################################################################

// ACTUAL MAIN PROGRAM ################################################
void main_main()
{

    // HEADING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# AMREX & DG PROJECT                                                   " << std::endl;
amrex::Print() << "# Author: Vincenzo Gulizzi (vgulizzi@lbl.gov)                          " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# SUMMARY:                                                             " << std::endl;
amrex::Print() << "# In this tutorial, we are solving an advection equation of the form   " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# phi_{,t}+(u_i phi)_{,i} = 0                                          " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# where phi is the advected field and u = {u_i} is a divergence-free   " << std::endl;
amrex::Print() << "# velocity field.                                                      " << std::endl;
amrex::Print() << "# The problem is subjected to periodic boundary conditions and the     " << std::endl;
amrex::Print() << "# following initial conditions:                                        " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# phi(0,x) = phi0(x)                                                   " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // GENERAL VARIABLES =============
    amrex::Real start_time, stop_time;
    // ===============================

    // READ INPUT PARAMETERS ==========================================
    // DATA STRUCTURES
    inputs_struct inputs;
    {
        // INPUT PARSER -------------------------
        amrex::ParmParse pp;
        amrex::ParmParse pp_space("space");
        amrex::ParmParse pp_time("time");
        amrex::ParmParse pp_mesh("mesh");
        amrex::ParmParse pp_dG("dG");
        // --------------------------------------
        
        // LOCAL VARIABLES -----------
        amrex::Vector<int> vector_int;
        // ---------------------------

        // SPACE ------------------------------------------------------
        pp_space.queryarr("is_periodic", inputs.space.is_periodic);
        pp_space.query("coord_sys", inputs.space.coord_sys);
        pp_space.getarr("prob_lo", inputs.space.prob_lo);
        pp_space.getarr("prob_hi", inputs.space.prob_hi);
        pp_space.queryarr("bc_lo", inputs.space.bc_lo);
        pp_space.queryarr("bc_hi", inputs.space.bc_hi);
        // ------------------------------------------------------------

        // TIME -------------------------------------------------------
        pp_time.get("T", inputs.time.T);
        // ------------------------------------------------------------

        // SPACE AND TIME DISCRETIZATION ------------------------------
        vector_int.resize(AMREX_SPACEDIM);
        pp_mesh.getarr("n_cells", vector_int);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) inputs.mesh.n_cells[d] = vector_int[d];
        pp_mesh.get("max_grid_size", inputs.mesh.max_grid_size);
        pp_mesh.get("n_time_steps", inputs.mesh.n_time_steps);

        // DG
        pp_dG.get("phi_space_p", inputs.dG.phi_space_p);
        pp_dG.get("space_p", inputs.dG.space_p);
        pp_dG.get("time_p", inputs.dG.time_p);

        inputs.dG.space_q = std::max(inputs.dG.phi_space_p+2, inputs.dG.space_p+3);

        pp_dG.query("use_slope_limiter", inputs.dG.use_slope_limiter);
        if      (inputs.dG.use_slope_limiter == "true")  inputs.dG.use_slope_limiter_flag = true;
        else inputs.dG.use_slope_limiter_flag = false;
        // ------------------------------------------------------------

        // POST-PROCESSING INFO ---------------------------------------
        pp.query("plot_int", inputs.plot_int);

        //SX ========
        pp.query("post_processing_by_convolution", inputs.dG.post_processing_by_convolution);
        if (inputs.dG.post_processing_by_convolution == "true") inputs.dG.post_processing_by_convolution_flag = true;
        else inputs.dG.post_processing_by_convolution_flag = false;
        //SX ========

        // ------------------------------------------------------------
        
    }
    // ================================================================

    // REPORT TO SCREEN ===============================================
amrex::Print() << "# ANALYSIS PARAMETERS                                                  " << std::endl;
amrex::Print() << "# SPACE ===============================================================" << std::endl;
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "# space.is_periodic = " << inputs.space.is_periodic[0] << std::endl;
amrex::Print() << "# space.coord_sys   = " << inputs.space.coord_sys << std::endl;
amrex::Print() << "# space.prob_lo     = " << inputs.space.prob_lo[0] << std::endl;
amrex::Print() << "# space.prob_hi     = " << inputs.space.prob_hi[0] << std::endl;
amrex::Print() << "# space.bc_lo       = " << inputs.space.bc_lo[0] << std::endl;
amrex::Print() << "# space.bc_hi       = " << inputs.space.bc_hi[0] << std::endl;
#endif
#if (AMREX_SPACEDIM == 2)
amrex::Print() << "# space.is_periodic = " << inputs.space.is_periodic[0] << ", " << inputs.space.is_periodic[1] << std::endl;
amrex::Print() << "# space.coord_sys   = " << inputs.space.coord_sys << std::endl;
amrex::Print() << "# space.prob_lo     = " << inputs.space.prob_lo[0] << ", " << inputs.space.prob_lo[1] << std::endl;
amrex::Print() << "# space.prob_hi     = " << inputs.space.prob_hi[0] << ", " << inputs.space.prob_hi[1] << std::endl;
amrex::Print() << "# space.bc_lo       = " << inputs.space.bc_lo[0] << ", " << inputs.space.bc_lo[1] << std::endl;
amrex::Print() << "# space.bc_hi       = " << inputs.space.bc_hi[0] << ", " << inputs.space.bc_hi[1] << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "# space.is_periodic = " << inputs.space.is_periodic[0] << ", " << inputs.space.is_periodic[1] << ", " << inputs.space.is_periodic[2] << std::endl;
amrex::Print() << "# space.coord_sys   = " << inputs.space.coord_sys << std::endl;
amrex::Print() << "# space.prob_lo     = " << inputs.space.prob_lo[0] << ", " << inputs.space.prob_lo[1] << ", " << inputs.space.prob_lo[2] << std::endl;
amrex::Print() << "# space.prob_hi     = " << inputs.space.prob_hi[0] << ", " << inputs.space.prob_hi[1] << ", " << inputs.space.prob_hi[2] << std::endl;
amrex::Print() << "# space.bc_lo       = " << inputs.space.bc_lo[0] << ", " << inputs.space.bc_lo[1] << ", " << inputs.space.bc_lo[2] << std::endl;
amrex::Print() << "# space.bc_hi       = " << inputs.space.bc_hi[0] << ", " << inputs.space.bc_hi[1] << ", " << inputs.space.bc_hi[2] << std::endl;
#endif
amrex::Print() << "#" << std::endl;
amrex::Print() << "# TIME ================================================================" << std::endl;
amrex::Print() << "# time.T = " << inputs.time.T << std::endl;
amrex::Print() << "#" << std::endl;
amrex::Print() << "# =====================================================================" << std::endl;
amrex::Print() << "# SPACE AND TIME DISCRETIZATION =======================================" << std::endl;
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "# mesh.n_cells       = " << inputs.mesh.n_cells[0] << std::endl;
#endif
#if (AMREX_SPACEDIM == 2)
amrex::Print() << "# mesh.n_cells       = " << inputs.mesh.n_cells[0] << ", " << inputs.mesh.n_cells[1] << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "# mesh.n_cells       = " << inputs.mesh.n_cells[0] << ", " << inputs.mesh.n_cells[1] << ", " << inputs.mesh.n_cells[2] << std::endl;
#endif
amrex::Print() << "# mesh.max_grid_size = " << inputs.mesh.max_grid_size << std::endl;
amrex::Print() << "# mesh.n_time_steps  = " << inputs.mesh.n_time_steps << std::endl;
amrex::Print() << "#" << std::endl;
amrex::Print() << "# dG.phi_space_p       = " << inputs.dG.phi_space_p << std::endl;
amrex::Print() << "# dG.space_p           = " << inputs.dG.space_p << std::endl;
amrex::Print() << "# dG.time_p            = " << inputs.dG.time_p << std::endl;
amrex::Print() << "# dG.use_slope_limiter = " << inputs.dG.use_slope_limiter << std::endl;
amrex::Print() << "#" << std::endl;
amrex::Print() << "# =====================================================================" << std::endl;
amrex::Print() << "# POSTPROCESSING INFO =================================================" << std::endl;
amrex::Print() << "# plot_int = " << inputs.plot_int << std::endl;
amrex::Print() << "#" << std::endl;
amrex::Print() << "# =====================================================================" << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // TIC ==========================================
    start_time = amrex::second();
    // ==============================================

    // PHYSICAL BOX ===================================================
    amrex::RealBox real_box(inputs.space.prob_lo.data(),
                            inputs.space.prob_hi.data());
    // ================================================================

    // BOX ARRAY ======================================================
    amrex::Box indices_box({AMREX_D_DECL(0, 0, 0)}, inputs.mesh.n_cells-1);

    amrex::BoxArray ba;
    ba.define(indices_box);
    ba.maxSize(inputs.mesh.max_grid_size);
    // ================================================================

    // GEOMETRY =======================================================
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if ((inputs.space.bc_lo[d] == amrex::BCType::int_dir) && (inputs.space.bc_hi[d] == amrex::BCType::int_dir))
        {
            inputs.space.is_periodic[d] = 1;
        }
        else if ((inputs.space.bc_lo[d] != amrex::BCType::int_dir) && (inputs.space.bc_hi[d] != amrex::BCType::int_dir))
        {
            inputs.space.is_periodic[d] = 0;
        }
    }

    amrex::Geometry geom;
    geom.define(indices_box, &real_box, inputs.space.coord_sys, inputs.space.is_periodic.data());
    // ================================================================
    
    // BOXES DISTRIBUTION AMONG MPI PROCESSES =========================
    amrex::DistributionMapping dm(ba);
    // ================================================================

    // INIT DG DATA STRUCTURES ========================================
    amrex::DG::ImplicitGeometry<N_PHI, N_DOM> iGeom(indices_box, real_box, ba, dm, geom, inputs.dG.phi_space_p, inputs.dG.space_q);
    amrex::DG::MatrixFactory<N_PHI, N_DOM> MatFactory(indices_box, real_box, ba, dm, geom, inputs.dG.space_p, inputs.dG.time_p, inputs.dG.space_q, 0);
    amrex::DG::DG<N_PHI, N_DOM, N_U> dG("Hyperbolic", "Runge-Kutta");
    dG.InitData(iGeom, MatFactory);
    // ================================================================

    // DESTINATION FOLDER =============================================
    const std::string dG_order = "p"+std::to_string(inputs.dG.space_p);
    std::string ics_type;
    std::string bcs_type;
    std::string dst_folder;

    ics_type = "ICS_Gaussian";
    bcs_type = "BCS_periodic";
    dst_folder = "./IBVP_"+std::to_string(AMREX_SPACEDIM)+"d_Linear_Advection/"+ics_type+"_"+bcs_type+"_"+dG_order+"/";
    // ================================================================

    // INIT IBVP DATA STRUCTURE =======================================
    LINADV LinAdv;
    // ================================================================

    // INIT FIELDS' DATA WITH INITIAL CONDITIONS ======================
    iGeom.ProjectDistanceFunctions(LinAdv);
    iGeom.EvalImplicitMesh(LinAdv);
    MatFactory.Eval(iGeom);
    dG.SetICs(iGeom, MatFactory, LinAdv);

    
    // WRITE TO OUTPUT
    
    if (inputs.plot_int > 0)
    {
        int n = 0;
        amrex::Real time = 0.0;
        std::vector<int> field_domains = {0, 1};
        std::vector<std::string> field_names = {"U0", "U1"};
        amrex::Print()<<"test"<<std::endl;

        iGeom.Export_VTK_Mesh(dst_folder, "Mesh", n, inputs.mesh.n_time_steps);
        dG.Export_VTK(dst_folder, "Solution", n, inputs.mesh.n_time_steps, field_domains, field_names, time, iGeom, MatFactory, LinAdv);
    }
    
    // ================================================================

    // BOUNDARY CONDITIONS ============================================
    // ================================================================

    // START THE ANALYSIS (ADVANCE IN TIME) ===========================
amrex::Print() << "# START OF THE ANALYSIS                                                " << std::endl;
    
    // VARIABLES --------
    int n;
    amrex::Real time, dt;
    // ------------------

    // ADVANCE IN TIME ------------------------------------------------
    n = 0;
    time = 0.0;
    while ((time < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.mesh.n_time_steps))
    {
        // COMPUTE NEXT TIME STEP
        dt = dG.Compute_dt(time+0.5*dt, iGeom, MatFactory, LinAdv);
        dt = std::min(time+dt, inputs.time.T) - time; 

        // REPORT TO SCREEN
amrex::Print() << "| COMPUTING TIME STEP: n = " << n+1 << " time step: " << dt << ", time = " << time+dt << std::endl;//std::min(time+dt, inputs.time.T)

        // TIME STEP
        
        dG.TakeTimeStep_Hyperbolic(dt, time, iGeom, MatFactory, LinAdv);

        // UPDATE TIME AND STEP
        n += 1;
        time = std::min(time+dt, inputs.time.T);

        // COMPUTE ERROR
        if (std::abs(time/inputs.time.T-1.0) < 1.0e-12)
        {
            amrex::Real err;
            err = dG.EvalErrorNorm(time, iGeom, MatFactory, LinAdv);
amrex::Print() << "| Error: " << std::scientific << std::setprecision(5) << std::setw(12) << err << std::endl;
        }

        // WRITE TO OUTPUT
        
        if (inputs.plot_int > 0 && n%inputs.plot_int == 0)
        {
            std::vector<int> field_domains = {0, 1};
            std::vector<std::string> field_names = {"U0", "U1"};
            dG.Export_VTK(dst_folder, "Solution", n, inputs.mesh.n_time_steps, field_domains, field_names, time, iGeom, MatFactory, LinAdv);
        }
        

    }
    // ----------------------------------------------------------------

    //SX ========
    // Post-prrocessing ==============================================
    

    if (inputs.dG.post_processing_by_convolution_flag)
    {
        amrex::Print()<<"Start postprocessing~~~~~"<<std::endl;
        // post-process the dG solution by convolution filtering
          //dG.Convolution_Postprocessing(iGeom,MatFactory);
        //dG.Convolution_Postprocessing_OneSideKernel(iGeom,MatFactory);
        dG.Convolution_Postprocessing_ImplicitMesh_OneSideKernel(inputs.time.T, iGeom, MatFactory, LinAdv);

        //amrex::Real err2;
        //amrex::Print()<<"TEST~~~~~"<<std::endl;

        //err2 = dG.PostProcessedEvalErrorNorm(time,iGeom, MatFactory, LinAdv);
        //amrex::Print() << "| Postprocessed Error 2: " << std::scientific << std::setprecision(5) << std::setw(12) << err2 << std::endl;

    }


amrex::Print() << "# END OF THE ANALYSIS                                                  " << std::endl;
    // ================================================================

    // TOC ============================================================
    stop_time = amrex::second();
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
    // ================================================================

    // CLOSING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# END OF TUTORIAL                                                      " << std::endl;
amrex::Print() << "# Time = " << stop_time-start_time << " s" << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

}
// ####################################################################


// DUMMY MAIN #########################################################
int main(int argc, char* argv[])
{
    // INIT AMREX ================
    amrex::Initialize(argc, argv);
    // ===========================

    main_main();

    // END AMREX =====
    amrex::Finalize();
    // ===============

    return 0;
}
// ####################################################################
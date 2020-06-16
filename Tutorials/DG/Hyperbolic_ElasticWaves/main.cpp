#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>

#include <AMReX_BC_TYPES.H>

#include <AMReX_Print.H>

#include <AMReX_DG_InputReader.H>
#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we are solving elastic wave propagations
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include <IBVP_ElasticWaves.H>
// ====================================================================
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
amrex::Print() << "# In this tutorial, we are solving elastic wave propagations.          " << std::endl;
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

    // DG MODEL =======================================================
    amrex::DG::InputReader dG_inputs;
    // ================================================================

    // REPORT TO SCREEN ===============================================
    // ================================================================

    // TIC ==========================================
    start_time = amrex::second();
    // ==============================================

    // PHYSICAL BOX ===================================================
    amrex::RealBox real_box(dG_inputs.space[0].prob_lo.data(),
                            dG_inputs.space[0].prob_hi.data());
    // ================================================================

    // BOX ARRAY ======================================================
    amrex::Box indices_box({AMREX_D_DECL(0, 0, 0)}, dG_inputs.mesh[0].n_cells-1);

    amrex::BoxArray ba;
    ba.define(indices_box);
    ba.maxSize(dG_inputs.mesh[0].max_grid_size);
    // ================================================================

    // GEOMETRY =======================================================
    amrex::Geometry geom;
    geom.define(indices_box, &real_box, dG_inputs.space[0].coord_sys, dG_inputs.space[0].is_periodic.data());
    // ================================================================
    
    // BOXES DISTRIBUTION AMONG MPI PROCESSES =========================
    amrex::DistributionMapping dm(ba);
    // ================================================================

    // INIT DG DATA STRUCTURES ========================================
    amrex::DG::ImplicitGeometry<N_PHI, N_DOM> iGeom(indices_box, real_box, ba, dm, geom, dG_inputs.dG[0].phi_space_p, dG_inputs.dG[0].space_q);
    amrex::DG::MatrixFactory<N_PHI, N_DOM> MatFactory(indices_box, real_box, ba, dm, geom, dG_inputs.dG[0].space_p, dG_inputs.dG[0].time_p, dG_inputs.dG[0].space_q, dG_inputs.dG[0].time_q);
    amrex::DG::DG<N_PHI, N_DOM, N_U> dG("Hyperbolic", "Runge-Kutta");
    dG.InitData(iGeom, MatFactory);
    // ================================================================

    // DESTINATION FOLDER =============================================
    const std::string dG_order = "p"+std::to_string(dG_inputs.dG[0].space_p);
    std::string ics_type;
    std::string bcs_type;
    std::string dst_folder;

    ics_type = "ICS_Gaussian";
    bcs_type = "BCS_periodic";
    dst_folder = "./IBVP_"+std::to_string(AMREX_SPACEDIM)+"d/"+dG_order+"/";
    // ================================================================

    // INIT IBVP DATA STRUCTURE =======================================
    ELASTIC_WAVES IsotropicWaves("Isotropic", {1.0, 1.0, 0.33});

    ELASTIC_WAVES const & Waves = IsotropicWaves;
    // ================================================================

    // INIT FIELDS' DATA WITH INITIAL CONDITIONS ======================
    iGeom.ProjectDistanceFunctions(Waves);
    iGeom.EvalImplicitMesh(Waves);
    MatFactory.Eval(iGeom);
    dG.SetICs(iGeom, MatFactory, Waves);
    
    // WRITE TO OUTPUT
    
    if (dG_inputs.plot_int > 0)
    {
        int n = 0;
        amrex::Real time = 0.0;
        std::vector<int> field_domains = {0, 0, 0, 0, 0};
        std::vector<std::string> field_names = {"V0", "V1", "S11", "S22", "S12"};

        iGeom.Export_VTK_Mesh(dst_folder, "Mesh", n, dG_inputs.time.n_steps);
        dG.Export_VTK(dst_folder, "Solution", n, dG_inputs.time.n_steps, field_domains, field_names, time, iGeom, MatFactory, Waves);
    }
    
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
    while ((time < dG_inputs.time.T*(1.0-1.0e-12)) && (n < dG_inputs.time.n_steps))
    {
        // COMPUTE NEXT TIME STEP
        dt = dG.Compute_dt(time+0.5*dt, iGeom, MatFactory, Waves);
        dt = std::min(time+dt, dG_inputs.time.T) - time;

        // REPORT TO SCREEN
amrex::Print() << "| COMPUTING TIME STEP: n = " << n+1 << " time step: " << dt << ", time = " << time+dt << std::endl;

        // TIME STEP
        dG.TakeTimeStep_Hyperbolic(dt, time, iGeom, MatFactory, Waves);

        // UPDATE TIME AND STEP
        n += 1;
        time = std::min(time+dt, dG_inputs.time.T);

         // COMPUTE ERROR
        if (std::abs(time/dG_inputs.time.T-1.0) < 1.0e-12)
        {
            amrex::Real err;
            err = dG.EvalErrorNorm(time, iGeom, MatFactory, Waves);
amrex::Print() << "| Error: " << std::scientific << std::setprecision(5) << std::setw(12) << err << std::endl;
        }


        // WRITE TO OUTPUT
        
        if (dG_inputs.plot_int > 0 && n%dG_inputs.plot_int == 0)
        {
            std::vector<int> field_domains = {0, 0, 0, 0, 0};
            std::vector<std::string> field_names = {"V0", "V1", "S11", "S22", "S12"};
            dG.Export_VTK(dst_folder, "Solution", n, dG_inputs.time.n_steps, field_domains, field_names, time, iGeom, MatFactory, Waves);
        }
        
    }
    // ----------------------------------------------------------------

     //SX ========
    // Post-prrocessing ==============================================
    // Shiqiang Xia 05/15/2020

    if (dG_inputs.post_processing_by_convolution_flag)
    {
        // post-process the dG solution by convolution filtering
        
        dG.Convolution_Postprocessing(iGeom,MatFactory);

        amrex::Real err2;
        amrex::Print()<<"TEST~~~~~"<<std::endl;

        err2 = dG.PostProcessedEvalErrorNorm(time,iGeom, MatFactory, Waves);
        amrex::Print() << "| Postprocessed Error 2: " << std::scientific << std::setprecision(5) << std::setw(12) << err2 << std::endl;


        // WRITE TO OUTPUT


        

    }
     // ================================================================
    //SX ========






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
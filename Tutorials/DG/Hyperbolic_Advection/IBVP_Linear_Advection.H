/* LINEAR ADVECTION EQUATION */

// IBVP PARAMETERS ####################################################
// NUMBER OF DISTANCE FUNCTIONS/DOMAINS/UNKNOWN FIELDS
#define N_PHI 1
#define N_DOM 1
#define N_U 1
// ---------------------------------------------------

// PDES PARAMETERS --------
#define CONSTANT_VELOCITY 1
#define SINGLE_VORTEX 2
#define VELOCITY_FIELD 1
// ------------------------
// ####################################################################

// IBVP CLASS #########################################################
class LINADV
{
private:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES

    // AUXILIARY DATA MEMBERS
    // ================================================================
public:
    // CONSTRUCTOR ====================================================
    // ================================================================

    // DISTRUCTOR =====================================================
    // ================================================================

    // DISTANCE FUNCTION ==============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
        const amrex::Real PHI[N_PHI] =
        {
            -1.0
            //x[0]
        };
        return PHI[ph];
    }
    // ================================================================

    // RELATIONSHIP AMONG DISTANCE FUNCTIONS AND DOMAINS ==============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int & dom, int * phi_info) const
    {
        const int PHI_INFO[2*N_DOM] =
        {
            // dom = 0
            0, -1
        };

        phi_info[0] = PHI_INFO[2*dom];
        phi_info[1] = PHI_INFO[2*dom+1];
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int & u, const amrex::Real * x) const
    {
        const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(0.5, 0.75, 0.5)};
#if (AMREX_SPACEDIM == 1)
        //amrex::Real r2 = 100.0*((x[0]-xc[0])*(x[0]-xc[0]));
        amrex::Real r2 = std::sin(2*3.14159265358979*x[0]);
#endif
#if (AMREX_SPACEDIM == 2)
         //amrex::Real r2 = 100.0*((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1]));
          amrex::Real r2 = x[0];
          //amrex::Real r2 = std::sin(2*3.14159265358979*x[0]) * std::sin(2*3.14159265358979*x[1]) ;
#endif
#if (AMREX_SPACEDIM == 3)
        // amrex::Real r2 = 100.0*((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1])+(x[2]-xc[2])*(x[2]-xc[2]));
        amrex::Real r2 = std::sin(2*3.14159265358979*x[0]) * std::sin(2*3.14159265358979*x[1])*std::sin(2*3.14159265358979*x[2]) ;
#endif
           //amrex::Real phi0 = std::exp(-r2);
           amrex::Real phi0 = r2;

        return phi0;
    }
    // ================================================================
    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
        const int DOM[N_U] =
        {
            0
        };

        return DOM[u];
    }
    // ================================================================

    // EXACT SOLUTION =================================================
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real & t, const amrex::Real * x,
                 amrex::Real * U) const
    {
        U[0] = this->F_U0(0, x); //SX ???? U0(0,x-t)??
    }
    // ================================================================

    // BOUNDARY CONDITIONS ============================================
    AMREX_GPU_HOST_DEVICE
    int F_DOM2NBRDOM(const int& dom) const
    {
        const int NBRDOM[N_DOM] =
        {
            -1
        };
        
        return NBRDOM[dom];
    }
    // ================================================================

    // PDES SYSTEM ====================================================
    AMREX_GPU_HOST_DEVICE
    void F_V(const amrex::Real & t, const amrex::Real * x,
             amrex::Real * V) const
    {
#if (VELOCITY_FIELD == CONSTANT_VELOCITY)
        AMREX_D_TERM
        (
            V[0] = 1;,
            V[1] = 0;,
            V[2] = 0.0;
        )
#endif
#if (VELOCITY_FIELD == SINGLE_VORTEX)
        const amrex::Real ct = -std::cos(0.5*M_PI*t);
        const amrex::Real sx = std::sin(M_PI*x[0]);
        const amrex::Real sy = std::sin(M_PI*x[1]);
        const amrex::Real s2x = std::sin(2.0*M_PI*x[0]);
        const amrex::Real s2y = std::sin(2.0*M_PI*x[1]);
        AMREX_D_TERM
        (
            V[0] = -ct*sx*sx*s2y;,
            V[1] = +ct*s2x*sy*sy;,
            V[2] = 1.0;
        )
#endif
    }

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_aV(const amrex::Real & t, const amrex::Real * x) const
    {
        amrex::Real aV, V[AMREX_SPACEDIM];
        this->F_V(t, x, V);

        aV = std::sqrt(AMREX_D_PICK(V[0]*V[0], V[0]*V[0]+V[1]*V[1], V[0]*V[0]+V[1]*V[1]+V[2]*V[2]));
        return aV;
    }
    
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_DT(const amrex::Real * dx,
                     const amrex::Real & t, const amrex::Real * x,
                     const amrex::Real * U) const
    {
        const amrex::Real aV = this->F_aV(t, x);
#if (AMREX_SPACEDIM == 1)
        const amrex::Real h = dx[0];
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = std::min(dx[0], dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = std::min(dx[0], std::min(dx[1],dx[2]));
#endif
        amrex::Real dt;
        
        dt = 0.5*h/aV;
        
        return dt;
    }

    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real & t, const amrex::Real * x,
             const amrex::Real * U,
             AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        amrex::Real V[AMREX_SPACEDIM];
        this->F_V(t, x, V);

        AMREX_D_TERM
        (
            F1[0] = V[0]*U[0];,
            F2[0] = V[1]*U[0];,
            F3[0] = V[2]*U[0];
        )
    }
    // ================================================================

    // DG-RELATED AUXILIARY FUNCTIONS: NUMERICAL FLUX =================
    // INTRAPHASE
    // Lax–Friedrichs Flux
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS_LF(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);
        amrex::Real aV;

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));
        aV = this->F_aV(t, x);

        // dom = 0
        AMREX_D_TERM
        (
            NF1[0] = 0.5*(F1[0]+nbr_F1[0])+0.5*aV*(U[0]-nbr_U[0])*un[0];,
            NF2[0] = 0.5*(F2[0]+nbr_F2[0])+0.5*aV*(U[0]-nbr_U[0])*un[1];,
            NF3[0] = 0.5*(F3[0]+nbr_F3[0])+0.5*aV*(U[0]-nbr_U[0])*un[2];
        )
    }

    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                         const amrex::Real * U, const amrex::Real * nbr_U,
                         AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);
        amrex::Real V[AMREX_SPACEDIM];
        this->F_V(t, x, V);
        amrex::Real V_dot_un = std::abs(AMREX_D_PICK(V[0]*un[0], V[0]*un[0]+V[1]*un[1], V[0]*un[0]+V[1]*un[1]+V[2]*un[2]));

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        // dom = 0
        AMREX_D_TERM
        (
            NF1[0] = 0.5*(F1[0]+nbr_F1[0])+0.5*V_dot_un*(U[0]-nbr_U[0])*un[0];,
            NF2[0] = 0.5*(F2[0]+nbr_F2[0])+0.5*V_dot_un*(U[0]-nbr_U[0])*un[1];,
            NF3[0] = 0.5*(F3[0]+nbr_F3[0])+0.5*V_dot_un*(U[0]-nbr_U[0])*un[2];
        )
    }

    // GRID BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {

    }

    // INTERNAL BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        
    }

    // INTERNAL INTERFACE
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        
    }
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
        // PARAMETERS

        // FIELD EXPRESSIONS
        F[0] = U[0];
    }
    // ================================================================
};
// ####################################################################
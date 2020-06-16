/* ELASTIC WAVE EQUATION */

// IBVP PARAMETERS ####################################################
// NUMBER OF DISTANCE FUNCTIONS/DOMAINS/UNKNOWN FIELDS ----------------
#define N_PHI 2
//#define N_PHI 1
#define N_DOM 1

// NUMBER OF VELOCITY COMPONENTS AND STRESS COMPONENTS
#define N_V AMREX_SPACEDIM
#if (AMREX_SPACEDIM == 2)
#define N_S 3
#endif
#if (AMREX_SPACEDIM == 3)
#define N_S 6
#endif

#define N_U (N_V+N_S)  // velocity + stress symmetric part : d+ d*(d+1)/2)
// --------------------------------------------------------------------

// PDES PARAMETERS ----------------------------------------------------
// --------------------------------------------------------------------

// DG-METHOD PARAMETERS -----------------------------------------------
// --------------------------------------------------------------------
// ####################################################################


// PATCH INFORMATIONS #################################################
// ####################################################################


// IBVP CLASS #########################################################
class ELASTIC_WAVES
{
private:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES
    amrex::Real C[N_S*N_S];
#if (AMREX_SPACEDIM == 2)
    const amrex::Real I1[N_S*N_V] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const amrex::Real I2[N_S*N_V] = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0};
#endif
#if (AMREX_SPACEDIM == 3)
    const amrex::Real I1[N_S*N_V] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    const amrex::Real I2[N_S*N_V] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    const amrex::Real I3[N_S*N_V] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
#endif
    amrex::Real AMREX_D_DECL(A1[N_U*N_U], A2[N_U*N_U], A3[N_U*N_U]);
    amrex::Real AMREX_D_DECL(A1_w[N_U], A2_w[N_U], A3_w[N_U]);
    amrex::Real AMREX_D_DECL(A1_v[N_U*N_U], A2_v[N_U*N_U], A3_v[N_U*N_U]);
    amrex::Real AMREX_D_DECL(iA1_v[N_U*N_U], iA2_v[N_U*N_U], iA3_v[N_U*N_U]);

    // VARIABLES FOR INITIAL CONDITIONS
    amrex::Real ICs_un[AMREX_SPACEDIM];
    amrex::Real ICs_P[N_U*N_U];
    amrex::Real ICs_P_w[N_U];
    amrex::Real ICs_P_v[N_U*N_U];

    // LARGEST WAVE SPEED
    amrex::Real max_wave_speed;

    // AUXILIARY DATA MEMBERS
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES(const std::string & material_type,
                  const amrex::Real * material_properties,
                  const bool plane_stress_flag = false,
                  const amrex::Real shear_factor = 1.0)
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real rho = material_properties[0];
        const amrex::Real inv_rho = 1.0/rho;
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        // MATERIAL PROPERTIES
        amrex::Real s3D[36], c3D[36];
#if (AMREX_SPACEDIM == 2)
        amrex::Real c2D[9];
#endif

        // AUXILIARY MATRICES
        AMREX_D_TERM
        (
            amrex::Real I1T[N_V*N_S];
            amrex::Real cI1[N_S*N_V];,
            amrex::Real I2T[N_V*N_S];
            amrex::Real cI2[N_S*N_V];,
            amrex::Real I3T[N_V*N_S];
            amrex::Real cI3[N_S*N_V];
        )
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        std::fill(s3D, s3D+36, 0.0);
        std::fill(c3D, c3D+36, 0.0);
#if (AMREX_SPACEDIM == 2)
        std::fill(c2D, c2D+9, 0.0);
#endif

        AMREX_D_TERM
        (
            amrex::DG_utils::transpose(N_S, N_V, this->I1, I1T);,
            amrex::DG_utils::transpose(N_S, N_V, this->I2, I2T);,
            amrex::DG_utils::transpose(N_S, N_V, this->I3, I3T);
        )

        AMREX_D_TERM
        (
            std::fill(A1, A1+N_U*N_U, 0.0);,
            std::fill(A2, A2+N_U*N_U, 0.0);,
            std::fill(A3, A3+N_U*N_U, 0.0);
        )
        // ------------------------------------------------------------

        // COMPUTE THE COMPLIANCE/STIFFNESS MATRIX ON THE BASIS OF THE
        // INPUT MATERIAL TYPE ----------------------------------------
        if (material_type.compare("Isotropic") == 0)
        {
            // Input properties
            const amrex::Real E = material_properties[1];
            const amrex::Real nu = material_properties[2];
            const amrex::Real G = 0.5*E/(1.0+nu);

            s3D[0+0*6] = 1.0/E; s3D[0+1*6] = -nu/E; s3D[0+2*6] = -nu/E;
            s3D[1+0*6] = -nu/E; s3D[1+1*6] = 1.0/E; s3D[1+2*6] = -nu/E;
            s3D[2+0*6] = -nu/E; s3D[2+1*6] = -nu/E; s3D[2+2*6] = 1.0/E;
            s3D[3+3*6] = 1.0/G;
            s3D[4+4*6] = 1.0/G;
            s3D[5+5*6] = 1.0/G;
        }
        else
        {
            amrex::Print() << std::endl;
            amrex::Print() << "ERROR: IBVP_ElasticWaves.H - ELASTIC_WAVES::ELASTIC_WAVES" << std::endl;
            amrex::Print() << "| Unexpected material type: " << material_type << std::endl;
            exit(-1);
        }

        if ((material_type.compare("Isotropic") == 0) || (material_type.compare("Orthotropic") == 0))
        {
            if (plane_stress_flag)
            {
                s3D[0+2*6] = 0.0;
                s3D[1+2*6] = 0.0;
                s3D[2+0*6] = 0.0;
                s3D[2+1*6] = 0.0;

                s3D[3+3*6] = s3D[3+3*6]/shear_factor;
                s3D[4+4*6] = s3D[4+4*6]/shear_factor;
            }

            amrex::DG_utils::matinv(6, s3D, c3D);
        }

#if (AMREX_SPACEDIM == 2)
        c2D[0+0*3] = c3D[0+0*6]; c2D[0+1*3] = c3D[0+1*6]; c2D[0+2*3] = c3D[0+5*6];
        c2D[1+0*3] = c3D[1+0*6]; c2D[1+1*3] = c3D[1+1*6]; c2D[1+2*3] = c3D[1+5*6];
        c2D[2+0*3] = c3D[5+0*6]; c2D[2+1*3] = c3D[5+1*6]; c2D[2+2*3] = c3D[5+5*6];
#endif
        // ------------------------------------------------------------

        // AUXILIARY MATRICES -----------------------------------------
#if (AMREX_SPACEDIM == 2)
        amrex::DG_utils::matmul(N_S, N_S, N_V, c2D, this->I1, cI1);
        amrex::DG_utils::matmul(N_S, N_S, N_V, c2D, this->I2, cI2);
#endif
#if (AMREX_SPACEDIM == 3)
        amrex::DG_utils::matmul(N_S, N_S, N_V, c3D, this->I1, cI1);
        amrex::DG_utils::matmul(N_S, N_S, N_V, c3D, this->I2, cI2);
        amrex::DG_utils::matmul(N_S, N_S, N_V, c3D, this->I3, cI3);
#endif
        // ------------------------------------------------------------

        // DIFFERENTIAL OPERATORS -------------------------------------

        
        for (int c = N_V; c < (N_V+N_S); ++c)
        for (int r = 0; r < N_V; ++r)
        {
            AMREX_D_TERM
            (
                this->A1[r+c*N_U] = -inv_rho*I1T[r+(c-N_V)*N_V];,
                this->A2[r+c*N_U] = -inv_rho*I2T[r+(c-N_V)*N_V];,
                this->A3[r+c*N_U] = -inv_rho*I3T[r+(c-N_V)*N_V];
            )
        }

        for (int c = 0; c < N_V; ++c)
        for (int r = N_V; r < (N_V+N_S); ++r)
        {
            AMREX_D_TERM
            (
                this->A1[r+c*N_U] = -cI1[r-N_V+c*N_S];,
                this->A2[r+c*N_U] = -cI2[r-N_V+c*N_S];,
                this->A3[r+c*N_U] = -cI3[r-N_V+c*N_S];
            )
        }

        

       /*

       for (int c = 0; c < (N_U); ++c)
        {
            this->A1[c+c*N_U] = 1.0 ;
            this->A2[c+c*N_U] = 1.0 ;
        }

        */

       

        // AND THEIR EIGENVALUES
        {
            amrex::Real wIm[N_U];

            AMREX_D_TERM
            (
                amrex::DG_utils::eig(false, true, N_U, this->A1, this->A1_w, wIm, nullptr, this->A1_v);,
                amrex::DG_utils::eig(false, true, N_U, this->A2, this->A2_w, wIm, nullptr, this->A2_v);,
                amrex::DG_utils::eig(false, true, N_U, this->A3, this->A3_w, wIm, nullptr, this->A3_v);
            )
            AMREX_D_TERM
            (
                amrex::DG_utils::matinv(N_U, this->A1_v, this->iA1_v);,
                amrex::DG_utils::matinv(N_U, this->A2_v, this->iA2_v);,
                amrex::DG_utils::matinv(N_U, this->A3_v, this->iA3_v);
            )
        }
        // ------------------------------------------------------------

        // INITIAL CONDITIONS -----------------------------------------
        AMREX_D_TERM
        (
            this->ICs_un[0] = 1.0;,
            this->ICs_un[1] = 0.0;,
            this->ICs_un[2] = 0.0;
        )
        {
            amrex::Real tmp = this->ICs_un[0]*this->ICs_un[0];
            for (int dim = 1; dim < AMREX_SPACEDIM; ++dim)
                tmp += this->ICs_un[dim]*this->ICs_un[dim];
            tmp = 1.0/std::sqrt(tmp);

            AMREX_D_TERM
            (
                this->ICs_un[0] *= tmp;,
                this->ICs_un[1] *= tmp;,
                this->ICs_un[2] *= tmp;
            )
        }

        for (int c = 0; c < N_U; ++c)
        for (int r = 0; r < N_U; ++r)
        {
            AMREX_D_TERM
            (
                this->ICs_P[r+c*N_U]  = this->A1[r+c*N_U]*this->ICs_un[0];,
                this->ICs_P[r+c*N_U] += this->A2[r+c*N_U]*this->ICs_un[1];,
                this->ICs_P[r+c*N_U] += this->A3[r+c*N_U]*this->ICs_un[2];
            )
        }

        // COMPUTE THE EIGENVALUES AND EIGENVECTOR
        // The i-th eigenvector will be stored in the i-th column of
        // the array ICs_P_v. Remember that we use row-major format.
        {
            amrex::Real wIm[N_U];

            amrex::DG_utils::eig(false, true, N_U, this->ICs_P, this->ICs_P_w, wIm, nullptr, this->ICs_P_v);
        }
        // ------------------------------------------------------------

        // FIND THE LARGEST WAVE SPEED --------------------------------
        // We span the unit 2d or 3d sphere
        // ------------------------------------------------------------
        {
            this->max_wave_speed = 0.0;

            amrex::Real un[AMREX_SPACEDIM], P[N_U*N_U], wRe[N_U], wIm[N_U];
            const int n_theta = 180;
            const amrex::Real dtheta = 2.0*M_PI/n_theta;
            amrex::Real theta;
#if (AMREX_SPACEDIM == 3)
            const int n_phi = n_theta/2;
            const amrex::Real dphi = M_PI/n_phi;
            amrex::Real phi;
#endif
#if (AMREX_SPACEDIM == 2)
            for (int i = 0; i < n_theta; ++i)
            {
                theta = i*dtheta;
                un[0] = std::cos(theta);
                un[1] = std::sin(theta);
                this->_eval_P_(un, P);
                amrex::DG_utils::eig(N_U, P, wRe, wIm);
                for (int u = 0; u < N_U; ++u) wRe[u] = std::abs(wRe[u]);

                this->max_wave_speed = std::max(*std::max_element(wRe, wRe+N_U), this->max_wave_speed);

            }
#endif
#if (AMREX_SPACEDIM == 3)
            for (int j = 0; j < n_phi; ++j)
            for (int i = 0; i < n_theta; ++i)
            {
                theta = i*dtheta;
                phi = j*dphi;
                un[0] = std::cos(theta)*std::sin(phi);
                un[1] = std::sin(theta)*std::sin(phi);
                un[2] = std::cos(phi);

                this->_eval_P_(un, P);
                amrex::DG_utils::eig(N_U, P, wRe, wIm);
                for (int u = 0; u < N_U; ++u) wRe[u] = std::abs(wRe[u]);

                this->max_wave_speed = std::max(*std::max_element(wRe, wRe+N_U), this->max_wave_speed);
            }
#endif
        }
        // ------------------------------------------------------------
    }

    ELASTIC_WAVES(const std::string & material_type,
                  const std::vector<amrex::Real> & material_properties,
                  const bool plane_stress_flag = false,
                  const amrex::Real shear_factor = 1.0)
    :
    ELASTIC_WAVES(material_type, material_properties.data(), plane_stress_flag, shear_factor)
    {}
    // ================================================================

    // DISTRUCTOR =====================================================
    // ================================================================

    // AUXILIARY FUNCTIONS ============================================
    AMREX_GPU_HOST_DEVICE
    void _eval_T_(const amrex::Real * un, amrex::Real * T) const
    {
        for (int k = 0; k < N_U*N_U; ++k)
        {
            T[k] = 0.0;
        }
#if (AMREX_SPACEDIM == 2)
        T[0+0*N_U] =  un[0]; T[0+1*N_U] = un[1];
        T[1+0*N_U] = -un[1]; T[1+1*N_U] = un[0];
        
        T[2+2*N_U] = un[0]*un[0]; T[2+3*N_U] = un[1]*un[1]; T[2+4*N_U] = 2.0*un[0]*un[1];
        T[3+2*N_U] = un[1]*un[1]; T[3+3*N_U] = un[0]*un[0]; T[3+4*N_U] = -2.0*un[0]*un[1];
        T[4+2*N_U] = -un[0]*un[1]; T[4+3*N_U] = un[0]*un[1]; T[4+4*N_U] = un[0]*un[0]-un[1]*un[1];
#endif
    }
    
    AMREX_GPU_HOST_DEVICE
    void _eval_T_(const amrex::Real * un, amrex::Real * T, amrex::Real * iT) const
    {
        for (int k = 0; k < N_U*N_U; ++k)
        {
            T[k] = 0.0;
            iT[k] = 0.0;
        }
#if (AMREX_SPACEDIM == 2)
        T[0+0*N_U] =  un[0]; T[0+1*N_U] = un[1];
        T[1+0*N_U] = -un[1]; T[1+1*N_U] = un[0];
        
        T[2+2*N_U] = un[0]*un[0]; T[2+3*N_U] = un[1]*un[1]; T[2+4*N_U] = 2.0*un[0]*un[1];
        T[3+2*N_U] = un[1]*un[1]; T[3+3*N_U] = un[0]*un[0]; T[3+4*N_U] = -2.0*un[0]*un[1];
        T[4+2*N_U] = -un[0]*un[1]; T[4+3*N_U] = un[0]*un[1]; T[4+4*N_U] = un[0]*un[0]-un[1]*un[1];

        iT[0+0*N_U] = un[0]; iT[0+1*N_U] = -un[1];
        iT[1+0*N_U] = un[1]; iT[1+1*N_U] = un[0];
        
        iT[2+2*N_U] = un[0]*un[0]; iT[2+3*N_U] = un[1]*un[1]; iT[2+4*N_U] = -2.0*un[0]*un[1];
        iT[3+2*N_U] = un[1]*un[1]; iT[3+3*N_U] = un[0]*un[0]; iT[3+4*N_U] = 2.0*un[0]*un[1];
        iT[4+2*N_U] = un[0]*un[1]; iT[4+3*N_U] = -un[0]*un[1]; iT[4+4*N_U] = un[0]*un[0]-un[1]*un[1];
#endif
    }
    
    AMREX_GPU_HOST_DEVICE
    void _eval_P_(const amrex::Real * un, amrex::Real * P) const
    {
        for (int c = 0; c < N_U; ++c)
        for (int r = 0; r < N_U; ++r)
        {
            AMREX_D_TERM
            (
                P[r+c*N_U]  = this->A1[r+c*N_U]*un[0];,
                P[r+c*N_U] += this->A2[r+c*N_U]*un[1];,
                P[r+c*N_U] += this->A3[r+c*N_U]*un[2];
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_Pn_(const amrex::Real * un, amrex::Real * Pn) const
    {
        amrex::Real P[N_U*N_U], T[N_U*N_U], iT[N_U*N_U];
        this->_eval_P_(un, Pn);
        this->_eval_T_(un, T, iT);

        amrex::DG_utils::matmul(N_U, N_U, N_U, Pn, iT, P);
        amrex::DG_utils::matmul(N_U, N_U, N_U, T, P, Pn);
    }
    // ================================================================

    // GRID MAPPING ===================================================
    AMREX_GPU_HOST_DEVICE
    void F_MAP(const amrex::Real & t, const amrex::Real * xi,  amrex::Real * x) const
    {

    }
    // ================================================================

    // DISTANCE FUNCTION ==============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
#if (AMREX_SPACEDIM == 2)
        const int n_holes = 1;
        const amrex::Real r[n_holes] = {0.15};
        const amrex::Real c0[n_holes] = {2.0};
        const amrex::Real c1[n_holes] = {0.5};
        amrex::Real phi;
        
        phi = r[0]*r[0]-(x[0]-c0[0])*(x[0]-c0[0])-(x[1]-c1[0])*(x[1]-c1[0]);
#endif
#if (AMREX_SPACEDIM == 3)
#endif
        const amrex::Real PHI[N_PHI] =
        {
            -1.0,
            -1.0
            //phi
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
#define ICS_NONE -1
#define ICS_EIGEN_STATE 0
#define ICS_INITIAL_STRAIN 1
#define ICS_INITIAL_STRESS 2
#define ICS 0

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int & u, const amrex::Real * x) const
    {
        amrex::Real res;
#if (ICS == ICS_NONE)
        res = 0.0;
#endif
#if (ICS == ICS_EIGEN_STATE)
        const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(this->ICs_un[0], this->ICs_un[1], this->ICs_un[2])};
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
        //const amrex::Real d = 0.2;
        //const amrex::Real f0 = std::exp(-100.0*(x_dot_un-d)*(x_dot_un-d));
        const amrex::Real f0 = std::sin(2*M_PI*x_dot_un);
        

        const int iv = 4; 
        
        res = this->ICs_P_v[u+iv*N_U]*f0;
#endif
#if (ICS == ICS_INITIAL_STRAIN)
        const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(0.5, 0.5, 0.5)};
#if (AMREX_SPACEDIM == 2)
        const amrex::Real r2 = 500.0*((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1]));
#endif
        //const amrex::Real f0 = std::exp(-r2);
        AMREX_D_TERM
        (
            const amrex::Real df0dx1 = -1000.0*(x[0]-xc[0])*std::exp(-r2);,
            const amrex::Real df0dx2 = -1000.0*(x[1]-xc[1])*std::exp(-r2);,
            const amrex::Real df0dx3 = -1000.0*(x[2]-xc[2])*std::exp(-r2);
        )
        const amrex::Real a[N_V] = {AMREX_D_DECL(1.0, 0.0, 0.0)};
        amrex::Real AMREX_D_DECL(dudx1[N_V], dudx2[N_V], dudx3[N_V]);
        amrex::Real AMREX_D_DECL(s1[N_S], s2[N_S], s3[N_S]);
        for (int r = 0; r < N_V; ++r)
        {
            AMREX_D_TERM
            (
                dudx1[r] = a[r]*df0dx1;,
                dudx2[r] = a[r]*df0dx2;,
                dudx3[r] = a[r]*df0dx3;
            )
        }

        for (int r = 0; r < N_S; ++r)
        {
            AMREX_D_TERM
            (
                s1[r] = 0.0;,
                s2[r] = 0.0;,
                s3[r] = 0.0;
            )
        }

        for (int c = 0; c < N_V; ++c)
        for (int r = 0; r < N_S; ++r)
        {
            AMREX_D_TERM
            (
                s1[r] -= this->A1[r+N_V+c*N_U]*dudx1[c];,
                s2[r] -= this->A2[r+N_V+c*N_U]*dudx2[c];,
                s3[r] -= this->A3[r+N_V+c*N_U]*dudx3[c];
            )
        }

        amrex::Real U0[N_U];

        for (int r = 0; r < N_V; ++r) U0[r] = 0.0;

        for (int r = 0; r < N_S; ++r)
        {
            AMREX_D_TERM
            (
                U0[r+N_V]  = s1[r];,
                U0[r+N_V] += s2[r];,
                U0[r+N_V] += s3[r];
            )
        }

        res = U0[u];
#endif

        return res;
    }
    // ================================================================
    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
        const int DOM[N_U] =
        {
            0, 0, 0, 0, 0
        };

        return DOM[u];
    }
    // ================================================================

    // EXACT SOLUTION =================================================
    void F_EXACT(const amrex::Real & t, const amrex::Real * x,
                 amrex::Real * U) const
    {
        const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(this->ICs_un[0], this->ICs_un[1], this->ICs_un[2])};
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);

        const int iv = 4;
        const amrex::Real speed = this->ICs_P_w[iv]; // eigenvaule

        
        const amrex::Real y = x_dot_un - speed*t;

        amrex::Real temp;
        if (y>=0)
        {
            const amrex::Real z= std::floor(y);
            temp = y - z ;
        }
        else
        {
            const amrex::Real z = std::floor( std::abs(y) );
            temp = y + z +1.0;

        }
        

        //const amrex::Real d = 0.2;
        //const amrex::Real f0 = std::exp(-100.0*(temp-d)*(temp-d));
        const amrex::Real f0 = std::sin(2*M_PI*y);
        

        for(int ru = 0; ru < N_U; ++ru)
        {  
            U[ru] = this->ICs_P_v[ru+iv*N_U] * f0; 
        }


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
    amrex::Real F_DT(const amrex::Real * dx,
                     const amrex::Real & t, const amrex::Real * x,
                     const amrex::Real * U) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = std::min(dx[0], dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = std::min(dx[0], std::min(dx[1], dx[2]));
#endif
        amrex::Real dt;
        
        dt = 0.5*h/(this->max_wave_speed);
        
        return dt;
    }

    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real & t, const amrex::Real * x,
             const amrex::Real * U,
             AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        AMREX_D_TERM
        (
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A1, U, F1);,
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A2, U, F2);,
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A3, U, F3);
        )
    }
    // ================================================================

    // DG-RELATED AUXILIARY FUNCTIONS: NUMERICAL FLUX =================
#define GRID_BCS_ABSORBING 0
#define GRID_BCS_ENCASTRE 1
#define GRID_BCS_TRACTION_FREE 2
#define GRID_BCS 2

    AMREX_GPU_HOST_DEVICE
    void _NF_base_(const amrex::Real * un,
                   const amrex::Real * U, const amrex::Real * nbr_U,
                   AMREX_D_DECL(const amrex::Real * F1, const amrex::Real * F2, const amrex::Real * F3),
                   AMREX_D_DECL(const amrex::Real * nbr_F1, const amrex::Real * nbr_F2, const amrex::Real * nbr_F3),
                   AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        const amrex::Real mu = this->max_wave_speed;

        // dom = 0
        for (int u = 0; u < N_U; ++u)
        {
            AMREX_D_TERM
            (
                NF1[u] = 0.5*(F1[u]+nbr_F1[u])+0.5*mu*(U[u]-nbr_U[u])*un[0];,
                NF2[u] = 0.5*(F2[u]+nbr_F2[u])+0.5*mu*(U[u]-nbr_U[u])*un[1];,
                NF3[u] = 0.5*(F3[u]+nbr_F3[u])+0.5*mu*(U[u]-nbr_U[u])*un[2];
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void _NF_PHI_base_(const amrex::Real * un,
                       const amrex::Real * U, const amrex::Real * nbr_U,
                       AMREX_D_DECL(const amrex::Real * F1, const amrex::Real * F2, const amrex::Real * F3),
                       AMREX_D_DECL(const amrex::Real * nbr_F1, const amrex::Real * nbr_F2, const amrex::Real * nbr_F3),
                       AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
    }
    
    // INTRAPHASE
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        this->_NF_base_(un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
    }

    // GRID BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);

#if (GRID_BCS == GRID_BCS_ABSORBING)
        amrex::Real cv[N_U], nbr_U[N_U];
        
        if (std::abs(un[0]) > 0.5)
        {
            amrex::DG_utils::matmul(N_U, N_U, 1, this->iA1_v, U, cv);
            for (int u = 0; u < N_U; ++u)
                if (this->A1_w[u]*un[0] < 0.0) cv[u] = 0.0;
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A1_v, cv, nbr_U);
            
        }
        else if (std::abs(un[1]) > 0.5)
        {
            amrex::DG_utils::matmul(N_U, N_U, 1, this->iA2_v, U, cv);
            for (int u = 0; u < N_U; ++u)
                if (this->A2_w[u]*un[1] < 0.0) cv[u] = 0.0;
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A2_v, cv, nbr_U);
        }
#if (AMREX_SPACEDIM > 2)
        else if (std::abs(un[2]) > 0.5)
        {

        }
#endif
#endif

#if (GRID_BCS == GRID_BCS_ENCASTRE)
        amrex::Real nbr_U[N_U];
        for (int u = 0; u < N_V; ++u) nbr_U[u] = -U[u];
        for (int u = N_V; u < (N_V+N_S); ++u) nbr_U[u] = +U[u];
#endif

#if (GRID_BCS == GRID_BCS_TRACTION_FREE)
        amrex::Real T[N_U*N_U], iT[N_U*N_U];
        amrex::Real nbr_U[N_U], Un[N_U];

        this->_eval_T_(un, T, iT);
        amrex::DG_utils::matmul(N_U, N_U, 1, T, U, Un);
#if (AMREX_SPACEDIM == 2)
        Un[2] = -Un[2]; // Snn
        Un[4] = -Un[4]; // Sns
#endif
#if (AMREX_SPACEDIM == 3)
        Un[3] = -Un[3]; // Snn
        Un[7] = -Un[7]; // Sns
        Un[8] = -Un[8]; // Snt
#endif
        amrex::DG_utils::matmul(N_U, N_U, 1, iT, Un, nbr_U);
#endif
    
        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        this->_NF_base_(un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
    }

#define PHI_BCS_TRACTION_FREE 2
#define PHI_BCS 2

    // INTERNAL BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);
        
#if (PHI_BCS == PHI_BCS_TRACTION_FREE)
        amrex::Real T[N_U*N_U], iT[N_U*N_U];
        amrex::Real nbr_U[N_U], Un[N_U];

        this->_eval_T_(un, T, iT);
        amrex::DG_utils::matmul(N_U, N_U, 1, T, U, Un);
#if (AMREX_SPACEDIM == 2)
        Un[2] = -Un[2]; // Snn
        Un[4] = -Un[4]; // Sns
#endif
#if (AMREX_SPACEDIM == 3)
        Un[3] = -Un[3]; // Snn
        Un[7] = -Un[7]; // Sns
        Un[8] = -Un[8]; // Snt
#endif
        amrex::DG_utils::matmul(N_U, N_U, 1, iT, Un, nbr_U);
#endif

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        this->_NF_base_(un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
    }

    // INTERNAL INTERFACE
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
#if (!AMREX_USE_CUDA)
amrex::Print() << "Hello! F_NF_PHI_ICS" << std::endl;
exit(-1);
#endif
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
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];
    }
    // ================================================================
};
// ####################################################################

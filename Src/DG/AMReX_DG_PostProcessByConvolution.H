
#include <AMReX_REAL.H>
#include <AMReX_DG_BasisFunctions.H>

namespace amrex
{
namespace DG
{
    const Real cr_1[3] = {-1.0/12,7.0/6,-1.0/12};
    const Real cr_2[5] = {37.0/1920, -(97.0/480), 437.0/320, -(97.0/480), 37.0/1920};
    const Real cr_3[7] = {-(41.0/7560), 311.0/5040, -(919.0/2520), 12223.0/7560, -(919.0/2520), 311.0/5040, -(41.0/7560)};
    const Real cr_4[9] = {153617.0/92897280, -(35411.0/1658880), 3153959.0/23224320,-(6803459.0/11612160), 18017975.0/9289728, -(6803459.0/11612160), 3153959.0/23224320, -(35411.0/1658880), 153617.0/92897280};
    const Real cr_5[11] = {-(4201.0/7983360), 30773.0/3991680, -(20813.0/380160), 2825.0/11088, -(1179649.0/1330560), 1569217.0/665280, -(1179649.0/1330560), 2825.0/11088, -(20813.0/380160), 30773.0/3991680, -(4201.0/7983360)};

    const Real cr_oneside_1[5] = {1.0/9.0, -19.0/36.0, 11.0/6.0, -19.0/36.0, 1.0/9.0};
    const Real cr_oneside_2[9] = {651657.0/11454080.0, -(26263297.0/77315040.0), 94794683.0/103086720.0, -(2667869.0/1610730.0), 14685479.0/4832190.0, -(2667869.0/1610730.0), 94794683.0/103086720.0, -(26263297.0/77315040.0), 651657.0/11454080.0};
    const Real cr_oneside_3[13]={726301.0/20455680.0, -(3305063.0/11932480.0),1876575341.0/1933061760.0,-(1985274419.0/966530880.0),11824911607.0/3866123520.0, -(52483121.0/13807584.0),141784865.0/27615168.0, -(52483121.0/13807584.0), 11824911607.0/3866123520.0, -(1985274419.0/966530880.0), 1876575341.0/1933061760.0, -(3305063.0/11932480.0), 726301.0/20455680.0};

    void Bspline(Real x, int k, Real & out)
    {
        // Recusive define Bspline
        // Bspline_{k+1}(x) = 1/k[ ((k+1)/2+x) * Bspline_{k}(x+1/2)+ ((k+1)/2-x) * Bspline_{k}(x-1/2) ]

        if (k==1)
        {
            if (x<=0.5 && x>= -0.5) out = 1.0;
            else out = 0.0;
        }
        else
        {
            Real temp1, temp2, s1, s2;
            Bspline(x+0.5, k-1, temp1);
            Bspline(x-0.5, k-1, temp2);
            s1 = k/2.0 + x;
            s2 = k/2.0 - x;
            out = 1.0/(k-1) * ( s1 * temp1 + s2 * temp2 );
        }
        
    }
    void Bspline_breaks(Real * bp_breaks, int & break_flag, int deg_Bspine, Real center_point,Real iupper, Real ilower)
    {
        // figure out the actual integral domain of
        // Int_{ilower, iupper} phi(center_point - z/2)*legendre(z)dz
        //-1 <= ilower <= iupper <= 1
        // based on the support of the Bspline and the center_point
        // Output:
        // break_flag: type of integral we will deal with
        // bp_breaks: store the actual x_low and x_high of the integral

        // lower and upper bound of the integral domain
        const Real sl = center_point - 0.5*iupper;
        const Real sr = center_point - 0.5*ilower;

        Real knots[deg_Bspine+1];
        // knots of the Bspline
        for(int ii = 0; ii<deg_Bspine+1; ++ii)
        {
            knots[ii] = (-deg_Bspine + 2.0 * ii) * 0.5;

        }
        //##########################

        // Find the acutal integral domain: 5 cases
        if (sl>=knots[deg_Bspine] || sr<= knots[0])
        {
            break_flag = 1;
            bp_breaks[0] = 0;
            bp_breaks[1] = 0;
            bp_breaks[2] = 0;

        }
        else if (sl< knots[0] && sr>knots[0] )
        {
            break_flag = 2;
            bp_breaks[0] = knots[0];
            bp_breaks[1] = sr;
            bp_breaks[2] = 0;

        }
        else if (sl<knots[deg_Bspine] && sr>knots[deg_Bspine])
        {
            break_flag = 2;
            bp_breaks[0] = sl;
            bp_breaks[1] = knots[deg_Bspine];
            bp_breaks[2] = 0;

        }
        else
        {
            int position = 0;
            for (int ii =0; ii < (deg_Bspine + 1); ++ii)
            {
                if (knots[ii] >= sl)
                {
                    position = ii;
                    break;

                }

            }
            if (knots[position] >= sr)
            {
                break_flag = 4;
                bp_breaks[0] = sl;
                bp_breaks[1] = sr;
                bp_breaks[2] = 0;
            }
            else
            {
                break_flag = 3;
                bp_breaks[0] = sl;
                bp_breaks[1] = knots[position];
                bp_breaks[2] = sr;
            }
            
        }
        
    }
    void Bspine_break_quadrature(Real * rlt, int deg_Bspine, Real center_point,Real xleft,Real xright, Gpu::ManagedVector<Real>  quad_points, Gpu::ManagedVector<Real>  quad_weights, int Np)
    {
        // Actual integral of Bspline and DG basis function on [xleft, xright]
        //Output:
        // rlt: array of Np x 1, contains integral (Bspline, basis_i) for 1<=i<=Np

        Real half_h =  ( xright-xleft ) *0.5;
        Real mid = (xright + xleft ) *0.5;
        int Nq = quad_points.size();
        Real spline_vec[Nq] ;
        Real spline_pts[Nq];
        Real basis_pts[Nq];



        for (int ii = 0; ii < Nq; ++ii)
        {
            Real out;
            spline_pts[ii] = half_h * quad_points[ii] + mid;
            basis_pts[ii] = ( center_point - spline_pts[ii] ) * 2.0 ;
            Bspline(spline_pts[ii], deg_Bspine, out);
            spline_vec[ii] = out ;
        }

        // evalue basis (up to Np) at basis_pts
        Real  BF[Nq*Np] ;
        DG_BF_1d_Table_NxL(Nq, Np-1, basis_pts, -1.0, 1.0, BF);

        // perform numerical quadratue int_{xleft, xright}(Bspline, basis_function)

        for (int nn = 0; nn < Np; ++nn)
        {
            rlt[nn] = 0;
            for (int jj = 0; jj < Nq; ++jj)
            {
                rlt[nn] += quad_weights[jj] * spline_vec[jj] * BF[jj+nn*Nq];
            }
            rlt[nn] *=2.0*half_h;

        }



    }
    void Bspline_integral(Real * rlt, int  Np, Real center_point, Real int_upper, Real int_lower, Gpu::ManagedVector<Real> quad_points, Gpu::ManagedVector<Real> quad_weights)
    {
        // Compute the integral (Bspline, DG_basis_function) for each basis function
        //
        // Integral_{lower, upper} Bspline( center_point - z/2)*Basis_m(z)dz for m = 0,1,2,..Poly degree
        //
        // normally upper = 1.0, lower = -1.0 (integral over one whole element)
        //
        //Output
        // rlt: array of Np x 1, integral for each basis function

        const int deg_Bspine = Np;
        Real bp_breaks[3];
        int  break_flag = 0;
        
        // Figure out the actual integral domain and type
        
        Bspline_breaks(bp_breaks,break_flag,deg_Bspine,center_point, int_upper, int_lower);

        if (break_flag == 1)
        {
            for(int nn = 0; nn < Np; ++nn)
            {
                rlt[nn] = 0.0;
            }

        }
        else if (break_flag == 2 || break_flag == 4)
        {
            // in this case there is no break
            // just have one integral
            Bspine_break_quadrature(rlt, deg_Bspine,center_point,bp_breaks[0],bp_breaks[1], quad_points,quad_weights, Np);

        }
        else
        {
            // in this case there is a break
            // need to add two integrals together
            Real out1[Np];
            Real out2[Np];
            Bspine_break_quadrature(out1, deg_Bspine,center_point,bp_breaks[0],bp_breaks[1], quad_points,quad_weights,Np);
            Bspine_break_quadrature(out2, deg_Bspine,center_point,bp_breaks[1],bp_breaks[2], quad_points,quad_weights,Np);

            for(int nn = 0; nn < Np; ++nn)
            {
                rlt[nn]= out1[nn] + out2[nn];
            }
        }
        

    }
    void compute_convolution_matrix(Gpu::ManagedVector<Real> & AA, int sp,  Gpu::ManagedVector<Real> points, Gpu::ManagedVector<Real> quad_points, Gpu::ManagedVector<Real> quad_weights )
    {
        // compute the convolution matrix based on the DG method and points of interests
        //Output: AA

        const int Npoints = points.size();
        const int Np = sp+1;
        const int Nr = 2*sp+1;
        const int NJ = 4*sp+1;
        const Real iupper = 1.0;
        const Real ilower = -1.0;

        AA.resize(Npoints*Np*Nr*NJ);
        Real center_point; 

        for (int nn = 0; nn < Npoints; ++nn)
        {
            for (int jj = 0; jj <  NJ; ++jj)
            {
                for (int rr = 0; rr < Nr; ++rr )
                {
                    center_point = points[nn]*0.5 - (rr-sp) - (jj - 2*sp);
                    Real rlt[Np];

                    Bspline_integral(rlt, Np, center_point,iupper, ilower, quad_points, quad_weights);
                    
                    /*
                    if(nn==0 && rr==0 &&jj==5)
                    {
                        Print()<<"point  "<<points[nn]<<std::endl;
                        Print()<<"center_point  "<<center_point<<std::endl;
                        for (int mm = 0; mm < Np; ++mm)
                        {
                            Print()<<"rlt "<<rlt[mm]<<std::endl;
                        }

                    }
                    */

                    for (int mm = 0; mm < Np; ++mm)
                    {
                        AA[ jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ] = rlt[mm];

                    }
                }
            }
        }

     


    }

    void get_convolution_matrix(Gpu::ManagedVector<Real> & matrix, Gpu::ManagedVector<Real> points, int sp, Gpu::ManagedVector<Real> quad_points, Gpu::ManagedVector<Real> quad_weights)
    {   
        // Precompute the convolution matrix needed for post-processing

        // basic info 
        const int Npoints = points.size();
        const int Np = sp+1;
        const int Nr = 2*sp+1;
        const int NJ = 4*sp+1;

        matrix.resize(Npoints*Np*NJ);
        
        //step 1. load cr

        const Real * cr ;

        if (sp == 1) cr = cr_1 ;
        else if (sp == 2) cr = cr_2;
        else if (sp == 3) cr = cr_3;
        else if (sp == 4) cr = cr_4;
        else if (sp == 5) cr =cr_5;
        else
        {
            Print()<<"Order "<<sp<<" is not implemented"<< std::endl;
        }

        // setp 2. compute A_{n,m,r,j}
        

        Gpu::ManagedVector<Real> AA;

        compute_convolution_matrix(AA, sp, points, quad_points, quad_weights);

        // step 3. sum A_{n,m,r,j} * cr/2 over r

        for (int nn = 0; nn < Npoints; ++nn)
        {
            for (int mm = 0; mm < Np; ++mm)
            {
                for (int jj = 0; jj <  NJ; ++jj)
                {
                    matrix[jj+mm*NJ + nn*NJ*Np] = 0.0;

                    for (int rr = 0; rr < Nr; ++rr )
                    {
                       matrix[ jj + mm*NJ + nn*NJ*Np ] += 0.5 * cr[rr] * AA[ jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ];

                    }
                }
            }
        }
        
        
    }


// one-side kernel

    void compute_convolution_matrix_one_side_kernel(Gpu::ManagedVector<Real> & AA,Gpu::ManagedVector<Real> & BB, int sp,  Gpu::ManagedVector<Real> points, Gpu::ManagedVector<Real> quad_points, Gpu::ManagedVector<Real> quad_weights)
    {
        // compute the convolution matrix based on the DG method and points of interests
        //Output: AA,BB
        // AA right side kernle
        // BB left side kernl
        // Example of right side kernel
        //                       j=0          j=1          j=2 .....    j= 3k
        //
        //   boundary |_______x____________|______________|_____________|______________|______________|______________|
        //                    ~~~~~~~~~~~~~~~support of right side kernel~~~~~~~~~~~~~~~

        const int Npoints = points.size();
        const int Np = sp+1; // k+1
        const int Nr = 4*sp+1;  //  -2k <= r <= 2k
        const int NJ = 3*sp+1; //  0<= j <= 3k
        const Real iupper = 1.0;
        const Real ilower = -1.0;

        AA.resize(Npoints*Np*Nr*NJ);
        BB.resize(Npoints*Np*Nr*NJ);

        Real center_point_A, center_point_B; 

        for (int nn = 0; nn < Npoints; ++nn)
        {
            for (int jj = 0; jj <  NJ; ++jj)
            {
                for (int rr = 0; rr < Nr; ++rr )
                {
                    center_point_A = points[nn]*0.5 - (rr-2*sp) - (jj);
                    center_point_B = points[nn]*0.5 - (rr-2*sp) + (jj);

                    Real rlt_A[Np];
                    Real rlt_B[Np];

                    if (jj == 0)
                    {// has to split the integral (because we are considering one side kernel)
                        Bspline_integral(rlt_A, Np, center_point_A, iupper, points[nn], quad_points, quad_weights);
                        Bspline_integral(rlt_B, Np, center_point_B, points[nn], ilower, quad_points, quad_weights);

                    }
                    else
                    {
                        Bspline_integral(rlt_A, Np, center_point_A, iupper, ilower, quad_points, quad_weights);
                        Bspline_integral(rlt_B, Np, center_point_B, iupper, ilower, quad_points, quad_weights);
                    }
                    

                    for (int mm = 0; mm < Np; ++mm)
                    {
                        AA[ jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ] = rlt_A[mm];
                        BB[ jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ] = rlt_B[mm];
   

                    }
                }
            }
        }

    }


    void get_convolution_matrix_one_side_kernel(Gpu::ManagedVector<Real> & matrix_r,Gpu::ManagedVector<Real> & matrix_l, Gpu::ManagedVector<Real> points, int sp, Gpu::ManagedVector<Real> quad_points, Gpu::ManagedVector<Real> quad_weights)
    {   
        // Precompute the convolution matrix needed for post-processing

        // basic info 
        const int Npoints = points.size();
        const int Np = sp+1;
        const int Nr = 4*sp+1;
        const int NJ = 3*sp+1;

        matrix_r.resize(Npoints*Np*NJ);
        matrix_l.resize(Npoints*Np*NJ);
        
        //step 1. load cr

        const Real * cr ;

        if (sp == 1) cr = cr_oneside_1 ;
        else if (sp == 2) cr = cr_oneside_2;
        else if (sp == 3) cr = cr_oneside_3;
        //else if (sp == 4) cr = cr_oneside_4;
        //else if (sp == 5) cr =cr_oneside_5;
        else
        {
            Print()<<"Order "<<sp<<" is not implemented"<< std::endl;
        }

        // setp 2. compute A_{n,m,r,j}
        

        Gpu::ManagedVector<Real> AA;
        Gpu::ManagedVector<Real> BB;

        compute_convolution_matrix_one_side_kernel(AA, BB, sp, points, quad_points, quad_weights);

        // step 3. sum A_{n,m,r,j} * cr/2 over r

        for (int nn = 0; nn < Npoints; ++nn)
        {
            for (int mm = 0; mm < Np; ++mm)
            {
                for (int jj = 0; jj <  NJ; ++jj)
                {
                    matrix_r[jj+mm*NJ + nn*NJ*Np] = 0.0;
                    matrix_l[jj+mm*NJ + nn*NJ*Np] = 0.0;

                    for (int rr = 0; rr < Nr; ++rr )
                    {
                       matrix_r[ jj + mm*NJ + nn*NJ*Np ] += 2.0*0.5 * cr[rr] * AA[ jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ];

                        //  notice we need to switch the order of BB matrix
                        // because BB is computed in order 0, -1, -2, .. -3k
                        // however all the other coefficients are listed in increasing order, namely, -3k, ...,-2,-1,0
                       matrix_l[ jj + mm*NJ + nn*NJ*Np ] += 2.0*0.5 * cr[rr] * BB[ NJ-1-jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ];

                       
                       

                    }
                    /*
                    if (nn == 2 && jj==0)
                        {
                            Print()<<"check!~~~"<<std::endl;
                            
                            Print()<<"jj "<< jj  << " mm "<< mm << " nn "<< nn << " point[nn] "<<points[nn] <<std::endl;
                            Print()<<"matrix_r  "<< matrix_r[ jj + mm*NJ + nn*NJ*Np ] << " matrix_l "<< matrix_l[ jj + mm*NJ + nn*NJ*Np ] << std::endl;
                        }
                        */
                    
                    
                    
                }
            }
        }
        
        
    }


// implicit mesh
    void get_num_kernel_support_one_direction(int * kernel_support, const bool * kernel_flag, const int & symmetric_kernl_one_side_support, const int & oneside_kernl_one_side_support)
    {
        if (kernel_flag[0] && kernel_flag[1])
        {
            kernel_support[0] = symmetric_kernl_one_side_support;
        }
        else if (kernel_flag[0] || kernel_flag[1])
        {
            kernel_support[0] = oneside_kernl_one_side_support;
        }
        else
        {
            // not enough neighbors on both sides. This is not what we want, refine the mesh
            Print()<<"ERROR: not enough elements to post-process in x directions! Refine the mesh"<<std::endl;
        }
#if(AMREX_SPACEDIM > 1)
        if (kernel_flag[2] && kernel_flag[3])
        {
            kernel_support[1] = symmetric_kernl_one_side_support;
        }
        else if (kernel_flag[2] || kernel_flag[3])
        {
            kernel_support[1] = oneside_kernl_one_side_support;
        }
        else
        {
            // not enough neighbors on both sides. This is not what we want, refine the mesh
            Print()<<"ERROR: not enough elements to post-process in y directions! Refine the mesh"<<std::endl;
        }

#endif

#if(AMREX_SPACEDIM > 2)

        if (kernel_flag[4] && kernel_flag[5])
        {
            kernel_support[2] = symmetric_kernl_one_side_support;
        }
        else if (kernel_flag[4] || kernel_flag[5])
        {
            kernel_support[2] = oneside_kernl_one_side_support;
        }
        else
        {
            // not enough neighbors on both sides. This is not what we want, refine the mesh
            Print()<<"ERROR: not enough elements to post-process in z directions! Refine the mesh"<<std::endl;
        }

#endif
        

    }

    void check_avaible_nbrs_IM_one_side_kernel(bool * kernel_flag, const int & symmetric_kernl_one_side_support, const int & oneside_kernl_one_side_support, const Box & domain, Array4<int const> const & eType_fab, const int & i, const int & j , const int & k,const int & ru,const int & dom )
    {
        // in the code we onlu considered entire element
        
        const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
        const bool element_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE) ;

        // 1) check avaible directions (away from the boundary)
        for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
        {
            kernel_flag[b] = true;

            const int dir = b/2;
            int fi, fj, fk;

            GRID_FACE(i, j, k, b, fi, fj, fk);
            // Print()<<"dir: "<< dir<<" smallend: "<< domain.smallEnd(dir)<<" bigend: "<<domain.bigEnd(dir)<<std::endl;

            // XX: only works for rectangular domain?
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

        // 2) check neighbors on the available directions are entire element (don't go cross the interface).
        //         |
        // -------[ ]----------
        //
        int num_kernel_support[AMREX_SPACEDIM];

        get_num_kernel_support_one_direction(num_kernel_support, kernel_flag, symmetric_kernl_one_side_support, oneside_kernl_one_side_support);

        for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
        {
            const int dir = b/2;
            int nbr_i, nbr_j, nbr_k, nbr_b;
            int nbr_etype;
            int tempi,tempj,tempk;
            tempi = i;
            tempj = j;
            tempk = k;

            if(kernel_flag[b]) // if this direction will be used, check no partial elements.
            {
                for (int nbr_iter = 0; nbr_iter < num_kernel_support[dir]; ++nbr_iter )
                {
                    NBR_CELL(tempi, tempj, tempk, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);

                    if (element_is_entire)
                    {   
                        // entire cell: we want only use all neighbors that are entire 

                        const bool nbr_is_entire = (nbr_etype%10 == DG_ELM_TYPE_ENTIRE);
                        if (nbr_is_entire)
                        {
                            tempi = nbr_i;
                            tempj = nbr_j;
                            tempk = nbr_k;
                        }
                        else
                        {
                        kernel_flag[b] = false;
                        break;
                        }

                    }

                    /*
                    if (etype%10 == DG_ELM_TYPE_LARGE || etype%10 == DG_ELM_TYPE_SMALL) 
                    {
                        // partial cell, we want to use all avaibale neightbors
                        const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                        const bool nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
                        if (nbr_is_small || nbr_is_empty)//????
                        {
                            kernel_flag[b] = false;
                            break;
                        }
                        else
                        {
                            tempi = nbr_i;
                            tempj = nbr_j;
                            tempk = nbr_k;
                            
                        }
                        


                    }
                    */
                    

                    
                }
            }

        }

        // 3) check all the neighbors are entire for entire cell
        // 2D : four Quadrants
        // 3D : eight Quadrants
        if (element_is_entire)
        {
        
#if (AMREX_SPACEDIM == 2)

        int nbr_i, nbr_j, nbr_k;
        int nbr_etype;
        
        bool quadrant_check[4];

        nbr_k = k;

        // check (+,+)
        quadrant_check[0] = false; // initilize

        if(kernel_flag[1] && kernel_flag[3])
        {
            quadrant_check[0] = true;
            for (int ii = 1; ii <= num_kernel_support[0]; ++ii)
            {
                nbr_i = i + ii;
                for (int jj = 1; jj <= num_kernel_support[1]; ++jj)
                {
                    nbr_j = j + jj;
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    if (nbr_etype%10 != DG_ELM_TYPE_ENTIRE)
                    {  
                        quadrant_check[0] = false;
                        
                        break;
                    }
                }
                if (quadrant_check[0] == false) break;
            }

        }
        
        // check (-,+)
        quadrant_check[1] = false; // initilize
        if(kernel_flag[0] && kernel_flag[3])
        {
            quadrant_check[1] = true;
            for (int ii = 1; ii <= num_kernel_support[0]; ++ii)
            {
                nbr_i = i - ii;
                for (int jj = 1; jj <= num_kernel_support[1]; ++jj)
                {
                    nbr_j = j + jj;
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    if (nbr_etype%10 != DG_ELM_TYPE_ENTIRE)
                    {  
                        quadrant_check[1] = false;
                        
                        break;
                    }
                }
                if (quadrant_check[1] == false) break;
            }

        }
        
        // check (-,-)
        quadrant_check[2] = false; // initilize
        if(kernel_flag[0] && kernel_flag[2])
        {
            quadrant_check[2] = true;
            for (int ii = 1; ii <= num_kernel_support[0]; ++ii)
            {
                nbr_i = i - ii;
                for (int jj = 1; jj <= num_kernel_support[1]; ++jj)
                {
                    nbr_j = j - jj;
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    if (nbr_etype%10 != DG_ELM_TYPE_ENTIRE)
                    {  
                        quadrant_check[2] = false;
                        
                        break;
                    }
                }
                if (quadrant_check[2] == false) break;
            }
            
        }
        
        //check (+,-)
        quadrant_check[3] = false; // initilize
        if(kernel_flag[1] && kernel_flag[2])
        {
            quadrant_check[3] = true;
            for (int ii = 1; ii <= num_kernel_support[0]; ++ii)
            {
                nbr_i = i + ii;
                for (int jj = 1; jj <= num_kernel_support[1]; ++jj)
                {
                    nbr_j = j - jj;
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    if (nbr_etype%10 != DG_ELM_TYPE_ENTIRE)
                    {  
                        quadrant_check[3] = false;
                        
                        break;
                    }
                }
                if (quadrant_check[3] == false) break;
            }
            
            
        }

        // update kernel_flag
        
        int temp_cout=0;

        for (int ii = 0; ii<4; ii++)
        {
            if (quadrant_check[ii]) temp_cout+=1;
        }

        if (temp_cout == 0)
        {
            kernel_flag[0]= false;
            kernel_flag[1]= false;
            kernel_flag[2]= false;
            kernel_flag[3]= false;

        }
        else if (temp_cout == 1)
        {
            if (quadrant_check[0])
                {
                    kernel_flag[0] = false;
                    kernel_flag[2] = false;
                }
                if (quadrant_check[1])
                {
                    kernel_flag[1] = false;
                    kernel_flag[2] = false;
                }
                if (quadrant_check[2])
                {
                    kernel_flag[1] = false;
                    kernel_flag[3] = false;
                }
                if (quadrant_check[3])
                {
                    kernel_flag[0] = false;
                    kernel_flag[3] = false;
                }

        }
        else if (temp_cout == 2)
        {
            // if diagonal
            if ( quadrant_check[0]==quadrant_check[2])
            // manually decide to use which one side kernel
                    if (quadrant_check[0] == false)
                    {
                        //            |3
                        //       X    |      X
                        //    0------[ ]----------1
                        //       X    |
                        //            |2
                        kernel_flag[0]= false;
                        kernel_flag[3]= false;

                    }
                    else
                    {
                        //            |3
                        //       X    |      
                        //    0------[ ]----------1
                        //       X    |      X
                        //            |2
                        kernel_flag[0]= false;
                        kernel_flag[2]= false;
                        
                    }
            else
            {
                if (quadrant_check[0] == false && quadrant_check[1] == false ) kernel_flag[3]= false;
                if (quadrant_check[1] == false && quadrant_check[2] == false ) kernel_flag[0]= false;
                if (quadrant_check[2] == false && quadrant_check[3] == false ) kernel_flag[2]= false;
                if (quadrant_check[3] == false && quadrant_check[0] == false ) kernel_flag[1]= false;

            }

        }
        else if (temp_cout == 3)
        {
            if (quadrant_check[0]==false)
                {
                    kernel_flag[3] = false;
                    
                }
                if (quadrant_check[1]==false)
                {
                    kernel_flag[0] = false;
                   
                }
                if (quadrant_check[2]==false)
                {
                    kernel_flag[2] = false;
                    
                }
                if (quadrant_check[3]==false)
                {
                    kernel_flag[1] = false;
                    
                }

        }
        else
        {
            // do nothing
            
        }

        


#endif

#if (AMREX_SPACEDIM == 3)
        // check (+,+,+)
        if(kernel_flag[1] && kernel_flag[3] && kernel_flag[5])
        {

        }
        // check (-,+,+)
        if(kernel_flag[0] && kernel_flag[3] && kernel_flag[5])
        {

        }
        // check (-,-,+)
        if(kernel_flag[0] && kernel_flag[2] && kernel_flag[5])
        {
            
        }
        //check (+,-,+)
        if(kernel_flag[1] && kernel_flag[2] && kernel_flag[5])
        {
            
        }
        ///////////

        // check (+,+,-)
        if(kernel_flag[1] && kernel_flag[3] && kernel_flag[4])
        {

        }
        // check (-,+,-)
        if(kernel_flag[0] && kernel_flag[3] && kernel_flag[4])
        {

        }
        // check (-,-,-)
        if(kernel_flag[0] && kernel_flag[2] && kernel_flag[4])
        {
            
        }
        //check (+,-,-)
        if(kernel_flag[1] && kernel_flag[2] && kernel_flag[4])
        {
            
        }
        
#endif
        }



    }


    void cut_ele_compute_convolution_matrix_one_side_kernel(Gpu::ManagedVector<Real> & AA, const int & i, const int & j, const int & k, const int & kernel_choice,  int sp,  Gpu::ManagedVector<Real> points, Gpu::ManagedVector<Real> quad_points, Gpu::ManagedVector<Real> quad_weights)
    {
        const int Npoints = points.size();
        const int Np = sp+1; // k+1
        const int Nr = 4*sp+1;  //  -2k <= r <= 2k
        const int NJ = 3*sp+1; //  0<= j <= 3k
        const Real iupper = 1.0;
        const Real ilower = -1.0;

        AA.resize(Npoints*Np*Nr*NJ);
        

        Real center_point_A; 

        for (int nn = 0; nn < Npoints; ++nn)
        {
            for (int jj = 0; jj <  NJ; ++jj)
            {
                for (int rr = 0; rr < Nr; ++rr )
                {
                     Real rlt_A[Np];
                    if (kernel_choice == 1) // right kernle
                    {
                        center_point_A = points[nn]*0.5 - (rr-2*sp) - (jj);

                        if (jj == 0)
                        {// has to split the integral (because we are considering one side kernel)
                            Bspline_integral(rlt_A, Np, center_point_A, iupper, points[nn], quad_points, quad_weights);
                        
                        }
                        else
                        {
                            Bspline_integral(rlt_A, Np, center_point_A, iupper, ilower, quad_points, quad_weights);
                        
                        }
                    }
                    else // left kernel
                    {
                        center_point_A = points[nn]*0.5 - (rr-2*sp) + (jj);

                        if (jj == 0)
                        {// has to split the integral (because we are considering one side kernel)
                            Bspline_integral(rlt_A, Np, center_point_A, points[nn], ilower, quad_points, quad_weights);
                        
                        }
                        else
                        {
                            Bspline_integral(rlt_A, Np, center_point_A, iupper, ilower, quad_points, quad_weights);
                        
                        }

                    }

                    
                    for (int mm = 0; mm < Np; ++mm)
                    {
                        AA[ jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ] = rlt_A[mm];
                       
   

                    }
                }
            }
        }

    }

    void cut_ele_get_convolution_matrix_one_side_kernel(Gpu::ManagedVector<Real> & matrix, const int & i, const int & j, const int & k, const int & kernel_choice, Gpu::ManagedVector<Real> points, int sp, Gpu::ManagedVector<Real> quad_points, Gpu::ManagedVector<Real> quad_weights)
    {
        if(kernel_choice == 0)
        {
            Print()<<"ERROR: partial cell can't use symmetric kernel!!!"<<std::endl;
        }
        // basic info 
        const int Npoints = points.size();
        const int Np = sp+1;
        const int Nr = 4*sp+1;
        const int NJ = 3*sp+1;

         matrix.resize(Npoints*Np*NJ);

         //step 1. load cr

        const Real * cr ;

        if (sp == 1) cr = cr_oneside_1 ;
        else if (sp == 2) cr = cr_oneside_2;
        //else if (sp == 3) cr = cr_oneside_3;
        //else if (sp == 4) cr = cr_oneside_4;
        //else if (sp == 5) cr =cr_oneside_5;
        else
        {
            Print()<<"Order "<<sp<<" is not implemented"<< std::endl;
        }

        // setp 2. compute A_{n,m,r,j}
        

        Gpu::ManagedVector<Real> AA;

        cut_ele_compute_convolution_matrix_one_side_kernel(AA, i,j,k, kernel_choice, sp, points, quad_points, quad_weights);
        // step 3. sum A_{n,m,r,j} * cr/2 over r

        for (int nn = 0; nn < Npoints; ++nn)
        {
            for (int mm = 0; mm < Np; ++mm)
            {
                for (int jj = 0; jj <  NJ; ++jj)
                {
                    matrix[jj+mm*NJ + nn*NJ*Np] = 0.0;
                    
                    for (int rr = 0; rr < Nr; ++rr )
                    {
                        if (kernel_choice == 1)
                        {
                            matrix[ jj + mm*NJ + nn*NJ*Np ] += 2.0*0.5 * cr[rr] * AA[ jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ];
                        }
                        else if (kernel_choice == -1)
                        {
                             matrix[ jj + mm*NJ + nn*NJ*Np ] += 2.0*0.5 * cr[rr] * AA[ NJ-1-jj + rr*NJ + mm*NJ*Nr + nn*NJ*Nr*Np ];
                        }
  
                    }
                       
                }
            }
        }
        




    }





}

}
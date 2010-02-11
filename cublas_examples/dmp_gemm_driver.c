/***************************************************************************

                  COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
 + 2009 University of Chicago

Permission is hereby granted to use, reproduce, prepare derivative works, and
to redistribute to others.  This software was authored by:

Jeff R. Hammond
Leadership Computing Facility
Argonne National Laboratory
Argonne IL 60439 USA
phone: (630) 252-5381
e-mail: jhammond@mcs.anl.gov

                  GOVERNMENT LICENSE

Portions of this material resulted from work developed under a U.S.
Government Contract and are subject to the following license: the Government
is granted for itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable worldwide license in this computer software to reproduce, prepare
derivative works, and perform publicly and display publicly.

                  DISCLAIMER

This computer code material was prepared, in part, as an account of work
sponsored by an agency of the United States Government.  Neither the United
States, nor the University of Chicago, nor any of their employees, makes any
warranty express or implied, or assumes any legal liability or responsibility
for the accuracy, completeness, or usefulness of any information, apparatus,
product, or process disclosed, or represents that its use would not infringe
privately owned rights.

 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "blas_gemm_test.h"
#include "cublas_gemm_test.h"

#include "ga_utils.h"

void gagpu_sgemm(int rows, int cols, int blksz, float alpha, int g_a, int g_b, float beta, int g_d);
void gagpu_dgemm(int rows, int cols, int blksz, double alpha, int g_a, int g_b, double beta, int g_d);

int main(int argc, char** argv)
{
    int me, nproc;

    start_parallel(&argc,&argv,&me,&nproc);

    int precision = 2;
    int rows  = 15;
    int cols  = 15;
    int blksz = 15;

    int g_a  = alloc_global_2d(precision,rows,cols,me);
    int g_b  = alloc_global_2d(precision,rows,cols,me);
    int g_c1 = alloc_global_2d(precision,rows,cols,me);
    int g_c2 = alloc_global_2d(precision,rows,cols,me);
    int g_d  = alloc_global_2d(precision,rows,cols,me);
    int g_e  = alloc_global_2d(precision,rows,cols,me);

    randomize_global(g_a);
    randomize_global(g_b);
//     GA_Symmetrize(g_a);
//     GA_Symmetrize(g_b);
    zero_global(g_c1);
    zero_global(g_c2);
    zero_global(g_d);
    zero_global(g_e);

/*******************************************************/

    int ndim = 2;
    int dims[2];
    float  f_alpha = 1.0;
    float  f_beta  = 1.0;
    double d_alpha = 1.0;
    double d_beta  = 1.0;
    dims[0] = rows;
    dims[1] = cols;
    if (precision==1)      GA_Sgemm('T','T',dims[0],dims[0],dims[0],f_alpha,g_a,g_b,f_beta,g_c1);
    else if (precision==2) GA_Dgemm('T','T',dims[0],dims[0],dims[0],d_alpha,g_a,g_b,d_beta,g_c1);

    GA_Transpose(g_c1,g_c2);

/*******************************************************/

    if (precision==1)      gagpu_sgemm(rows,cols,blksz,f_alpha,g_a,g_b,f_beta,g_d);
    else if (precision==2) gagpu_dgemm(rows,cols,blksz,d_alpha,g_a,g_b,d_beta,g_d);

/*******************************************************/

//     GA_Print(g_c2);
//     GA_Print(g_d);

//     f_alpha = 1.0;
//     f_beta = -1.0;
//     d_alpha = 1.0;
//     d_beta = -1.0;

//     if (precision==1)      GA_Add(&f_alpha,g_c2,&f_beta,g_d,g_e);
//     else if (precision==2) GA_Add(&f_alpha,g_c2,&f_beta,g_d,g_e);

//     double error = 0.0;

//     GA_Norm1(g_e,&error);
//     GA_Norm_infinity(g_e,&error);
//     if (me == 0) printf("\n! error = %20.10f\n",error);

    int ii,jj,i,j;

    int lo_a[ndim],lo_b[ndim];
    int hi_a[ndim],hi_b[ndim];
    int ld_a[ndim-1],ld_b[ndim-1];

    double* p_a = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    double* p_b = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));

    int nblock = rows/blksz;

    double a,b;

    for (ii = 0 ; ii < nblock; ii++){
        for (jj = 0 ; jj < nblock; jj++){

            lo_a[0] = blksz * ii;
            hi_a[0] = blksz * (ii + 1) - 1;
            lo_a[1] = blksz * jj;
            hi_a[1] = blksz * (jj + 1) - 1;
            ld_a[0] = blksz;

            lo_b[0] = blksz * ii;
            hi_b[0] = blksz * (ii + 1) - 1;
            lo_b[1] = blksz * jj;
            hi_b[1] = blksz * (jj + 1) - 1;
            ld_b[0] = blksz;

            NGA_Get(g_c2,lo_a,hi_a,p_a,ld_a);
            NGA_Get(g_d ,lo_b,hi_b,p_b,ld_b);

            for (i = 0 ; i < blksz ; i++ ){
                for (j = 0 ; j < blksz ; j++ ){
                    a = p_a[ blksz * i + j ];
                    b = p_b[ blksz * i + j ];
                    printf("(%d,%d): a = %e b = %e (a-b) = %e\n",ii+i,jj+j,a,b,a-b);
                } // j
            } // i
        } // jj
    } // ii

    ARMCI_Free_local(p_b);
    ARMCI_Free_local(p_a);

/*******************************************************/

    free_global(g_e);
    free_global(g_d);
    free_global(g_c2);
    free_global(g_c1);
    free_global(g_b);
    free_global(g_a);

    stop_parallel();

    fprintf(stderr,"# the test driver has finished!!!\n");
    fflush(stderr);

    return 0;
}

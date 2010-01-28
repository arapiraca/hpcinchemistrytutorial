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

#include "blas_gemm_test.h"
#include "cublas_gemm_test.h"

int main(int argc, char** argv)
{
#ifdef CUDA
    cublasStatus status;
#endif

    int threads;
    int i, t;
    int ntests = 100;
    int dim[ntests];
    float f_alpha = 1.0;
    float f_beta = 1.0;
    double d_alpha = 1.0;
    double d_beta = 1.0;
    double blas_time[ntests];
    double blas_Gflops[ntests];
    double cublas_excl_time[ntests];
    double cublas_excl_Gflops[ntests];
    double cublas_incl_time[ntests];
    double cublas_incl_Gflops[ntests];

    t = 0;
    dim[t++] = 1;
    for ( i = 1 ; i < 20 ; i++ ) dim[t++] = i*10;  //   10 to  200 by 10
    for ( i = 4 ; i < 10 ; i++ ) dim[t++] = i*50;  //  200 to  450 by 50
//     for ( i = 5 ; i < 20 ; i++ ) dim[t++] = i*100; //  500 to 1900 by 100
//     for ( i = 4 ; i <  8 ; i++ ) dim[t++] = i*500; // 2000 to 3500 by 500
    ntests = t;

    for ( t = 0 ; t < ntests ; t++ ) fprintf(stderr,"@ dim[%d] = %d\n",t,dim[t]);
    fflush(stderr);

    threads = 1;

    for ( t = 0 ; t < ntests ; t++)
    {
        run_blas_sgemm_test(threads, dim[t], f_alpha, f_beta, &blas_time[t], &blas_Gflops[t]);
    }
    for ( t = 0 ; t < ntests ; t++)
    {
//         run_blas_dgemm_test(threads, dim[t], d_alpha, d_beta, &blas_time[t], &blas_Gflops[t]);
    }

#ifdef CUDA
    status = cublasInit();
    if (status == CUBLAS_STATUS_SUCCESS) {
//         printf("cublasInit succeeded\n");
    } else {
        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cublasInit failed\n");
        fflush(stdout);
    }

    for ( t = 0 ; t < ntests ; t++)
    {
        run_cublas_sgemm_test(dim[t], f_alpha, f_beta, &cublas_excl_time[t], &cublas_excl_Gflops[t],
                                                       &cublas_incl_time[t], &cublas_incl_Gflops[t]);
    }

    status = cublasShutdown();
    if (status == CUBLAS_STATUS_SUCCESS) {
//         printf("cublasShutdown succeeded\n");
    } else {
        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cublasShutdown failed\n");
        fflush(stdout);
    }
#endif

    printf("    d        BLAS         CUBLAS (incl)   CUBLAS (excl)\n");
    for ( t = 0 ; t < ntests ; t++ )
    {
        printf("%6d %8.3f Gflops %8.3f Gflops %8.3f Gflops\n",
                dim[t],blas_Gflops[t],cublas_incl_Gflops[t],cublas_excl_Gflops[t]);
    }
    fflush(stdout);

    fprintf(stderr,"# the test driver has finished!!!\n");
    fflush(stderr);

    return 0;
}
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

#ifdef CUDA
#include "cublas_gemm_test.h"
#endif

int main(int argc, char** argv)
{
    int threads;
    int i, t;
    int precision;
    int ntests = 100;
    int dim[ntests];
    float f_alpha = 1.0;
    float f_beta = 0.0;
    double d_alpha = 1.0;
    double d_beta = 0.0;
    double blas_time[ntests];
    double blas_Gflops[ntests];

    /* default to single precision or command-line override */
    if ( argc > 1 ) precision = atoi(argv[1]);
    else            precision = 1;
    if      ( precision == 1 ) printf("You have requested single-precision.\n");
    else if ( precision == 2 ) printf("You have requested double-precision.\n");
    else    {                  printf("Defaulting to single-precision.\n"); }
    fflush(stdout);

#ifdef CUDA
    int cudaDevice;
    struct cudaDeviceProp cudaProp;
    cudaGetDevice( &cudaDevice );
    cudaGetDeviceProperties( &cudaProp, cudaDevice );
    if ( cudaProp.major==1 && cudaProp.minor==0 && precision == 2 ){
        precision = 1;
        printf("CUDA device does not support double-precision\n");
        printf("Changing to single-precision\n");
        fflush(stdout);
    }
#endif

    t = 0;
    dim[t++] = 1;
    for ( i = 1 ; i < 20 ; i++ ) dim[t++] = i*10;  //   10 to  200 by 10
    for ( i = 4 ; i < 10 ; i++ ) dim[t++] = i*50;  //  200 to  450 by 50
    for ( i = 5 ; i < 11 ; i++ ) dim[t++] = i*100; //  500 to  900 by 100
#ifdef BIGTESTS
    for ( i = 6 ; i < 10 ; i++ ) dim[t++] = i*200; // 1200 to 1800 by 200
    for ( i = 4 ; i < 11 ; i++ ) dim[t++] = i*500; // 2000 to 5000 by 500
#endif
    ntests = t;

    for ( t = 0 ; t < ntests ; t++ ) fprintf(stderr,"@ dim[%d] = %d\n",t,dim[t]);
    fflush(stderr);

    threads = 2;
#ifdef OPENMP
    if ( threads > 0 ){ omp_set_num_threads(threads); }
    else { omp_set_num_threads( omp_get_max_threads() ); }
    printf("Using %d OpenMP threads with BLAS (if applicable)\n",omp_get_num_threads());
#else
    printf("Not using OpenMP threads with BLAS\n");
#endif
    fflush(stdout);

    for ( t = 0 ; t < ntests ; t++)
    {
        if ( precision==1 ) { 
            run_blas_sgemm_test(dim[t], f_alpha, f_beta, &blas_time[t], &blas_Gflops[t]); }
        else if (precision==2 ) {
            run_blas_dgemm_test(dim[t], d_alpha, d_beta, &blas_time[t], &blas_Gflops[t]); }
    }

#ifdef CUDA

    double cublas_excl_time[ntests];
    double cublas_excl_Gflops[ntests];
    double cublas_incl_time[ntests];
    double cublas_incl_Gflops[ntests];
    double ratio;

    cublasStatus status;

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
        if ( precision==1 ) {
            run_cublas_sgemm_test(dim[t], f_alpha, f_beta, &cublas_excl_time[t], &cublas_excl_Gflops[t],
                                                           &cublas_incl_time[t], &cublas_incl_Gflops[t]); }
        else if  (precision==2 ) {
            run_cublas_dgemm_test(dim[t], d_alpha, d_beta, &cublas_excl_time[t], &cublas_excl_Gflops[t],
                                                           &cublas_incl_time[t], &cublas_incl_Gflops[t]); }
    }

    status = cublasShutdown();
    if (status == CUBLAS_STATUS_SUCCESS) {
//         printf("cublasShutdown succeeded\n");
    } else {
        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cublasShutdown failed\n");
        fflush(stdout);
    }

    printf("=========================================================\n");
    printf("CUDA device properties:\n");
    printf("name:                 %20s\n",cudaProp.name);
    printf("major version:        %20d\n",cudaProp.major);
    printf("minor version:        %20d\n",cudaProp.minor);
    printf("canMapHostMemory:     %20d\n",cudaProp.canMapHostMemory);
    printf("totalGlobalMem:       %20ld MiB\n",cudaProp.totalGlobalMem/(1024*1024));
    printf("sharedMemPerBlock:    %20ld\n",cudaProp.sharedMemPerBlock);
    printf("clockRate:            %20d\n",cudaProp.clockRate);
    printf("regsPerBlock:         %20d\n",cudaProp.regsPerBlock);
    printf("warpSize:             %20d\n",cudaProp.warpSize);
    printf("maxThreadsPerBlock:   %20d\n",cudaProp.maxThreadsPerBlock);
    printf("=========================================================\n");
//     struct cudaDeviceProp {
//         size_t memPitch;
//         int maxThreadsDim[3];
//         int maxGridSize[3];
//         size_t totalConstMem;
//         size_t textureAlignment;
//         int deviceOverlap;
//         int multiProcessorCount;
//         int kernelExecTimeoutEnabled;
//         int integrated;
//         int computeMode;
//     }

#endif

    if ( precision==1 ) printf("   dim        SGEMM         RATIO        CUBLAS (incl)   CUBLAS (excl)\n");
    if ( precision==2 ) printf("   dim        DGEMM         RATIO        CUBLAS (incl)   CUBLAS (excl)\n");
    for ( t = 0 ; t < ntests ; t++ )
    {
#ifdef CUDA
        ratio = cublas_incl_Gflops[t] / blas_Gflops[t];
        if ( blas_Gflops[t] > cublas_incl_Gflops[t] ){
            printf("%6d %8.3f Gflops <== %6.3f     %8.3f Gflops %8.3f Gflops\n",
                    dim[t],blas_Gflops[t],ratio,cublas_incl_Gflops[t],cublas_excl_Gflops[t]); }

        else if ( blas_Gflops[t] < cublas_incl_Gflops[t] ) {
            printf("%6d %8.3f Gflops     %6.3f ==> %8.3f Gflops %8.3f Gflops\n",
                    dim[t],blas_Gflops[t],ratio,cublas_incl_Gflops[t],cublas_excl_Gflops[t]); }
        else {
            printf("%6d %8.3f Gflops <== %6.3f ==> %8.3f Gflops %8.3f Gflops\n",
                    dim[t],blas_Gflops[t],ratio,cublas_incl_Gflops[t],cublas_excl_Gflops[t]); }
#else
            printf("%6d %8.3f Gflops <== %6.3f ==> %8.3f Gflops %8.3f Gflops\n",
                   dim[t],blas_Gflops[t],0.0,0.0,0.0);
#endif
    }
    fflush(stdout);

    fprintf(stderr,"# the test driver has finished!!!\n");
    fflush(stderr);

    return 0;
}

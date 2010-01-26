/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 * Modifications from original source (simpleCUBLAS.c) by
 * Jeff Hammond, Argonne National Laboratory, 2010.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_runtime.h"
#include "cublas.h"
#include "mkl.h"
#include <omp.h>

unsigned long long getticks(void);

int main(int argc, char** argv)
{    
    cublasStatus status;
    cublasStatus status0;
    cublasStatus status1;
    cublasStatus status2;
    cublasStatus status3;
    double* h_A;
    double* h_B;
    double* h_C;
    double* h_D;
    double* d_A = 0;
    double* d_B = 0;
    double* d_C = 0;
    double alpha = 1.0f;
    double beta = 1.0f;
    int N;
    int n2;
    int i;
    int use_GPU = 0;
    long long nflops;
    double error_norm;
    double ref_norm;
    double diff;
    unsigned long long tt0, tt1, tt2, tt3, tt_blas, tt_cublas;
    double rt0, rt1, rt2, rt3, rt_blas, rt_cublas;

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n"); fflush(stdout);

    if ( argc > 1 ) use_GPU = atoi(argv[1]);

    N = 100;
    printf("attempting to get matrix rank from arguments\n");
    fflush(stdout);
    if ( argc > 2 ) N = atoi(argv[2]);

    printf("SGEMM of %d by %d matrices\n",N,N); fflush(stdout);

    n2 = N * N;
    nflops = 2 * N * N * N; // C = C + A * B
    printf("SGEMM requires %8.3e flops\n",1.0*nflops); fflush(stdout);

    /* Allocate host memory for the matrices */
    printf("malloc h_A\n"); fflush(stdout);
    h_A = (double*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf(stderr, "!!!! host memory allocation error (A)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    printf("malloc h_B\n");
    fflush(stdout);
    h_B = (double*)malloc(n2 * sizeof(h_B[0]));
    if (h_B == 0) {
        fprintf(stderr, "!!!! host memory allocation error (B)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    printf("malloc h_C\n"); fflush(stdout);
    h_C = (double*)malloc(n2 * sizeof(h_C[0]));
    if (h_C == 0) {
        fprintf(stderr, "!!!! host memory allocation error (C)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    printf("malloc h_D\n"); fflush(stdout);
    h_D = (double*)malloc(n2 * sizeof(h_D[0]));
    if (h_D == 0) {
        fprintf(stderr, "!!!! host memory allocation error (D)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    printf("rand() initialization A\n"); fflush(stdout);
    for (i = 0; i < n2; i++) h_A[i] = rand() / (double)RAND_MAX;
    printf("rand() initialization B\n"); fflush(stdout);
    for (i = 0; i < n2; i++) h_D[i] = rand() / (double)RAND_MAX;
    printf("rand() initialization C and D\n"); fflush(stdout);
    for (i = 0; i < n2; i++) h_C[i] = rand() / (double)RAND_MAX;
    for (i = 0; i < n2; i++) h_D[i] = h_C[i];

    /* Performs operation using plain C code */
    //simple_dgemm(N, alpha, h_A, h_B, beta, h_D);
    printf("calling dgemm\n"); fflush(stdout);
    if (N<1000) dgemm("n", "n", &N, &N, &N, &alpha, h_A, &N, h_B, &N, &beta, h_D, &N);
    rt0 = omp_get_wtime();
    tt0 = getticks();
    dgemm("n", "n", &N, &N, &N, &alpha, h_A, &N, h_B, &N, &beta, h_D, &N);
    tt1 = getticks();
    rt1 = omp_get_wtime();
    tt_blas = tt1 - tt0;
    rt_blas = rt1 - rt0;
    //printf("dgemm took %lld ticks\n",tt_blas);
    printf("# dgemm took %f seconds\n",rt_blas);
    //printf("dgemm Mflops %f\n",1e-6 * nflops / rt_blas); fflush(stdout);
    printf("# dgemm Gflops %f\n",1e-9 * nflops / rt_blas); fflush(stdout);

    if ( use_GPU == 0){
        /* Memory clean up */
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_D);
        printf("quitting early since no GPU\n"); fflush(stdout);
        return(0);
    } // use_GPU

    printf("cublasInit()\n");
    fflush(stdout);
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n"); fflush(stderr);
        return EXIT_FAILURE;
    }

    /* Allocate device memory for the matrices */
    printf("cublasAlloc A\n");
    fflush(stdout);
    status = cublasAlloc(n2, sizeof(d_A[0]), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device memory allocation error (A)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    printf("cublasAlloc B\n");
    fflush(stdout);
    status = cublasAlloc(n2, sizeof(d_B[0]), (void**)&d_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device memory allocation error (B)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    printf("cublasAlloc C\n");
    fflush(stdout);
    status = cublasAlloc(n2, sizeof(d_C[0]), (void**)&d_C);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device memory allocation error (C)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    printf("starting cublasDgemm input transfers\n"); fflush(stdout);

    cudaThreadSynchronize();
    rt2 = omp_get_wtime();
    tt2 = getticks();

    status0 = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
    status1 = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    status2 = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
            
    /* Clear last error */
    cublasGetError();

    /* Performs operation using cublas */
    cudaThreadSynchronize(); 
    rt0 = omp_get_wtime();
    tt0 = getticks();

    cublasDgemm('n', 'n', N, N, N, alpha, d_A, N, d_B, N, beta, d_C, N);
    status = cublasGetError();

    cudaThreadSynchronize(); 
    tt1 = getticks();
    rt1 = omp_get_wtime();

    /* Read the result back */
    status3 = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

    cudaThreadSynchronize(); 
    rt3 = omp_get_wtime();
    tt3 = getticks();

    if (status0 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write C)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    if (status1 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write A)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    if (status2 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write B)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    if (status3 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read C)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    
    tt_cublas = tt1 - tt0;
    rt_cublas = rt1 - rt0;
    //printf("cublasDgemm took %lld ticks excluding transfer\n",tt_cublas);
    printf("  cublasDgemm took %f seconds excluding transfer\n",rt_cublas);
    //printf("cublasDgemm Mflops %f excluding transfer\n",1e-6 * nflops / rt_cublas); fflush(stdout);
    printf("  cublasDgemm Gflops %f excluding transfer\n",1e-9 * nflops / rt_cublas); fflush(stdout);
    tt_cublas = tt3 - tt2;
    rt_cublas = rt3 - rt2;
    //printf("cublasDgemm took %lld ticks including transfer\n",tt_cublas);
    printf("# cublasDgemm took %f seconds including transfer\n",rt_cublas);
    //printf("cublasDgemm Mflops %f including transfer\n",1e-6 * nflops / rt_cublas); fflush(stdout);
    printf("# cublasDgemm Gflops %f including transfer\n",1e-9 * nflops / rt_cublas); fflush(stdout);

    /* Check result against reference */
    if ( N < 20 ){
        printf("CPU              GPU\n");
        for (i = 0; i < n2; ++i) printf("%20.10f %20.10f\n",h_C[i],h_D[i]);
    } fflush(stdout);

    error_norm = 0;
    ref_norm = 0;
    for (i = 0; i < n2; ++i) {
        diff = h_D[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_D[i] * h_D[i];
    }
    error_norm = (double)sqrt((double)error_norm);
    ref_norm = (double)sqrt((double)ref_norm);
    if (fabs(ref_norm) < 1e-7) {
        fprintf(stderr, "!!!! reference norm is 0\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    printf( "Test %s\n", (error_norm / ref_norm < 1e-6) ? "PASSED" : "FAILED"); fflush(stdout);
    //printf( " error_norm = %f\n ref_norm = %f\n", error_norm, ref_norm); fflush(stdout);

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    status = cublasFree(d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! memory free error (A)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    status = cublasFree(d_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! memory free error (B)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }
    status = cublasFree(d_C);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! memory free error (C)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n"); fflush(stderr);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

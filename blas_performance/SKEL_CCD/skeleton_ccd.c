/***************************************************************************

                  COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
+2009 University of Chicago

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
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#else
#warning OpenMP not enabled!
#endif

#include "mymath.h"

inline double gettime(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    clock_t time;
    time = clock();
    return (double) (time/CLOCKS_PER_SEC);
//    return (double) time(NULL);
#endif
}

inline int imax(int a, int b)
{
    return ( (a>b) ? a : b );
}

/****** BLAS documentation ******
 *
 * syntax:
 * 
 * dgemm(transa, transb, 
 *       m, n, k,
 *       alpha, 
 *       a, lda, 
 *       b, ldb, 
 *       beta, 
 *       c, ldc);
 *
 * usage:
 * 
 * dgemm_("n","n",
 *        &rowa,&colb,&cola,
 *        &alpha,
 *        p_a,&rowa,
 *        p_b,&rowb,
 *        &beta,
 *        p_d,&rowc);
 *
 ********************************/

void cc_gemm(int M, int N, int K, double* rtime)
{
    int rowa = M;
    int cola = K;
    int rowb = K;
    int colb = N;
    int rowc = M;
    int colc = N;

    double* p_a;
    double* p_b;
    double* p_c;

    double start,finish,time;

    double alpha = 1.0;
    double beta = 1.0;

    p_a = (double *) malloc(rowa*cola*sizeof(double));
    p_b = (double *) malloc(rowb*colb*sizeof(double));
    p_c = (double *) malloc(rowc*colc*sizeof(double));

    time = 0.0;
    start = gettime();
    dzero(rowa*cola,p_a);
    dzero(rowb*colb,p_b);
    dzero(rowc*colc,p_c);
    finish = gettime();
    // fprintf(stderr,"time for dzero = %30.14lf\n",finish-start);

    time = 0.0;
    start = gettime();
    drand(rowa*cola,p_a);
    drand(rowb*colb,p_b);
    drand(rowc*colc,p_c);
    finish = gettime();
    // fprintf(stderr,"time for drand = %30.14lf\n",finish-start);

    time = 0.0;
    start = gettime();
    dgemm_("n","n",&rowa,&colb,&cola,&alpha,p_a,&rowa,p_b,&rowb,&beta,p_c,&rowc);
    finish = gettime();
    time = (finish - start);
    // fprintf(stderr,"time for dgemm = %30.14lf\n",finish-start);

    free(p_c);
    free(p_b);
    free(p_a);

    *rtime = time;

    return;
}

int main(int argc, char **argv)
{
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    fprintf(stdout,"using %d OpenMP threads\n",num_threads);
#endif

    if (argc!=4) fprintf(stderr,"./dgemm_performance.x <nocc> <nvir> <iter>\n");

    int nocc = ( argc>1 ? atoi(argv[1]) : 10 );
    int nvir = ( argc>2 ? atoi(argv[2]) : 60 );
    int iter = ( argc>3 ? atoi(argv[3]) : 0 );

    fprintf(stdout,"nocc = %d\n",nocc);
    fprintf(stdout,"nvir = %d\n",nvir);
    fprintf(stdout,"iter = %d\n",iter);

    if ((nocc<2) || (nvir<2)) return(1);

    int no2 = nocc*nocc;
    int nov = nocc*nvir;
    int nv2 = nvir*nvir;
    int no2v = nocc*nocc*nvir;
    int nov2 = nocc*nvir*nvir;

    int n;
    int dim1, dim2, dim3;

    double start,finish,time;
    double ngf;
    double rate;

    /********************************
     *
     *          M       N       K
     * count    dim1    dim2    dim3
     *  1       o^2     o^2     v^2
     *  1       o^2     v^2     v^2
     *  1       o^2     v^2     o^2
     *  6       ov      ov      ov
     *  1       v       v       o^2v
     *  1       v       o^2v    v
     *  1       o       o       ov^2
     *  1       o       ov^2    o
     *
     ********************************/

    for (n=0;n<=iter;n++)
    {
        if (n==0)
        {
            fprintf(stdout,"!!!!!!!! DRY RUN - NO COMPUTATION !!!!!!!!\n");
        }
        else
        {
            fprintf(stdout,"iteration %d\n",iter);
        }

        time = 0.0;

        fprintf(stdout,"%20s %4s %4s %4s %14s %14s %14s\n","term","M","N","K","gigaflops","seconds","gigaflop/s");

        dim1 = no2;    dim2 = no2;    dim3 = nv2;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n","o^2 o^2 v^2",dim1,dim2,dim3,ngf,time,rate);

        dim1 = no2;    dim2 = nv2;    dim3 = nv2;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n","o^2 v^2 v^2",dim1,dim2,dim3,ngf,time,rate);

        dim1 = no2;    dim2 = nv2;    dim3 = no2;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n","o^2 v^2 o^2",dim1,dim2,dim3,ngf,time,rate);

        dim1 = nov;    dim2 = nov;    dim3 = nov;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n","ov ov ov",dim1,dim2,dim3,ngf,time,rate);

        dim1 = nvir;    dim2 = nvir;    dim3 = no2v;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n",dim1,dim2,dim3,ngf,time,rate);

        dim1 = nvir;    dim2 = no2v;    dim3 = nvir;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n",dim1,dim2,dim3,ngf,time,rate);

        dim1 = nocc;    dim2 = nocc;    dim3 = nov2;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n",dim1,dim2,dim3,ngf,time,rate);

        dim1 = nocc;    dim2 = nov2;    dim3 = nocc;
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
        if (n>0) cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
        fprintf(stdout,"%20s %4d %4d %4d %14.6lf %14.6lf %14.6lf\n",dim1,dim2,dim3,ngf,time,rate);
    }

    return(0);
}

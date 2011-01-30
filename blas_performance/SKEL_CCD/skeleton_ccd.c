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

inline long imax(long a, long b)
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

double gemm_perf_est(long M, long N, long K)
{
    return 0.5;
}

void cc_gemm(long M, long N, long K, double* rtime)
{
    long rowa = M;
    long cola = K;
    long rowb = K;
    long colb = N;
    long rowc = M;
    long colc = N;

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
//    fprintf(stderr,"time for dzero = %30.14lf\n",finish-start);

//    time = 0.0;
//    start = gettime();
//    drand(rowa*cola,p_a);
//    drand(rowb*colb,p_b);
//    drand(rowc*colc,p_c);
//    finish = gettime();
//    fprintf(stderr,"time for drand = %30.14lf\n",finish-start);

    time = 0.0;
    start = gettime();
    dgemm_("n","n",&rowa,&colb,&cola,&alpha,p_a,&rowa,p_b,&rowb,&beta,p_c,&rowc);
    finish = gettime();
    time = (finish - start);
//    fprintf(stderr,"time for dgemm = %30.14lf\n",finish-start);

    free(p_c);
    free(p_b);
    free(p_a);

    *rtime = time;

    return;
}

void cc_perf(int n, int dim1, int dim2, int dim3, double perf, char** name)
{
    double time;
    double ngf;
    double rate;
    double factor;

    ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
    if (n>0) {
        cc_gemm(dim1,dim2,dim3, &time);
        rate = ( time>0.0 ? ngf/time : 0.0 );
    } else {
        factor = gemm_perf_est(dim1, dim2, dim3);
        rate = perf*factor;
        time = ngf/rate;
    }
    fprintf(stdout,"%12s %9ld %9ld %9ld %10.2lf %10.6lf %10.3lf\n",*name,dim1,dim2,dim3,ngf,time,rate);
    fflush(stdout);
    return;
}

int main(int argc, char **argv)
{
#ifdef _OPENMP
    long num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    fprintf(stdout,"using %ld OpenMP threads\n",num_threads);
#endif

    if (argc!=5) fprintf(stderr,"./dgemm_performance.x <nocc> <nvir> <iter> <perf>\n");

    long nocc = ( argc>1 ? atoi(argv[1]) : 10 );
    long nvir = ( argc>2 ? atoi(argv[2]) : 60 );
    long iter = ( argc>3 ? atoi(argv[3]) : 0 );
    double perf = ( argc>4 ? atoi(argv[4]) : 20.0 );

    fprintf(stdout,"nocc = %ld\n",nocc);
    fprintf(stdout,"nvir = %ld\n",nvir);
    fprintf(stdout,"iter = %ld\n",iter);
    fprintf(stdout,"perf = %lf\n",perf);
    fprintf(stdout,"\n");

    if ( (nocc<1) || (nvir<1) || (perf<0.0) ) return(1);

    long no2 = nocc*nocc;
    long nov = nocc*nvir;
    long nv2 = nvir*nvir;
    long no2v = nocc*nocc*nvir;
    long nov2 = nocc*nvir*nvir;

    long n;
    long dim1, dim2, dim3;
    char* name = malloc(12*sizeof(char));

    double memtotal;
    double time;
    double ngf;
    double rate;
    double factor;

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
            memtotal = 0.0;
            fprintf(stdout,"double-precision memory usage:\n");
            fprintf(stdout,"T1      = %16.2lf GB\n",1e-9*nocc*nvir*sizeof(double));
            fprintf(stdout,"T2      = %16.2lf GB\n",1e-9*no2*nv2*sizeof(double));
            fprintf(stdout,"F1      = %16.2lf GB\n",1e-9*(nocc+nvir)*(nocc+nvir)*sizeof(double));
            fprintf(stdout,"V2oooo  = %16.2lf GB\n",1e-9*no2*no2*sizeof(double));
            fprintf(stdout,"V2ooov  = %16.2lf GB\n",1e-9*no2*nov*sizeof(double));
            fprintf(stdout,"V2ovov  = %16.2lf GB\n",1e-9*nov*nov*sizeof(double));
            fprintf(stdout,"V2oovv  = %16.2lf GB\n",1e-9*no2*nv2*sizeof(double));
            fprintf(stdout,"V2ovvv  = %16.2lf GB\n",1e-9*nov*nv2*sizeof(double));
            fprintf(stdout,"V2vvvv  = %16.2lf GB\n",1e-9*nv2*nv2*sizeof(double));
            memtotal += 1e-9*no2*nv2*sizeof(double);
            memtotal += 1e-9*(nocc+nvir)*(nocc+nvir)*sizeof(double);
            memtotal += 1e-9*no2*no2*sizeof(double);
            memtotal += 1e-9*no2*nov*sizeof(double);
            memtotal += 1e-9*nov*nov*sizeof(double);
            memtotal += 1e-9*no2*nv2*sizeof(double);
            fprintf(stdout,"Total 1 = %16.2lf GB\n",memtotal);
            memtotal += 1e-9*nov*nv2*sizeof(double);
            memtotal += 1e-9*nv2*nv2*sizeof(double);
            fprintf(stdout,"Total 2 = %16.2lf GB\n",memtotal);
            fprintf(stdout,"\n");

            fprintf(stdout,"!!!!!!!! DRY RUN - PERFORMANCE ESTIMATED CRUDELY !!!!!!!!\n");
            fprintf(stdout,"\n");
        }
        else
        {
            fprintf(stdout,"iteration %ld\n",iter);
            fprintf(stdout,"\n");
        }
        fflush(stdout);

        time = 0.0;

        fprintf(stdout,"%10s %9s %9s %9s %10s %10s %10s\n","term","M","N","K","gigaflops","seconds","gigaflop/s");
        fflush(stdout);

        dim1 = nocc;    dim2 = nov2;    dim3 = nocc;
        name="o ov^2 o";
        cc_perf(n,dim1,dim2,dim3,perf,&name);
        dim1 = nocc;    dim2 = nocc;    dim3 = nov2;
        name="o o ov^2";
        cc_perf(n,dim1,dim2,dim3,perf,&name);
        dim1 = nvir;    dim2 = no2v;    dim3 = nvir;
        name="v o^2v v";
        cc_perf(n,dim1,dim2,dim3,perf,&name);
        dim1 = nvir;    dim2 = nvir;    dim3 = no2v;
        name="v v o^2v";
        cc_perf(n,dim1,dim2,dim3,perf,&name);
        dim1 = no2;    dim2 = nv2;    dim3 = no2;
        name="o^2 v^2 o^2";
        cc_perf(n,dim1,dim2,dim3,perf,&name);
        dim1 = no2;    dim2 = no2;    dim3 = nv2;
        name="o^2 o^2 v^2";
        cc_perf(n,dim1,dim2,dim3,perf,&name);
        dim1 = no2;    dim2 = nv2;    dim3 = nv2;
        name="o^2 v^2 v^2";
        cc_perf(n,dim1,dim2,dim3,perf,&name);
        dim1 = nov;    dim2 = nov;    dim3 = nov;
        name="6 ov ov ov";
        cc_perf(n,dim1,dim2,dim3,perf,&name);

        fprintf(stdout,"\n");
    }

    return(0);
}

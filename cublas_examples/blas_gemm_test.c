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

#include "blas_gemm_test.h"

void run_blas_sgemm_test(int dim, float alpha, float beta, double* time, double* Gflops)
{
    int i;
    int count = 10;
    int N = dim;
    float myalpha = alpha;
    float mybeta = beta;

    double tt_start, tt_end, tt_blas;

    float* a;
    float* b;
    float* c;

    a = alloc_host_floats(dim*dim);
    b = alloc_host_floats(dim*dim);
    c = alloc_host_floats(dim*dim);

    fprintf(stderr,"# calling sgemm for %d by %d matrices\n",N,N);
    fflush(stderr);

    /* warm-up */
    sgemm_("n", "n", &N, &N, &N, &myalpha, a, &N, b, &N, &mybeta, c, &N);

    tt_blas = 0;
    for ( i = 0 ; i < count ; i++ )
    {
        /* run the timing */
        tt_start = gettime();
        sgemm_("n", "n", &N, &N, &N, &myalpha, a, &N, b, &N, &mybeta, c, &N);
        tt_end = gettime();
        tt_blas += ( tt_end - tt_start );
    }

    tt_blas /= (double) count;

    *time = tt_blas;
    *Gflops = 2*(1e-3*dim)*(1e-3*dim)*(1e-3*dim) / tt_blas;

    fprintf(stderr,"# sgemm took %f seconds\n",*time);
    fprintf(stderr,"# sgemm Gflops %f\n",*Gflops);
    fflush(stderr);

    free_host_floats(a);
    free_host_floats(b);
    free_host_floats(c);

}

void run_blas_dgemm_test(int dim, double alpha, double beta, double* time, double* Gflops)
{
    int i;
    int count = 10;
    int N = dim;
    double myalpha = alpha;
    double mybeta = beta;

    double tt_start, tt_end, tt_blas;

    double* a;
    double* b;
    double* c;

    a = alloc_host_doubles(dim*dim);
    b = alloc_host_doubles(dim*dim);
    c = alloc_host_doubles(dim*dim);

    fprintf(stderr,"# calling dgemm for %d by %d matrices\n",N,N);
    fflush(stderr);

    /* warm-up */
    dgemm_("n", "n", &N, &N, &N, &myalpha, a, &N, b, &N, &mybeta, c, &N);

    tt_blas = 0;
    for ( i = 0 ; i < count ; i++ )
    {
        /* run the timing */
        tt_start = gettime();
        dgemm_("n", "n", &N, &N, &N, &myalpha, a, &N, b, &N, &mybeta, c, &N);
        tt_end = gettime();
        tt_blas += ( tt_end - tt_start );
    }

    tt_blas /= (double) count;

    *time = tt_blas;
    *Gflops = 2*(1e-3*dim)*(1e-3*dim)*(1e-3*dim) / tt_blas;

    fprintf(stderr,"# dgemm took %f seconds\n",*time);
    fprintf(stderr,"# dgemm Gflops %f\n",*Gflops);
    fflush(stderr);

    free_host_doubles(a);
    free_host_doubles(b);
    free_host_doubles(c);

}

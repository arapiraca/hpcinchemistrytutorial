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

#include "cublas_gemm_test.h"

void run_cublas_sgemm_test(int dim, float alpha, float beta, double* time_excl, double* Gflops_excl,
                                                             double* time_incl, double* Gflops_incl)
{
    cublasStatus status;
    cublasStatus statusX;

    int i;
    int count = 10;
    int N = dim;
    float myalpha = alpha;
    float mybeta = beta;

    long long nflops;
    double tt_incl, tt_incl_start, tt_incl_end;
    double tt_excl, tt_excl_start, tt_excl_end;

    float* ha;
    float* hb;
    float* hc;

    float* da;
    float* db;
    float* dc;

    nflops = 0;
    if      (alpha==0.0){  nflops += 0; }
    else if (alpha==1.0){  nflops += dim*dim*dim; }
    else                {  nflops += 2*dim*dim*dim; }

    if      (beta==0.0){  nflops += 0; }
    else if (beta==1.0){  nflops += dim*dim; }
    else               {  nflops += 2*dim*dim; }

    ha = alloc_host_floats(dim*dim);
    hb = alloc_host_floats(dim*dim);
    hc = alloc_host_floats(dim*dim);

    randomize_floats(dim*dim, ha);
    randomize_floats(dim*dim, hb);
    randomize_floats(dim*dim, hc);

    da = alloc_device_floats(dim*dim);
    db = alloc_device_floats(dim*dim);
    dc = alloc_device_floats(dim*dim);

    fprintf(stderr,"# calling cublasSgemm for %d by %d matrices\n",N,N);
    fflush(stderr);

    /* warm-up */
    push_floats(dim*dim, ha, da);
    push_floats(dim*dim, hb, db);
    if ( myalpha != 0.0) push_floats(dim*dim, hc, dc);
    cublasSgemm('n', 'n', N, N, N, myalpha, da, N, db, N, mybeta, dc, N);
    pull_floats(dim*dim, hc, dc);

    /* run the timing */
    tt_incl = 0;
    tt_excl = 0;

    for ( i = 0 ; i < count ; i++ )
    {
        cudaThreadSynchronize();
        tt_incl_start = gettime();

        push_floats(dim*dim, ha, da);
        push_floats(dim*dim, hb, db);
        if ( myalpha != 0.0) push_floats(dim*dim, hc, dc);

        tt_excl_start = gettime();
        cublasSgemm('n', 'n', N, N, N, myalpha, da, N, db, N, mybeta, dc, N);
        cudaThreadSynchronize();

        /* check for errors */
        statusX = cublasGetError();
        tt_excl_end = gettime();

        /* read the result back */
        pull_floats(dim*dim, hc, dc);
        tt_incl_end = gettime();

        tt_incl += ( tt_incl_end - tt_incl_start );
        tt_excl += ( tt_excl_end - tt_excl_start );
    }

    if (statusX == CUBLAS_STATUS_SUCCESS) {

        tt_incl /= (double) count;
        tt_excl /= (double) count;

        *time_incl = tt_incl;
        *time_excl = tt_excl;

        *Gflops_incl = 1e-9 * nflops / *time_incl;
        *Gflops_excl = 1e-9 * nflops / *time_excl;

        fprintf(stderr,"# cublasSgemm took %f seconds (exclusive)\n",*time_excl);
        fprintf(stderr,"# cublasSgemm took %f seconds (inclusive)\n",*time_incl);
        fprintf(stderr,"# cublasSgemm Gflops %f (exclusive)\n",*Gflops_excl);
        fprintf(stderr,"# cublasSgemm Gflops %f (inclusive)\n",*Gflops_incl);
        fflush(stderr);

    } else {

        *time_incl = 1e9;
        *time_excl = 1e9;

        *Gflops_incl = 0;
        *Gflops_excl = 0;

        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cublasSgemm failed\n");
        fflush(stdout);
    }

    free_host_floats(ha);
    free_host_floats(hb);
    free_host_floats(hc);

    free_device_floats(da);
    free_device_floats(db);
    free_device_floats(dc);

}

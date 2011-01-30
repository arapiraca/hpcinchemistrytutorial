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

#include "mymath.h"

void dzero(const int n, double* p)
{
    int i;
    #pragma omp parallel private(i)
    #pragma omp parallel for schedule(static)
    for (i=0;i<n;i++) p[i] = 0.0;
}

void dscal(const int n, double* p, double scale)
{
    int i;
    #pragma omp parallel private(i)
    #pragma omp parallel for schedule(static)
    for (i=0;i<n;i++) p[i] *= scale;
}

void drand(const int n, double* p)
{
#if 0
/* my serial RNG */
    int i;
    for (i=0;i<n;i++) p[i] = ( (double)rand() )/RAND_MAX;
#endif

#if 1
/* my threaded RNG */
    int i;
    unsigned int seed;
    #pragma omp parallel private(i,seed)
    seed = omp_get_thread_num();
    #pragma omp parallel for schedule(static)
    for (i=0;i<n;i++) p[i] = ( (double)rand_r(&seed) )/RAND_MAX;
#endif

#if 0
/* ESSL RNG */
    double seed = 1.0;
    durand(&seed,&n,p);
#endif
}

void dprint(int n, double* a)
{
    int i;
    for (i=0;i<n;i++) fprintf(stdout,"a[%6d] = %30.14lf\n",i,a[i]);
}

void dprint2(int n, double* a, double* b)
{
    int i;
    for (i=0;i<n;i++) fprintf(stdout,"a[%6d] = %30.14lf    b[%6d] = %30.14lf\n",i,a[i],i,b[i]);
}

double dmax(int n, double* a)
{
    int i;
    double max = 0.0;
    for (i=0;i<n;i++) max = ( max > a[i] ? max : a[i] );
    return max;
}

double ddiff(int n, double* a, double* b)
{
    int i;
    double diff = 0.0;
    #pragma omp parallel private(i)
    #pragma omp parallel for schedule(static) reduction(+:diff)
    for (i=0;i<n;i++) diff += fabs(a[i] - b[i]);
    return diff;
}

void dmatmul_noalpha_nobeta(const int M, const int N, const int K, double* A, double* B, double* C)
{
    int i,j,k;
    double c;

    for ( i=0 ; i<M ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            c = 0.0;
            for ( k=0 ; k<K ; k++ ) {
                c += A[i+k*M] * B[k+j*K];
            }
            C[i+j*M] = c;
        }
    }
}

void dmatmul_noalpha_beta(const int M, const int N, const int K, double* A, double* B, const double beta, double* C)
{
    int i,j,k;
    double c;

    for ( i=0 ; i<M ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            c = beta * C[i+j*M];
            for ( k=0 ; k<K ; k++ ) c += A[i+k*M] * B[k+j*K];
            C[i+j*M] = c;
        }
    }
}

void dmatmul_alpha_nobeta(const int M, const int N, const int K, const double alpha, double* A, double* B, double* C)
{
    int i,j,k;
    double c;

    for ( i=0 ; i<M ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            c = 0.0;
            for ( k=0 ; k<K ; k++ ) c += alpha * A[i+k*M] * B[k+j*K];
            C[i+j*M] = c;
        }
    }
}

void dmatmul_alpha_beta(const int M, const int N, const int K, const double alpha, double* A, double* B, const double beta, double* C)
{
    int i,j,k;
    double c;

    for ( i=0 ; i<M ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            c = beta * C[i+j*M];
            for ( k=0 ; k<K ; k++ ) c += alpha * A[i+k*M] * B[k+j*K];
            C[i+j*M] = c;
        }
    }
}

void dmatmul(int M, int N, int K, const double alpha, double* A, double* B, const double beta, double* C)
{
    const int cM = M;
    const int cN = N;
    const int cK = K;

    /* special case optimizations */
    if (alpha==0.0)
    {
        if (beta==0.0)
        {
            dzero(M*N,C);
        }
        else if (beta!=1.0)
        {
            dscal(M*N,C,beta);
        }
        return;
    } 

    /* actually doing something */
    if (alpha==1.0)
    {
        if (beta==0.0)
        {
            //fprintf(stderr,"dmatmul_noalpha_nobeta\n");
            dmatmul_noalpha_nobeta(cM, cN, cK, A, B, C);
        }
        else
        {
            dmatmul_noalpha_beta(cM, cN, cK, A, B, beta, C);
        }
    } 
    else /* alpha not 0.0 or 1.0 */
    {
        if (beta==0.0)
        {
            dmatmul_alpha_nobeta(cM, cN, cK, alpha, A, B, C);
        }
        else
        {
            dmatmul_alpha_beta(cM, cN, cK, alpha, A, B, beta, C);
        }
    }
    return;
}


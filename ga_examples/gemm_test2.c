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
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#ifdef USE_GSL
    #include "gsl/gsl_math.h"
    #include "gsl/gsl_cblas.h"
#endif

unsigned long long getticks(void);

//void dgemm_(char transa, char transb,int* rank, int* rank, int* rank, double* alpha, double* p_b, int* rank, double* p_a, int* rank, double* beta, double* p_c, int *rank);
#ifdef BLAS_USES_LONG
    void dgemm_(char* , char* ,long* , long* , long* , double* , double* , long* , double* , long* , double* , double* , long* );
    #define BLAS_INT long
#else
    void dgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );
    #define BLAS_INT int
#endif

#ifdef GOTO
    #define BLAS_NAME "GotoBLAS"
#elif defined(MKL)
    #define BLAS_NAME "Intel MKL"
#elif defined(NETLIB)
    #define BLAS_NAME "Netlib"
#else
    #error "What BLAS are you using?"
#endif

void* amalloc(size_t size);

#define MEMORY_ALLOCATOR amalloc

int gemm_test2(int dim1, int dim2, int dim3)
{
    if ((dim1>0) && (dim2>0) && (dim3>0)){
        printf("dim1=%d\n",dim1);
        printf("dim2=%d\n",dim2);
        printf("dim3=%d\n",dim3);
    } else { return(1); }

    unsigned long long start,finish;
    unsigned long long t_loops,t_dgemm;

    int i,j,k;

    double alpha=(double)rand()/RAND_MAX;
    double beta =(double)rand()/RAND_MAX;

    BLAS_INT rowc = dim1;
    BLAS_INT rowa = dim1;
    BLAS_INT cola = dim3;
    BLAS_INT rowb = dim3;
    BLAS_INT colb = dim2;
    BLAS_INT colc = dim2;

    double* p_a =(double *) MEMORY_ALLOCATOR(rowa*cola*sizeof(double));
    for (i=0;i<(rowa*cola);i++) p_a[i]=(double)rand()/RAND_MAX;
    double* p_b =(double *) MEMORY_ALLOCATOR(rowb*colb*sizeof(double));
    for (i=0;i<(rowb*colb);i++) p_b[i]=(double)rand()/RAND_MAX;

    double* p_c=(double *) MEMORY_ALLOCATOR(rowc*colc*sizeof(double));
    for (i=0;i<(rowc*colc);i++) p_c[i]=0.0;
    start = getticks();
    for (i=0;i<dim1;i++ ){
        for (j=0;j<dim2;j++ ){
            p_c[i+j*rowc] *= beta;
            for (k=0;k<dim3;k++ ){
                p_c[i+j*rowc]+=alpha*p_a[i+k*rowa]*p_b[k+j*rowb];
            }
        }
    }
    finish = getticks();
    t_loops = finish - start;
    printf("! time for %15s dgemm=%12llu ticks\n","triple-loops",t_loops);

    double* p_d=(double *) MEMORY_ALLOCATOR(rowc*colc*sizeof(double));
    for (i=0;i<(rowc*colc);i++) p_d[i]=0.0;
    start = getticks();
    dgemm_("n","n",&rowa,&colb,&cola,&alpha,p_a,&rowa,p_b,&rowb,&beta,p_d,&rowc);
    finish = getticks();
    t_dgemm = finish - start;
    printf("! time for %15s dgemm=%12llu ticks\n",BLAS_NAME,t_dgemm);
    printf("! %s is %6.2f times faster than loops\n",BLAS_NAME,((double)t_loops)/t_dgemm);
    double error3=0.0;
    for (i=0;i<rowc;i++ ){
        for (j=0;j<colc;j++ ){
//             printf("%4d %4d %20.14f %20.14f\n",i,j,p_c[i+j*rowc],p_d[i+j*rowc]);
            error3+=abs(p_c[i+j*rowc]-p_d[i+j*rowc]);
            assert(abs(p_c[i+j*rowc]-p_d[i+j*rowc])<1e-14);
         }
    }
    printf("! dgemm error=%20.14f\n",error3);
    free(p_d);

    free(p_c);
    free(p_b);
    free(p_a);

    return(0);
}

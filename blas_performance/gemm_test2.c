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

#ifdef BLAS_USES_LONG
    void sgemm_(char* , char* ,long* , long* , long* , float* , float* , long* , float* , long* , float* , float* , long* );
    #define BLAS_INT long
#else
    void sgemm_(char* , char* ,int* , int* , int* , float* , float* , int* , float* , int* , float* , float* , int* );
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

#define MEMORY_ALLOCATOR malloc

#include "blas_utils.h"

int main(int argc, char **argv)
{

    if (argc!=4) printf("./gemm_test2.x <dim1> <dim2> <dim3>\n");

    int dim1 = ( argc>1 ? atoi(argv[1]) : 50 );
    int dim2 = ( argc>2 ? atoi(argv[2]) : 50 );
    int dim3 = ( argc>3 ? atoi(argv[3]) : 50 );

    if ((dim1>0) && (dim2>0) && (dim3>0)){
        printf("dim1=%d\n",dim1);
        printf("dim2=%d\n",dim2);
        printf("dim3=%d\n",dim3);
    } else { return(1); }

    double start,finish;
    double t_loops,t_sgemm;

    int i,j,k;

    float alpha=(float)rand()/RAND_MAX;
    float beta =(float)rand()/RAND_MAX;

    BLAS_INT rowc = dim1;
    BLAS_INT rowa = dim1;
    BLAS_INT cola = dim3;
    BLAS_INT rowb = dim3;
    BLAS_INT colb = dim2;
    BLAS_INT colc = dim2;

    float* p_a =(float *) MEMORY_ALLOCATOR(rowa*cola*sizeof(float));
    for (i=0;i<(rowa*cola);i++) p_a[i]=(float)rand()/RAND_MAX;
    float* p_b =(float *) MEMORY_ALLOCATOR(rowb*colb*sizeof(float));
    for (i=0;i<(rowb*colb);i++) p_b[i]=(float)rand()/RAND_MAX;

    float* p_c=(float *) MEMORY_ALLOCATOR(rowc*colc*sizeof(float));
    for (i=0;i<(rowc*colc);i++) p_c[i]=0.0;
    start = gettime();
    for (i=0;i<dim1;i++ ){
        for (j=0;j<dim2;j++ ){
            p_c[i+j*rowc] *= beta;
            for (k=0;k<dim3;k++ ){
                p_c[i+j*rowc]+=alpha*p_a[i+k*rowa]*p_b[k+j*rowb];
            }
        }
    }
    finish = gettime();
    t_loops = finish - start;
    printf("! time for %15s sgemm=%14.7f seconds\n","triple-loops",t_loops);

    float* p_d=(float *) MEMORY_ALLOCATOR(rowc*colc*sizeof(float));
    for (i=0;i<(rowc*colc);i++) p_d[i]=0.0;
    start = gettime();
    sgemm_("n","n",&rowa,&colb,&cola,&alpha,p_a,&rowa,p_b,&rowb,&beta,p_d,&rowc);
    finish = gettime();
    t_sgemm = finish - start;
    printf("! time for %15s sgemm=%14.7f seconds\n",BLAS_NAME,t_sgemm);
    printf("! %15s is %6.2f times faster than loops\n",BLAS_NAME,t_loops/t_sgemm);
    float error3=0.0;
    for (i=0;i<rowc;i++ ){
        for (j=0;j<colc;j++ ){
//             printf("%4d %4d %20.14f %20.14f\n",i,j,p_c[i+j*rowc],p_d[i+j*rowc]);
            error3+=abs(p_c[i+j*rowc]-p_d[i+j*rowc]);
            assert(abs(p_c[i+j*rowc]-p_d[i+j*rowc])<1e-14);
         }
    }
    printf("! sgemm error=%20.14f\n",error3);
    free(p_d);

    free(p_c);
    free(p_b);
    free(p_a);

    return(0);
}

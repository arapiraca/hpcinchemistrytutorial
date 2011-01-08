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

#ifdef BLAS_USES_LONG
    void dgemm_(char* , char* ,long* , long* , long* , double* , double* , long* , double* , long* , double* , double* , long* );
    #define BLAS_INT long
#else
    void dgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );
    #define BLAS_INT int
#endif

#if defined(PGOTO)
    #define BLAS_NAME "GotoBLAS (parallel)"
#elif defined(SGOTO)
    #define BLAS_NAME "GotoBLAS (serial)"
#elif defined(PMKL)
    #define BLAS_NAME "Intel MKL (parallel)"
#elif defined(SMKL)
    #define BLAS_NAME "Intel MKL (serial)"
#elif defined(NETLIB)
    #define BLAS_NAME "Netlib"
#else
    #error "What BLAS are you using?"
#endif

#ifdef OPENMP
#include <omp.h>
#endif

inline double gettime(void)
{
#ifdef OPENMP
    return omp_get_wtime();
#else
    return (double) time(NULL);
#endif
}

int main(int argc, char **argv)
{
    if (argc!=4) fprintf(stderr,"./dgemm_performance.x <dim1> <dim2> <dim3>\n");
    int dim1 = ( argc>1 ? atoi(argv[1]) : 50 );
    int dim2 = ( argc>2 ? atoi(argv[2]) : 50 );
    int dim3 = ( argc>3 ? atoi(argv[3]) : 50 );
    if ((dim1<1) || (dim2<1) || (dim3<1)) return(1);

    BLAS_INT rowc = dim1;
    BLAS_INT rowa = dim1;
    BLAS_INT cola = dim3;
    BLAS_INT rowb = dim3;
    BLAS_INT colb = dim2;
    BLAS_INT colc = dim2;

    int i,j,k;

    double* p_a = (double *) malloc(rowa*cola*sizeof(double));
    double* p_b = (double *) malloc(rowb*colb*sizeof(double));
    double* p_c = (double *) malloc(rowc*colc*sizeof(double));
    double* p_d = (double *) malloc(rowc*colc*sizeof(double));
    for (i=0;i<(rowa*cola);i++) p_a[i] = (double)rand()/RAND_MAX;
    for (i=0;i<(rowb*colb);i++) p_b[i] = (double)rand()/RAND_MAX;

    //if ((dim1==1) && (dim2==1) && (dim3==1))
        fprintf(stdout,"%20s %4s %4s %4s %14s %14s\n","BLAS_NAME","dim1","dim2","dim3","seconds","Gflop/s");

    double start,finish;
    double t_loops,t_dgemm;

    long nflops = 2*dim1*dim2*dim3;

    int count = 20;
    if ((dim1<400) && (dim2<400) && (dim3<400)) count = 1; /* disable if checking for correctness */

    double alpha;
    double beta;

    alpha = 1.0;
    beta  = 1.0;

    if ((dim1<400) && (dim2<400) && (dim3<400)){
        for (i=0;i<(rowc*colc);i++) p_c[i]=0.0;
        start = gettime();
        for (i=0;i<dim1;i++){
            for (j=0;j<dim2;j++){
                p_c[i+j*rowc] *= beta;
                for (k=0;k<dim3;k++ ) p_c[i+j*rowc]+=alpha*p_a[i+k*rowa]*p_b[k+j*rowb];
            }
        }
        finish = gettime();
        t_loops = finish - start;
        fprintf(stdout,"%20s %4d %4d %4d %14.7f %14.7f\n","triple-loops",dim1,dim2,dim3,t_loops,((double)nflops/(1000*1000*1000))/t_loops);
    }

    for (i=0;i<(rowc*colc);i++) p_d[i]=0.0;
    start = gettime();
    for (i=0;i<count;i++) dgemm_("n","n",&rowa,&colb,&cola,&alpha,p_a,&rowa,p_b,&rowb,&beta,p_d,&rowc);
    finish = gettime();
    t_dgemm = (finish - start)/count;
    fprintf(stdout,"%20s %4d %4d %4d %14.7f %14.7f\n",BLAS_NAME,dim1,dim2,dim3,t_dgemm,((double)nflops/(1000*1000*1000))/t_dgemm);

    if ((dim1<400) && (dim2<400) && (dim3<400)){
        double error=0.0;
        for (i=0;i<rowc;i++ ){
            for (j=0;j<colc;j++ ){
                error+=abs(p_c[i+j*rowc]-p_d[i+j*rowc]);
                if (abs(p_c[i+j*rowc]-p_d[i+j*rowc])>1e-14) printf("error in output(%3d,%3d): %f vs %f\n",i,j,p_c[i+j*rowc],p_d[i+j*rowc]);
                //assert(abs(p_c[i+j*rowc]-p_d[i+j*rowc])<1e-12);
             }
        }
        fprintf(stderr,"dgemm error=%20.14f\n",error);
    }

    alpha = 1.0;
    beta  = 0.0;

    if ((dim1<400) && (dim2<400) && (dim3<400)){
        for (i=0;i<(rowc*colc);i++) p_c[i]=0.0;
        start = gettime();
        for (i=0;i<dim1;i++){
            for (j=0;j<dim2;j++){
                p_c[i+j*rowc] *= beta;
                for (k=0;k<dim3;k++ ) p_c[i+j*rowc]+=alpha*p_a[i+k*rowa]*p_b[k+j*rowb];
            }
        }
        finish = gettime();
        t_loops = finish - start;
        fprintf(stdout,"%20s %4d %4d %4d %14.7f %14.7f\n","triple-loops",dim1,dim2,dim3,t_loops,((double)nflops/(1000*1000*1000))/t_loops);
    }

    for (i=0;i<(rowc*colc);i++) p_d[i]=0.0;
    start = gettime();
    for (i=0;i<count;i++) dgemm_("n","n",&rowa,&colb,&cola,&alpha,p_a,&rowa,p_b,&rowb,&beta,p_d,&rowc);
    finish = gettime();
    t_dgemm = (finish - start)/count;
    fprintf(stdout,"%20s %4d %4d %4d %14.7f %14.7f\n",BLAS_NAME,dim1,dim2,dim3,t_dgemm,((double)nflops/(1000*1000*1000))/t_dgemm);

    if ((dim1<400) && (dim2<400) && (dim3<400)){
        double error=0.0;
        for (i=0;i<rowc;i++ ){
            for (j=0;j<colc;j++ ){
                error+=abs(p_c[i+j*rowc]-p_d[i+j*rowc]);
                if (abs(p_c[i+j*rowc]-p_d[i+j*rowc])>1e-14) printf("error in output(%3d,%3d): %f vs %f\n",i,j,p_c[i+j*rowc],p_d[i+j*rowc]);
                //assert(abs(p_c[i+j*rowc]-p_d[i+j*rowc])<1e-12);
             }
        }
        fprintf(stderr,"dgemm error=%20.14f\n",error);
    }

    free(p_d);
    free(p_c);
    free(p_b);
    free(p_a);

    return(0);
}

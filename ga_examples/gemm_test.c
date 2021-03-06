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

//void dgemm_(char transa, char transb,int* rank, int* rank, int* rank, double* alpha, double* p_b, int* rank, double* p_a, int* rank, double* beta, double* p_c, int *rank);
void dgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );

int gemm_test(int rank)
{

    printf("rank=%d\n",rank);

    int i,j,k;

    double alpha=(double)rand()/RAND_MAX;
    double beta =(double)rand()/RAND_MAX;

    double* p_a =(double *) malloc(rank*rank*sizeof(double));
    for (i=0;i<(rank*rank);i++) p_a[i]=(double)rand()/RAND_MAX;
    double* p_b =(double *) malloc(rank*rank*sizeof(double));
    for (i=0;i<(rank*rank);i++) p_b[i]=(double)rand()/RAND_MAX;

    double* p_c0=(double *) malloc(rank*rank*sizeof(double));
    for (i=0;i<(rank*rank);i++) p_c0[i]=0.0;
    for (i=0;i<rank;i++ ){
        for (j=0;j<rank;j++ ){
            p_c0[rank*i+j] *= beta;
            for (k=0;k<rank;k++ ){
                p_c0[rank*i+j]+=alpha*p_a[rank*i+k]*p_b[rank*k+j];
            }
        }
    }

#ifdef USE_LOOPS
    double* p_c1=(double *) malloc(rank*rank*sizeof(double));
    for (i=0;i<(rank*rank);i++) p_c1[i]=0.0;
    double aik;
    for (i=0;i<rank;i++ ){
        for (k=0;k<rank;k++ ){
            aik=alpha*p_a[rank*i+k];
            for (j=0;j<rank;j++ ){
                p_c1[rank*i+j]+=aik*p_b[rank*k+j];
            }
        }
    }
    double error1=0.0;
    for (i=0;i<rank;i++ ){
        for (j=0;j<rank;j++ ){
            error1+=abs(p_c0[i*rank+j] - p_c1[i*rank+j]);
            assert(p_c0[i*rank+j]==p_c1[i*rank+j]);
        }
    }
    printf("! loops error=%20.14f\n",error1);
    free(p_c1);
#endif

#ifdef USE_GSL
    double* p_c2=(double *) malloc(rank*rank*sizeof(double));
    for (i=0;i<(rank*rank);i++) p_c2[i]=0.0;
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,rank,rank,rank,
                alpha,p_a,rank,p_b,rank,beta,p_c2,rank);
    double error2=0.0;
    for (i=0;i<rank;i++ ){
        for (j=0;j<rank;j++ ){
            error2+=abs(p_c0[i*rank+j] - p_c2[i*rank+j]);
            assert(p_c0[i*rank+j]==p_c2[i*rank+j]);
        }
    }
    printf("! cblas_dgemm error=%20.14f\n",error2);
    free(p_c2);
#endif

    double* p_c3=(double *) malloc(rank*rank*sizeof(double));
    for (i=0;i<(rank*rank);i++) p_c3[i]=0.0;
   	dgemm_("n","n",&rank,&rank,&rank,&alpha,p_b,&rank,p_a,&rank,&beta,p_c3,&rank);
    double error3=0.0;
    for (i=0;i<rank;i++ ){
        for (j=0;j<rank;j++ ){
            error3+=abs(p_c0[i*rank+j] - p_c3[i*rank+j]);
            assert(p_c0[i*rank+j]==p_c3[i*rank+j]);
        }
    }
    printf("! dgemm error=%20.14f\n",error3);
    free(p_c3);

    free(p_c0);
    free(p_b);
    free(p_a);

    return(0);
}

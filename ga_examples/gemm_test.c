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


#define USE_GSL
#define USE_BLAS

#include "driver.h"

//void dgemm_(char transa, char transb,int* rank, int* rank, int* rank, double* alpha, double* p_b, int* rank, double* p_a, int* rank, double* beta, double* p_c, int *rank);
void dgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );

/***************************************************************************
 *                                                                         *
 * gemm_test:                                                              *
 *       -serial matrix multiplication									   *
 *                                                                         *
 ***************************************************************************/

int gemm_test(int rank)
{
    int i,j,k;
    int status;
    double alpha = 1.0;
    double beta  = 0.0;
    double temp;
    double error2,error3;
    double* p_a;  // input matrix A
    double* p_b;  // input matrix B
    double* p_c1; // my dgemm output
    double* p_c2; // GSL CBLAS output
    double* p_c3; // Fortran BLAS output

    printf("rank = %d\n",rank);

    p_a  = (double *)ARMCI_Malloc_local((armci_size_t) rank * rank * sizeof(double));
    p_b  = (double *)ARMCI_Malloc_local((armci_size_t) rank * rank * sizeof(double));
    p_c1 = (double *)ARMCI_Malloc_local((armci_size_t) rank * rank * sizeof(double));
    p_c2 = (double *)ARMCI_Malloc_local((armci_size_t) rank * rank * sizeof(double));
    p_c3 = (double *)ARMCI_Malloc_local((armci_size_t) rank * rank * sizeof(double));

    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            //p_a[i*rank + j] = (double)(i*rank + j);
            p_a[i*rank + j] = (double) ( rand() * 1e-9);
        }
    }

    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            //p_b[i*rank + j] = (double)(i*rank + j);
            p_b[i*rank + j] = (double) ( rand() * 1e-9);
        }
    }

    start = MPI_Wtime(); 

#define LOOP_ALG_2

#ifdef LOOP_ALG_1
    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            temp = 0;
            for (k = 0 ; k < rank ; k++ ){
                temp += p_a[ rank * i + k ] * p_b[ rank * k + j ];
            }
            p_c1[ rank * i + j ] = temp;
        }
    }
#endif

#ifdef LOOP_ALG_2
    double aik;

    memset(p_c1,0,rank * rank * sizeof(double));

    for (i = 0 ; i < rank ; i++ ){
        for (k = 0 ; k < rank ; k++ ){
            aik = p_a[ rank * i + k ];
            for (j = 0 ; j < rank ; j++ ){
                p_c1[ rank * i + j ] += aik * p_b[ rank * k + j ];
            }
        }
    }
#endif

	/**************************************/

    finish = MPI_Wtime(); 

    printf("! My dgemm took %f seconds\n",(double) (finish - start) );
	fflush(stdout);\

#ifdef USE_GSL

    start = MPI_Wtime(); 

	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,rank,rank,rank,
                alpha,p_a,rank,p_b,rank,beta,p_c2,rank);

    finish = MPI_Wtime(); 

   	printf("! GSL CBLAS dgemm took %f seconds\n",(double) (finish - start) );
    fflush(stdout);\

#endif

#ifdef USE_BLAS

    start = MPI_Wtime(); 

   	//dgemm_("n","n",&rank,&rank,&rank,&alpha,p_a,&rank,p_b,&rank,&beta,p_c3,&rank);
   	//dgemm_("t","t",&rank,&rank,&rank,&alpha,p_a,&rank,p_b,&rank,&beta,p_c3,&rank);
   	//dgemm_("n","n",&rank,&rank,&rank,&alpha,p_b,&rank,p_a,&rank,&beta,p_c3,&rank);
   	dgemm_("n","n",&rank,&rank,&rank,&alpha,p_b,&rank,p_a,&rank,&beta,p_c3,&rank);

    finish = MPI_Wtime(); 

   	printf("! BLAS dgemm took %f seconds\n",(double) (finish - start) );
    fflush(stdout);

#endif

/*
 * begin error evaluation
 */

/*
    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            printf("a(%d,%d) = %f\n",i,j,p_a[i*rank + j]);
        }
    }

    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            printf("b(%d,%d) = %f\n",i,j,p_b[i*rank + j]);
        }
    }

    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            printf("(%d,%d) c1 c2 c3 (c2-c3) = %f %f %f %f\n",i,j,p_c1[i*rank + j],p_c2[i*rank + j],p_c3[i*rank + j],p_c2[i*rank + j]-p_c3[i*rank + j]);
        }
    }

*/

    error2 = 0.0;

    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            error2 += pow(p_c1[i*rank + j] - p_c2[i*rank + j],2);
        }
    }

    printf("! error2 = %f\n",error2);

    error3 = 0.0;

    for (i = 0 ; i < rank ; i++ ){
        for (j = 0 ; j < rank ; j++ ){
            error3 += pow(p_c1[i*rank + j] - p_c3[i*rank + j],2);
        }
    }

    printf("! error3 = %f\n",error3);

/*
 * terminate data structures
 */

    status = ARMCI_Free_local(p_c3);
    if(status != 0){
    	 printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_c2);
    if(status != 0){
    	 printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_c1);
    if(status != 0){
    	 printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_b);
    if(status != 0){
    	 printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_a);
    if(status != 0){
    	 printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };

    return(0);
}

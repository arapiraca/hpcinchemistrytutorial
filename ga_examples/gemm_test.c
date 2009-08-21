/***************************************************************************
 *   Copyright (C) 2009 by Jeff Hammond                                    *
 *   jeff.science@gmail.com                                                *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

//#define USE_GSL
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

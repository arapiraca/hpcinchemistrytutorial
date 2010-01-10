/***************************************************************************
 *   Copyright (C) 2009 by Jeff Hammond                                    *
 *   jeff.science@gmail.com                                                *
 *                                                                         *
 * Redistribution and use in source and binary forms, with or without      *
 * modification, are permitted provided that the following conditions      *
 * are met:                                                                *
 * 1. Redistributions of source code must retain the above copyright       *
 *    notice, this list of conditions and the following disclaimer.        *
 * 2. Redistributions in binary form must reproduce the above copyright    *
 *    notice, this list of conditions and the following disclaimer in the  *
 *    documentation and/or other materials provided with the distribution. *
 * 3. The name of the author may not be used to endorse or promote         *
 *    products derived from this software without specific prior written   *
 *    permission.                                                          *
 *                                                                         *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR    *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED          *
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE  *
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,      *
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES      *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR      *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)      *
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,     *
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING   *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE      *
 * POSSIBILITY OF SUCH DAMAGE.                                             *
 *                                                                         *
 ***************************************************************************/

//#define USE_LOOPS
//#define USE_GSL
//#define USE_BLAS
#define USE_GOTO

#include "driver.h"

//void dgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );
void dgemm(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );

/***************************************************************************
 *                                                                         *
 * matmul:                                                                  *
 *       -demonstrates how to create a GA using the new API                *
 *       -matrix multiplication									           *
 *                                                                         *
 ***************************************************************************/

int matmul(int rank, int blksz)
{
	int me,nproc,ntask,t;
    int ii,jj,kk;
    int i,j,k;
    int g_a,g_b,g_c1,g_c2,g_d,g_error; // GA handles
    int status;
    int ndim = 2;
    int dims[2];
    int chunk[2];
    int nblock;
    int lo_a[2],lo_b[2],lo_d[2];
    int hi_a[2],hi_b[2],hi_d[2];
    int rng_a[2],rng_b[2];//,rng_c[2];
    int ld_a[1],ld_b[1],ld_d[1];
    int pg_world;   // world processor group
    double alpha,beta,error;
    double zero = 0.0;
    double one  = 1.0;
    double temp;
    double t_get_a,t_get_b,t_acc_d, t_dgemm; // timers
    double* p_in; // pointers for local access to GAs
    double* p_a;  // pointers for local access to GAs
    double* p_b;  // pointers for local access to GAs
    double* p_d;  // pointers for local access to GAs
    bool myturn;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    t_get_a = 0.0;
    t_get_b = 0.0;
    t_acc_d = 0.0;
    t_dgemm = 0.0;

    dims[0] = rank;
    dims[1] = rank;
    chunk[0] = -1;
    chunk[1] = -1;
    nblock = rank/blksz;

    if (me == 0){
      printf("! matmul: rank %d matrix with block size %d\n",rank,blksz);
    }

    pg_world = GA_Pgroup_get_world();

    g_a= GA_Create_handle();
    GA_Set_array_name(g_a,"matrix A");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,pg_world);

    status = GA_Allocate(g_a);
    if(status == 0){
    	if (me == 0) printf("%s: GA_Allocate failed at line %d\n",__FILE__,__LINE__);
    };
    
    g_b  = GA_Duplicate(g_a,"matrix B");
    if(g_b == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    g_c1  = GA_Duplicate(g_a,"matrix C1");
    if(g_c1 == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    g_c2  = GA_Duplicate(g_a,"matrix C2");
    if(g_c2 == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    g_d  = GA_Duplicate(g_a,"matrix D");
    if(g_d == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    g_error  = GA_Duplicate(g_a,"error");
    if(g_error == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    GA_Sync();

/*
    GA_Zero(g_a);
    GA_Zero(g_b);
    GA_Zero(g_c1);
    GA_Zero(g_c2);
    GA_Zero(g_error);
    if (me == 0){
        printf("\n");
        GA_Print_distribution(g_a);
        printf("\n");
    }
*/

/*
 * begin initialization with random values using local access
 */

    NGA_Distribution(g_a,me,lo_a,hi_a);
    NGA_Access(g_a,lo_a,hi_a,&p_in,&ld_a[0]);

    rng_a[0] = hi_a[0] - lo_a[0] + 1;
    rng_a[1] = hi_a[1] - lo_a[1] + 1;

    double scale = 0.00001/sqrt(RAND_MAX);

    for(i=0; i<rng_a[0]; i++){
    	for(j=0; j<rng_a[1]; j++){
    		p_in[ ld_a[0] * i + j ] = (double) ( rand() * scale );
//    		p_in[ ld_a[0] * i + j ] = (double) ( 1 );
    	}
    }

    NGA_Release_update(g_b,lo_a,hi_a); /* this function does nothing as of GA 4.2 */

    NGA_Distribution(g_b,me,lo_b,hi_b);
    NGA_Access(g_b,lo_b,hi_b,&p_in,&ld_b[0]);

    rng_b[0] = hi_b[0] - lo_b[0] + 1;
    rng_b[1] = hi_b[1] - lo_b[1] + 1;

    for(i=0; i<rng_b[0]; i++){
    	for(j=0; j<rng_b[1]; j++){
    		    		p_in[ ld_a[0] * i + j ] = (double) ( rand() * scale );
//    		    		p_in[ ld_a[0] * i + j ] = (double) ( 1 );
    	}
    }

    NGA_Release_update(g_b,lo_b,hi_b); /* this function does nothing as of GA 4.2 */

#ifdef DEBUG
	GA_Print(g_a);
	GA_Print(g_b);
#endif

/*
 * begin hand-written transposition
 */

    GA_Zero(g_d);

    GA_Sync();

    start = MPI_Wtime(); 

    ntask = nblock * nblock * nblock;

    if (me == 0) {
    	printf("! nproc     = %10d\n",nproc);
    	printf("! ntask     = %10d\n",ntask);
    	printf("! task/proc = %8.1f\n",(1.0*ntask)/nproc);
	    fflush(stdout);\
    }
	//printf("proc %d is here\n",me);
	//fflush(stdout);

    p_a = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_b = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_d = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));

    t = 0;

    for (ii = 0 ; ii < nblock; ii++){
    	for (jj = 0 ; jj < nblock; jj++){
    		for (kk = 0 ; kk < nblock; kk++){

//    			printf("t mod nproc = %d\n",t % nproc);
//    			fflush(stdout);
    			myturn = ( me == ( t % nproc ) );

    			if (myturn){

#ifdef DEBUG
    				printf("proc %d doing work tuple (%d,%d,%d)\n",me,ii,jj,kk);
    				fflush(stdout);
#endif

    				lo_a[0] = blksz * ii;
    				hi_a[0] = blksz * (ii + 1) - 1;
    				lo_a[1] = blksz * kk;
    				hi_a[1] = blksz * (kk + 1) - 1;
    				ld_a[0] = blksz;

    				lo_b[0] = blksz * kk;
    				hi_b[0] = blksz * (kk + 1) - 1;
    				lo_b[1] = blksz * jj;
    				hi_b[1] = blksz * (jj + 1) - 1;
    				ld_b[0] = blksz;

    				lo_d[0] = blksz * ii;
    				hi_d[0] = blksz * (ii + 1) - 1;
    				lo_d[1] = blksz * jj;
    				hi_d[1] = blksz * (jj + 1) - 1;
    				ld_d[0] = blksz;

                    
                    temp = MPI_Wtime(); 
    			    NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
                    t_get_a += (double) (MPI_Wtime() - temp);


                    temp = MPI_Wtime(); 
    				NGA_Get(g_b,lo_b,hi_b,p_b,ld_b);
                    t_get_b += (double) (MPI_Wtime() - temp);

    				/**************************************/

//    				memset(p_d,0,blksz * blksz * sizeof(double));

#define LOOP_ALG_3

#ifdef USE_LOOPS
  #ifdef LOOP_ALG_1
    				for (i = 0 ; i < blksz ; i++ ){
    					for (j = 0 ; j < blksz ; j++ ){
    						temp = 0;
    						for (k = 0 ; k < blksz ; k++ ){
    							temp += p_a[ blksz * i + k ] * p_b[ blksz * k + j ];
      						}
    						p_d[ blksz * i + j ] = temp;
    					}
    				}
  #endif

  #ifdef LOOP_ALG_2
                    double aik;

    				memset(p_d,0,blksz * blksz * sizeof(double));

    				for (i = 0 ; i < blksz ; i++ ){
    					for (k = 0 ; k < blksz ; k++ ){
    					    aik = p_a[ blksz * i + k ];
    					    for (j = 0 ; j < blksz ; j++ ){
    						    p_d[ blksz * i + j ] += aik * p_b[ blksz * k + j ];
      						}
    					}
    				}
  #endif

  #ifdef LOOP_ALG_3
                    double aik;

                    if ( 0 != (blksz % 4) ){
    	                if (me == 0) printf("%s: blksz is not a multiple of 4\n",__FILE__);
                        return(1);
                    }

    				memset(p_d,0,blksz * blksz * sizeof(double));

    				for (i = 0 ; i < blksz ; i++ ){
    					for (k = 0 ; k < blksz ; k++ ){
    					    aik = p_a[ blksz * i + k ];
    					    for (j = 0 ; j < blksz ; j+=4 ){
    						    p_d[ blksz * i + j     ] += aik * p_b[ blksz * k + j     ];
    						    p_d[ blksz * i + j + 1 ] += aik * p_b[ blksz * k + j + 1 ];
    						    p_d[ blksz * i + j + 2 ] += aik * p_b[ blksz * k + j + 2 ];
    						    p_d[ blksz * i + j + 3 ] += aik * p_b[ blksz * k + j + 3 ];
      						}
    					}
    				}
  #endif

  #ifdef LOOP_ALG_4
                    double aik;
                    int blksz2,peel;

                    peel = (blksz % 4);
                    blksz2 = blksz - peel;

                    memset(p_d,0,blksz * blksz * sizeof(double));

                    for (i = 0 ; i < blksz ; i++ ){
                        for (k = 0 ; k < blksz ; k++ ){
                            aik = p_a[ blksz * i + k ];
                            for (j = 0 ; j < blksz2 ; j+=4 ){
                                p_d[ blksz * i + j     ] += aik * p_b[ blksz * k + j     ];
                                p_d[ blksz * i + j + 1 ] += aik * p_b[ blksz * k + j + 1 ];
                                p_d[ blksz * i + j + 2 ] += aik * p_b[ blksz * k + j + 2 ];
                                p_d[ blksz * i + j + 3 ] += aik * p_b[ blksz * k + j + 3 ];
                            }
                            for (j = 0 ; j < peel ; j++ ){
                                p_d[ blksz * i + j ] += aik * p_b[ blksz * k + j ];
                            }
                        }
                    }
  #endif

#endif

#ifdef USE_GSL
                    temp = MPI_Wtime(); 
    				cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,blksz,blksz,blksz,
    				              one,p_a,blksz,p_b,blksz,zero,p_d,blksz);
                    t_dgemm += (double) (MPI_Wtime() - temp);
#endif

#ifdef USE_BLAS
                    temp = MPI_Wtime(); 
                    dgemm_("n","n",&blksz,&blksz,&blksz,&one,p_b,&blksz,p_a,&blksz,&zero,p_d,&blksz);
                    t_dgemm += (double) (MPI_Wtime() - temp);
#endif

#ifdef USE_ESSL
                    temp = MPI_Wtime(); 
    				dgemm_("n","n",&blksz,&blksz,&blksz,&one,p_a,&blksz,p_b,&blksz,&zero,p_d,&blksz);
                    t_dgemm += (double) (MPI_Wtime() - temp);
#endif

#ifdef USE_GOTO
                    temp = MPI_Wtime(); 
    				dgemm_("n","n",&blksz,&blksz,&blksz,&one,p_a,&blksz,p_b,&blksz,&zero,p_d,&blksz);
                    t_dgemm += (double) (MPI_Wtime() - temp);
#endif

    				/**************************************/

                    temp = MPI_Wtime(); 
    				NGA_Acc(g_d,lo_d,hi_d,p_d,ld_d,&one);
                    t_acc_d += (double) (MPI_Wtime() - temp);

    			} // myturn

    			t += 1;

    		} // kk
    	} // jj
    } // ii

    status = ARMCI_Free_local(p_d);
    if(status != 0){
    	if (me == 0) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_b);
    if(status != 0){
    	if (me == 0) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_a);
    if(status != 0){
    	if (me == 0) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };

    GA_Sync();

    finish = MPI_Wtime(); 

    if (me == 0){
    	printf("! My dgemm took %f seconds\n",(double) (finish - start) );
	    fflush(stdout);\
    }

    for (i = 0 ; i < nproc; i++){
        if (me==i){
            printf("! GA_Get A time on node %d = %f seconds\n", me, (double) t_get_a );
            printf("! GA_Get B time on node %d = %f seconds\n", me, (double) t_get_b );
            printf("! GA_Acc C time on node %d = %f seconds\n", me, (double) t_acc_d );
            printf("!   DGEMM  time on node %d = %f seconds\n", me, (double) t_dgemm );
	        fflush(stdout);\
        }
        GA_Sync();
    }


#ifdef DEBUG
	GA_Print(g_d);
#endif

/*
 * GA reference matrix multiplication
 */

	alpha = 1.0;
    beta  = 0.0;

    GA_Sync();

    start = MPI_Wtime(); 

    // GA_Dgemm uses Fortran ordering, hence the double 'T'
    GA_Dgemm('T','T',dims[0],dims[0],dims[0],alpha,g_a,g_b,beta,g_c1);

    finish = MPI_Wtime(); 

    GA_Sync();

    if (me == 0){
    	printf("! GA_Dgemm took %f seconds\n",(double) (finish - start) );
	    fflush(stdout);\
    }

    //GA_Transpose(g_c1,g_c2);

#ifdef DEBUG
	GA_Print(g_c2);
#endif

/*
 * begin error evaluation
 */

    alpha = 1.0;
    beta = -1.0;
    //GA_Add(&alpha,g_c2,&beta,g_d,g_error);

    //GA_Norm1(g_error,&error);

    if (me == 0) printf("! error = %f\n",error);

/*
 * terminate data structures
 */

    GA_Destroy(g_error);
    GA_Destroy(g_d);
    GA_Destroy(g_c2);
    GA_Destroy(g_c1);
    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}

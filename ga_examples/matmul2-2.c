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

#include "driver.h"

#define USE_LOOPS
//#define USE_BLAS

#ifdef USE_BLAS
void dgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* ); 
#endif

/***************************************************************************
 *                                                                         *
 * matmul2:                                                                  *
 *       -demonstrates how to create a GA using the new API                *
 *       -matrix multiplication									           *
 *                                                                         *
 ***************************************************************************/

int matmul2(int rank, int blksz)
{
	int me,nproc,ntask,t;
    int ii,jj,kk;
    int i,j,k;
    int g_a,g_b;
    int g_c,g_d;
    int g_error;
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
    double* p_in; // pointers for local access to GAs
    double* p_a;  // pointers for local access to GAs
    double* p_b;  // pointers for local access to GAs
    double* p_d1;  // pointers for local access to GAs
    double* p_d2;  // pointers for local access to GAs
    double d1,d2;
    bool myturn;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = rank;
    dims[1] = rank;
    chunk[0] = -1;
    chunk[1] = -1;
    nblock = rank/blksz;

    if (me == 0){
      printf("matmul2: rank %d matrix with block size %d\n",rank,blksz);
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

    g_c  = GA_Duplicate(g_a,"matrix C");
    if(g_c == 0){
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
    GA_Zero(g_c);
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
    GA_Symmetrize(g_a);

    NGA_Distribution(g_b,me,lo_b,hi_b);
    NGA_Access(g_b,lo_b,hi_b,&p_in,&ld_b[0]);

    rng_b[0] = hi_b[0] - lo_b[0] + 1;
    rng_b[1] = hi_b[1] - lo_b[1] + 1;

    for(i=0; i<rng_b[0]; i++){
    	for(j=0; j<rng_b[1]; j++){
    		    		p_in[ ld_a[0] * i + j ] = (double) ( rand() * scale );
    	}
    }

    NGA_Release_update(g_b,lo_b,hi_b); /* this function does nothing as of GA 4.2 */
    GA_Symmetrize(g_b);

#ifdef DEBUG
	GA_Print(g_a);
	GA_Print(g_b);
#endif

/*
 * end initialization
 */

/*
 * GA reference matrix multiplication
 */
	alpha = 1.0;
    beta  = 0.0;

    GA_Sync();

    start = clock();

    // GA_Dgemm uses Fortran ordering, hence the double 'T'
    GA_Dgemm('t','t',dims[0],dims[0],dims[0],alpha,g_b,g_a,beta,g_c);

    finish = clock();

    GA_Sync();

    if (me == 0){
    	printf("GA_Dgemm took %f seconds\n",(double) (finish - start) / CLOCKS_PER_SEC);
    }

#ifdef DEBUG
	GA_Print(g_c);
#endif

/*
 * begin hand-written transposition
 */

    GA_Zero(g_d);

    GA_Sync();

    start = clock();

    ntask = nblock * nblock * nblock;

    if (me == 0) {
    	printf("ntask = %d\n",ntask);
    	printf("nproc = %d\n",nproc);
    }
	//printf("proc %d is here\n",me);
	//fflush(stdout);

    p_a  = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_b  = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_d1 = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_d2 = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));

    t = 0;

    for (ii = 0 ; ii < nblock; ii++){
    	for (jj = 0 ; jj < nblock; jj++){
    		for (kk = 0 ; kk < nblock; kk++){

//    			printf("t mod nproc = %d\n",t % nproc);
//    			fflush(stdout);
    			myturn = ( me == ( t % nproc ) );

    			if (myturn){
#ifdef DEBUG
                    //printf("proc %d doing work tuple (%d,%d,%d)\n",me,ii,jj,kk); fflush(stdout);
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

    			    NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
    				NGA_Get(g_b,lo_b,hi_b,p_b,ld_b);

    				/**************************************/

//    				memset(p_d,0,blksz * blksz * sizeof(double));
#ifdef USE_LOOPS
    				for (i = 0 ; i < blksz ; i++ ){
    					for (j = 0 ; j < blksz ; j++ ){
    						temp = 0;
    						for (k = 0 ; k < blksz ; k++ ){
    							// temp += p_a[ blksz * i + k ] * p_b[ blksz * k + j ]; // double transpose version
    							temp += p_a[ blksz * k + i ] * p_b[ blksz * j + k ];
      						}
    						//p_d[ blksz * i + j ] = temp; // double transpose version
    						p_d1[ blksz * j + i ] = temp;
    					}
    				}
#endif

#ifdef USE_GSL
    				cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,blksz,blksz,blksz,
    				              one,p_a,blksz,p_b,blksz,zero,p_d,blksz);
#endif

#ifdef USE_BLAS
    				dgemm_("n","n",&blksz,&blksz,&blksz,&one,p_a,&blksz,p_b,&blksz,&zero,p_d2,&blksz);
#endif

    				/**************************************/

#ifdef SUPERDEBUG
    				for (i = 0 ; i < blksz ; i++ ){
    					for (j = 0 ; j < blksz ; j++ ){
    						d1 = p_d1[ blksz * i + j ];
    						d2 = p_d2[ blksz * i + j ];
                            printf("%4d %4d: d1=%e d2=%e (d1-d2)=%e\n",i,j,d1,d2,d1-d2);
    					}
    				}
#endif

    				/**************************************/

    				NGA_Acc(g_d,lo_d,hi_d,p_d1,ld_d,&one);

    			} // myturn

    			t += 1;

    		} // kk
    	} // jj
    } // ii

    status = ARMCI_Free_local(p_d2);
    status = ARMCI_Free_local(p_d1);
    status = ARMCI_Free_local(p_b);
    status = ARMCI_Free_local(p_a);

    GA_Sync();

    finish = clock();

    if (me == 0){
    	printf("My DGEMM took %f seconds\n",(double) (finish - start) / CLOCKS_PER_SEC);
    }

#ifdef DEBUG
	GA_Print(g_d);
#endif

/*
 * end hand-written transposition
 */

/*
 * begin error evaluation
 */

    alpha = 1.0;
    beta = -1.0;
    GA_Add(&alpha,g_c,&beta,g_d,g_error);

    GA_Norm1(g_error,&error);

    if (me == 0) printf("error = %f\n",error);


/*
 * end error evaluation
 */

    GA_Destroy(g_error);
    GA_Destroy(g_d);
    GA_Destroy(g_c);
    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}

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

#include "driver.h"

/***************************************************************************
 *                                                                         *
 * test3:                                                                  *
 *       -demonstrates how to create a GA using the new API                *
 *       -matrix multiplication									           *
 *                                                                         *
 ***************************************************************************/

void transpose_patch(double* input, double* output);

int test3()
{
	int me,nproc;
    int i,j,k,ii,jj,kk;
    int g_a,g_b,g_c1,g_c2,g_d,g_error; // GA handles
    int status;
    const int ndim = 2;
    const int rank = 600;
	const int blksz = 30;
    int dims[2];
    int chunk[2];
    int nblock;
    int lo_a[2],lo_b[2],lo_d[2];
    int hi_a[2],hi_b[2],hi_d[2];
    int rng_a[2],rng_b[2];//,rng_c[2];
    int ld_a[1],ld_b[1],ld_d[1];
    int pg_world;   // world processor group
    double alpha,beta,error;
    double* p_in; // pointers for local access to GAs
    double* p_a;  // pointers for local access to GAs
    double* p_b;  // pointers for local access to GAs
    double* p_d;  // pointers for local access to GAs
    bool myturn = true;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = rank;
    dims[1] = rank;
    chunk[0] = 2;
    chunk[1] = 2;
    nblock = rank/blksz;

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

#ifdef DEBUG
    GA_Zero(g_a);
    GA_Zero(g_b);
    GA_Zero(g_c1);
    GA_Zero(g_c2);
    GA_Zero(g_d);
    GA_Zero(g_error);
    if (me == 0) printf("\n");
    GA_Print_distribution(g_a);
    if (me == 0) printf("\n");
#endif

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
//    		    		p_in[ ld_a[0] * i + j ] = (double) ( 1 );
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
    GA_Dgemm('N','N',dims[0],dims[0],dims[0],alpha,g_a,g_b,beta,g_c1);
    GA_Transpose(g_c1,g_c2);

#ifdef DEBUG
	GA_Print(g_c2);
#endif

/*
 * begin hand-written transposition
 */

    double temp;

    p_a = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_b = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_d = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));

    for (ii = 0 ; ii < nblock; ii++){
    	for (jj = 0 ; jj < nblock; jj++){
    		memset(p_d,0,blksz * blksz * sizeof(double));
    		for (kk = 0 ; kk < nblock; kk++){

    			myturn = ( me == ( (nblock * nblock * nblock) % nproc ) );

    			if (myturn){

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
    				for (i = 0 ; i < blksz ; i++){
    					for (j = 0 ; j < blksz ; j++){
    						temp = 0.0;
    						for (k = 0 ; k < blksz ; k++){
    							temp += p_a[ blksz * i + k ] * p_b[ blksz * k + j ];
//    							p_d[ blksz * i + j ] += p_a[ blksz * i + k ] * p_b[ blksz * k + j ];
    						}
    						p_d[ blksz * i + j ] = temp;
    					}
    				}
    				/**************************************/

    				NGA_Put(g_d,lo_d,hi_d,p_d,ld_d);

    			}
    		}
    	}
    }

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
    GA_Add(&alpha,g_c2,&beta,g_d,g_error);

    GA_Norm1(g_error,&error);

    if (me == 0) printf("error = %f\n",error);


/*
 * end error evaluation
 */

    GA_Destroy(g_error);
    GA_Destroy(g_d);
    GA_Destroy(g_c2);
    GA_Destroy(g_c1);
    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}

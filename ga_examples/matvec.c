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

#define USE_LOOPS

/***************************************************************************
 *                                                                         *
 * matvec:                                                                 *
 *       -demonstrates how to create a GA using the new API                *
 *       -sparse matrix-vector product									   *
 *                                                                         *
 ***************************************************************************/

int matvec(int rank, int blksz)
{
	int me,nproc,ntask,t;
    int iii,jjj;
    int ii,jj;
    int i,j;
    int g_x,g_y; // GA handles
    int status;
    int ndim = 1;
    int dims[1];
    int chunk[1];
    int nblock;
    int lo_x[1],lo_y[1];
    int hi_x[1],hi_y[1];
//    int rng_x[1],rng_y[1];
    int ld_x[1],ld_y[1];
    int pg_world;   // world processor group
    double zero = 0.0;
    double one  = 1.0;
    double* p_x;  // pointers for local access to GAs
    double* p_y;  // pointers for local access to GAs
    double* p_A;  // pointers for local access to GAs
    bool myturn;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = rank;
    dims[1] = rank;
    chunk[0] = -1;
    chunk[1] = -1;
    nblock = rank/blksz;

    if (me == 0){
      printf("! matvec: rank %d vector with block size %d\n",rank,blksz);
    }

    pg_world = GA_Pgroup_get_world();

    g_x= GA_Create_handle();
    GA_Set_array_name(g_x,"vector x");
    GA_Set_data(g_x,ndim,dims,MT_DBL);
    GA_Set_chunk(g_x,chunk);
    GA_Set_pgroup(g_x,pg_world);

    status = GA_Allocate(g_x);
    if(status == 0){
    	if (me == 0) printf("%s: GA_Allocate failed at line %d\n",__FILE__,__LINE__);
    };
    
    g_y  = GA_Duplicate(g_x,"vector y");
    if(g_y == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    GA_Sync();

    GA_Fill(g_x,&one);
    GA_Zero(g_y);

#ifdef DEBUG
    GA_Print(g_x);
    GA_Print(g_y);
#endif

/*
 *  doing a fake sparse matrix-vector product y=A(x)
 *
 *  -2d parallel decomposition of work
 *  -loop over all 1d blocks of x (columns of A) and y (rows of A)
 *  -get local piece of x, form local y=A(x), accumulate y to global vector
 *
 *  y[i] = A(x[i])
 *
 */

    GA_Sync();
    start = MPI_Wtime(); 

    ntask = nblock * nblock;

    if (me == 0) {
    	printf("! nproc     = %10d\n",nproc);
    	printf("! ntask     = %10d\n",ntask);
    	printf("! task/proc = %8.1f\n",(1.0*ntask)/nproc);
	    fflush(stdout);\
    }

    p_x = (double *)ARMCI_Malloc_local((armci_size_t) blksz * sizeof(double));
    p_y = (double *)ARMCI_Malloc_local((armci_size_t) blksz * sizeof(double));
    p_A = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));

    t = 0;

    for (ii = 0 ; ii < nblock; ii++){
    	for (jj = 0 ; jj < nblock; jj++){

//    		printf("t mod nproc = %d\n",t % nproc);
//    		fflush(stdout);
    		myturn = ( me == ( t % nproc ) );

    		if (myturn){

#ifdef DEBUG
    			printf("proc %d doing work tuple (%d,%d)\n",me,ii,jj);
    			fflush(stdout);
#endif

    			lo_x[0] = blksz * jj;
    			hi_x[0] = blksz * (jj + 1) - 1;

    		    NGA_Get(g_x,lo_x,hi_x,p_x,ld_x);

    			lo_y[0] = blksz * ii;
    			hi_y[0] = blksz * (ii + 1) - 1;

    			/**************************************/

    			memset(p_y,0,blksz * sizeof(double));
    			memset(p_A,0,blksz * blksz * sizeof(double));

                // trivial example
                for (i = 0 ; i < blksz ; i++ ){
                    iii = lo_y[0] + i;
                    for (j = 0 ; j < blksz ; j++ ){
                        jjj = lo_x[0] + j;
                        if (iii == jjj) p_A[i*blksz + j] = 1.0;
                    }
                }

                // y = A*x
                for (i = 0 ; i < blksz ; i++ ){
                    for (j = 0 ; j < blksz ; j++ ){
                        p_y[i] += p_A[i*blksz + j]*p_x[j];
                    }
                }

   				/**************************************/

   				NGA_Acc(g_y,lo_y,hi_y,p_y,ld_y,&one);

    		} // myturn

    		t += 1;

    	} // jj
    } // ii

    status = ARMCI_Free_local(p_y);
    if(status != 0){
    	if (me == 0) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_x);
    if(status != 0){
    	if (me == 0) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };

    GA_Sync();
    finish = MPI_Wtime(); 

    if (me == 0){
    	printf("! My matvec took %f seconds\n",(double) (finish - start) );
	    fflush(stdout);\
    }

#ifdef DEBUG
    GA_Print(g_x);
    GA_Print(g_y);
#endif

/*
 * terminate data structures
 */

    GA_Destroy(g_y);
    GA_Destroy(g_x);

    return(0);
}

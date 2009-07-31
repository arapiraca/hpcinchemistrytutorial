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
 * test2:                                                                  *
 *       -demonstrates how to create a GA using the new API                *
 *       -implements in-place matrix transpose and compares this to        *
 *        GA_Transpose (out-of-place)                                      *
 *                                                                         *
 ***************************************************************************/

void transpose_patch(double* input, double* output);

int test2()
{
	int me;
    int i,j;
    int g_before,g_trans1,g_trans2,g_error; // GA handles
    int status;
    const int ndim=2;
    int dims[ndim];
    int chunk[ndim];
    int lo[ndim];
    int hi[ndim];
    int range[ndim];
    int ld[ndim-1];
    int pg_world;   // world processor group
    double* p_in;   // pointers for local access to GAs
    double* p_buf1; // pointers for local access to GAs
    double* p_buf2; // pointers for local access to GAs

    me=GA_Nodeid();

    dims[0] = 6;
    dims[1] = 6;
    chunk[0] = -1;
    chunk[1] = chunk[0];

    pg_world = GA_Pgroup_get_world();

    g_before = GA_Create_handle();
    GA_Set_array_name(g_before,"input and in-place transpose");
    GA_Set_data(g_before,ndim,dims,MT_DBL);
    GA_Set_chunk(g_before,chunk);
    GA_Set_pgroup(g_before,pg_world);
    status = GA_Allocate(g_before);
    if(status == 0){
    	if (me == 0) printf("%s: GA_Allocate failed at line %d\n",__FILE__,__LINE__);
    };
    
    g_trans1  = GA_Duplicate(g_before,"reference transpose");
    if(g_trans1 == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    g_trans2  = GA_Duplicate(g_before,"test transpose");
    if(g_trans2 == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

    g_error  = GA_Duplicate(g_before,"error");
    if(g_error == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

#ifdef DEBUG
    GA_Zero(g_before);
    GA_Zero(g_trans1);
    GA_Zero(g_trans2);
    GA_Zero(g_error);
    if (me == 0) printf("\n");
    GA_Print_distribution(g_before);
    if (me == 0) printf("\n");
#endif

/*
 * begin initialization with random values using local access
 */

    NGA_Distribution(g_before,me,lo,hi);
    NGA_Access(g_before,lo,hi,&p_in,&ld[0]);

    range[0] = hi[0] - lo[0] + 1;

//    double scale = 0.001/sqrt(RAND_MAX);

    for(i=0; i<range[0]; i++){
    	for(j=0; j<range[1]; j++){
//    		p_in[ ld[0] * i + j ] = (double) ( rand() * scale );
    		p_in[ ld[0] * i + j ] = (double) ( 1000 * me + ld[0] * i + j );
    	}
    }

    NGA_Release_update(g_before,lo,hi); /* this function does nothing as of GA 4.2 */

#ifdef DEBUG
	if (dims[0]<100) GA_Print(g_before);
#endif

/*
 * end initialization
 */

/*
 * GA reference transposition
 */

    GA_Transpose(g_before,g_trans1);

#ifdef DEBUG
    if (dims[0]<100) GA_Print(g_trans1);
#endif

/*
 * begin hand-written transposition
 */

    int lo2[ndim];
    int hi2[ndim];
    int range2[ndim];

    int blksz;
    if (dims[0] > 1000){
    	blksz = 100;
    } else if(dims[0] > 100){
		blksz = 20;
    } else {
    	blksz = 1;
    }

    p_buf1 = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    p_buf2 = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));

    NGA_Get(g_before,lo,hi,p_buf1,ld);

    lo2[0] = lo[1];
    lo2[1] = lo[0];
    hi2[0] = hi[1];
    hi2[1] = hi[0];

    range2[0] = hi2[0] - lo2[0] + 1;

    NGA_Get(g_before,lo2,hi2,p_buf2,ld);

    GA_Sync();

    status = ARMCI_Free_local(p_buf2);
    if(status != 0){
    	if (me == 0) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };
    status = ARMCI_Free_local(p_buf1);
    if(status != 0){
    	if (me == 0) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    };

#ifdef DEBUG
    if (dims[0]<100) GA_Print(g_trans2);
#endif

/*
 * end hand-written transposition
 */

/*
 * begin error evaluation
 */

    double alpha,beta,error;

    alpha = 1.0;
    beta = -1.0;
    GA_Add(&alpha,g_trans1,&beta,g_trans2,g_error);

    GA_Norm1(g_error,&error);

    printf("error = %f\n",error);


/*
 * end error evaluation
 */

    GA_Destroy(g_error);
    GA_Destroy(g_trans2);
    GA_Destroy(g_trans1);
    GA_Destroy(g_before);

    return(0);
}

void transpose_patch(double* input, double* output){

    





}


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
    int d,i,j; //
    int g_a,g_b,g_c; // GA handles
    int status;
    const int ndim=2;
    int dims[ndim];
    int chunk[ndim];
    int lo[ndim];
    int hi[ndim];
    int range[ndim];
    int ld[ndim-1];
    int pg_world; // world processor group
    double* p_in,p_out; // pointers for local access to GAs

    me=GA_Nodeid();

    dims[0] = 600;
    dims[1] = dims[0];
    chunk[0] = -1;
    chunk[1] = chunk[0];

    pg_world = GA_Pgroup_get_world();

    g_a = GA_Create_handle();
    GA_Set_array_name(g_a,"input and in-place transpose");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,pg_world);
    status = GA_Allocate(g_a);
    if(status == 0){
    	if (me == 0) printf("%s: GA_Allocate failed at line %d\n",__FILE__,__LINE__);
    };
    
    g_b  = GA_Duplicate(g_a,"reference transpose");
    if(g_b == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };
    g_c  = GA_Duplicate(g_a,"difference");
    if(g_c == 0){
    	if (me == 0) printf("%s: GA_Duplicate failed at line %d\n",__FILE__,__LINE__);
    };

#ifdef DEBUG
    GA_Zero(g_a);
    GA_Zero(g_b);
    GA_Zero(g_c);
    if (me == 0) printf("\n");
    GA_Print_distribution(g_a);
    if (me == 0) printf("\n");
    GA_Print_distribution(g_b);
    if (me == 0) printf("\n");
    GA_Print_distribution(g_c);
    if (me == 0) printf("\n");
#endif

/*
 * begin nitialization with random values using local access
 */

    NGA_Distribution(g_a,me,lo,hi);
    NGA_Access(g_a,lo,hi,&p_in,&ld[0]);

    for(d=0; d<ndim; d++){
        range[d] = hi[d] - lo[d] + 1;
    }

    double scale = 0.001/sqrt(RAND_MAX);

    for(i=0; i<range[0]; i++){
    	for(j=0; j<range[1]; j++){
    		p_in[ ld[0] * i + j ] = (double) ( rand() * scale );
    	}
    }

    NGA_Release_update(g_a,lo,hi); /* this function does nothing as of GA 4.2 */

#ifdef DEBUG
	if (dims[0]<100) GA_Print(g_a);
#endif

/*
 * end initialization
 */

/*
 * GA out-of-place transposition
 */

    GA_Transpose(g_a,g_b);

#ifdef DEBUG
    if (dims[0]<100) GA_Print(g_b);
#endif


/*
 * begin in-place transposition
 */



/*
 * end in-place transposition
 */

/*
 * begin error evaluation
 */

    double alpha,beta,error;

    alpha = 1.0;
    beta = -1.0;
    GA_Add(&alpha,g_a,&beta,g_b,g_c);

    GA_Norm1(g_c,&error);

    printf("error = %f\n",error);


/*
 * end error evaluation
 */

    GA_Destroy(g_c);
    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}

void transpose_patch(double* input, double* output){

    





}


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
    int d,i,j;
    int g_a;
    int status;
    const int ndim=2;
    int dims[ndim];
    int chunk[ndim];
    int lo[ndim];
    int hi[ndim];
    int range[ndim];
    int ld[ndim-1];
    int pg_world;
    double* p_in,p_out;

    dims[0] = 6;
    dims[1] = 8;
    chunk[0] = -1;
    chunk[1] = -1;

    pg_world = GA_Pgroup_get_world();

    g_a = GA_Create_handle();
    GA_Set_array_name(g_a,"test array A");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,pg_world);
    status = GA_Allocate(g_a);
    if(0 != status){};
    
#ifdef DEBUG
    GA_Zero(g_a);
    GA_Print_distribution(g_a);
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
    GA_Print(g_a);
#endif

/*
 * end initialization
 */

/*
 * begin in-place transposition
 */



/*
 * end in-place transposition
 */


    GA_Destroy(g_a);

    return(0);
}

void transpose_patch(double* input, double* output){

    





}


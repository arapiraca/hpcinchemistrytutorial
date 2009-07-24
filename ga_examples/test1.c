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
 * test1:                                                                  *
 *       -demonstrates how to create a GA using the new API                *
 *       -demonstrates how to obtain local access to GA memory             *
 *                                                                         *
 ***************************************************************************/


int test1()
{
    int p,d,i,j;
    int g_a;
    int status;
    int ndim=2;
    int dims[ndim];
    int chunk[ndim];
    int lo[ndim];
    int hi[ndim];
    int range[ndim];
    int ld[ndim-1];
    int pg_world;
    int* pg_list;
    double* p_a;
    double val;

    for(i=0; i<ndim; i++){
        dims[i] = 12;
        chunk[i] = -1;
    }

    pg_list = (int*) malloc(nproc*sizeof(int));
    for(p=0; p<nproc; p++){ pg_list[p]=p; }
    pg_world = GA_Pgroup_create(pg_list,nproc);

    g_a = GA_Create_handle();
    GA_Set_array_name(g_a,"test array A");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,pg_world);
    status = GA_Allocate(g_a);
    if(0 != status){};
    
    GA_Zero(g_a);
    val = -1.0;
    GA_Fill(g_a,&val);

    NGA_Distribution(g_a,me,lo,hi);
    GA_Print_distribution(g_a);
    for(p=0; p<ndim; p++){
        printf("proc %d: lo[%d] = %d hi[%d] = %d\n",me,p,lo[p],p,hi[p]);
        fflush(stdout);
    }

    NGA_Access(g_a,lo,hi,&p_a,&ld[0]);

    for(d=0; d<ndim; d++){
        range[d] = hi[d] - lo[d] + 1;
    }

    if (ndim == 2){
        for(i=0; i<range[0]; i++){
            for(j=0; j<range[1]; j++){
                p_a[ i*ld[0]+j ] = (double)(me+1);
            }
        }
    }

    NGA_Release_update(g_a,lo,hi); /* this function does nothing as of GA 4.2 */

    GA_Print(g_a);

    GA_Destroy(g_a);

    if(0 != GA_Pgroup_destroy(pg_world)){};

    return(0);
}




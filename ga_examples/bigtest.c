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
 * bigtest:                                                                *
 *       -demonstrates how to create a GA using the new API                *
 *       -demonstrates how to obtain local access to GA memory             *
 *                                                                         *
 ***************************************************************************/

INTEGER bigtest(INTEGER rank)
{
	INTEGER me,nproc;
    //INTEGER p,d,i,j;
    INTEGER g_a;
    INTEGER status;
    INTEGER ndim=1;
    INTEGER dims[ndim];
    INTEGER chunk[ndim];
    INTEGER pg_world;
    //INTEGER* pg_list;
    double val;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = pow(2,rank);
    chunk[0] = -1;

    //pg_list = (INTEGER*) malloc(nproc*sizeof(INTEGER));
    //for(p=0; p<nproc; p++){ pg_list[p]=p; }
    //pg_world = GA_Pgroup_create(pg_list,nproc);
    /* This is an easier way to get the world group */
    pg_world = GA_Pgroup_get_world();

    g_a = GA_Create_handle();
    GA_Set_array_name(g_a,"test array A");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,pg_world);
    status = GA_Allocate(g_a);
    if(0 != status){ GA_Error("GA_Allocate failed",0); };
    
    GA_Zero(g_a);
    val = -1.0;
    GA_Fill(g_a,&val);

    GA_Print_distribution(g_a);

    GA_Destroy(g_a);

    /* cannot do this if using pg_world = GA_Pgroup_get_world() */
    //if(0 != GA_Pgroup_destroy(pg_world)){};

    return(0);
}




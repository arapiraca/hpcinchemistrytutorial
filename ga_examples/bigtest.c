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


/***************************************************************************
 *                                                                         *
 * bigtest:                                                                *
 *       -demonstrates how to create a GA using the new API                *
 *       -demonstrates how to obtain local access to GA memory             *
 *                                                                         *
 ***************************************************************************/

int bigtest(int rank)
{
	int me,nproc;
    //int p,d,i,j;
    int g_a;
    int status;
    int ndim=1;
    int dims[ndim];
    int chunk[ndim];
    int pg_world;
    //int* pg_list;
    double val;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = pow(2,rank);
    chunk[0] = -1;

    //pg_list = (int*) malloc(nproc*sizeof(int));
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




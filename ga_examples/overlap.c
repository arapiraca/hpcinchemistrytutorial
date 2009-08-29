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
 * overlap:                                                                *
 *       -tests how well GA allows overlap of comm. and comp.	           *
 *                                                                         *
 ***************************************************************************/

unsigned long long int getticks();

void delay(unsigned long long delay_ticks)
{
  unsigned long long start, end;

  start = getticks();
  end = start + delay_ticks;

  while (start < end)
    start = getticks();
}

int overlap(int len)
{
	int me,nproc,status;
    int g_a,g_b; // GA handles
    int ndim = 1;
    int dims[1];
    int chunk[1];
    int lo_a[1],lo_b[1];
    int hi_a[1],hi_b[1];
    int ld_a[1],ld_b[1];
    unsigned long long int temp;
    unsigned long long int t_get_a; // timers
    unsigned long long int t_get_b; // timers
    double* p_a;  // pointers for local access to GAs
    double* p_b;  // pointers for local access to GAs
    ga_nbhdl_t nbh;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = len*nproc;
    chunk[0] = len;

    if (me == 0) printf("Process %5d: running overlap with length %d vector\n",me,len);

    g_a= GA_Create_handle();
    GA_Set_array_name(g_a,"matrix A");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,GA_Pgroup_get_world());

    status = GA_Allocate(g_a);
    if ((status != 1) && (me == 0)) printf("%s: GA_Allocate failed at line %d with status %d\n",__FILE__,__LINE__,status);
    g_b = GA_Duplicate(g_a,"matrix B");
    if ((g_b == 0) && (me == 0)) printf("%s: GA_Duplicate failed at line %d with status %d\n",__FILE__,__LINE__,g_b);
    GA_Zero(g_a);
    GA_Zero(g_b);
    GA_Sync();
    GA_Print_distribution(g_a);
    fflush(stdout);

    start = getticks(); 

    p_a = (double *)ARMCI_Malloc_local((armci_size_t) len * sizeof(double));
    p_b = (double *)ARMCI_Malloc_local((armci_size_t) len * sizeof(double));

    if (me == 0) printf("\nProcess %5d: doing the blocking version...\n",me);
    GA_Sync();
    if (me == 0){
        lo_a[0] = (nproc - 1) * len;
        hi_a[0] = (nproc * len - 1);
    } else {
        lo_a[0] = (me - 1) * len;
        hi_a[0] = (me * len - 1);
    }

    temp = getticks(); 
    NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
    t_get_a = getticks() - temp;
    printf("t_get_a = %lld\n",t_get_a);

    GA_Sync();
    printf("Process %5d: lo_a [0] = %12d hi_a [0] = %12d\n",me,lo_a[0],hi_a[0]); fflush(stdout);
    GA_Sync();
    printf("Process %5d: NGA_Get+delay total time = %12lld\n",me,t_get_a); fflush(stdout);
    GA_Sync();

    if (me == 0) printf("\nProcess %d: doing the non-blocking version...\n",me);
    GA_Sync();
    if (me == 0){
        lo_b[0] = (nproc - 1) * len;
        hi_b[0] = (nproc * len - 1);
    } else {
        lo_b[0] = (me - 1) * len;
        hi_b[0] = (me * len - 1);
    }

    temp = getticks();
    NGA_NbGet(g_b,lo_b,hi_b,p_b,ld_b,&nbh);
    NGA_NbWait(&nbh);
    t_get_b = getticks() - temp;
    printf("t_get_b = %lld\n",t_get_b);

    GA_Sync();
    printf("Process %5d: lo_b [0] = %12d hi_b [0] = %12d\n",me,lo_b[0],hi_b[0]); fflush(stdout);
    GA_Sync();
    printf("Process %5d: NGA_NbGet+delay total time = %12lld\n",me,t_get_b); fflush(stdout);
    GA_Sync();

    if ((ARMCI_Free_local(p_b) != 0) && (me == 0)) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    if ((ARMCI_Free_local(p_a) != 0) && (me == 0)) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    fflush(stdout);

    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}

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
 * overlap:                                                                *
 *       -tests how well GA allows overlap of comm. and comp.	           *
 *                                                                         *
 ***************************************************************************/

#define REPS 28

void delay(unsigned long long delay_ticks)
{
  unsigned long long start, end;

  start = DCMF_Timebase();
  end = start + delay_ticks;

  while (start < end)
    start = DCMF_Timebase();
}

int overlap(int len)
{
    int i;
	int me,nproc,status;
    int g_a,g_b; // GA handles
    int ndim = 1;
    int dims[1];
    int chunk[1];
    int lo_a[1],lo_b[1];
    int hi_a[1],hi_b[1];
    int ld_a[1],ld_b[1];
    unsigned long long int tt, cp, cm, t0, t1;
    unsigned long long int delays[ REPS ];
    double ov;
    double* p_a;  // pointers for local access to GAs
    double* p_b;  // pointers for local access to GAs
    ga_nbhdl_t nbh;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    /* setup delays */
    delays[0] = 0;
    for ( i = 1; i < REPS; i++ )
    {
        delays[i] = pow(2,i) - 1;
    }

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
    //GA_Print_distribution(g_a);

    start = DCMF_Timebase(); 

    p_a = (double *)ARMCI_Malloc_local((armci_size_t) len * sizeof(double));
    p_b = (double *)ARMCI_Malloc_local((armci_size_t) len * sizeof(double));

    if (me == 0) printf("\nProcess %5d: doing the BLOCKING version with length %d vector\n",me,len);
    GA_Sync();

    if (me == 0){
        lo_a[0] = (nproc - 1) * len;
        hi_a[0] = (nproc * len - 1);
    } else {
        lo_a[0] = (me - 1) * len;
        hi_a[0] = (me * len - 1);
    }

    for ( i = 0; i < REPS; i++ ){

        t0 = DCMF_Timebase(); 
        NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
        delay( delays[i] );
        t1 = DCMF_Timebase();

        if (me == 0){
            tt = t1 - t0;
            cp = delays[i];
            cm = tt - cp;
            ov = (double)cp / (double)tt;
            printf("BLOCKING %5d: comp, comm, total, ratio:  %16lld  %16lld  %16lld  %18.8lf\n",me,cp,cm,tt,ov);
        }
        fflush(stdout);
        GA_Sync();
    }

    //printf("Process %5d: lo_a [0] = %12d hi_a [0] = %12d\n",me,lo_a[0],hi_a[0]); fflush(stdout);
    //GA_Sync();

    if (me == 0) printf("\nProcess %5d: doing the NONBLOCK version with length %d vector\n",me,len);
    GA_Sync();

    if (me == 0){
        lo_b[0] = (nproc - 1) * len;
        hi_b[0] = (nproc * len - 1);
    } else {
        lo_b[0] = (me - 1) * len;
        hi_b[0] = (me * len - 1);
    }

    for ( i = 0; i < REPS; i++ ){

        t0 = DCMF_Timebase();
        NGA_NbGet(g_b,lo_b,hi_b,p_b,ld_b,&nbh);
        delay( delays[i] );
        NGA_NbWait(&nbh);
        t1 = DCMF_Timebase();

        if (me == 0){
            tt = t1 - t0;
            cp = delays[i];
            cm = tt - cp;
            ov = (double)cp / (double)tt;
            printf("NONBLOCK %5d: comp, comm, total, ratio:  %16lld  %16lld  %16lld  %18.8lf\n",me,cp,cm,tt,ov);
        }
        fflush(stdout);
        GA_Sync();
    }

    //printf("Process %5d: lo_b [0] = %12d hi_b [0] = %12d\n",me,lo_b[0],hi_b[0]); fflush(stdout);
    //GA_Sync();

    if ((ARMCI_Free_local(p_b) != 0) && (me == 0)) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    if ((ARMCI_Free_local(p_a) != 0) && (me == 0)) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    fflush(stdout);

    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}

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
 * overlap_b_ring:                                                         *
 *       -test of overlapping communication and computation for Vitali     *
 *                                                                         *
 ***************************************************************************/

#define REPS 28

unsigned long long int getticks();

int overlap_b_ring(int me, int nproc, int len)
{
    int status;
    int n, i;
    unsigned long long int delays[ REPS ];
    unsigned long long int t0, t1;
    unsigned long long int cp, cm, tt;
    double ov;
    armci_hdl_t nb_handle;

    /* setup delays */
    delays[0] = 0;
    for ( i = 1; i < REPS; i++ )
    {
        delays[i] = pow(2,i) - 1;
    }

    /* register remote pointers */
    double** addr_vec1 = (double **) malloc(sizeof(double *) * nproc);
    double** addr_vec2 = (double **) malloc(sizeof(double *) * nproc);
    ARMCI_Malloc((void **) addr_vec1, len*sizeof(double));
    ARMCI_Malloc((void **) addr_vec2, len*sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);

    /* initialization of local segments */
    for( i=0 ; i<len ; i++ ){
       addr_vec1[me][i] = (double) +1*(1000*me+i);    
    }
    for( i=0 ; i<len ; i++ ){
       addr_vec2[me][i] = (double) -1*(1000*me+i);    
    }

#ifdef DEBUG
    /* print before exchange */
    for( n=0 ; n<nproc ; n++){
       MPI_Barrier(MPI_COMM_WORLD);
       if (n==me){
          printf("values before exchange\n");
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec1[%d][%d] = %f\n", n, n, i, addr_vec1[n][i]);
          }
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec2[%d][%d] = %f\n", n, n, i, addr_vec2[n][i]);
          }
          fflush(stdout);
       }
       MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    for ( i = 0; i < REPS; i++ ){

        MPI_Barrier(MPI_COMM_WORLD);

        t0 = getticks();

        if (me == (nproc-1) ){

           status = ARMCI_Get(addr_vec1[0], addr_vec2[nproc-1], len*sizeof(double), 0);
           delay( delays[i] );

        } else {

           status = ARMCI_Get(addr_vec1[me+1], addr_vec2[me], len*sizeof(double), me+1);
           delay( delays[i] );

        }

        if((status != 0) && (me == 0)) printf("%s: ARMCI_Get failed at line %d\n",__FILE__,__LINE__);

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = getticks();

        if (me == 0){
           //printf("Iter %6d Proc %6d: (t0,t1) = %16lld %16lld\n",i,me,t0,t1);
           tt = t1 - t0;
           cp = delays[i];
           cm = tt - cp;
           ov = (double)cp / (double)(tt);
           printf("BLOCKING %6d: comp, comm, total, ratio:  %16lld  %16lld  %16lld  %18.8lf\n", me, cp, cm, tt, ov );
        }
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG
    /* print after exchange */
    for( n=0 ; n<nproc ; n++){
       MPI_Barrier(MPI_COMM_WORLD);
       if (n==me){
          printf("values after exchange\n");
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec1[%d][%d] = %f\n", n, n, i, addr_vec1[n][i]);
          }
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec2[%d][%d] = %f\n", n, n, i, addr_vec2[n][i]);
          }
          fflush(stdout);
       }
       MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    return(0);
}

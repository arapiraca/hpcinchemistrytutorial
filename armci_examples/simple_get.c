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
 * simple_get:                                                             *
 *       -demonstrates how to allocate some shared segements with ARMCI    *
 *       -demonstrates how to do one-sided point-to-point communication    *
 *       -inspired by armci/examples/features/non-blocking/simple/simple.c *
 *                                                                         *
 ***************************************************************************/

int simple_get(int me, int nproc, int len)
{
    int status;
    int n,i;
    double t0,t1;

    double** addr_vec = (double **) malloc( len * sizeof(double *) );
    ARMCI_Malloc((void **) addr_vec, len*sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);

    /* initialization of local segments */
    for( i=0 ; i<len ; i++ ){
       addr_vec[me][i] = (double) (1000*me+i);    
    }

    /* print before exchange */
    for( n=0 ; n<nproc ; n++){
       MPI_Barrier(MPI_COMM_WORLD);
       if (n==me){
          printf("values before exchange\n");
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec[%d][%d] = %f\n", n, n, i, addr_vec[n][i]);
          }
          fflush(stdout);
       }
       MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* even processes get from odd right neighbors */
    if (me%2 == 0){
       t0 = MPI_Wtime();
       status = ARMCI_Get(addr_vec[me+1], addr_vec[me], len*sizeof(double), me+1);
       t1 = MPI_Wtime();
       if(status != 0){
    	  if (me == 0) printf("%s: ARMCI_Get failed at line %d\n",__FILE__,__LINE__);
       }
       printf("Proc %d: Get Latency=%lf microseconds\n",me,1e6*(t1-t0)/len);
       fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    /* print after exchange */
    for( n=0 ; n<nproc ; n++){
       MPI_Barrier(MPI_COMM_WORLD);
       if (n==me){
          printf("values after exchange\n");
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec[%d][%d] = %f\n", n, n, i, addr_vec[n][i]);
          }
          fflush(stdout);
       }
       MPI_Barrier(MPI_COMM_WORLD);
    }

    return(0);
}

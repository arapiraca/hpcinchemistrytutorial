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

int main(int argc, char **argv)
{
	int me,nproc;
    int status;
    int rank;

    /* initialization */
    MPI_Init(&argc, &argv);
    ARMCI_Init();

#ifdef HPC_PROFILING
    HPM_Init();
#endif

    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

#ifdef DEBUG
    if(me == 0){
       printf("The result of MPI_Comm_size is %d\n",nproc);
       fflush(stdout);
    }
#endif

    /* get the matrix parameters */
    if (argc > 1){
        rank = atoi(argv[1]);
    } else {
        rank = 8;
    }
    if (me == 0){
        printf("Running matmul.x with rank = %d\n",rank);
        fflush(stdout);
    }

    /* register remote pointers */
    double** addr_A = (double **) malloc(sizeof(double *) * nproc);
    if (addr_A == NULL) ARMCI_Error("malloc A failed at line",0);

    double** addr_B = (double **) malloc(sizeof(double *) * nproc);
    if (addr_B == NULL) ARMCI_Error("malloc B failed at line",0);

    double** addr_C = (double **) malloc(sizeof(double *) * nproc);
    if (addr_C == NULL) ARMCI_Error("malloc C failed at line",0);

#ifdef DEBUG
    if(me == 0) printf("ARMCI_Malloc A requests %lu bytes\n",rank*rank*sizeof(double));
    fflush(stdout);
#endif
    status = ARMCI_Malloc((void **) addr_A, rank*rank*sizeof(double));
    if (status != 0) ARMCI_Error("ARMCI_Malloc A failed",status);

#ifdef DEBUG
    if(me == 0) printf("ARMCI_Malloc B requests %lu bytes\n",rank*rank*sizeof(double));
    fflush(stdout);
#endif
    status = ARMCI_Malloc((void **) addr_B, rank*rank*sizeof(double));
    if (status != 0) ARMCI_Error("ARMCI_Malloc B failed",status);

#ifdef DEBUG
    if(me == 0) printf("ARMCI_Malloc C requests %lu bytes\n",rank*rank*sizeof(double));
    fflush(stdout);
#endif
    status = ARMCI_Malloc((void **) addr_C, rank*rank*sizeof(double));
    if (status != 0) ARMCI_Error("ARMCI_Malloc C failed",status);

    MPI_Barrier(MPI_COMM_WORLD);






    /* free ARMCI pointers */
    free(addr_C);
    free(addr_B);
    free(addr_A);

#ifdef HPC_PROFILING
    HPM_Print();
#endif

    /* the end */
    ARMCI_Finalize();
    MPI_Finalize();

    return(0);
}




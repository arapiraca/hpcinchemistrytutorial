/***************************************************************************

                COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
+ 2009 University of Chicago

Permission is hereby granted to use, reproduce, prepare derivative works, and
to redistribute to others.  This software was authored by:

Jeff R. Hammond
Leadership Computing Facility
Argonne National Laboratory
Argonne IL 60439 USA
phone: (630) 252-5381
e-mail: jhammond@mcs.anl.gov

                GOVERNMENT LICENSE

Portions of this material resulted from work developed under a U.S.
Government Contract and are subject to the following license: the Government
is granted for itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable worldwide license in this computer software to reproduce, prepare
derivative works, and perform publicly and display publicly.

                DISCLAIMER

This computer code material was prepared, in part, as an account of work
sponsored by an agency of the United States Government.  Neither the United
States, nor the University of Chicago, nor any of their employees, makes any
warranty express or implied, or assumes any legal liability or responsibility
for the accuracy, completeness, or usefulness of any information, apparatus,
product, or process disclosed, or represents that its use would not infringe
privately owned rights.

***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

#ifdef GA
  #include "../armci/src/armci.h"
#endif

#ifdef MPI
  #include "mpi.h"
#endif

void sgemm_(char* , char* ,int* , int* , int* , float* , float* , int* , float* , int* , float* , float* , int* );
void dgemm_(char* , char* ,int* , int* , int* , double*, double*, int* , double*, int* , double*, double*, int* );

int main(int argc, char **argv)
{
    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);
    int me, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    ARMCI_Init();

    if (me==0 && argc!=4) printf("./new_armci_cpu_sgemm.x <m> <n> <k>\n");
//     int a;
//     if (me==0) for (a=0;a<argc;a++) printf("argv[%1d] = %s\n",a,argv[a]);

    int m  = ( argc>1 ? atoi(argv[1]) : 512 );
    int n  = ( argc>2 ? atoi(argv[2]) : 512 );
    int k  = ( argc>3 ? atoi(argv[3]) : 512 );

    printf("m=%4d   n=%d   k=%4d\n",m,n,k);

    int sizeA = m*k;
    int sizeB = k*n;
    int sizeC = m*n;

    float** winA = (float**) malloc( nproc*sizeof(void*) ); assert(winA!=NULL);
    float** winB = (float**) malloc( nproc*sizeof(void*) ); assert(winB!=NULL);
    float** winC = (float**) malloc( nproc*sizeof(void*) ); assert(winC!=NULL);

    int status;
    status = ARMCI_Malloc( (void**) winA, sizeA*sizeof(float) ); assert(status==0);
    status = ARMCI_Malloc( (void**) winB, sizeB*sizeof(float) ); assert(status==0);
    status = ARMCI_Malloc( (void**) winC, sizeC*sizeof(float) ); assert(status==0);
    ARMCI_Barrier();

    double* bufA = (double*) ARMCI_Malloc_local( sizeA*sizeof(double) ); assert(bufA!=NULL);
    double* bufB = (double*) ARMCI_Malloc_local( sizeB*sizeof(double) ); assert(bufB!=NULL);
    double* bufC = (double*) ARMCI_Malloc_local( sizeC*sizeof(double) ); assert(bufC!=NULL);

    int i;
    for (i=0;i<sizeA;i++) bufA[i]=(float)  i;
    for (i=0;i<sizeB;i++) bufB[i]=(float) -i;
    for (i=0;i<sizeC;i++) bufC[i]=(float)  0;

    status = ARMCI_Put(bufA, winA[me], sizeA*sizeof(float), me); assert(status==0);
    status = ARMCI_Put(bufB, winB[me], sizeB*sizeof(float), me); assert(status==0);
    status = ARMCI_Put(bufC, winC[me], sizeC*sizeof(float), me); assert(status==0);
    ARMCI_AllFence();





    status = ARMCI_Free_local(bufA); assert(status==0);
    status = ARMCI_Free_local(bufB); assert(status==0);
    status = ARMCI_Free_local(bufC); assert(status==0);

    status = ARMCI_Free(winA[me]); assert(status==0);
    status = ARMCI_Free(winB[me]); assert(status==0);
    status = ARMCI_Free(winC[me]); assert(status==0);
    ARMCI_Barrier();

    ARMCI_Finalize();
    MPI_Finalize();
    return(0);
}

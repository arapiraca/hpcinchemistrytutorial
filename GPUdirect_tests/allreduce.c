#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mpi.h"

#if USE_CUDA
    #include "cuda.h"
    #include "cuda_runtime.h"
#endif

int main(int argc, char* argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i;
    int n = ( argc>1 ? atoi(argv[1]) : 1000 );
    if (rank==0) printf("testing MPI_Allreduce for dimension %d\n",n);

    double t0, t1;

    double* a;
    double* b;

    t0 = MPI_Wtime();
#ifdef USE_MALLOC
    if (rank==0) printf("using malloc\n");
    a = (double *) malloc(n*sizeof(double)); assert(a!=NULL);
    b = (double *) malloc(n*sizeof(double)); assert(b!=NULL);
#elif USE_MEMALIGN
    if (rank==0) printf("using posix_memalign\n");
    int status;
    status = posix_memalign((void**)&a,4096,n*sizeof(double)); assert(status==0 && a!=NULL);
    status = posix_memalign((void**)&b,4096,n*sizeof(double)); assert(status==0 && b!=NULL);
#elif USE_CUDA
    if (rank==0) printf("using cudaMallocHost\n");
    cudaError_t cuStatus;
    cuStatus = cudaMallocHost((void**)&a,n*sizeof(double)); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaMallocHost((void**)&b,n*sizeof(double)); assert(cuStatus==CUDA_SUCCESS);
#else
    #error You need to set one of {USE_MALLOC, USE_MEMALIGN, USE_CUDA}!
#endif
    t1 = MPI_Wtime();
    if (rank==0) printf("allocation time = %10.5f\n",t1-t0);

    for (i=0; i<n; i++) a[i] = (double)i;
    for (i=0; i<n; i++) b[i] = (double)0;

#ifdef DEBUG
    if (rank==0) for (i=0; i<n; i++) printf("a[%4d] = %30.15f\n",i,a[i]);
    if (rank==0) for (i=0; i<n; i++) printf("b[%4d] = %30.15f\n",i,b[i]);
#endif

    t0 = MPI_Wtime();
    MPI_Allreduce(a,b,n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    if (rank==0) printf("MPI_Allreduce time = %10.5f\n",t1-t0);

#ifdef DEBUG
    if (rank==0) for (i=0; i<n; i++) printf("a[%4d] = %30.15f\n",i,a[i]);
    if (rank==0) for (i=0; i<n; i++) printf("b[%4d] = %30.15f\n",i,b[i]);
#else
    for (i=0; i<n; i++) assert(b[i]==(size*a[i]));
    if (rank==0) printf("the result is correct\n");
#endif

#ifdef USE_MALLOC
    free(a);
    free(b);
#elif USE_MEMALIGN
    free(a);
    free(b);
#elif USE_CUDA
    cuStatus = cudaFreeHost(a); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaFreeHost(b); assert(cuStatus==CUDA_SUCCESS);
#else
    #error You need to set one of {USE_MALLOC, USE_MEMALIGN, USE_CUDA}!
#endif

    MPI_Finalize();

    return 0;
}


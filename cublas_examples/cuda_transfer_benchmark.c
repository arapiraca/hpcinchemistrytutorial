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

#ifdef CUDA
  #include "cuda.h"
  #include "cuda_runtime.h"
  #include "cublas.h"
#endif

#ifdef OPENMP
  #include "omp.h"
#else
  #include <time.h>
#endif

inline double gettime(void)
{
#ifdef OPENMP
    return omp_get_wtime();
#else
    return (double) time(NULL);
#endif
}

unsigned long long getticks(void);

int main(int argc, char **argv)
{
    printf("Testing CUDA transfer bandwidth...\n");

    int status;
    cudaError_t cuStatus;

    double start, finish, time1, time2, bw1, bw2;

    size_t bufSize;
    size_t maxSize = ( argc>1 ? atoi(argv[1]) : 1 );

    unsigned char* h_ptr;
    unsigned char* d_ptr;

    size_t alignment = getpagesize();

    for (size_t i=1; pow(2,i)<=maxSize; i++){
        bufSize=pow(2,i);

        status = posix_memalign(&h_ptr, alignment, bufSize);
        assert(status==0 && h_ptr!=NULL);

        cuStatus = cudaMalloc((void**)&d_ptr,bufSize);
        assert(cuStatus==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ d_ptr, /* src */ h_ptr, bufSize, cudaMemcpyHostToDevice);
        finish = gettime();
        assert(status==CUDA_SUCCESS);

        time1 = finish - start;
        bw1 = (double) bufSize;
        bw1 /= time1;
        bw1 /= (1024*1024);

        cuStatus = cudaMemset(d_ptr,2,bufSize);
        assert(status==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ h_ptr, /* src */ d_ptr, bufSize, cudaMemcpyDeviceToHost);
        finish = gettime();
        assert(status==CUDA_SUCCESS);

        time2 = finish - start;
        bw2 = (double) bufSize;
        bw2 /= time2;
        bw2 /= (1024*1024);
        printf("%10lu bytes IN: %8.6f seconds (%8.3f MB/s) OUT: %8.6f seconds (%8.3f MB/s)\n",
                bufSize,        time1,         bw1,             time2,         bw2);

        cuStatus = cudaFree(d_ptr);
        assert(status==CUDA_SUCCESS);

        free(h_ptr);
    }

    for (size_t i=1; pow(2,i)<=maxSize; i++){
        bufSize=pow(2,i);

        cuStatus = cudaMallocHost((void**)&h_ptr, bufSize);
        assert(cuStatus==CUDA_SUCCESS);
        memset(h_ptr,1,bufSize);

        cuStatus = cudaMalloc((void**)&d_ptr,bufSize);
        assert(cuStatus==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ d_ptr, /* src */ h_ptr, bufSize, cudaMemcpyHostToDevice);
        finish = gettime();
        assert(status==CUDA_SUCCESS);

        time1 = finish - start;
        bw1 = (double) bufSize;
        bw1 /= time1;
        bw1 /= (1024*1024);

        cuStatus = cudaMemset(d_ptr,2,bufSize);
        assert(status==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ h_ptr, /* src */ d_ptr, bufSize, cudaMemcpyDeviceToHost);
        finish = gettime();
        assert(status==CUDA_SUCCESS);

        time2 = finish - start;
        bw2 = (double) bufSize;
        bw2 /= time2;
        bw2 /= (1024*1024);
        printf("%10lu bytes IN: %8.6f seconds (%8.3f MB/s) OUT: %8.6f seconds (%8.3f MB/s)\n",
                bufSize,        time1,         bw1,             time2,         bw2);

        cuStatus = cudaFree(d_ptr);
        assert(status==CUDA_SUCCESS);

        cuStatus = cudaFreeHost(h_ptr);
        assert(status==CUDA_SUCCESS);
    }


    printf("...all done.\n");
    return(0);
}

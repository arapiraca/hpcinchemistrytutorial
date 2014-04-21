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
int getpagesize(void);
int posix_memalign(void **memptr, size_t alignment, size_t size);

#ifdef CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <cublas.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#else
  #include <time.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

/******************************************************************
 * author: azutomo Yoshii kazutomo@mcs.anl.gov
 * from http://www.mcs.anl.gov/~kazutomo/getticks.html 
 ******************************************************************/

#if defined(__i386__)

static __inline__ unsigned long long getticks(void)
{
  unsigned long long int x;
     __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
     return x;
}

#elif defined(__x86_64__)

__inline__ unsigned long long getticks(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#elif defined(__powerpc__)

static __inline__ unsigned long long getticks(void)
{
  unsigned long long int result=0;
  unsigned long int upper, lower,tmp;
  __asm__ volatile(
                "0:                  \n"
                "\tmftbu   %0           \n"
                "\tmftb    %1           \n"
                "\tmftbu   %2           \n"
                "\tcmpw    %2,%0        \n"
                "\tbne     0b         \n"
                : "=r"(upper),"=r"(lower),"=r"(tmp)
                );
  result = upper;
  result = result<<32;
  result = result|lower;

  return(result);
}

#endif

/******************************************************************/

inline double gettime(void)
{
#if defined(MPI_VERSION) && defined(USE_MPI)
    return MPI_Wtime();
#elif defined(_OPENMP)
    return omp_get_wtime();
#elif defined(USE_TIME)
    return (double) time(NULL);
#elif defined(USE_GETTICKS)
    /* this is a terrible hack */
    return (double) getticks()/2.67e9;
#else
#warning No timer available!
#endif
}
int main(int argc, char **argv)
{
    printf("Testing CUDA transfer bandwidth...\n");

    int status;
    cudaError_t cuStatus;

    double start, finish, time1, time2, bw1, bw2;

    size_t bufSize;
    size_t maxSize = ( argc>1 ? atoi(argv[1]) : 0 );

    unsigned char* h_ptr;
    unsigned char* d_ptr;

    size_t alignment = getpagesize();

    printf("...posix_memalign...\n");
    size_t i;
    for (i=10; pow(2,i)<=maxSize; i++){
        bufSize=pow(2,i);

        status = posix_memalign((void**)&h_ptr, alignment, bufSize);
        assert(status==0 && h_ptr!=NULL);

        cuStatus = cudaMalloc((void**)&d_ptr,bufSize);
        assert(cuStatus==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ d_ptr, /* src */ h_ptr, bufSize, cudaMemcpyHostToDevice);
        finish = gettime();
        assert(cuStatus==CUDA_SUCCESS);

        time1 = finish - start;
        bw1 = (double) bufSize;
        bw1 /= time1;
        bw1 /= (1024*1024);

        cuStatus = cudaMemset(d_ptr,2,bufSize);
        assert(cuStatus==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ h_ptr, /* src */ d_ptr, bufSize, cudaMemcpyDeviceToHost);
        finish = gettime();
        assert(cuStatus==CUDA_SUCCESS);

        time2 = finish - start;
        bw2 = (double) bufSize;
        bw2 /= time2;
        bw2 /= (1024*1024);
        printf("%10lu KiB IN: %8.6f seconds ( %8.3f MiB/s) OUT: %8.6f seconds ( %8.3f MiB/s)\n",
                bufSize>>10,  time1,         bw1,              time2,         bw2);

        cuStatus = cudaFree(d_ptr);
        assert(cuStatus==CUDA_SUCCESS);

        free(h_ptr);
    }

    printf("...cudaMallocHost...\n");
    for (i=10; pow(2,i)<=maxSize; i++){
        bufSize=pow(2,i);

        cuStatus = cudaMallocHost((void**)&h_ptr, bufSize);
        assert(cuStatus==CUDA_SUCCESS);
        memset(h_ptr,1,bufSize);

        cuStatus = cudaMalloc((void**)&d_ptr,bufSize);
        assert(cuStatus==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ d_ptr, /* src */ h_ptr, bufSize, cudaMemcpyHostToDevice);
        finish = gettime();
        assert(cuStatus==CUDA_SUCCESS);

        time1 = finish - start;
        bw1 = (double) bufSize;
        bw1 /= time1;
        bw1 /= (1024*1024);

        cuStatus = cudaMemset(d_ptr,2,bufSize);
        assert(cuStatus==CUDA_SUCCESS);

        start = gettime();
        cuStatus = cudaMemcpy(/* dst */ h_ptr, /* src */ d_ptr, bufSize, cudaMemcpyDeviceToHost);
        finish = gettime();
        assert(cuStatus==CUDA_SUCCESS);

        time2 = finish - start;
        bw2 = (double) bufSize;
        bw2 /= time2;
        bw2 /= (1024*1024);
        printf("%10lu KiB IN: %8.6f seconds ( %8.3f MiB/s) OUT: %8.6f seconds ( %8.3f MiB/s)\n",
                bufSize>>10,  time1,         bw1,              time2,         bw2);


        cuStatus = cudaFree(d_ptr);
        assert(cuStatus==CUDA_SUCCESS);

        cuStatus = cudaFreeHost(h_ptr);
        assert(cuStatus==CUDA_SUCCESS);
    }

    printf("...all done.\n");
    return(0);
}

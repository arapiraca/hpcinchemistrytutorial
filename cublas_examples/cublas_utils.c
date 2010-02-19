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

#include "cublas_utils.h"

#ifdef CUDA

void start_cublas(int printMask)
{
//     static int cublas_instances = 0;
// 
//     if ( cublas_instances == 0 )
//     {
//         printf("CUBLAS initialized\n");
// 
//         cublasStatus status;
//         status = cublasInit();
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             printf("failure at line %d of %s\n",__LINE__,__FILE__);
//             printf("CUBLAS initialization failed\n");
//             fflush(stdout);
//         }
//         cublas_instances = 1;
//     } else {
//         cublas_instances += 1;
//     }
    cublasStatus status;

    status = cublasInit();
    if (status == CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"! cublasInit succeeded\n");
    } else {
        fprintf(stderr,"! failure at line %d of %s\n",__LINE__,__FILE__);
        fprintf(stderr,"! cublasInit failed\n");
    }
    fflush(stderr);

    if (printMask==0)
    {
        int cudaDevice;
        struct cudaDeviceProp cudaProp;
        cudaGetDevice( &cudaDevice );
        cudaGetDeviceProperties( &cudaProp, cudaDevice );
        if ( cudaProp.major==1 && cudaProp.minor==0 ){
            fprintf(stderr,"! CUDA device does not support double-precision\n");
            fflush(stderr);
        }

        printf("=========================================================\n");
        printf("CUDA device properties:\n");
        printf("name:                 %20s\n",cudaProp.name);
        printf("major version:        %20d\n",cudaProp.major);
        printf("minor version:        %20d\n",cudaProp.minor);
        printf("canMapHostMemory:     %20d\n",cudaProp.canMapHostMemory);
        printf("totalGlobalMem:       %20lu MiB\n",cudaProp.totalGlobalMem/(1024*1024));
        printf("sharedMemPerBlock:    %20lu\n",cudaProp.sharedMemPerBlock);
        printf("clockRate:            %20.3f GHz\n",cudaProp.clockRate/1.0e6); /* kHz is base unit */
        printf("regsPerBlock:         %20d\n",cudaProp.regsPerBlock);
        printf("warpSize:             %20d\n",cudaProp.warpSize);
        printf("maxThreadsPerBlock:   %20d\n",cudaProp.maxThreadsPerBlock);
        printf("=========================================================\n");
    }
}

void stop_cublas(void)
{
//     static int cublas_instances = 0;
// 
//     if ( cublas_instances == 0 )
//     {
//         printf("CUBLAS terminated\n");
// 
//         cublasStatus status;
//         status = cublasShutdown();
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             printf("failure at line %d of %s\n",__LINE__,__FILE__);
//             printf("CUBLAS shutdown failed\n");
//             fflush(stdout);
//         }
//     } else {
//         cublas_instances -= 1;
//     }

    cublasStatus status;
    status = cublasShutdown();
    if (status == CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"! cublasShutdown succeeded\n");
    } else {
        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cublasShutdown failed\n");
        fflush(stdout);
    }

}

float* alloc_device_floats(int num)
{
    cublasStatus status;
    float* ptr;
    status = cublasAlloc(num, sizeof(float), (void**)&ptr);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cannot allocate %d floats on device\n",num);
        fflush(stdout);
        return NULL;
    }
    return ptr;
}

double* alloc_device_doubles(int num)
{
    cublasStatus status;
    double* ptr;
    status = cublasAlloc(num, sizeof(double), (void**)&ptr);
    assert(status==CUBLAS_STATUS_SUCCESS);
    return ptr;
}

void free_device_floats(float* ptr)
{
    cublasStatus status;
    status = cublasFree(ptr);
    assert(status==CUBLAS_STATUS_SUCCESS);
}

void free_device_doubles(double* ptr)
{
    cublasStatus status;
    status = cublasFree(ptr);
    assert(status==CUBLAS_STATUS_SUCCESS);
}

/*
 * NOTATION
 * h = host
 * d = device
 */
void push_floats(int num, float* h_ptr, float* d_ptr)
{
    cublasStatus status;
    status = cublasSetVector(num, sizeof(float), h_ptr, 1, d_ptr, 1);
    assert(status==CUBLAS_STATUS_SUCCESS);
}

void push_doubles(int num, double* h_ptr, double* d_ptr)
{
    cublasStatus status;
    status = cublasSetVector(num, sizeof(double), h_ptr, 1, d_ptr, 1);
    assert(status==CUBLAS_STATUS_SUCCESS);
}

void pull_floats(int num, float* h_ptr, float* d_ptr)
{
    cublasStatus status;
    status = cublasGetVector(num, sizeof(float), d_ptr, 1, h_ptr, 1);
    assert(status==CUBLAS_STATUS_SUCCESS);
}

void pull_doubles(int num, double* h_ptr, double* d_ptr)
{
    cublasStatus status;
    status = cublasGetVector(num, sizeof(double), d_ptr, 1, h_ptr, 1);
    assert(status==CUBLAS_STATUS_SUCCESS);
}

#else

void start_cublas(int printMask)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

void stop_cublas(void)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

float* alloc_device_floats(int num)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

double* alloc_device_doubles(int num)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

void free_device_floats(float* ptr)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

void free_device_doubles(double* ptr)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

void push_floats(int num, float* h_ptr, float* d_ptr)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

void push_doubles(int num, double* h_ptr, double* d_ptr)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

void pull_floats(int num, float* d_ptr, float* h_ptr)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

void pull_doubles(int num, double* d_ptr, double* h_ptr)
{
    printf("! CUDA not enabled\n");
    fflush(stdout);
}

#endif

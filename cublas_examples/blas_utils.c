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

#include "blas_utils.h"

inline double gettime(void)
{
#ifdef OPENMP
    return omp_get_wtime();
#else
    return (double) time(NULL);
#endif
}

void zero_host_floats(int num, float* ptr)
{
    int i;
    for ( i = 0 ; i < num ; i++ ) ptr[i] = 0.0;
}

void zero_host_doubles(int num, double* ptr)
{
    int i;
    for ( i = 0 ; i < num ; i++ ) ptr[i] = 0.0;
}

float* alloc_host_floats(int num)
{
    float* ptr = (float*) malloc( num * sizeof(float) );
    if (ptr == NULL) {
        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cannot allocate %d floats on host\n",num);
        fflush(stdout);
        return NULL;
    }
    int i;
    zero_host_floats(num, ptr);
    return ptr;
}

double* alloc_host_doubles(int num)
{
    double* ptr = (double*) malloc( num * sizeof(double) );
    if (ptr == NULL) {
        printf("! failure at line %d of %s\n",__LINE__,__FILE__);
        printf("! cannot allocate %d doubles on host\n",num);
        fflush(stdout);
        return NULL;
    }
    int i;
    zero_host_doubles(num, ptr);
    return ptr;
}

void copy_host_floats(int num, float* in_ptr, float* out_ptr)
{
    int i;
    for ( i = 0 ; i < num ; i++ ) out_ptr[i] = in_ptr[i];
}

void copy_host_doubles(int num, double* in_ptr, double* out_ptr)
{
    int i;
    for ( i = 0 ; i < num ; i++ ) out_ptr[i] = in_ptr[i];
}

void free_host_floats(float* ptr)
{
    free(ptr);
}

void free_host_doubles(double* ptr)
{
    free(ptr);
}

void randomize_floats(size_t num, float* ptr)
{
    int i;
    for ( i = 0 ; i < num ; i++ ) ptr[i] = (float) rand() / (float) RAND_MAX;
}

void randomize_doubles(size_t num, double* ptr)
{
    int i;
    for ( i = 0 ; i < num ; i++ ) ptr[i] = (double) rand() / (double) RAND_MAX;
}


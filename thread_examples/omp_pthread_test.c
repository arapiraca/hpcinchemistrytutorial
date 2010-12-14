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
#include <assert.h>
#include "pthread.h"
#include "omp.h"

#define MAX_OMP_THREADS 2
#define MAX_POSIX_THREADS 2
static pthread_t thread_pool[MAX_POSIX_THREADS];

void* foo(void* dummy)
{
    pthread_t my_pthread = pthread_self();
    int i,my_pth;

    for (i=0 ; i<MAX_POSIX_THREADS ; i++){
        if (my_pthread==thread_pool[i]) my_pth = i;
    }
    fprintf(stderr, "hello from pthread %d!\n" , my_pth );

    int my_omp, num_omp, max_omp;
    #pragma omp parallel private(my_omp,num_omp,max_omp) shared(my_pth)
    {
        my_omp  = omp_get_thread_num();
        num_omp = omp_get_num_threads();
        max_omp = omp_get_max_threads();
        fprintf(stderr,"hello from OpenMP thread %d of %d (max=%d) on pthread %d\n",my_omp,num_omp,max_omp,my_pth);
    }

    pthread_exit(0);
}

int main(int argc, char** argv)
{
    int i,rc;

    fprintf(stderr,"forcing OpenMP to use only %d threads\n",MAX_OMP_THREADS);
    omp_set_num_threads(MAX_OMP_THREADS);
    fprintf(stderr,"omp_get_max_threads() = %d\n",omp_get_max_threads());

    fprintf(stderr,"creating %d threads\n",MAX_POSIX_THREADS);
    for (i=0 ; i<MAX_POSIX_THREADS ; i++){
        rc = pthread_create(&thread_pool[i], NULL, foo, NULL);
        assert(rc==0);
    }

    for (i=0 ; i<MAX_POSIX_THREADS ; i++){
        rc = pthread_join(thread_pool[i],NULL);
        assert(rc==0);
    }
    fprintf(stderr,"joined %d threads\n",MAX_POSIX_THREADS);

    fflush(stderr);
    return(0);
}
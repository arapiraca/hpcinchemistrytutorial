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
#include <unistd.h>

#include "blas_gemm_test.h"

#ifdef CUDA
#include "cublas_gemm_test.h"
#endif

#include "ga_utils.h"

int main(int argc, char** argv)
{
    int me, nproc;

    start_parallel(&argc,&argv,&me,&nproc);

    int precision = 1;
    int rows = 10000;
    int cols = 10000;

    int g_a = alloc_global_2d(precision,rows,cols,me);
    int g_b = alloc_global_2d(precision,rows,cols,me);
    int g_c = alloc_global_2d(precision,rows,cols,me);

    zero_global(g_a);
    zero_global(g_b);
    zero_global(g_c);

    free_global(g_c);
    free_global(g_b);
    free_global(g_a);

    stop_parallel(me);

    fprintf(stderr,"# the test driver has finished!!!\n");
    fflush(stderr);

    return 0;
}

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
#include "ga_utils.h"

void sgemm_(char* , char* ,int* , int* , int* , float* , float* , int* , float* , int* , float* , float* , int* );
void dgemm_(char* , char* ,int* , int* , int* , double*, double*, int* , double*, int* , double*, double*, int* );

int main(int argc, char **argv)
{
    int me, nproc;
    int armci_not_ga = 0;
    start_parallel(&argc,&argv,&me,&nproc,armci_not_ga,0);

    if (me==0 && argc!=4) printf("./new_ga_cpu_sgemm.x <m> <n> <k>\n");
    int a;
    if (me==0) for (a=0;a<argc;a++) printf("argv[%1d] = %s\n",a,argv[a]);

    int m  = ( argc>1 ? atoi(argv[1]) : 512 );
    int n  = ( argc>2 ? atoi(argv[2]) : 512 );
    int k  = ( argc>3 ? atoi(argv[3]) : 512 );
    int blksz = 64;

    int g_a = alloc_global_2d(1, m, k, me);
    int g_b = alloc_global_2d(1, k, n, me);
    int g_c = alloc_global_2d(1, m, n, me);
    parallel_sync();

    float* l_a = alloc_host_floats(m*k);
    float* l_b = alloc_host_floats(k*n);
    float* l_c = alloc_host_floats(m*n);

    int i;
    for (i=0;i<(m*k);i++) l_a[i] = (float)i;
    local_to_global((void*)l_a,g_a);
    for (i=0;i<(k*n);i++) l_b[i] = (float)i;
    local_to_global((void*)l_b,g_b);
    for (i=0;i<(m*n);i++) l_c[i] = (float)i;
    local_to_global((void*)l_c,g_c);

    global_to_local(g_a, (void*)l_a);
    global_to_local(g_b, (void*)l_b);
    global_to_local(g_c, (void*)l_c);
//     for (i=0;i<(m*n);i++) printf("l_c[%3d] = %f\n",i,l_c[i]);
//     printf("fucking assholes\n");

//     free_host_floats(l_a);
//     free_host_floats(l_b);
//     free_host_floats(l_c);

    free_global(g_a);
    free_global(g_b);
    free_global(g_c);

    parallel_sync();
    stop_parallel(0);
    return(0);
}

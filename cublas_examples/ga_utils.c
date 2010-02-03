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

#include "ga_utils.h"
#include "cublas_utils.h"

int gethostname(char *name, size_t len);

void print_hostname(int printMask)
{
    if (printMask==0)
    {
        char buf[256];
        if (gethostname(buf, (int) sizeof(buf)) != 0) buf[0] = 0;
        printf("Hostname: %s\n",buf);
        fflush(stdout);
    }
}

void start_parallel(int* argc, char*** argv, int* me, int* nproc)
{
    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(argc, argv, desired, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD,me);
    MPI_Comm_size(MPI_COMM_WORLD,nproc);

    if ( (provided!=MPI_THREAD_MULTIPLE) && (*me==0) )
       fprintf(stderr,"MPI_THREAD_MULTIPLE not provided\n");
    else fprintf(stderr,"! MPI_Init_thread succeeded\n");

    GA_Initialize();
    fprintf(stderr,"! GA_Initialize succeeded\n");

    const int ma_stack = 32*1024*1024;
    const int ma_heap  =  2*1024*1024;
    MA_init(MT_DBL, ma_stack, ma_heap);

    start_cublas(*me);

    print_hostname(*me);

}

void stop_parallel(int me)
{
    if (me==0) GA_Print_stats();

    stop_cublas();

    GA_Terminate();
    fprintf(stderr,"! GA_Terminate succeeded\n");
    MPI_Finalize();
    fprintf(stderr,"! MPI_Finalize succeeded\n");
}

void zero_global(int g_a)
{
    GA_Zero(g_a);
}

int alloc_global_2d(int precision, int rows, int cols, int printMask)
{
    int status;
    int g_in;
    int ga_type;
    const int ndim = 2;
    int dims[ndim];
    int chunk[ndim];

    int pg_world;

    dims[0] = rows;
    dims[1] = cols;
    chunk[0] = -1;
    chunk[1] = -1;

    g_in= GA_Create_handle();

    GA_Set_array_name(g_in,"null");

    pg_world = GA_Pgroup_get_world();
    GA_Set_pgroup(g_in,pg_world);

    if (precision==1) ga_type=MT_REAL;
    else if (precision==2) ga_type=MT_DBL;

    GA_Set_data(g_in,ndim,dims,ga_type);
    GA_Set_chunk(g_in,chunk);

    status = GA_Allocate(g_in);
    if ((status==0) && (printMask==0))
        printf("%s: GA_Allocate failed at line %d\n",__FILE__,__LINE__);

    return g_in;
}

void copy_global(int g_in, int g_out)
{
    GA_Copy(g_in, g_out);
}

void free_global(int g_in)
{
    GA_Destroy(g_in);
}
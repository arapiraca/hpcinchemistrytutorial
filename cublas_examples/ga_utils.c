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

/* declared for use only in the following function */
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

int parallel_nproc(void)
{
    int nproc;
#ifdef MPI
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
#else
  #ifdef GA
    nproc = GA_Nnodes();
  #else
    nproc = 1;
  #endif
#endif
    return nproc;
}

int parallel_me(void)
{
    int me;
#ifdef MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
#else
  #ifdef GA
    me    = GA_Nodeid();
  #else
    me    = 0;
  #endif
#endif
    return me;
}

void parallel_sync(void)
{
#ifdef GA
    GA_Sync();
#else
  #ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
#endif

}

void start_parallel(int* argc, char*** argv, int* me, int* nproc, int armci_not_ga, int use_cuda)
{
#if defined(MPI)
    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(argc, argv, desired, &provided);
#endif

    *me    = parallel_me();
    *nproc = parallel_nproc();

#if defined(MPI)
    if (*me==0) fprintf(stderr,"! MPI_Init_thread succeeded\n");
    switch (provided)
    {
        case MPI_THREAD_MULTIPLE:
            if (*me==0) printf("%d: provided = MPI_THREAD_MULTIPLE\n",*me);
            break;

        case MPI_THREAD_SERIALIZED:
            if (*me==0) printf("%d: provided = MPI_THREAD_SERIALIZED\n",*me);
            break;

        case MPI_THREAD_FUNNELED:
            if (*me==0) printf("%d: provided = MPI_THREAD_FUNNELED\n",*me);
            break;

        case MPI_THREAD_SINGLE:
            if (*me==0) printf("%d: provided = MPI_THREAD_SINGLE\n",*me);
            break;

        default:
            if (*me==0) printf("%d: MPI_Init_thread returned an invalid value of <provided>.\n",*me);
    }
#endif

#if defined(GA)
    if (armci_not_ga>0){
        ARMCI_Init();
        fprintf(stderr,"! ARMCI_Init succeeded\n");
    } else {
        GA_Initialize();
        fprintf(stderr,"! GA_Initialize succeeded\n");
        const int ma_stack =   8*1024*1024;
        const int ma_heap  =   8*1024*1024;
        MA_init(MT_REAL, ma_stack, ma_heap);
    }
#else

#endif

#if defined(CUDA)
    if (use_cuda!=0) { start_cublas(*me); cuda_active=1; }
#endif

    print_hostname(*me);

}

void stop_parallel(int stats)
{
    int me = parallel_me();

#ifdef GA
    if (me==0 && stats!=0) GA_Print_stats();
#endif

#if defined(CUDA)
    if (cuda_active==1) stop_cublas();
#endif

#ifdef GA
    GA_Terminate();
    fprintf(stderr,"! GA_Terminate succeeded\n");
#endif

#if defined(MPI)
    MPI_Finalize();
    fprintf(stderr,"! MPI_Finalize succeeded\n");
#endif
}

void zero_global(int g_a)
{
#ifdef GA
    GA_Zero(g_a);
#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
}

int alloc_global_2d(int precision, int rows, int cols, int printMask)
{
    int g_in;
#ifdef GA
    int status;
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

    GA_Set_pgroup(g_in,GA_Pgroup_get_world());

    if (precision==1)      ga_type=MT_REAL;
    else if (precision==2) ga_type=MT_DBL;

    GA_Set_data(g_in,ndim,dims,ga_type);
    GA_Set_chunk(g_in,chunk);

    status = GA_Allocate(g_in);
    if ((status==0) && (printMask==0))
        printf("%s: GA_Allocate failed at line %d\n",__FILE__,__LINE__);
#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
    return g_in;
}

int clone_global(int g_in)
{
    int g_out;
#ifdef GA
    g_out = GA_Duplicate(g_in,"null");
    assert(g_out!=0);
#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
    return g_out;
}

void copy_global(int g_in, int g_out)
{
#ifdef GA
    GA_Copy(g_in, g_out);
#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
}

void free_global(int g_in)
{
#ifdef GA
    GA_Destroy(g_in);
#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
}

void randomize_global(int g_in)
{
    double one = 1.0;
#ifdef GA
    GA_Randomize(g_in,&one);
#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
}

void global_to_local(int g_in, void* l_out)
{
#ifdef GA
    printf("global_to_local\n");
    int i;
    int type;
    int ndim = GA_Ndim(g_in);
    int* dims = malloc(ndim*sizeof(int));
    NGA_Inquire(g_in, &type, &ndim, dims);
    for (i=0;i<ndim;i++) printf("dims[%1d] = %d\n",i,dims[i]);

    int* lo = malloc(ndim*sizeof(int));
    for (i=0;i<ndim;i++) lo[i]=0;
    for (i=0;i<ndim;i++) printf("lo[%1d] = %d\n",i,lo[i]);

    int* hi = malloc(ndim*sizeof(int));
    for (i=0;i<ndim;i++) hi[i]=dims[i]-1;
    for (i=0;i<ndim;i++) printf("hi[%1d] = %d\n",i,hi[i]);

    int* ld = malloc((ndim-1)*sizeof(int));
    for (i=0;i<(ndim-1);i++) ld[i]=dims[i];
    for (i=0;i<(ndim-1);i++) printf("ld[%1d] = %d\n",i,ld[i]);

    NGA_Get(g_in, lo, hi, l_out, ld);

    free(dims);
    free(lo);
    free(hi);
    free(ld);

#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
}

void local_to_global(void* l_in, int g_out)
{
#ifdef GA
    printf("local_to_global\n");
    int i;
    int type;
    int ndim = GA_Ndim(g_out);
    int* dims = malloc(ndim*sizeof(int));
    NGA_Inquire(g_out, &type, &ndim, dims);
    for (i=0;i<ndim;i++) printf("dims[%1d] = %d\n",i,dims[i]);

    int* lo = malloc(ndim*sizeof(int));
    for (i=0;i<ndim;i++) lo[i]=0;
    for (i=0;i<ndim;i++) printf("lo[%1d] = %d\n",i,lo[i]);

    int* hi = malloc(ndim*sizeof(int));
    for (i=0;i<ndim;i++) hi[i]=dims[i]-1;
    for (i=0;i<ndim;i++) printf("hi[%1d] = %d\n",i,hi[i]);

    int* ld = malloc((ndim-1)*sizeof(int));
    for (i=0;i<(ndim-1);i++) ld[i]=dims[i];
    for (i=0;i<(ndim-1);i++) printf("ld[%1d] = %d\n",i,ld[i]);

    NGA_Put(g_out, lo, hi, l_in, ld);

    free(dims);
    free(lo);
    free(hi);
    free(ld);

#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
}
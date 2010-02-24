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

/* internal functions */
#ifdef GA
void randomize_global_2d_float(int g_in);
void randomize_global_2d_double(int g_in);
#endif

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
    int me;
#ifdef GA
    GA_Sync();
#else
  #ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
#endif

}

void start_parallel(int* argc, char*** argv, int* me, int* nproc, int armci_not_ga)
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

    start_cublas(*me);

    print_hostname(*me);

}

void stop_parallel()
{
    int me = parallel_me();

#ifdef GA
    if (me==0) GA_Print_stats();
#endif

    stop_cublas();

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

    pg_world = GA_Pgroup_get_world();
    GA_Set_pgroup(g_in,pg_world);

    if (precision==1) ga_type=MT_REAL;
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
#ifdef GA
    int ga_type;
    int ndim;
    const int maxdim=8;
    int dims[maxdim];

    NGA_Inquire(g_in, &ga_type, &ndim, dims);

    if (ndim==2 && ga_type==MT_REAL) randomize_global_2d_float(g_in);
    if (ndim==2 && ga_type==MT_DBL)  randomize_global_2d_double(g_in);
#else
    printf("! GA not enabled\n");
    fflush(stdout);
#endif
}

#ifdef GA
void randomize_global_2d_float(int g_in)
{
    int me = parallel_me();

    const int ndim = 2;

    int lo[ndim];
    int hi[ndim];
    int rng[ndim];
    int ld[ndim-1];

    float* p_in;

    NGA_Distribution(g_in,me,lo,hi);
    NGA_Access(g_in,lo,hi,&p_in,&ld[0]);

    rng[0] = hi[0] - lo[0] + 1;
    rng[1] = hi[1] - lo[1] + 1;

    int i,j;

    for(i=0; i<rng[0]; i++){
        for(j=0; j<rng[1]; j++){
            p_in[ ld[0] * i + j ] = (float) rand() * 0.000000001;
        }
    }
    NGA_Release_update(g_in,lo,hi); /* this function does nothing as of GA 4.2 */

}

void randomize_global_2d_double(int g_in)
{
    int me = parallel_me();

    const int ndim = 2;

    int lo[ndim];
    int hi[ndim];
    int rng[ndim];
    int ld[ndim-1];

    double* p_in;

    NGA_Distribution(g_in,me,lo,hi);
    NGA_Access(g_in,lo,hi,&p_in,&ld[0]);

    rng[0] = hi[0] - lo[0] + 1;
    rng[1] = hi[1] - lo[1] + 1;

    int i,j;

    for(i=0; i<rng[0]; i++){
        for(j=0; j<rng[1]; j++){
            p_in[ ld[0] * i + j ] = (double) rand() * 0.000000001;
        }
    }
    NGA_Release_update(g_in,lo,hi); /* this function does nothing as of GA 4.2 */

}
#endif


// int multiply_globals(int precision, int rows, int cols, )
// {
//     int g_a, g_b, g_c;
// #ifdef GA
//     GA_Dgemm('T','T',dims[0],dims[0],dims[0],alpha,g_a,g_b,beta,g_c1);
// #else
//     printf("! GA not enabled\n");
//     fflush(stdout);
// #endif
// }

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

****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "dcmf.h"

#ifdef HPM_PROFILING
#include "mpi.h"
#endif

/***************************************************************/

#define DCMF_CRITICAL( call ) \
    { \
    DCMF_CriticalSection_enter(0); \
    call \
    DCMF_CriticalSection_exit(0); \
    }

/***************************************************************/

#ifdef HPM_PROFILING
    void HPM_Init(void);
    void HPM_Start(char *);
    void HPM_Stop(char *);
    void HPM_Print(void);
    void HPM_Print_flops(void);
#endif

/***************************************************************
 * DCMF_Send callback functions                                *
 ***************************************************************/

size_t rank;
size_t size;

volatile unsigned _recv_active;

static void cb_recv_new_short(void           * clientdata,
                              const DCQuad   * msginfo,
                              unsigned         count,
                              size_t           peer,
                              const char     * src,
                              size_t           bytes)
{
    printf("rank %d: inside cb_recv_new_short\n",rank);
}

static DCMF_Request_t * cb_recv_new(void                 * clientdata,
                                    const DCQuad         * msginfo,
                                    unsigned               count,
                                    size_t                 senderID,
                                    size_t                 sndlen,
                                    size_t               * rcvlen,
                                    char                ** rcvbuf,
                                    DCMF_Callback_t      * cb_info)
{
    DCMF_Request_t empty;
    printf("rank %d: inside cb_recv_new_short\n",rank);
    return &empty;
}

/***************************************************************/

int main(int argc, char **argv)
{
#ifdef HPM_PROFILING
    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);
    if ( provided != MPI_THREAD_MULTIPLE ) printf("provided != MPI_THREAD_MULTIPLE\n");

    HPM_Init();
    HPM_Start("all");
#endif

    size_t rc;

/***************************************************************/

    DCMF_CRITICAL( rc = DCMF_Messager_initialize(); )

/***************************************************************/

    rank = DCMF_Messager_rank();
    size = DCMF_Messager_size();
    if (rank==0) printf("rank %d: DCMF_Messager_size() = %d\n" , rank , size );

/***************************************************************/

    DCMF_Protocol_t           registration;
    DCMF_Send_Configuration_t configuration = { DCMF_DEFAULT_SEND_PROTOCOL, DCMF_DEFAULT_NETWORK, cb_recv_new_short, NULL, cb_recv_new, NULL };
    printf("rank %d: Trying DCMF_Send_register...\n",rank);
    DCMF_Result result = DCMF_Send_register( &registration  /* out */ ,
                                             &configuration /* in  */ );
    printf("rank %d: DCMF_Send_register returned %d\n",rank,result);

/***************************************************************/

    DCMF_Request_t   request;
    DCMF_Callback_t  cb_done;
    DCQuad           msginfo;

    size_t target = 1;
    size_t bytes  = 1;
    size_t count  = 1;
    char*  src    = "a";

    printf("rank %d: before DCMF_Send\n",rank);
    if (rank==0)
    {
        printf("rank %d: Trying DCMF_Send...\n",rank);
        result = DCMF_Send( &registration, &request, cb_done, DCMF_MATCH_CONSISTENCY,
                            target, bytes, src, &msginfo, count /* all in */ );   
        printf("rank %d: DCMF_Send returned %d\n",rank,result);
    }
    printf("rank %d: after DCMF_Send\n",rank);

/***************************************************************/

    DCMF_CRITICAL( rc = DCMF_Messager_finalize(); )

/***************************************************************/

#ifdef HPM_PROFILING
    HPM_Stop("all");
    HPM_Print();

    MPI_Finalize();
#endif

    printf("rank %d: ~ THE END ~ \n",rank);

    return(0);
}



